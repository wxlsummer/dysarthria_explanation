"""
run_mlp_reweight_experiment.py

目的
----
用 MLP fold 1 的 self-influence 分数，验证以下假设：
高 self-influence 样本（top 5% / top 10%）是标签可靠性低的噪声样本，
对它们降权或移除后，模型在少数类（mild、moderate、severe）上的分类能力应该提升。

实验条件（共 41 组，统一从 baseline 起点对比）
------------------------------------------------
  百分比档位：top 5% / 10% / 15% / 20%
  每个档位包含：
    hard_remove      — 直接移除 top k% high self-influence 样本
    random_remove    — 随机移除等数量样本（对照组）
    down_0.2/0.4/0.6/0.8  — 下加权（假设高 SI = 噪声标签）
    up_1.2/1.4/1.6/1.8    — 上加权（假设高 SI = 难例）
  共 4 档 × 10 条件 + 1 baseline = 41 条件

输入
----
  self-influence:  tracin_results/selfinfluence_fold1/self_influences.csv
  features:        /srv/xw4e25/Data/torgo_fbanks_83/*.npy  (shape 80,)
  fold split:      StratifiedKFold(n_splits=5, shuffle=True, random_state=42), fold 1

输出
----
  控制台打印每个条件的 overall acc + per-class acc 汇总表

用法
----
  conda run -n tracin_env python run_mlp_reweight_experiment.py
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from model import MultiTaskMLP
from torgo_dataloaders import SeverityDataset

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_DIR        = '/srv/xw4e25/Data/torgo_fbanks_83'
SELF_INF_CSV    = 'tracin_results/selfinfluence_fold1/self_influences.csv'
DEVICE          = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
EPOCHS          = 30
LR              = 0.0003
BATCH_SIZE      = 32
DIM             = 80
SEED            = 42

SEVERITY_MAPPING = {
    'F01': 2, 'M01': 2, 'M02': 2, 'M04': 2, 'M05': 2,
    'F03': 1,
    'F04': 0, 'M03': 0,
    'FC01': 3, 'FC02': 3, 'FC03': 3,
    'MC01': 3, 'MC02': 3, 'MC03': 3, 'MC04': 3,
}
CLASS_NAMES = ['mild', 'moderate', 'severe', 'typical']

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────

class WeightedFileDataset(Dataset):
    """
    (path, label, weight) 三元组构成的 dataset。
    weight 用于在训练时对 per-sample loss 加权。
    """
    def __init__(self, triples):
        self.triples = triples  # [(path, label, weight), ...]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        path, label, weight = self.triples[idx]
        x = torch.tensor(np.load(path).astype(np.float32))
        return x, label, weight


def collate_weighted(batch):
    xs, ys, ws = zip(*batch)
    return (
        pad_sequence(xs, batch_first=True),
        torch.tensor(ys, dtype=torch.long),
        torch.tensor(ws, dtype=torch.float32),
    )


# ── 数据准备 ───────────────────────────────────────────────────────────────────
# 用 SeverityDataset（os.walk，不排序），保证文件顺序和原 MLP 训练完全一致，
# 从而 StratifiedKFold fold 1 划出的 train/test 和原 baseline 是同一批样本。

def get_fold1_split(data_dir):
    ds = SeverityDataset(data_dir)                        # 同原 MLP 训练的顺序
    all_pairs = ds.files                                  # [(path, label), ...]
    labels    = [lb for _, lb in all_pairs]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_idx, (tr, te) in enumerate(skf.split(range(len(all_pairs)), labels), start=1):
        if fold_idx == 1:
            train_pairs = [all_pairs[i] for i in tr]
            test_pairs  = [all_pairs[i] for i in te]
            return train_pairs, test_pairs
    raise RuntimeError("fold 1 not found")


# ── 训练 & 评估 ────────────────────────────────────────────────────────────────

def train_and_eval(train_triples, test_pairs, condition_name):
    """
    train_triples: [(path, label, weight), ...]
    test_pairs:    [(path, label), ...]
    """
    train_ds = WeightedFileDataset(train_triples)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_weighted)

    # test loader — no weighting needed
    test_ds = WeightedFileDataset([(p, lb, 1.0) for p, lb in test_pairs])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_weighted)

    model     = MultiTaskMLP(dim=DIM, task='severity').to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(reduction='none')  # per-sample loss for weighting

    for epoch in range(EPOCHS):
        model.train()
        for x, y, w in train_loader:
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = (criterion(logits, y) * w).mean()  # weighted mean
            loss.backward()
            optimizer.step()

    # ── evaluate ──────────────────────────────────────────────────────────────
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            trues.extend(y.cpu().tolist())

    arr_t = np.array(trues)
    arr_p = np.array(preds)
    overall = accuracy_score(trues, preds)
    f1      = f1_score(trues, preds, average='macro', zero_division=0)

    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = arr_t == c
        per_class[name] = float((arr_p[mask] == c).mean()) if mask.sum() > 0 else float('nan')

    print(f"  {condition_name:<22s}  acc={overall:.4f}  f1={f1:.4f}  "
          + "  ".join(f"{name[:3]}={per_class[name]:.3f}" for name in CLASS_NAMES))

    return {'condition': condition_name, 'accuracy': overall, 'f1': f1, **per_class}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    # 1. 构建 fold 1 split（和原 MLP 训练相同的文件顺序）
    train_pairs, test_pairs = get_fold1_split(DATA_DIR)
    n_train = len(train_pairs)
    print(f"Fold 1 — train: {n_train}  test: {len(test_pairs)}")

    # 2. 加载 self-influence 分数，建立 filename → score 的映射
    si_df = pd.read_csv(SELF_INF_CSV)
    si_map = dict(zip(si_df['filename'], si_df['self_influence']))

    # 3. 给每个训练样本打上 self-influence 分数
    train_scores = []
    for path, lb in train_pairs:
        fname = os.path.basename(path)
        score = si_map.get(fname, 0.0)
        train_scores.append(score)
    train_scores = np.array(train_scores)

    # 4. 确定 top 5% / 10% / 15% / 20% 的高分 index 集合
    PCTS = [5, 10, 15, 20]
    high_idx = {}
    for pct in PCTS:
        threshold = np.percentile(train_scores, 100 - pct)
        idx = set(np.where(train_scores >= threshold)[0].tolist())
        high_idx[pct] = idx
        print(f"Top {pct:2d}%: {len(idx):4d} samples  (score >= {threshold:.1f})")

    # 随机移除对照：从最低分的 80% 样本里随机选，避免和高分集重叠
    low_idx = [i for i in range(n_train) if i not in high_idx[20]]
    rand_idx = {}
    for pct in PCTS:
        rand_idx[pct] = set(random.sample(low_idx, len(high_idx[pct])))

    # ── 定义所有实验条件 ──────────────────────────────────────────────────────

    def make_triples(remove_set=None, weight_map=None):
        triples = []
        for i, (path, lb) in enumerate(train_pairs):
            if remove_set and i in remove_set:
                continue
            w = weight_map.get(i, 1.0) if weight_map else 1.0
            triples.append((path, lb, w))
        return triples

    DOWN_WEIGHTS = [0.2, 0.4, 0.6, 0.8]
    UP_WEIGHTS   = [1.2, 1.4, 1.6, 1.8]

    conditions = [('baseline', make_triples())]

    for pct in PCTS:
        tag = f'top{pct}%'
        hidx = high_idx[pct]
        ridx = rand_idx[pct]

        # 移除实验
        conditions.append((f'hard_remove_{tag}',   make_triples(remove_set=hidx)))
        conditions.append((f'random_remove_{tag}',  make_triples(remove_set=ridx)))

        # 下加权
        for w in DOWN_WEIGHTS:
            conditions.append((f'down_{w}_{tag}', make_triples(weight_map={i: w for i in hidx})))

        # 上加权
        for w in UP_WEIGHTS:
            conditions.append((f'up_{w}_{tag}',   make_triples(weight_map={i: w for i in hidx})))

    # ── 运行所有条件 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*85}")
    print(f"  {'condition':<22s}  {'acc':>6}  {'f1':>6}  "
          + "  ".join(f"{n[:3]:>5}" for n in CLASS_NAMES))
    print(f"{'─'*85}")

    results = []
    for name, triples in conditions:
        print(f"  running {name} ({len(triples)} train samples)...", end='\r')
        r = train_and_eval(triples, test_pairs, name)
        results.append(r)

    # ── 汇总表 + 保存 ─────────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print("SUMMARY TABLE")
    print(f"{'='*85}")
    header = f"{'condition':<22s}  {'acc':>6}  {'f1':>6}  " + "  ".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    print('─' * 85)
    base = results[0]
    for r in results:
        diff_acc = r['accuracy'] - base['accuracy']
        diff_f1  = r['f1']      - base['f1']
        row = (f"{r['condition']:<22s}  {r['accuracy']:.4f}  {r['f1']:.4f}  "
               + "  ".join(f"{r[n]:>8.4f}" for n in CLASS_NAMES))
        if r['condition'] != 'baseline':
            row += f"  (Δacc={diff_acc:+.5f}  Δf1={diff_f1:+.5f})"
        print(row)

    # 保存结果到 CSV
    out_dir = 'tracin_results/reweight_experiment'
    os.makedirs(out_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    # 加上相对 baseline 的差值列
    for col in ['accuracy', 'f1'] + CLASS_NAMES:
        results_df[f'delta_{col}'] = results_df[col] - results_df.loc[0, col]
    csv_path = os.path.join(out_dir, 'results_fold1.csv')
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\nResults saved → {csv_path}")


if __name__ == '__main__':
    main()
