"""
verify_linear_baseline.py

用和 train_cnn_1d.py 完全相同的训练/评估循环，
但模型换成 Linear(80, 4)（和已有 MLP checkpoint 等价）。

如果这个脚本跑出来的 accuracy 和已有 MLP epoch30 的 0.7556 接近
→ 训练/评估代码是正确的，CNN 高准确率是模型容量导致的。

如果这个脚本也给出 99%
→ 训练代码本身有 bug，和模型无关。

用法：
    conda run -n tracin_env python verify_linear_baseline.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# ── 和 train_cnn_1d.py 完全相同的数据部分 ────────────────────────────────────
SEVERITY_MAPPING = {
    'F01': 2, 'M01': 2, 'M02': 2, 'M04': 2, 'M05': 2,
    'F03': 1,
    'F04': 0, 'M03': 0,
    'FC01': 3, 'FC02': 3, 'FC03': 3,
    'MC01': 3, 'MC02': 3, 'MC03': 3, 'MC04': 3,
}
CLASS_NAMES = ['mild', 'moderate', 'severe', 'typical']
DATA_DIR = '/srv/xw4e25/Data/torgo_fbanks_83'


class FileListDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        x = torch.tensor(np.load(path).astype(np.float32))
        return x, label


def collate(batch):
    xs, ys = zip(*batch)
    return pad_sequence(xs, batch_first=True), torch.tensor(ys, dtype=torch.long)


def build_full_file_list(data_dir):
    pairs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.npy'):
            continue
        sp = fname.split('_')[0]
        lb = SEVERITY_MAPPING.get(sp, -1)
        if lb != -1:
            pairs.append((os.path.join(data_dir, fname), lb))
    return pairs


# ── Linear(80,4) — 等价于已有的 MLP ─────────────────────────────────────────
class LinearBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(80, 4)

    def forward(self, x):
        # x: [B, 80]  (pad_sequence on (80,) tensors gives [B, 80])
        return self.fc(x)


# ── 和 train_cnn_1d.py 完全相同的训练/评估循环 ────────────────────────────────
def run(train_pairs, test_pairs, epochs=10, lr=0.0003, batch_size=32,
        device=torch.device('cpu')):

    train_ds = FileListDataset(train_pairs)
    test_ds  = FileListDataset(test_pairs)

    # 验证无交叉
    assert len({p for p,_ in train_pairs} & {p for p,_ in test_pairs}) == 0

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate)

    model     = LinearBaseline().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f'Train={len(train_ds)}  Test={len(test_ds)}')
    print(f'{"ep":>4}  {"train_loss":>10}  {"test_loss":>9}  {"acc":>6}  {"f1":>6}')

    for epoch in range(1, epochs+1):
        # TRAIN
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        # EVAL — identical to train_cnn_1d.py
        model.eval()
        te_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                te_loss += criterion(logits, y).item()
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                trues.extend(y.cpu().tolist())
        te_loss /= max(1, len(test_loader))

        acc = accuracy_score(trues, preds)
        f1  = f1_score(trues, preds, average='macro', zero_division=0)
        print(f'{epoch:4d}  {tr_loss:10.4f}  {te_loss:9.4f}  {acc:6.4f}  {f1:6.4f}')

    # per-class accuracy at end
    arr_t = np.array(trues); arr_p = np.array(preds)
    print('\nPer-class accuracy (final epoch):')
    for c, name in enumerate(CLASS_NAMES):
        mask = arr_t == c
        if mask.sum() > 0:
            ca = (arr_p[mask] == c).mean()
            print(f'  {name:10s}: {mask.sum():4d} samples  acc={ca:.4f}')


# ── main ─────────────────────────────────────────────────────────────────────
all_pairs  = build_full_file_list(DATA_DIR)
all_labels = [lb for _, lb in all_pairs]
print(f'Total: {len(all_pairs)}  dist: {Counter(all_labels)}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold_idx, (tr_idx, te_idx) in enumerate(
        skf.split(range(len(all_pairs)), all_labels), start=1):
    if fold_idx != 1:
        continue
    train_pairs = [all_pairs[i] for i in tr_idx]
    test_pairs  = [all_pairs[i] for i in te_idx]
    break

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')
print('=== Linear(80,4) baseline — fold 1, 10 epochs ===')
run(train_pairs, test_pairs, epochs=10, device=device)
