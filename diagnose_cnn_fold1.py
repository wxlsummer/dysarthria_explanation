"""
diagnose_cnn_fold1.py — 诊断 CNN fold 1 准确率是否异常

检查项：
  1. Fold 1 的 train/test 集大小、说话人分布
  2. 用已保存的 epoch_030.pth 重新在 test set 上算一次，看和 checkpoint 里存的是否一致
  3. 用同一个 checkpoint 在 train set 上也算一次 —— 如果 train acc ≈ test acc ≈ 99%，说明过拟合/泄漏
  4. 打印完整混淆矩阵，看预测是否退化成单一类别

用法：
    conda run -n tracin_env python diagnose_cnn_fold1.py --config configs/config_cnn_fold1.yaml
"""

import argparse
import os
import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from collections import Counter

from cnn_model_1d import SeverityCNN
from torgo_dataloaders import SeverityDataset, collate_fn_severity

severity_mapping = {
    'F01': 2, 'M01': 2, 'M02': 2, 'M04': 2, 'M05': 2,
    'F03': 1,
    'F04': 0, 'M03': 0,
    'FC01': 3, 'FC02': 3, 'FC03': 3, 'MC01': 3, 'MC02': 3, 'MC04': 3, 'MC03': 3,
}
id_to_name = {0: 'mild', 1: 'moderate', 2: 'severe', 3: 'typical'}


def evaluate(model, loader, device, label):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    acc = accuracy_score(all_true, all_preds)

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  Total samples : {len(all_true)}")
    print(f"  Overall acc   : {acc:.4f}")

    # per-class accuracy
    print("  Per-class accuracy:")
    for c in range(4):
        mask = (all_true == c)
        n = mask.sum()
        if n > 0:
            ca = (all_preds[mask] == c).mean()
            print(f"    {id_to_name[c]:10s} (label {c}): {n:4d} samples  →  acc={ca:.4f}")
        else:
            print(f"    {id_to_name[c]:10s} (label {c}):    0 samples")

    # confusion matrix
    cm = confusion_matrix(all_true, all_preds, labels=[0, 1, 2, 3])
    print("  Confusion matrix (rows=true, cols=pred):")
    header = "              " + "  ".join(f"pred{c}" for c in range(4))
    print("  " + header)
    for r in range(4):
        row_str = "  ".join(f"{cm[r,c]:6d}" for c in range(4))
        print(f"  true{r}({id_to_name[r][:3]}): {row_str}")

    # prediction distribution — if model always predicts one class, this shows it
    pred_counts = Counter(all_preds.tolist())
    print("  Prediction distribution:", {id_to_name[k]: v for k, v in sorted(pred_counts.items())})

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_cnn_fold1.yaml')
    parser.add_argument('--epoch', type=int, default=30)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 1. reconstruct fold 1 split ────────────────────────────────────────────
    dataset = SeverityDataset(cfg['data_dir'])
    targets = [dataset[i][1] for i in range(len(dataset))]
    print(f"\nDataset total: {len(dataset)} samples")
    dist = Counter(targets)
    print("Class distribution:", {id_to_name[k]: v for k, v in sorted(dist.items())})

    skf = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=cfg['random_state'])

    train_idx, test_idx = None, None
    for fold_idx, (tr, te) in enumerate(skf.split(range(len(dataset)), targets), start=1):
        if fold_idx == 1:
            train_idx, test_idx = tr, te
            break

    print(f"\nFold 1 — train: {len(train_idx)}  test: {len(test_idx)}")

    # speaker breakdown in test set
    print("\nTest set speaker breakdown:")
    test_speakers = Counter()
    for idx in test_idx:
        fname = os.path.basename(dataset.files[idx][0])
        spk = fname.split('_')[0]
        test_speakers[spk] += 1
    for spk, cnt in sorted(test_speakers.items()):
        lbl = severity_mapping.get(spk, -1)
        print(f"  {spk:6s}  ({id_to_name.get(lbl,'?'):8s})  {cnt} utterances")

    # ── 2. load checkpoint ─────────────────────────────────────────────────────
    ckpt_path = os.path.join(cfg['ckpt_root'], f"fold_1/epoch_{args.epoch:03d}.pth")
    print(f"\nLoading: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"Stored in checkpoint — train_loss={ck['train_loss']:.4f}  test_loss={ck['test_loss']:.4f}")
    print(f"Stored accuracy: {ck['metrics']['accuracy']:.4f}")

    model = SeverityCNN().to(device)
    model.load_state_dict(ck['model_state_dict'])

    # ── 3. build loaders ───────────────────────────────────────────────────────
    bs = cfg['batch_size']
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=bs,
                              shuffle=False, collate_fn=collate_fn_severity)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=bs,
                              shuffle=False, collate_fn=collate_fn_severity)

    # ── 4. evaluate ────────────────────────────────────────────────────────────
    test_acc  = evaluate(model, test_loader,  device, label=f"TEST  SET  (epoch {args.epoch})")
    train_acc = evaluate(model, train_loader, device, label=f"TRAIN SET  (epoch {args.epoch})  ← sanity check")

    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"  Test  accuracy : {test_acc:.4f}  (stored: {ck['metrics']['accuracy']:.4f})")
    print(f"  Train accuracy : {train_acc:.4f}")
    if abs(test_acc - ck['metrics']['accuracy']) > 0.001:
        print("  *** MISMATCH: re-evaluated acc differs from stored acc! ***")
    if train_acc - test_acc > 0.05:
        print("  INFO: train >> test  →  model overfits but test eval is independent")
    elif abs(train_acc - test_acc) < 0.02:
        print("  WARNING: train acc ≈ test acc  →  likely utterance-level speaker leakage")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
