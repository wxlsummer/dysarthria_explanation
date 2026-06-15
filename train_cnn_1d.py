"""
train_cnn_1d.py — train SeverityCNN on TORGO 80-dim FBank features

Usage
-----
    python train_cnn_1d.py --config configs/config_cnn_fold1.yaml

Split design
------------
  StratifiedKFold(n_splits=5, shuffle=True, random_state=42), same as MLP pipeline.
  The fold indices are used to split the raw FILE LIST into two separate lists
  (train_files / test_files) BEFORE any Dataset is created.
  Two completely independent Dataset objects are built from these two lists.
  There is no shared parent dataset, no Subset, no index indirection.

Checkpoint format (identical to MLP checkpoints for TracIn compatibility)
--------------------------------------------------------------------------
    fold, epoch, task_type, feature, dim,
    model_state_dict, optimizer_state_dict,
    train_loss, test_loss, metrics, is_best
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from cnn_model_1d import SeverityCNN

# ── label mapping (must match torgo_dataloaders.py severity_mapping) ──────────
SEVERITY_MAPPING = {
    'F01': 2, 'M01': 2, 'M02': 2, 'M04': 2, 'M05': 2,   # severe
    'F03': 1,                                               # moderate
    'F04': 0, 'M03': 0,                                    # mild
    'FC01': 3, 'FC02': 3, 'FC03': 3,                      # typical/control
    'MC01': 3, 'MC02': 3, 'MC03': 3, 'MC04': 3,
}
# 0=mild  1=moderate  2=severe  3=typical(control)
CLASS_NAMES = ['mild', 'moderate', 'severe', 'typical']


# ─────────────────────────────────────────────────────────────────────────────
# Dataset that owns its own explicit file list — no shared parent, no Subset
# ─────────────────────────────────────────────────────────────────────────────

class FileListDataset(Dataset):
    """Takes an explicit list of (file_path, label) pairs. Completely self-contained."""

    def __init__(self, file_label_pairs: list):
        self.pairs = file_label_pairs  # [(path, label), ...]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        x = torch.tensor(np.load(path).astype(np.float32))  # shape [80]
        return x, label


def collate(batch):
    xs, ys = zip(*batch)
    return pad_sequence(xs, batch_first=True), torch.tensor(ys, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Build the full file+label list from data_dir
# ─────────────────────────────────────────────────────────────────────────────

def build_full_file_list(data_dir: str) -> list:
    """Return sorted list of (file_path, label) for every valid .npy file."""
    pairs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.npy'):
            continue
        speaker_id = fname.split('_')[0]
        label = SEVERITY_MAPPING.get(speaker_id, -1)
        if label == -1:
            continue
        pairs.append((os.path.join(data_dir, fname), label))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, fold_idx, epoch_1based,
                    train_loss, test_loss, metrics, ckpt_dir, feature, dim):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"epoch_{epoch_1based:03d}.pth")
    torch.save({
        "fold":                 fold_idx,
        "epoch":                epoch_1based,
        "task_type":            "severity",
        "feature":              feature,
        "dim":                  dim,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss":           float(train_loss),
        "test_loss":            float(test_loss),
        "metrics":              metrics,
        "is_best":              False,
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(train_dataset: FileListDataset,
                   test_dataset:  FileListDataset,
                   fold_idx: int, cfg: dict, device):

    # ── sanity check: train and test must share zero files ────────────────────
    train_paths = {p for p, _ in train_dataset.pairs}
    test_paths  = {p for p, _ in test_dataset.pairs}
    overlap = train_paths & test_paths
    assert len(overlap) == 0, f"BUG: {len(overlap)} files appear in both train and test!"

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}")
    print(f"  Train : {len(train_dataset)} samples  ({len(train_paths)} unique files)")
    print(f"  Test  : {len(test_dataset)}  samples  ({len(test_paths)}  unique files)")
    print(f"  Overlap between train and test files: {len(overlap)}  ← must be 0")
    print(f"{'='*60}")

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                              shuffle=True,  collate_fn=collate)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg['batch_size'],
                              shuffle=False, collate_fn=collate)

    model     = SeverityCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    ckpt_dir  = os.path.join(cfg['ckpt_root'], f"fold_{fold_idx}")
    epochs    = cfg['epochs']

    for epoch in range(epochs):
        t0 = time.time()

        # ── TRAIN ─────────────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        train_loss = train_loss_sum / max(1, len(train_loader))

        # ── EVALUATE on test_loader only ──────────────────────────────────────
        model.eval()
        test_loss_sum = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                test_loss_sum += criterion(logits, labels).item()
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_true.extend(labels.cpu().tolist())
        test_loss = test_loss_sum / max(1, len(test_loader))

        all_true_arr  = np.array(all_true)
        all_preds_arr = np.array(all_preds)
        per_class_acc = {
            name: float((all_preds_arr[all_true_arr == c] == c).mean())
            if (all_true_arr == c).sum() > 0 else 0.0
            for c, name in enumerate(CLASS_NAMES)
        }
        metrics = {
            "accuracy":      float(accuracy_score(all_true, all_preds)),
            "precision":     float(precision_score(all_true, all_preds, average="macro", zero_division=0)),
            "recall":        float(recall_score(all_true, all_preds,    average="macro", zero_division=0)),
            "f1_macro":      float(f1_score(all_true, all_preds,        average="macro", zero_division=0)),
            "per_class_acc": per_class_acc,
        }

        save_checkpoint(
            model=model, optimizer=optimizer,
            fold_idx=fold_idx, epoch_1based=epoch + 1,
            train_loss=train_loss, test_loss=test_loss,
            metrics=metrics, ckpt_dir=ckpt_dir,
            feature=cfg['feature'], dim=cfg['dim'],
        )

        per_class_str = "  ".join(
            f"{name[:3]}={acc:.2f}" for name, acc in per_class_acc.items()
        )
        print(f"  [{epoch+1:2d}/{epochs}]  "
              f"train={train_loss:.4f}  test={test_loss:.4f}  "
              f"f1={metrics['f1_macro']:.4f}  acc={metrics['accuracy']:.4f}  "
              f"| {per_class_str}  ({time.time()-t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config_cnn_fold1.yaml')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("SeverityCNN Training")
    print("=" * 60)
    print(f"Config  : {args.config}")
    print(f"Device  : {device}")
    print(f"Data    : {cfg['data_dir']}")
    print(f"Output  : {cfg['ckpt_root']}")
    print(f"Folds   : {cfg.get('folds', [1])}")
    print(f"Epochs  : {cfg['epochs']}  |  LR: {cfg['learning_rate']}  (AdamW)")

    # ── Step 1: build the full sorted file list ───────────────────────────────
    all_pairs = build_full_file_list(cfg['data_dir'])
    all_labels = [label for _, label in all_pairs]
    print(f"\nTotal samples loaded: {len(all_pairs)}")
    from collections import Counter
    dist = Counter(all_labels)
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:10s} (label {c}): {dist[c]}")

    # ── Step 2: StratifiedKFold on the file list ──────────────────────────────
    skf = StratifiedKFold(
        n_splits=cfg['n_splits'],
        shuffle=True,
        random_state=cfg['random_state'],
    )

    folds_to_run = cfg.get('folds', [1])

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(range(len(all_pairs)), all_labels), start=1):

        if fold_idx not in folds_to_run:
            continue

        # ── Step 3: hard-split the file list into two independent lists ───────
        train_pairs = [all_pairs[i] for i in train_idx]   # list of (path, label)
        test_pairs  = [all_pairs[i] for i in test_idx]    # list of (path, label)

        # ── Step 4: build two completely independent Dataset objects ──────────
        train_dataset = FileListDataset(train_pairs)
        test_dataset  = FileListDataset(test_pairs)

        train_one_fold(train_dataset, test_dataset, fold_idx, cfg, device)

    print("\nDone. All checkpoints saved.")


if __name__ == '__main__':
    main()
