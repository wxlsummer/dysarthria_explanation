"""
cnn_model.py — SeverityCNN for dysarthria severity classification

Data reality
------------
Each .npy file in torgo_fbanks_83 has shape (80,) — already mean-pooled
over time frames. There is no time dimension. The MLP handles this trivially
with a single Linear(80, 4). The CNN here applies Conv1d along the frequency
axis (80 bins), capturing local spectral patterns instead.

Task    : 4-class severity  (0=typical, 1=mild, 2=moderate, 3=severe)
Input   : [B, 80]  — mean-pooled 80-dim FBank features
Output  : [B, 4]   — class logits

Architecture
------------
  unsqueeze  →  treat [B, 80] as [B, 1, 80]  (1 channel, freq-axis sequence)
  Conv1d(1→32, k=3) + BN + ReLU               →  [B, 32, 80]
  Conv1d(32→64, k=3) + BN + ReLU              →  [B, 64, 80]
  Flatten                                      →  [B, 5120]
  FC(5120→256) + Dropout
  FC(256→128)  + Dropout
  FC(128→4)                                    →  severity logits

NOTE: fc1 input size 5120 = 64 channels × 80 freq bins.
If the FBank dimension changes from 80, update fc1 accordingly.
"""

import torch
import torch.nn as nn


class SeverityCNN(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()

        # --- 1D convolutional backbone along the frequency axis ---
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.relu  = nn.ReLU(inplace=True)

        # flatten [B, 64, 80] → [B, 5120]
        self.fc1     = nn.Linear(64 * 80, 256)
        self.fc2     = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)

        self.head_severity = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 80] from collate, or [B, T, 80] if future data has time dim
        if x.ndim == 3:
            x = x.mean(dim=1)              # [B, 80]
        x = x.unsqueeze(1)                 # [B, 1, 80]
        x = self.relu(self.bn1(self.conv1(x)))   # [B, 32, 80]
        x = self.relu(self.bn2(self.conv2(x)))   # [B, 64, 80]
        x = x.view(x.size(0), -1)               # [B, 5120]
        x = self.dropout(self.relu(self.fc1(x))) # [B, 256]
        x = self.dropout(self.relu(self.fc2(x))) # [B, 128]
        return self.head_severity(x)             # [B, 4]
