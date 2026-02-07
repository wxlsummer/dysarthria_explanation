import torch
import numpy as np
import os
from torch.utils.data import Dataset

# Severity mapping
severity_mapping = {
    'F01': 2, 'M01': 2, 'M02': 2, 'M04': 2, 'M05': 2,  # Severe -> 2
    'F03': 1,  # Moderate -> 1
    'F04': 0, 'M03': 0,  # Mild -> 0
    'FC01': 3, 'FC02': 3, 'FC03': 3, 'MC01': 3, 'MC02': 3, 'MC04': 3, 'MC03': 3
}

class SeverityDataset(Dataset):
    def __init__(self, data_dir, severity_mapping=severity_mapping):
        self.data_dir = data_dir
        self.severity_mapping = severity_mapping
        self.files = self._load_files(data_dir)

    def _load_files(self, data_dir):
        files_and_labels = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.npy'):
                    speaker_id = file.split('_')[0]
                    label = self.severity_mapping.get(speaker_id, -1)
                    if label != -1:
                        file_path = os.path.join(root, file)
                        files_and_labels.append((file_path, label))
        return files_and_labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        embedding = np.load(file_path).astype(np.float32)
        embedding_tensor = torch.tensor(embedding)
        return embedding_tensor, label, file_path  # 返回文件路径方便追踪

# 加载数据
data_dir = '/srv/xw4e25/Data/fbanks_83'
dataset = SeverityDataset(data_dir)

print(f"数据集总数: {len(dataset)}")
print(f"\n前5个样本:")
for i in range(min(5, len(dataset))):
    emb, label, path = dataset[i]
    print(f"  样本 {i}: 形状={emb.shape}, 标签={label}, 文件={os.path.basename(path)}")

# 随机选择一个测试样本
import random
random.seed(42)
test_idx = random.randint(0, len(dataset)-1)
test_emb, test_label, test_path = dataset[test_idx]

print(f"\n选择的测试样本:")
print(f"  索引: {test_idx}")
print(f"  文件: {os.path.basename(test_path)}")
print(f"  标签: {test_label}")
print(f"  形状: {test_emb.shape}")

# 保存测试样本信息
import pickle
with open('test_sample_info.pkl', 'wb') as f:
    pickle.dump({
        'idx': test_idx,
        'path': test_path,
        'label': test_label
    }, f)
print("\n测试样本信息已保存到 test_sample_info.pkl")