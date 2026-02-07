import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from os.path import join

# Severity mapping
severity_mapping = {
    'F01': 2, 'M01': 2, 'M02': 2, 'M04': 2, 'M05': 2,
    'F03': 1,
    'F04': 0, 'M03': 0,
    'FC01': 3, 'FC02': 3, 'FC03': 3, 'MC01': 3, 'MC02': 3, 'MC04': 3, 'MC03': 3
}

# 1. Severity Dataset
class SeverityDataset(Dataset):
    def __init__(self, data_dir, severity_mapping=severity_mapping, transform=None):
        self.data_dir = data_dir
        self.severity_mapping = severity_mapping
        self.transform = transform
        self.files = self._load_files(data_dir)

    def _load_files(self, data_dir):
        files_and_labels = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.npy'):
                    speaker_id = file.split('_')[0]
                    label = self.severity_mapping.get(speaker_id, -1)
                    if label != -1:
                        files_and_labels.append((os.path.join(root, file), label))
        return files_and_labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        embedding = np.load(file_path).astype(np.float32)
        embedding_tensor = torch.tensor(embedding)
        if self.transform:
            embedding_tensor = self.transform(embedding_tensor)
        return embedding_tensor, label

    def get_file_info(self, idx):
        """获取样本的文件信息"""
        file_path, label = self.files[idx]
        return {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'label': label
        }

def collate_fn_severity(batch):
    embeddings, labels = zip(*batch)
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_embeddings, labels


# 2. Detection Dataset
def get_binary_label(speaker_id):
    if 'C' in speaker_id:
        return 0  # Control
    else:
        return 1  # Dysarthria

class DetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.files = self._load_files(data_dir)

    def _load_files(self, data_dir):
        files_and_labels = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.npy'):
                    speaker_id = file.split('_')[0]
                    label = get_binary_label(speaker_id)
                    files_and_labels.append((os.path.join(root, file), label))
        return files_and_labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        embedding = np.load(file_path).astype(np.float32)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        if self.transform:
            embedding_tensor = self.transform(embedding_tensor)
        return embedding_tensor, label
    def get_file_info(self, idx):
        """获取样本的文件信息"""
        file_path, label = self.files[idx]
        return {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'label': label
        }

def collate_fn_detection(batch):
    embeddings, labels = zip(*batch)
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_embeddings, labels


# 3. Multitask Dataset
def get_severity_label(speaker_id):
    severity = severity_mapping.get(speaker_id, None)
    if severity == 0:
        return 0
    elif severity == 1:
        return 1
    elif severity == 2:
        return 2
    else:
        return 3

def get_dysarthric_label(speaker_id):
    if 'C' in speaker_id:
        return 0
    return 1

class MultitaskDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = self._get_files()

    def _get_files(self):
        files = []
        for file in os.listdir(self.root_dir):
            if file.endswith('.npy'):
                speaker_id = file.split('_')[0]
                files.append(join(self.root_dir, file))
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        speaker_id = file_path.split('/')[-1].split('_')[0]
        embedding = np.load(file_path).astype(np.float32)
        severity_label = get_severity_label(speaker_id)
        dysarthric_label = get_dysarthric_label(speaker_id)
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(severity_label, dtype=torch.int64), torch.tensor(dysarthric_label, dtype=torch.int64)
    def get_file_info(self, idx):
        """获取样本的文件信息"""
        file_path = self.files[idx]
        speaker_id = file_path.split('/')[-1].split('_')[0]
        return {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'speaker_id': speaker_id,
            'severity_label': get_severity_label(speaker_id),
            'dysarthric_label': get_dysarthric_label(speaker_id)
        }

def collate_fn_multitask(batch):
    embeddings, severity_labels, dysarthric_labels = zip(*batch)
    embeddings = list(embeddings)
    embeddings = pad_sequence(embeddings, batch_first=True)
    severity_labels = torch.stack(severity_labels)
    dysarthric_labels = torch.stack(dysarthric_labels)
    return embeddings, severity_labels, dysarthric_labels