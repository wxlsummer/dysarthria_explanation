# ==============================================================
# 废物文件！中间过渡版本，test_idx 硬编码为 0，没有 mode 机制，已被 tracin_main.py 完全取代。
# ==============================================================

import torch
import os
import yaml
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import pickle

from model import MultiTaskMLP
from torgo_dataloaders import (
    SeverityDataset, DetectionDataset, MultitaskDataset,
    collate_fn_severity, collate_fn_detection, collate_fn_multitask
)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_dataset_and_collate(task, data_dir):
    """根据任务类型返回对应的dataset和collate_fn"""
    if task == 'severity':
        dataset = SeverityDataset(data_dir)
        collate_fn = collate_fn_severity
    elif task == 'dysarthria':
        dataset = DetectionDataset(data_dir)
        collate_fn = collate_fn_detection
    elif task == 'multitask':
        dataset = MultitaskDataset(data_dir)
        collate_fn = collate_fn_multitask
    else:
        raise ValueError(f"Unknown task: {task}")
    return dataset, collate_fn

def get_fold_split(dataset, fold_idx, n_splits, random_state, task):
    """重现训练时的fold split"""
    # 获取targets
    if task == 'multitask':
        targets = [dataset[i][1].item() for i in range(len(dataset))]  # severity label
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for current_fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), targets), start=1):
        if current_fold == fold_idx:
            return list(train_idx), list(test_idx)
    
    raise ValueError(f"fold_idx {fold_idx} 超出范围")

def load_checkpoint_paths(model_base_path, fold_idx, total_epochs):
    """加载所有checkpoint路径"""
    checkpoint_paths = []
    fold_dir = os.path.join(model_base_path, f'fold_{fold_idx}')
    
    if not os.path.exists(fold_dir):
        print(f"错误: fold目录不存在: {fold_dir}")
        return checkpoint_paths
    
    for epoch in range(1, total_epochs + 1):
        path = os.path.join(fold_dir, f'epoch_{epoch:03d}.pth')
        if os.path.exists(path):
            checkpoint_paths.append(path)
        else:
            print(f"警告: {path} 不存在")
    
    return checkpoint_paths

def main():
    # 加载配置
    config = load_config()
    
    # 提取配置
    DATA_DIR = config['data_dir']
    MODEL_BASE_PATH = config['model_base_path']
    TASK = config['task']
    DIM = config['dim']
    N_SPLITS = config['n_splits']
    RANDOM_STATE = config['random_state']
    FOLD_IDX = config['fold_idx']
    TOTAL_EPOCHS = config['total_epochs']
    DEVICE = config['device']
    BATCH_SIZE = config['batch_size']
    OUTPUT_DIR = config['output_dir']
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 1. 加载数据集
    print("=" * 60)
    print("步骤1: 加载数据集")
    print("=" * 60)
    full_dataset, collate_fn = get_dataset_and_collate(TASK, DATA_DIR)
    print(f"任务类型: {TASK}")
    print(f"总样本数: {len(full_dataset)}\n")
    
    # 2. 获取fold split
    print("=" * 60)
    print(f"步骤2: 获取fold {FOLD_IDX} 的split")
    print("=" * 60)
    train_indices, test_indices = get_fold_split(full_dataset, FOLD_IDX, N_SPLITS, RANDOM_STATE, TASK)
    print(f"训练集样本数: {len(train_indices)}")
    print(f"测试集样本数: {len(test_indices)}\n")
    
    # 创建subset
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 3. 创建DataLoader
    print("=" * 60)
    print("步骤3: 创建DataLoader")
    print("=" * 60)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"训练loader: {len(train_loader)} batches")
    print(f"测试loader: {len(test_loader)} batches\n")
    
    # 4. 加载checkpoint路径
    print("=" * 60)
    print("步骤4: 查找checkpoint文件")
    print("=" * 60)
    checkpoint_paths = load_checkpoint_paths(MODEL_BASE_PATH, FOLD_IDX, TOTAL_EPOCHS)
    print(f"找到 {len(checkpoint_paths)} 个checkpoint\n")
    
    if len(checkpoint_paths) == 0:
        print("错误: 没有找到任何checkpoint文件")
        return
    
    # 5. 测试加载第一个checkpoint
    print("=" * 60)
    print("步骤5: 测试加载checkpoint")
    print("=" * 60)
    model = MultiTaskMLP(dim=DIM, task=TASK)
    checkpoint = torch.load(checkpoint_paths[0], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"成功加载: {os.path.basename(checkpoint_paths[0])}")
    print(f"Checkpoint信息:")
    print(f"  - Fold: {checkpoint['fold']}")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Task: {checkpoint['task_type']}")
    print(f"  - Dim: {checkpoint['dim']}\n")
    
    # 6. 选择测试样本
    print("=" * 60)
    print("步骤6: 选择测试样本")
    print("=" * 60)
    test_idx = 0
    
    if TASK == 'multitask':
        test_emb, severity_label, dysarthria_label = test_dataset[test_idx]
        print(f"Severity标签: {severity_label.item()}")
        print(f"Dysarthria标签: {dysarthria_label.item()}")
    else:
        test_emb, test_label = test_dataset[test_idx]
        print(f"真实标签: {test_label}")
    
    print(f"测试样本索引 (在test_dataset中): {test_idx}")
    print(f"数据形状: {test_emb.shape}")
    
    # 模型预测
    test_emb_batch = test_emb.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(test_emb_batch)
        if TASK == 'multitask':
            pred_severity = pred[0].argmax(dim=1).item()
            pred_dysarthria = pred[1].argmax(dim=1).item()
            print(f"Severity预测: {pred_severity}")
            print(f"Dysarthria预测: {pred_dysarthria}")
        else:
            pred_class = pred.argmax(dim=1).item()
            print(f"模型预测: {pred_class}")
    print()
    
    print("=" * 60)
    print("准备工作完成！下一步将实现TracIn计算")
    print("=" * 60)

if __name__ == '__main__':
    main()