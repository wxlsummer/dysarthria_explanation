# ==============================================================
# 废物文件！单样本版本加了矩阵乘法加速，但仍被 tracin_main.py 完全取代。
# ==============================================================

import torch
import os
import yaml
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import pickle
from tqdm import tqdm

from model import MultiTaskMLP
from torgo_dataloaders import (
    SeverityDataset, DetectionDataset, MultitaskDataset,
    collate_fn_severity, collate_fn_detection, collate_fn_multitask
)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_dataset_and_collate(task, data_dir):
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
    if task == 'multitask':
        targets = [dataset[i][1].item() for i in range(len(dataset))]
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for current_fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), targets), start=1):
        if current_fold == fold_idx:
            return list(train_idx), list(test_idx)
    
    raise ValueError(f"fold_idx {fold_idx} 超出范围")

def load_checkpoint_paths(model_base_path, fold_idx, total_epochs):
    checkpoint_paths = []
    fold_dir = os.path.join(model_base_path, f'fold_{fold_idx}')
    
    if not os.path.exists(fold_dir):
        print(f"错误: fold目录不存在: {fold_dir}")
        return checkpoint_paths
    
    for epoch in range(1, total_epochs + 1):
        path = os.path.join(fold_dir, f'epoch_{epoch:03d}.pth')
        if os.path.exists(path):
            checkpoint_paths.append(path)
    
    return checkpoint_paths

def compute_loss(model, inputs, labels, task):
    """计算损失"""
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    outputs = model(inputs)
    
    if task == 'multitask':
        loss = criterion(outputs[0], labels)
    else:
        loss = criterion(outputs, labels)
    
    return loss

def compute_single_gradient(model, inputs, labels, task):
    """计算单个样本的梯度"""
    model.zero_grad()
    
    loss = compute_loss(model, inputs, labels, task)
    loss = loss.mean()
    loss.backward()
    
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.view(-1).clone())
    
    return torch.cat(gradients)

def compute_all_train_gradients(model, train_dataset, task, device, batch_size=1):
    """
    计算所有训练样本的梯度
    返回: [num_train, num_params] 的梯度矩阵
    """
    all_gradients = []
    
    # 使用 DataLoader 但 batch_size=1（因为每个样本要单独算梯度）
    for idx in tqdm(range(len(train_dataset)), desc="计算训练集梯度", leave=False):
        if task == 'multitask':
            sample, severity_label, _ = train_dataset[idx]
            label = severity_label.item()
        else:
            sample, label = train_dataset[idx]
        
        sample = sample.unsqueeze(0).to(device)
        label = torch.tensor([label], dtype=torch.long).to(device)
        
        grad = compute_single_gradient(model, sample, label, task)
        all_gradients.append(grad.cpu())
    
    return torch.stack(all_gradients)  # [num_train, num_params]

def compute_tracin_influence_optimized(
    train_dataset,
    test_sample,
    test_label,
    checkpoint_paths,
    task,
    dim,
    device,
    output_dir=None,
    fold_idx=None,
    test_idx=None,
    use_cache=True
):
    """
    优化版的 TracIn 影响力计算
    - 支持断点续传
    - 使用矩阵运算加速
    """
    num_train = len(train_dataset)
    
    # 检查缓存
    cache_path = None
    if use_cache and output_dir and fold_idx is not None and test_idx is not None:
        cache_path = os.path.join(output_dir, 
                                 f'tracin_cache_fold{fold_idx}_test{test_idx}.pkl')
        
        if os.path.exists(cache_path):
            print(f"\n发现缓存: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            
            if cache['num_checkpoints'] == len(checkpoint_paths):
                print("✅ 缓存完整，直接返回结果")
                return cache['influences']
            else:
                print(f"⚠️  缓存不完整，重新计算")
    
    influences = torch.zeros(num_train)
    
    # 1. 计算测试样本梯度（只算一次）
    print("\n计算测试样本梯度...")
    model = MultiTaskMLP(dim=dim, task=task).to(device)
    last_ckpt = torch.load(checkpoint_paths[-1], map_location=device, weights_only=False)
    model.load_state_dict(last_ckpt['model_state_dict'])
    model.train()
    
    test_sample = test_sample.unsqueeze(0).to(device)
    test_label_tensor = torch.tensor([test_label], dtype=torch.long).to(device)
    test_grad = compute_single_gradient(model, test_sample, test_label_tensor, task)
    test_grad = test_grad.cpu()
    
    print(f"梯度维度: {test_grad.shape[0]:,}")
    
    # 2. 遍历所有 checkpoint
    print(f"\n计算 TracIn 影响力（共 {len(checkpoint_paths)} 个 checkpoint）...")
    
    for ckpt_idx, ckpt_path in enumerate(tqdm(checkpoint_paths, desc="Checkpoints")):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()
        
        # ✅ 批量计算所有训练样本的梯度
        train_grads = compute_all_train_gradients(model, train_dataset, task, device)
        
        # ✅ 矩阵向量乘法：一次性计算所有点积
        batch_influences = torch.mv(train_grads, test_grad)  # [num_train]
        influences += batch_influences
        
        # 可选：每个 checkpoint 保存一次
        if cache_path and (ckpt_idx + 1) % 5 == 0:  # 每5个checkpoint保存一次
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'influences': influences.numpy(),
                    'completed_checkpoints': ckpt_idx + 1,
                    'num_checkpoints': len(checkpoint_paths)
                }, f)
    
    # 最终保存
    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'influences': influences.numpy(),
                'completed_checkpoints': len(checkpoint_paths),
                'num_checkpoints': len(checkpoint_paths)
            }, f)
    
    return influences.numpy()

def main():
    config = load_config()
    
    DATA_DIR = config['data_dir']
    MODEL_BASE_PATH = config['model_base_path']
    TASK = config['task']
    DIM = config['dim']
    N_SPLITS = config['n_splits']
    RANDOM_STATE = config['random_state']
    FOLD_IDX = config['fold_idx']
    TOTAL_EPOCHS = config['total_epochs']
    DEVICE = config['device']
    OUTPUT_DIR = config['output_dir']
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("TracIn 影响力分析（优化版）")
    print("=" * 60)
    print(f"任务: {TASK}")
    print(f"Fold: {FOLD_IDX}")
    print(f"设备: {device}\n")
    
    # 1. 加载数据
    full_dataset, collate_fn = get_dataset_and_collate(TASK, DATA_DIR)
    train_indices, test_indices = get_fold_split(full_dataset, FOLD_IDX, N_SPLITS, RANDOM_STATE, TASK)
    
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 2. 加载 checkpoints
    checkpoint_paths = load_checkpoint_paths(MODEL_BASE_PATH, FOLD_IDX, TOTAL_EPOCHS)
    print(f"Checkpoint 数量: {len(checkpoint_paths)}")
    
    if len(checkpoint_paths) == 0:
        print("没有找到 checkpoint，退出")
        return
    
    # 3. 选择测试样本
    test_idx = 0
    if TASK == 'multitask':
        test_sample, test_label, _ = test_dataset[test_idx]
        test_label = test_label.item()
    else:
        test_sample, test_label = test_dataset[test_idx]
    
    print(f"\n测试样本索引: {test_idx}")
    print(f"测试样本标签: {test_label}")
    
    # 4. 计算 TracIn 影响力（优化版）
    influences = compute_tracin_influence_optimized(
        train_dataset=train_dataset,
        test_sample=test_sample,
        test_label=test_label,
        checkpoint_paths=checkpoint_paths,
        task=TASK,
        dim=DIM,
        device=device,
        output_dir=OUTPUT_DIR,
        fold_idx=FOLD_IDX,
        test_idx=test_idx,
        use_cache=True
    )
    
    # 5. 保存结果
    results = {
        'influences': influences,
        'test_idx': test_idx,
        'test_label': test_label,
        'fold_idx': FOLD_IDX,
        'num_checkpoints': len(checkpoint_paths)
    }
    
    output_path = os.path.join(OUTPUT_DIR, f'tracin_influences_fold{FOLD_IDX}_test{test_idx}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n结果已保存到: {output_path}")
    
    # 6. 显示 top 影响样本
    print("\n" + "=" * 60)
    print("Top 10 最有帮助的训练样本:")
    print("=" * 60)
    top_helpful_indices = np.argsort(influences)[-10:][::-1]
    for rank, idx in enumerate(top_helpful_indices, 1):
        if TASK == 'multitask':
            _, label, _ = train_dataset[idx]
            label = label.item()
        else:
            _, label = train_dataset[idx]
        print(f"{rank}. 训练样本索引={idx}, 标签={label}, 影响力={influences[idx]:.6f}")
    
    print("\n" + "=" * 60)
    print("Top 10 最有害的训练样本:")
    print("=" * 60)
    top_harmful_indices = np.argsort(influences)[:10]
    for rank, idx in enumerate(top_harmful_indices, 1):
        if TASK == 'multitask':
            _, label, _ = train_dataset[idx]
            label = label.item()
        else:
            _, label = train_dataset[idx]
        print(f"{rank}. 训练样本索引={idx}, 标签={label}, 影响力={influences[idx]:.6f}")

if __name__ == '__main__':
    main()