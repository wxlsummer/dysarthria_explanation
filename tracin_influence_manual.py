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
        # 用severity的损失
        loss = criterion(outputs[0], labels)
    else:
        loss = criterion(outputs, labels)
    
    return loss

def compute_gradients(model, inputs, labels, task):
    """计算梯度并展平"""
    model.zero_grad()
    
    loss = compute_loss(model, inputs, labels, task)
    loss = loss.mean()
    loss.backward()
    
    # 收集所有参数的梯度
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.view(-1))
    
    return torch.cat(gradients)

def compute_tracin_influence(
    train_dataset,
    test_sample,
    test_label,
    checkpoint_paths,
    task,
    dim,
    device
):
    """
    计算训练集中每个样本对测试样本的TracIn影响力
    
    Returns:
        influences: shape [num_train_samples], 每个训练样本的影响力分数
    """
    num_train = len(train_dataset)
    influences = torch.zeros(num_train, device='cpu')
    
    # 1. 计算测试样本的梯度（在最后一个checkpoint）
    print("\n计算测试样本梯度...")
    model = MultiTaskMLP(dim=dim, task=task).to(device)
    
    # 加载最后一个checkpoint
    last_ckpt = torch.load(checkpoint_paths[-1], map_location=device, weights_only=False)
    model.load_state_dict(last_ckpt['model_state_dict'])
    model.train()
    
    test_sample = test_sample.unsqueeze(0).to(device)
    test_label = torch.tensor([test_label], dtype=torch.long).to(device)
    
    test_grad = compute_gradients(model, test_sample, test_label, task)
    test_grad = test_grad.cpu()
    
    print(f"测试样本梯度维度: {test_grad.shape}")
    
    # 2. 遍历所有checkpoint，累积影响力
    print(f"\n计算TracIn影响力（共 {len(checkpoint_paths)} 个checkpoint）...")
    
    for ckpt_idx, ckpt_path in enumerate(tqdm(checkpoint_paths, desc="Checkpoints")):
        # 加载checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()
        
        # 对每个训练样本计算梯度
        for train_idx in tqdm(range(num_train), desc=f"Epoch {ckpt_idx+1}", leave=False):
            if task == 'multitask':
                train_sample, train_severity_label, train_dysarthria_label = train_dataset[train_idx]
                train_label = train_severity_label.item()
            else:
                train_sample, train_label = train_dataset[train_idx]
            
            train_sample = train_sample.unsqueeze(0).to(device)
            train_label_tensor = torch.tensor([train_label], dtype=torch.long).to(device)
            
            # 计算训练样本梯度
            train_grad = compute_gradients(model, train_sample, train_label_tensor, task)
            train_grad = train_grad.cpu()
            
            # 计算点积
            influence = torch.dot(train_grad, test_grad).item()
            influences[train_idx] += influence
    
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
    print("TracIn 影响力分析")
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
    
    # 2. 加载checkpoints
    checkpoint_paths = load_checkpoint_paths(MODEL_BASE_PATH, FOLD_IDX, TOTAL_EPOCHS)
    print(f"Checkpoint数量: {len(checkpoint_paths)}")
    
    # 3. 选择测试样本
    test_idx = 0
    if TASK == 'multitask':
        test_sample, test_label, _ = test_dataset[test_idx]
        test_label = test_label.item()
    else:
        test_sample, test_label = test_dataset[test_idx]
    
    print(f"\n测试样本索引: {test_idx}")
    print(f"测试样本标签: {test_label}")
    
    # 4. 计算TracIn影响力
    influences = compute_tracin_influence(
        train_dataset=train_dataset,
        test_sample=test_sample,
        test_label=test_label,
        checkpoint_paths=checkpoint_paths,
        task=TASK,
        dim=DIM,
        device=device
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
    
    # 6. 显示top影响样本
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