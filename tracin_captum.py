import torch
import os
import yaml
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedKFold
import pickle
from captum.influence import TracInCP

from model import MultiTaskMLP
from torgo_dataloaders import SeverityDataset

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_fold_split(dataset, fold_idx, n_splits, random_state):
    targets = [dataset[i][1] for i in range(len(dataset))]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for current_fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), targets), start=1):
        if current_fold == fold_idx:
            return list(train_idx), list(test_idx)

def load_checkpoint_paths(model_base_path, fold_idx, total_epochs):
    checkpoint_paths = []
    fold_dir = os.path.join(model_base_path, f'fold_{fold_idx}')
    
    for epoch in range(1, total_epochs + 1):
        path = os.path.join(fold_dir, f'epoch_{epoch:03d}.pth')
        if os.path.exists(path):
            checkpoint_paths.append(path)
    
    return checkpoint_paths

def main():
    config = load_config()
    
    DATA_DIR = config['data_dir']
    MODEL_BASE_PATH = config['model_base_path']
    TASK = config['task']
    DIM = config['dim']
    FOLD_IDX = config['fold_idx']
    TOTAL_EPOCHS = config['total_epochs']
    BATCH_SIZE = config['batch_size']
    OUTPUT_DIR = config['output_dir']
    N_SPLITS = config['n_splits']
    RANDOM_STATE = config['random_state']
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("1. 加载数据...")
    full_dataset = SeverityDataset(DATA_DIR)
    train_indices, test_indices = get_fold_split(full_dataset, FOLD_IDX, N_SPLITS, RANDOM_STATE)
    
    train_data = []
    train_labels = []
    for idx in train_indices:
        sample, label = full_dataset[idx]
        train_data.append(sample)
        train_labels.append(label)
    
    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_dataset = TensorDataset(train_data, train_labels)
    
    print(f"训练集: {len(train_dataset)} 样本\n")
    
    print("2. 加载checkpoints...")
    checkpoint_paths = load_checkpoint_paths(MODEL_BASE_PATH, FOLD_IDX, TOTAL_EPOCHS)
    print(f"找到 {len(checkpoint_paths)} 个checkpoint\n")
    
    print("3. 初始化模型...")
    model = MultiTaskMLP(dim=DIM, task=TASK)

    for param in model.parameters():
        param.requires_grad = True
    
    # 按照tutorial: checkpoints_load_func接受(model, checkpoint_path)两个参数
    # 它应该原地修改model，不返回任何东西
    def checkpoints_load_func(model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        # 不返回任何东西
    
    print("4. 初始化TracInCP...\n")
    
    tracin = TracInCP(
        model=model,
        train_dataset=train_dataset,
        checkpoints=checkpoint_paths,
        checkpoints_load_func=checkpoints_load_func,
        loss_fn=torch.nn.CrossEntropyLoss(reduction='none'),
        batch_size=BATCH_SIZE
    )
    
    print("5. 准备测试样本...")
    test_idx = 0
    test_sample, test_label = full_dataset[test_indices[test_idx]]
    print(f"测试样本标签: {test_label}\n")
    
    print("6. 计算影响力...\n")
    
    influences = tracin.influence(
        (test_sample.unsqueeze(0), torch.tensor([test_label], dtype=torch.long))
    )
    
    influences = influences.cpu().numpy()
    
    print("\n完成！\n")
    
    # 保存
    output_path = os.path.join(OUTPUT_DIR, f'influences_fold{FOLD_IDX}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump({
            'influences': influences,
            'test_idx': test_idx,
            'test_label': test_label
        }, f)
    
    print(f"保存到: {output_path}\n")
    
    # Top 10
    print("Top 10 最有帮助:")
    top = np.argsort(influences)[-10:][::-1]
    for rank, idx in enumerate(top, 1):
        print(f"{rank}. idx={idx}, label={train_labels[idx].item()}, score={influences[idx]:.4f}")

if __name__ == '__main__':
    main()