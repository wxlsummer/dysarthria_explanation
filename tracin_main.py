import torch
import os
import yaml
import numpy as np
import argparse
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from model import MultiTaskMLP
from torgo_dataloaders import (
    SeverityDataset, DetectionDataset, MultitaskDataset,
    collate_fn_severity, collate_fn_detection, collate_fn_multitask
)
from tracin_utils import sample_by_mode, save_results_individual_csv


def parse_args():
    parser = argparse.ArgumentParser(description='TracIn Influence Analysis')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    return parser.parse_args()


def load_config(config_path):
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
    
    raise ValueError(f"fold_idx {fold_idx} out of range")


def load_checkpoint_paths(model_base_path, fold_idx, config):
    checkpoint_paths = []
    fold_dir = os.path.join(model_base_path, f'fold_{fold_idx}')
    
    if not os.path.exists(fold_dir):
        print(f"Error: fold directory not found: {fold_dir}")
        return checkpoint_paths
    
    if 'checkpoint_epochs' in config and config['checkpoint_epochs'] is not None:
        epochs = config['checkpoint_epochs']
        print(f"\nUsing specified checkpoint epochs: {epochs}")
        
        for epoch in epochs:
            path = os.path.join(fold_dir, f'epoch_{epoch:03d}.pth')
            if os.path.exists(path):
                checkpoint_paths.append(path)
            else:
                print(f"Warning: checkpoint not found: {path}")
    
    else:
        total_epochs = config.get('total_epochs', 30)
        print(f"\nLoading all checkpoints (epochs 1-{total_epochs})")
        
        for epoch in range(1, total_epochs + 1):
            path = os.path.join(fold_dir, f'epoch_{epoch:03d}.pth')
            if os.path.exists(path):
                checkpoint_paths.append(path)
    
    return checkpoint_paths


def compute_loss(model, inputs, labels, task):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    outputs = model(inputs)
    
    if task == 'multitask':
        loss = criterion(outputs[0], labels)
    else:
        loss = criterion(outputs, labels)
    
    return loss


def compute_single_gradient(model, inputs, labels, task):
    model.zero_grad()
    
    loss = compute_loss(model, inputs, labels, task)
    loss = loss.mean()
    loss.backward()
    
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.view(-1).clone())
    
    return torch.cat(gradients)


def compute_all_train_gradients(model, train_dataset, task, device):
    all_gradients = []
    
    for idx in tqdm(range(len(train_dataset)), desc="  Train gradients", leave=False):
        if task == 'multitask':
            sample, severity_label, _ = train_dataset[idx]
            label = severity_label.item()
        else:
            sample, label = train_dataset[idx]
        
        sample = sample.unsqueeze(0).to(device)
        label = torch.tensor([label], dtype=torch.long).to(device)
        
        grad = compute_single_gradient(model, sample, label, task)
        all_gradients.append(grad.cpu())
    
    return torch.stack(all_gradients)


def compute_tracin_for_multiple_tests(
    train_dataset,
    test_dataset,
    sampled_test_indices,
    checkpoint_paths,
    task,
    dim,
    device
):
    num_train = len(train_dataset)
    num_test = len(sampled_test_indices)
    
    influences_matrix = np.zeros((num_test, num_train))
    
    print("\n" + "="*60)
    print(f"Computing TracIn Influences")
    print("="*60)
    print(f"Train samples: {num_train}")
    print(f"Test samples: {num_test}")
    print(f"Checkpoints: {len(checkpoint_paths)}")
    
    for ckpt_idx, ckpt_path in enumerate(tqdm(checkpoint_paths, desc="Checkpoints")):
        print(f"\nProcessing Checkpoint {ckpt_idx+1}/{len(checkpoint_paths)}: {os.path.basename(ckpt_path)}")
        
        model = MultiTaskMLP(dim=dim, task=task).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()
        
        train_grads = compute_all_train_gradients(model, train_dataset, task, device)
        
        for i, test_idx in enumerate(tqdm(sampled_test_indices, desc=f"  Test samples", leave=False)):
            if task == 'multitask':
                test_sample, test_label, _ = test_dataset[test_idx]
                test_label = test_label.item()
            else:
                test_sample, test_label = test_dataset[test_idx]
            
            test_sample_batch = test_sample.unsqueeze(0).to(device)
            test_label_tensor = torch.tensor([test_label], dtype=torch.long).to(device)
            test_grad = compute_single_gradient(model, test_sample_batch, test_label_tensor, task)
            test_grad = test_grad.cpu()
            
            batch_influences = torch.mv(train_grads, test_grad).numpy()
            
            influences_matrix[i, :] += batch_influences
    
    print(f"\nTracIn computation complete")
    print(f"Influence matrix shape: {influences_matrix.shape}")
    print(f"Influence range: [{influences_matrix.min():.4f}, {influences_matrix.max():.4f}]")
    
    return influences_matrix

def compute_self_influence(
    train_dataset,
    checkpoint_paths,
    task,
    dim,
    device
):
    """
    计算所有训练样本的self-influence
    
    Returns:
        self_influences: [n_train] 数组
    """
    num_train = len(train_dataset)
    self_influences = np.zeros(num_train)
    
    print("\n" + "="*60)
    print(f"Computing Self-Influences")
    print("="*60)
    print(f"Train samples: {num_train}")
    print(f"Checkpoints: {len(checkpoint_paths)}")
    
    for ckpt_idx, ckpt_path in enumerate(tqdm(checkpoint_paths, desc="Checkpoints")):
        print(f"\nProcessing Checkpoint {ckpt_idx+1}/{len(checkpoint_paths)}: {os.path.basename(ckpt_path)}")
        
        model = MultiTaskMLP(dim=dim, task=task).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()
        
        # 计算所有训练样本的梯度
        train_grads = compute_all_train_gradients(model, train_dataset, task, device)
        
        # 计算每个样本的梯度范数平方（self-influence的简化版）
        # 或者计算梯度和自己的点积
        for idx in tqdm(range(num_train), desc="  Self-influences", leave=False):
            grad = train_grads[idx]
            # self_influence = grad · grad（点积）
            self_influence = torch.dot(grad, grad).item()
            self_influences[idx] += self_influence
    
    print(f"\nSelf-influence computation complete")
    print(f"Self-influence range: [{self_influences.min():.4f}, {self_influences.max():.4f}]")
    
    return self_influences

def main():
    args = parse_args()
    
    print("="*60)
    print(f"Loading config: {args.config}")
    print("="*60)
    config = load_config(args.config)
    
    DATA_DIR = config['data_dir']
    MODEL_BASE_PATH = config['model_base_path']
    TASK = config['task']
    DIM = config['dim']
    N_SPLITS = config['n_splits']
    RANDOM_STATE = config['random_state']
    FOLD_IDX = config['fold_idx']
    DEVICE = config['device']
    OUTPUT_DIR = config['output_dir']
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Task: {TASK}")
    print(f"  Fold: {FOLD_IDX}")
    print(f"  Mode: {config.get('mode', 'data_cleaning')}")
    
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    print("\n" + "="*60)
    print("Loading dataset")
    print("="*60)
    full_dataset, collate_fn = get_dataset_and_collate(TASK, DATA_DIR)
    train_indices, test_indices = get_fold_split(full_dataset, FOLD_IDX, N_SPLITS, RANDOM_STATE, TASK)
    
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    checkpoint_paths = load_checkpoint_paths(MODEL_BASE_PATH, FOLD_IDX, config)
    print(f"\nFound {len(checkpoint_paths)} checkpoints")
    
    if len(checkpoint_paths) == 0:
        print("No checkpoints found, exiting")
        return
    
    print("\n" + "="*60)
    print("Sampling test set")
    print("="*60)

    mode = config.get('mode', 'data_cleaning')

    if mode == 'self_influence':
        # Self-influence模式
        from tracin_utils import save_self_influence_results
        
        self_influences = compute_self_influence(
            train_dataset=train_dataset,
            checkpoint_paths=checkpoint_paths,
            task=TASK,
            dim=DIM,
            device=device
        )
        
        summary = save_self_influence_results(
            self_influences=self_influences,
            train_dataset=train_dataset,
            train_indices=train_indices,
            config=config,
            checkpoint_paths=checkpoint_paths,
            output_dir=OUTPUT_DIR
        )    
    else:
        sampled_test_indices, mode_info = sample_by_mode(test_dataset, TASK, config)
        print(f"\n{mode_info}")
        print(f"Selected test samples: {len(sampled_test_indices)}")
        
        influences_matrix = compute_tracin_for_multiple_tests(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            sampled_test_indices=sampled_test_indices,
            checkpoint_paths=checkpoint_paths,
            task=TASK,
            dim=DIM,
            device=device
        )
        
        summary = save_results_individual_csv(
            influences_matrix=influences_matrix,
            sampled_test_indices=sampled_test_indices,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_indices=train_indices,
            test_indices=test_indices,
            config=config,
            checkpoint_paths=checkpoint_paths,
            output_dir=OUTPUT_DIR
        )
    
    print("\n" + "="*60)
    print("Complete")
    print("="*60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"CSV files: {len(summary['saved_files'])}")
    print(f"Summary file: {os.path.join(OUTPUT_DIR, 'summary.pkl')}")


if __name__ == '__main__':
    main()