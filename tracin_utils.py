import numpy as np
import torch
import pickle
import os
import csv
from tqdm import tqdm
from torch.utils.data import Subset


def get_sample_metadata(dataset, idx, task):
    """获取样本的完整元数据"""
    if isinstance(dataset, Subset):
        original_idx = dataset.indices[idx]
        original_dataset = dataset.dataset
    else:
        original_idx = idx
        original_dataset = dataset
    
    if hasattr(original_dataset, 'get_file_info'):
        file_info = original_dataset.get_file_info(original_idx)
    else:
        file_info = {
            'filename': f'sample_{original_idx}',
            'file_path': 'unknown'
        }
    
    if task == 'multitask':
        _, severity_label, dysarthric_label = dataset[idx]
        label = severity_label.item()
    else:
        _, label = dataset[idx]
    
    metadata = {
        'subset_idx': idx,
        'original_idx': original_idx,
        'filename': file_info.get('filename', f'sample_{original_idx}'),
        'file_path': file_info.get('file_path', 'unknown'),
        'label': label
    }
    
    if task == 'multitask':
        metadata['severity_label'] = severity_label.item()
        metadata['dysarthric_label'] = dysarthric_label.item()
    
    return metadata


def sample_test_set_proportional(test_dataset, task, total_samples, random_state):
    """按原始类别分布比例采样测试集"""
    np.random.seed(random_state)
    
    label_counts = {}
    labels = []
    for idx in range(len(test_dataset)):
        if task == 'multitask':
            _, label, _ = test_dataset[idx]
            label = label.item()
        else:
            _, label = test_dataset[idx]
        labels.append(label)
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n测试集原始分布:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = count / len(labels) * 100
        print(f"  类别 {label}: {count} ({pct:.1f}%)")
    
    sampled_indices = []
    for label in sorted(label_counts.keys()):
        proportion = label_counts[label] / len(labels)
        n_sample = int(total_samples * proportion)
        
        indices = [i for i, l in enumerate(labels) if l == label]
        
        if len(indices) >= n_sample:
            sampled = np.random.choice(indices, n_sample, replace=False)
        else:
            sampled = indices
        
        sampled_indices.extend(sampled)
    
    if len(sampled_indices) < total_samples:
        remaining = total_samples - len(sampled_indices)
        all_indices = list(range(len(test_dataset)))
        available = [i for i in all_indices if i not in sampled_indices]
        extra = np.random.choice(available, remaining, replace=False)
        sampled_indices.extend(extra)
    
    print(f"\n采样后统计:")
    print(f"  总样本数: {len(sampled_indices)}")
    sampled_labels = [labels[i] for i in sampled_indices]
    for label in sorted(label_counts.keys()):
        count = sum(1 for l in sampled_labels if l == label)
        pct = count / len(sampled_indices) * 100
        print(f"  类别 {label}: {count} ({pct:.1f}%)")
    
    return sampled_indices


def sample_specific_tests(test_dataset, task, config):
    """根据指定的文件名选择测试样本"""
    if 'test_sample_files' not in config:
        raise ValueError("specific_tests 模式需要指定 test_sample_files")
    
    target_files = config['test_sample_files']
    print(f"\n查找指定的测试样本文件:")
    
    sampled_indices = []
    for idx in range(len(test_dataset)):
        metadata = get_sample_metadata(test_dataset, idx, task)
        if metadata['filename'] in target_files:
            print(f"  找到: {metadata['filename']} (索引={idx}, 标签={metadata['label']})")
            sampled_indices.append(idx)
    
    if len(sampled_indices) != len(target_files):
        found_files = [get_sample_metadata(test_dataset, i, task)['filename'] 
                      for i in sampled_indices]
        missing = set(target_files) - set(found_files)
        print(f"警告: 以下文件未找到: {missing}")
    
    return sampled_indices


def sample_class_level(test_dataset, task, n_per_class, random_state):
    """每个类别均衡采样"""
    np.random.seed(random_state)
    
    label_to_indices = {}
    for idx in range(len(test_dataset)):
        if task == 'multitask':
            _, label, _ = test_dataset[idx]
            label = label.item()
        else:
            _, label = test_dataset[idx]
        
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    print(f"\n测试集类别分布:")
    for label in sorted(label_to_indices.keys()):
        print(f"  类别 {label}: {len(label_to_indices[label])} 个样本")
    
    sampled_indices = []
    print(f"\n每个类别采样 {n_per_class} 个:")
    
    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        n_sample = min(n_per_class, len(indices))
        
        sampled = np.random.choice(indices, n_sample, replace=False)
        sampled_indices.extend(sampled)
        
        print(f"  类别 {label}: 采样 {n_sample} 个")
    
    print(f"\n总共采样: {len(sampled_indices)} 个样本")
    
    return sampled_indices

def save_self_influence_results(
    self_influences,
    train_dataset,
    train_indices,
    config,
    checkpoint_paths,
    output_dir
):
    """保存self-influence结果"""
    import pandas as pd
    
    task = config['task']
    
    print("\n" + "="*60)
    print("Saving self-influence results")
    print("="*60)
    
    # 收集训练样本信息
    results = []
    for idx in tqdm(range(len(train_dataset)), desc="Collecting metadata"):
        metadata = get_sample_metadata(train_dataset, idx, task)
        results.append({
            'filename': metadata['filename'],
            'label': metadata['label'],
            'self_influence': self_influences[idx]
        })
    
    # 保存为CSV（按self-influence排序）
    df = pd.DataFrame(results)
    df = df.sort_values('self_influence', ascending=False)
    
    csv_path = os.path.join(output_dir, 'self_influences.csv')
    df.to_csv(csv_path, index=False)
    
    # 保存为pkl
    summary = {
        'self_influences': self_influences,
        'train_samples': results,
        'config': config,
        'checkpoint_paths': checkpoint_paths,
        'index_mapping': {
            'train_subset_to_original': train_indices
        }
    }
    
    pkl_path = os.path.join(output_dir, 'self_influences.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\nResults saved:")
    print(f"  CSV: {csv_path}")
    print(f"  PKL: {pkl_path}")
    
    # 打印统计
    print(f"\nStatistics by class:")
    for label in sorted(df['label'].unique()):
        label_data = df[df['label'] == label]['self_influence']
        print(f"  Class {label}:")
        print(f"    Mean: {label_data.mean():.4f}")
        print(f"    Std: {label_data.std():.4f}")
        print(f"    Min: {label_data.min():.4f}")
        print(f"    Max: {label_data.max():.4f}")
    
    print(f"\nTop-10 highest self-influence samples:")
    for i, row in df.head(10).iterrows():
        print(f"  {row['filename']}, label={row['label']}, self_influence={row['self_influence']:.4f}")
    
    return summary

def sample_by_mode(test_dataset, task, config):
    """根据配置的mode选择测试样本"""
    mode = config.get('mode', 'data_cleaning')
    random_state = config.get('random_state', 42)
    
    if mode == 'data_cleaning':
        n_test_samples = config.get('n_test_samples', 200)
        sampled_indices = sample_test_set_proportional(
            test_dataset, task, n_test_samples, random_state
        )
        mode_info = f"数据清洗模式: 采样 {n_test_samples} 个（按比例）"
    
    elif mode == 'specific_tests':
        sampled_indices = sample_specific_tests(test_dataset, task, config)
        mode_info = f"指定样本模式: {len(sampled_indices)} 个样本"
    
    elif mode == 'class_level':
        n_per_class = config.get('n_per_class', 20)
        sampled_indices = sample_class_level(
            test_dataset, task, n_per_class, random_state
        )
        mode_info = f"类别级模式: 每类 {n_per_class} 个样本"
    
    else:
        raise ValueError(f"未知的模式: {mode}")
    
    return sampled_indices, mode_info


def save_results_individual_csv(
    influences_matrix,
    sampled_test_indices,
    train_dataset,
    test_dataset,
    train_indices,
    test_indices,
    config,
    checkpoint_paths,
    output_dir
):
    """为每个测试样本保存独立的CSV文件"""
    task = config['task']
    
    print("\n" + "="*60)
    print("保存结果（每个测试样本一个CSV文件）")
    print("="*60)
    
    print("\n收集训练样本信息...")
    train_samples_info = []
    for idx in tqdm(range(len(train_dataset)), desc="训练样本"):
        metadata = get_sample_metadata(train_dataset, idx, task)
        train_samples_info.append({
            'filename': metadata['filename'],
            'label': metadata['label']
        })
    
    print(f"\n为 {len(sampled_test_indices)} 个测试样本创建 CSV 文件...")
    
    saved_files = []
    
    for i, test_idx in enumerate(tqdm(sampled_test_indices, desc="保存CSV")):
        test_metadata = get_sample_metadata(test_dataset, test_idx, task)
        test_filename = test_metadata['filename']
        test_label = test_metadata['label']
        
        influences = influences_matrix[i, :]
        
        csv_filename = test_filename.replace('.npy', '.csv')
        csv_path = os.path.join(output_dir, csv_filename)
        
        # 写入 CSV（按影响力排序）
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow(['train_filename', 'train_label', 'influence'])
            
            # 按影响力从大到小排序
            sorted_indices = np.argsort(influences)[::-1]
            
            # 写入排序后的数据
            for train_idx in sorted_indices:
                train_info = train_samples_info[train_idx]
                writer.writerow([
                    train_info['filename'],
                    train_info['label'],
                    influences[train_idx]
                ])
        
        saved_files.append({
            'test_filename': test_filename,
            'test_label': test_label,
            'csv_path': csv_path
        })
    
    summary = {
        'config': config,
        'n_test_samples': len(sampled_test_indices),
        'n_train_samples': len(train_dataset),
        'num_checkpoints': len(checkpoint_paths),
        'checkpoint_paths': checkpoint_paths,
        'saved_files': saved_files,
        'train_samples_info': train_samples_info,
        'index_mapping': {
            'train_subset_to_original': train_indices,
            'test_subset_to_original': test_indices,
            'sampled_test_subset_indices': sampled_test_indices
        }
    }
    
    summary_path = os.path.join(output_dir, 'summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print("\n" + "="*60)
    print("保存完成")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"CSV 文件数: {len(saved_files)}")
    print(f"汇总文件: {summary_path}")
    print("\n示例文件:")
    for i in range(min(5, len(saved_files))):
        print(f"  - {saved_files[i]['test_filename']} -> {os.path.basename(saved_files[i]['csv_path'])}")
    
    return summary