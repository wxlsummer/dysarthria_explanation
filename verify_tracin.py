import pickle
import numpy as np

# 加载结果
with open('./tracin_results/tracin_influences_fold1_test0.pkl', 'rb') as f:
    results = pickle.load(f)

influences = results['influences']
test_label = results['test_label']

print("="*60)
print("TracIn结果验证")
print("="*60)

print(f"\n测试样本标签: {test_label}")
print(f"影响力数组形状: {influences.shape}")
print(f"影响力范围: [{influences.min():.6f}, {influences.max():.6f}]")
print(f"影响力均值: {influences.mean():.6f}")
print(f"影响力标准差: {influences.std():.6f}")

# 理论上的sanity checks:
print("\n" + "="*60)
print("Sanity Checks:")
print("="*60)

# 1. 同标签样本的平均影响应该是正的
from torgo_dataloaders import SeverityDataset
from sklearn.model_selection import StratifiedKFold

dataset = SeverityDataset('/srv/xw4e25/Data/fbanks_83')
targets = [dataset[i][1] for i in range(len(dataset))]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), targets), start=1):
    if fold_idx == 1:
        train_indices = train_idx
        break

# 获取训练样本的标签
train_labels = [dataset[i][1] for i in train_indices]

# 同标签 vs 不同标签的平均影响
same_label_influences = [influences[i] for i in range(len(train_labels)) if train_labels[i] == test_label]
diff_label_influences = [influences[i] for i in range(len(train_labels)) if train_labels[i] != test_label]

print(f"\n1. 同标签样本 (label={test_label}):")
print(f"   数量: {len(same_label_influences)}")
print(f"   平均影响: {np.mean(same_label_influences):.6f}")

print(f"\n2. 不同标签样本:")
print(f"   数量: {len(diff_label_influences)}")
print(f"   平均影响: {np.mean(diff_label_influences):.6f}")

print(f"\n3. 理论预期:")
print(f"   ✓ 同标签样本的平均影响 > 不同标签样本的平均影响")
print(f"   实际: {np.mean(same_label_influences):.6f} {'>' if np.mean(same_label_influences) > np.mean(diff_label_influences) else '<'} {np.mean(diff_label_influences):.6f}")

# 2. Top有帮助的样本应该主要是同标签
top_10_indices = np.argsort(influences)[-10:]
top_10_labels = [train_labels[i] for i in top_10_indices]
same_label_count = sum(1 for label in top_10_labels if label == test_label)

print(f"\n4. Top 10 最有帮助的样本:")
print(f"   同标签数量: {same_label_count}/10")
print(f"   理论预期: 应该大部分是同标签")

print("\n" + "="*60)
if np.mean(same_label_influences) > np.mean(diff_label_influences) and same_label_count >= 5:
    print("✓ TracIn结果看起来合理!")
else:
    print("⚠ TracIn结果可能有问题，需要进一步检查")
print("="*60)