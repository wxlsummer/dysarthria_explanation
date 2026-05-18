"""
简化版SHAP分析 - 只生成漂亮的图，没有标题
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
CSV_PATH = r'/srv/xw4e25/dysarthria_explanation/data/test_labels_with_egemaps.csv'
FIGURE_DIR = r'/srv/xw4e25/dysarthria_explanation/tracin_results/shap_figures'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 要分析的test_id
TARGET_TEST_IDS = ['F01_Session1_wav_arrayMic_0022', 'M01_Session1_wav_arrayMic_0042']

os.makedirs(FIGURE_DIR, exist_ok=True)

print("="*60)
print("SHAP分析")
print("="*60)

# ==================== 1. 加载数据 ====================
print("\n[1] 加载数据...")
df = pd.read_csv(CSV_PATH)

test_id = df.iloc[:, 0]
y = df.iloc[:, 1]
X = df.iloc[:, 2:90]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True).values
test_id = test_id.reset_index(drop=True)

feature_names = X.columns.tolist()
print(f"样本数: {len(X)}, 特征数: {len(feature_names)}")

# 处理数据类型
X = X.astype('float32')
X = X.fillna(X.mean())

# 编码标签
if isinstance(y[0], str):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

print(f"标签类别: {np.unique(y)}")

# ==================== 2. 分割数据 ====================
print("\n[2] 分割数据...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
test_id_train, test_id_test = train_test_split(
    test_id, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
test_id_test = test_id_test.reset_index(drop=True)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 找目标样本在测试集中的位置
target_positions = []
print(f"\n搜索目标样本...")
for target_id in TARGET_TEST_IDS:
    matches = np.where(test_id_test.values == target_id)[0]
    if len(matches) > 0:
        pos = matches[0]
        target_positions.append(pos)
        print(f"✓ 找到 '{target_id}' 在测试集位置 {pos}")
    else:
        print(f"✗ 未找到 '{target_id}'")

if len(target_positions) == 0:
    print("\n⚠ 警告：没有找到任何目标样本！")
    print("测试集中的前5个test_id:")
    for i in range(min(5, len(test_id_test))):
        print(f"  {i}: {test_id_test.iloc[i]}")
    exit()

# ==================== 3. 训练模型 ====================
print("\n[3] 训练XGBoost...")
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    verbosity=0
)
model.fit(X_train, y_train, verbose=False)

print(f"测试精度: {model.score(X_test, y_test):.4f}")

# ==================== 4. 计算SHAP值 ====================
print("\n[4] 计算SHAP值...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap_array = np.array(shap_values)

print(f"SHAP值形状: {shap_array.shape}")

# ==================== 5. 全局特征重要性（Summary plot + Bar plot）====================
print("\n[5] 生成全局特征重要性...")

num_classes = shap_array.shape[2] if len(shap_array.shape) == 3 else 1

for class_idx in range(num_classes):
    if len(shap_array.shape) == 3:
        shap_class = shap_array[:, :, class_idx]
    else:
        shap_class = shap_array
    
    # 1. 生成漂亮的Summary plot（红蓝点点）
    plt.figure(figsize=(14, 10))
    shap.summary_plot(
        shap_class,
        X_test,
        feature_names=feature_names,
        plot_type='dot',
        show=False
    )
    
    # 去掉标题
    ax = plt.gca()
    ax.set_title('')
    fig = plt.gcf()
    if fig._suptitle:
        fig._suptitle.remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'summary_plot_class_{class_idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 类别 {class_idx} Summary plot 完成")
    
    # 2. 生成柱状图（Bar plot）
    mean_abs_shap = np.abs(shap_class).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 10))
    top = feature_importance.head(20)
    plt.barh(range(len(top)), top['importance'].values, color='steelblue')
    plt.yticks(range(len(top)), top['feature'].values, fontsize=10)
    plt.xlabel('Mean |SHAP value|', fontsize=11)
    plt.gca().invert_yaxis()
    
    # 去掉标题
    ax = plt.gca()
    ax.set_title('')
    fig = plt.gcf()
    if fig._suptitle:
        fig._suptitle.remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'feature_importance_class_{class_idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 类别 {class_idx} Bar plot 完成")

# ==================== 6. 样本级分析（只要Force和Waterfall）====================
print("\n[6] 生成样本级分析...")

for pos in target_positions:
    test_id_val = test_id_test.iloc[pos]
    pred_label = model.predict(X_test.iloc[[pos]])[0]
    pred_prob = model.predict_proba(X_test.iloc[[pos]])[0]
    true_label = y_test[pos]
    
    print(f"\n分析: {test_id_val}")
    print(f"  真实: {true_label}, 预测: {pred_label}, 置信度: {pred_prob[pred_label]:.4f}")
    
    # 提取SHAP值
    if len(shap_array.shape) == 3:
        sample_shap = shap_array[pos, :, pred_label]
    else:
        sample_shap = shap_array[pos]
    
    # 特征重要性
    sample_importance = pd.DataFrame({
        'feature': feature_names,
        'shap': sample_shap,
        'abs_shap': np.abs(sample_shap),
        'value': X_test.iloc[pos].values
    }).sort_values('abs_shap', ascending=False)
    
    # 保存CSV
    sample_importance.to_csv(
        os.path.join(FIGURE_DIR, f'{test_id_val}_importance.csv'),
        index=False
    )
    
    # 简单柱状图
    plt.figure(figsize=(12, 8))
    top = sample_importance.head(15)
    colors = ['red' if x < 0 else 'blue' for x in top['shap']]
    plt.barh(range(len(top)), top['shap'].values, color=colors)
    plt.yticks(range(len(top)), top['feature'].values, fontsize=10)
    plt.xlabel('SHAP value', fontsize=11)
    plt.axvline(x=0, color='black', linewidth=0.8)
    
    # 反转Y轴，让大的在上面
    plt.gca().invert_yaxis()
    
    # 去掉标题
    ax = plt.gca()
    ax.set_title('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{test_id_val}_shap_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 柱状图: {test_id_val}_shap_bar.png")

print("\n" + "="*60)
print("✅ 完成！所有结果保存在:")
print(FIGURE_DIR)
print("="*60)