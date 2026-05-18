import matplotlib.pyplot as plt
import numpy as np
removal_pct = [0, 5, 10, 15, 20]  # 或者 [0, 5, 10, 15, 20] 如果你有0%的数据

# 为每个test label填入数据
# 格式: 列表中的每个值对应上面removal_pct中的百分比
data = {
    'Class-0 (Typical)': {
        'High': [93.7, 85.5, 80.0, 79.8, 80],  # 填入你的数据
        'Low': [93.7, 92.7, 93.3, 95.1, 95],    # 示例：从你的表格
        'Random': [93.7, 91.6, 92.3, 89.5, 92.9] # 填入你的数据
    },
    'Class-1 (Mild)': {
        'High': [5.5, 4.6, 0.0, 0.0, 0.0],
        'Low': [5.5, 9.3, 10.1, 16.9, 17.6],
        'Random': [5.5, 6.5, 5.5, 2.6, 4.9]
    },
    'Class-2 (Moderate)': {
        'High': [36.8, 14.7, 0.4, 1.0, 0.3],
        'Low': [36.8, 41.4, 47.7, 38.6, 47.0],
        'Random': [36.8, 32.3, 31.2, 28.4, 28.4]
    },
    'Class-3 (Severe)': {
        'High': [72.3, 60.0, 52.9, 46.1, 43.1],
        'Low': [72.3, 75.3, 77.5, 82.8, 81.1],
        'Random': [72.3, 67.1, 67.8, 71.2, 66.7]
    }
}

# ============================================================================
# 绘图代码 (不需要修改)
# ============================================================================

# 设置绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

# 配色方案
colors = {
    'High': '#D32F2F',      # 红色 - 删除重要样本
    'Low': '#388E3C',       # 绿色 - 删除不重要样本
    'Random': '#1976D2'     # 蓝色 - 随机删除
}

markers = {
    'High': 'o',
    'Low': 's',
    'Random': '^'
}

# 绘制每个test label
for idx, (title, label_data) in enumerate(data.items()):
    ax = axes[idx]
    
    # 绘制三种策略
    for strategy in ['High', 'Low', 'Random']:
        values = label_data[strategy]
        
        # 过滤None值
        plot_data = [(x, y) for x, y in zip(removal_pct, values) if y is not None]
        
        if len(plot_data) > 0:
            x_points = [p[0] for p in plot_data]
            y_points = [p[1] for p in plot_data]
            
            ax.plot(x_points, y_points,
                   marker=markers[strategy],
                   color=colors[strategy],
                   linewidth=2.5,
                   markersize=9,
                   label=f'{strategy} Influence',
                   alpha=0.85,
                   markeredgewidth=1.5,
                   markeredgecolor='white')
    
    # 设置标签和标题
    ax.set_xlabel('Removed Training Samples (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    
    # 网格
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 图例
    ax.legend(fontsize=9.5, loc='best', framealpha=0.95,
             edgecolor='gray', fancybox=True)
    
    # 设置x轴刻度
    ax.set_xticks(removal_pct)
    
    # 自动计算y轴范围
    all_values = []
    for s in ['High', 'Low', 'Random']:
        all_values.extend([v for v in label_data[s] if v is not None])
    
    if len(all_values) > 0:
        y_min = max(0, min(all_values) - 5)
        y_max = min(100, max(all_values) + 5)
        ax.set_ylim(y_min, y_max)

plt.tight_layout(pad=2.0)

# 保存图片
plt.savefig('/srv/xw4e25/dysarthria_explanation/tracin_results/Figures/severity_classification.png',
            dpi=400, bbox_inches='tight', facecolor='white')
plt.savefig('/srv/xw4e25/dysarthria_explanation/tracin_results/Figures/severity_classification.pdf',
            bbox_inches='tight', facecolor='white')

print("✓ 图片已保存!")
print("\n保存位置:")
print("  - /srv/xw4e25/dysarthria_explanation/tracin_results/Figures/severity_classification.png")
print("  - /srv/xw4e25/dysarthria_explanation/tracin_results/Figures/severity_classification.pdf")

# 打印数据摘要
print("\n" + "="*60)
print("数据摘要")
print("="*60)
for title, label_data in data.items():
    print(f"\n{title}:")
    for strategy in ['High', 'Low', 'Random']:
        values = [v for v in label_data[strategy] if v is not None]
        if len(values) >= 2:
            change = values[-1] - values[0]
            print(f"  {strategy:8s}: {values[0]:5.1f}% → {values[-1]:5.1f}% (Δ {change:+5.1f}%)")
        elif len(values) == 1:
            print(f"  {strategy:8s}: {values[0]:5.1f}% (单点)")
        else:
            print(f"  {strategy:8s}: 无数据")