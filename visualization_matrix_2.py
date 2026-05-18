import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Load data
# ============================================================================

# Load S matrix
df_S = pd.read_csv('/srv/xw4e25/dysarthria_explanation/tracin_results/Figures/influence_matrix_S_labeled.csv', index_col=0)

# Extract values
S_matrix = df_S.values

# Labels
test_labels = ['Typical: 0', 'Mild: 1', 'Moderate: 2', 'Severe: 3']
train_labels = ['Typical: 0', 'Mild: 1', 'Moderate: 2', 'Severe: 3']

print("S Matrix:")
print(df_S)

# ============================================================================
# Create heatmap
# ============================================================================

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# Create figure
fig, ax = plt.subplots(figsize=(8, 6.5))

# Create heatmap with better diverging colormap
# Use RdBu_r: Blue (high/positive) -> White (zero) -> Red (low/negative)
sns.heatmap(S_matrix, 
            annot=True,           # Show numbers
            fmt='.0f',            # Format as integer
            cmap='RdBu_r',        # Blue-White-Red (reversed)
            center=0,             # Center colormap at 0
            square=True,          # Square cells
            linewidths=1.5,       # Cell borders
            linecolor='gray',     # Border color (gray instead of white)
            cbar_kws={'label': 'Average Influence', 'shrink': 0.8},
            xticklabels=train_labels,
            yticklabels=test_labels,
            vmin=-8000,           # Set bounds for better contrast
            vmax=25000,
            ax=ax)

# Move x-axis (Training Label) to TOP
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# Labels
ax.set_xlabel('Training Label', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Test Label', fontsize=13, fontweight='bold', labelpad=10)
#ax.set_title('Class-Level Influence Matrix $\mathbf{S}$', fontsize=14, fontweight='bold', pad=15)

# Rotate labels for better readability
plt.xticks(rotation=0, ha='center')
plt.yticks(rotation=0)

# Adjust layout
plt.tight_layout()

# Save
plt.savefig('/srv/xw4e25/dysarthria_explanation/tracin_results/Figures/influence_matrix_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/srv/xw4e25/dysarthria_explanation/tracin_results/Figures/influence_matrix_heatmap.pdf', bbox_inches='tight', facecolor='white')

print("\n✓ Heatmap saved:")
print("  - influence_matrix_heatmap.png")
print("  - influence_matrix_heatmap.pdf")

# ============================================================================
# Create ordinal sensitivity bar chart
# ============================================================================

# Load S̄(d) data
df_sbar = pd.read_csv('/srv/xw4e25/dysarthria_explanation/tracin_results/Figures/ordinal_sensitivity.csv')
distances = df_sbar['distance'].values.astype(int)
S_bar = df_sbar['S_bar'].values

fig, ax = plt.subplots(figsize=(8, 5))

# Bar chart with gradient colors
colors = ['#1E88E5', '#FDD835', '#FB8C00', '#E53935']  # Blue->Yellow->Orange->Red
bars = ax.bar(distances, S_bar, color=colors, alpha=0.9, 
              edgecolor='black', linewidth=1.5, width=0.65)

# Add value labels on bars with adjusted positioning
for i, (d, val) in enumerate(zip(distances, S_bar)):
    # 增加顶部间距，防止标字被压到上边缘
    offset = 800 if val > 0 else -800
    ax.text(d, val + offset, 
            f'{val:.0f}', 
            ha='center', va='bottom' if val > 0 else 'top',
            fontsize=11, fontweight='bold')

# Labels
distance_names = ['Same-level\n(d=0)', 'Adjacent\n(d=1)', 
                  'Two-apart\n(d=2)', 'Opposite\n(d=3)']
ax.set_xticks(distances)
ax.set_xticklabels(distance_names, fontsize=11)

ax.set_ylabel('Average Influence $\\bar{S}(d)$', fontsize=12, fontweight='bold')
ax.set_xlabel('Ordinal Distance', fontsize=12, fontweight='bold')
# Title removed - 移除标题

# Grid
ax.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=0)
ax.set_axisbelow(True)

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linewidth=1.2, linestyle='-', alpha=0.7)

# 自动调整y轴范围，给顶部留足空间
ax.margins(y=0.15)  # 增加上下边距

# Adjust layout
plt.tight_layout()

# Save
plt.savefig('/srv/xw4e25/dysarthria_explanation/tracin_results/Figures/ordinal_sensitivity_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/srv/xw4e25/dysarthria_explanation/tracin_results/Figures/ordinal_sensitivity_bar.pdf', bbox_inches='tight', facecolor='white')

print("\n✓ Ordinal sensitivity chart saved:")
print("  - ordinal_sensitivity_bar.png")
print("  - ordinal_sensitivity_bar.pdf")

# ============================================================================
# Print statistics
# ============================================================================

print("\n" + "="*70)
print("KEY STATISTICS")
print("="*70)

print("\nDiagonal (same-level) values:")
for i in range(4):
    print(f"  S[{test_labels[i]}, {train_labels[i]}] = {S_matrix[i,i]:.0f}")

print("\nRatios:")
if S_bar[1] != 0:
    print(f"  S̄(0) / S̄(1) = {S_bar[0]/S_bar[1]:.2f}×")
if S_bar[2] != 0:
    print(f"  S̄(0) / |S̄(2)| = {abs(S_bar[0]/S_bar[2]):.2f}×")

print("\nMonotonic decrease:")
for i in range(len(S_bar)-1):
    direction = "✓" if S_bar[i] > S_bar[i+1] else "✗"
    print(f"  {direction} S̄({i}) = {S_bar[i]:.0f} > S̄({i+1}) = {S_bar[i+1]:.0f}")