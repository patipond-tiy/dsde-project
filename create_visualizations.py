#!/usr/bin/env python3
"""
Create model performance comparison visualizations.
Saves to /home/CHAIN/project/temp/dsde/docs/final-report/assets/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for professional-looking charts
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Output directory
output_dir = Path('/home/CHAIN/project/temp/dsde/docs/final-report/assets')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. MODEL EVOLUTION CHART
# ============================================================================
print("Creating Model Evolution Chart...")

# Define model names and extract relevant metrics
models_data = [
    {
        'name': 'Robust V1\n(No Urgency)',
        'mae': 23.97,
        'r2': 0.0999,
        'is_final': False
    },
    {
        'name': 'Robust V2\n(Linear Enc)',
        'mae': 23.97,
        'r2': 0.1039,
        'is_final': False
    },
    {
        'name': 'Hybrid\nResampling',
        'mae': 35.88,
        'r2': 0.3714,
        'is_final': False
    },
    {
        'name': 'Optimized\nFull Data',
        'mae': 53.13,
        'r2': 0.0160,
        'is_final': False
    },
    {
        'name': 'Hybrid V2\n(Final)',
        'mae': 34.47,
        'r2': 0.6132,
        'is_final': True
    }
]

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

x = np.arange(len(models_data))
width = 0.35

colors = ['#1f77b4' if not m['is_final'] else '#2ca02c' for m in models_data]

mae_values = [m['mae'] for m in models_data]
r2_values = [m['r2'] for m in models_data]

# Create bars for MAE (left axis)
bars1 = ax.bar(x - width/2, mae_values, width, label='MAE (days)', color=colors, alpha=0.8)

# Create second y-axis for R²
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, r2_values, width, label='R² Score', color='#ff7f0e', alpha=0.8)

# Customize axes
ax.set_xlabel('Model Version', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE (days)', fontsize=12, fontweight='bold', color='#1f77b4')
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold', color='#ff7f0e')
ax.set_title('Model Evolution: Performance Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([m['name'] for m in models_data], fontsize=10)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Color the y-axis labels
ax.tick_params(axis='y', labelcolor='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#ff7f0e')

# Add legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# Add grid
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(output_dir / 'model_evolution.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'model_evolution.png'}")
plt.close()

# ============================================================================
# 2. STRATIFIED PERFORMANCE CHART
# ============================================================================
print("Creating Stratified Performance Chart...")

time_bins = ['0-7d\n(Quick)', '7-30d\n(Fast)', '30-90d\n(Medium)', '90-180d\n(Slow)', '180-365d\n(Very Slow)']
mae_values = [15.73, 25.40, 40.15, 62.80, 70.04]
colors_strat = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#c0392b']

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

bars = ax.bar(time_bins, mae_values, color=colors_strat, alpha=0.85, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, mae_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Resolution Time Range', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Absolute Error (days)', fontsize=12, fontweight='bold')
ax.set_title('Stratified Performance: Model Error by Resolution Time Category',
             fontsize=14, fontweight='bold', pad=20)

# Set y-axis limit
ax.set_ylim(0, max(mae_values) * 1.15)

# Add grid
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)

# Add a reference line for overall MAE
overall_mae = 34.47
ax.axhline(y=overall_mae, color='red', linestyle='--', linewidth=2, label=f'Overall MAE: {overall_mae:.2f}')
ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig(output_dir / 'stratified_performance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'stratified_performance.png'}")
plt.close()

# ============================================================================
# 3. METRICS DASHBOARD
# ============================================================================
print("Creating Metrics Dashboard...")

fig = plt.figure(figsize=(14, 8), dpi=300)
gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

# Metrics data
metrics = {
    'MAE': {'value': 34.47, 'unit': 'days', 'color': '#3498db'},
    'RMSE': {'value': 57.55, 'unit': 'days', 'color': '#e74c3c'},
    'R² Score': {'value': 0.6132, 'unit': '', 'color': '#2ecc71'},
    'Balance Score': {'value': 0.5344, 'unit': '', 'color': '#f39c12'}
}

positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for (i, j), (metric_name, metric_data) in zip(positions, metrics.items()):
    ax = fig.add_subplot(gs[i, j])

    # Create background circle/gauge effect
    circle = plt.Circle((0.5, 0.5), 0.45, color=metric_data['color'], alpha=0.15, transform=ax.transAxes)
    ax.add_patch(circle)

    # Add metric value
    ax.text(0.5, 0.65, f"{metric_data['value']:.4f}".rstrip('0').rstrip('.') if metric_data['value'] < 1 else f"{metric_data['value']:.2f}",
            transform=ax.transAxes, ha='center', va='center',
            fontsize=48, fontweight='bold', color=metric_data['color'])

    # Add metric unit
    if metric_data['unit']:
        ax.text(0.5, 0.45, metric_data['unit'],
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='#555555')

    # Add metric name
    ax.text(0.5, 0.20, metric_name,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=16, fontweight='bold', color='#333333')

    # Style the subplot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Add border
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add a subtle background
    ax.set_facecolor('#f8f9fa')

# Add main title
fig.suptitle('Model Performance Metrics Dashboard\nHybrid Resampling V2 Final Model',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(output_dir / 'metrics_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {output_dir / 'metrics_dashboard.png'}")
plt.close()

# ============================================================================
# 4. FEATURE IMPORTANCE CHART
# ============================================================================
print("Creating Feature Importance Chart...")

feature_imp_path = '/home/CHAIN/project/temp/dsde/data/models/feature_importance_optimized.csv'
if Path(feature_imp_path).exists():
    df_imp = pd.read_csv(feature_imp_path)
    
    # Sort and take top 15
    df_imp = df_imp.sort_values('importance', ascending=False).head(15)
    
    # Clean up feature names for display
    df_imp['feature_clean'] = df_imp['feature'].str.replace('_', ' ').str.title()
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Horizontal bar chart
    bars = ax.barh(df_imp['feature_clean'], df_imp['importance'], color='#8e44ad', alpha=0.8)
    ax.invert_yaxis()  # Best feature at top
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (width * 0.01), bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'feature_importance.png'}")
    plt.close()
else:
    print(f"⚠ Warning: Feature importance file not found at {feature_imp_path}")

print("\n" + "="*80)
print("All visualizations created successfully!")
print(f"Output directory: {output_dir}")
print("="*80)
