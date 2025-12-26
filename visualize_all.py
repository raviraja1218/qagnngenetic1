import sys
sys.path.append('/home/raviraja/projects/qagnn/src')
from analysis.visualization import Phase1Visualizer
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

print("=== CREATING ALL VISUALIZATIONS ===")

# Create directory
viz_dir = 'results/figures/process/phase1'
os.makedirs(viz_dir, exist_ok=True)

# Initialize visualizer
viz = Phase1Visualizer()

# 1. Circuit schematic
print("1. Creating circuit schematic...")
viz.create_circuit_schematic()
print(f"   Saved: {viz_dir}/circuit_schematic.png")

# 2. Load batch 1 data and create time series
print("2. Loading batch data...")
viz.load_batch_data('data/raw/batch_0000_0999.h5')

print("3. Creating time series examples...")
viz.plot_time_series_examples(n_examples=3)
print(f"   Saved: {viz_dir}/ode_time_series_examples.png")

print("4. Creating parameter distributions...")
viz.plot_parameter_distributions()
print(f"   Saved: {viz_dir}/parameter_distributions.png")

# 5. Additional visualizations
print("5. Creating additional plots...")

# Accuracy distribution
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(viz.data['accuracy'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Circuit Accuracy', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('Distribution of Circuit Accuracies (Batch 1)', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/accuracy_distribution.png', dpi=300)
plt.close()
print(f"   Saved: {viz_dir}/accuracy_distribution.png")

# Parameter correlation
import pandas as pd
import seaborn as sns
data = pd.DataFrame({
    'w1': viz.data['w1'],
    'w2': viz.data['w2'], 
    'w3': viz.data['w3'],
    'accuracy': viz.data['accuracy']
})
corr_matrix = data.corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Parameter Correlation Matrix', fontsize=11)
plt.tight_layout()
plt.savefig(f'{viz_dir}/correlation_matrix.png', dpi=300)
plt.close()
print(f"   Saved: {viz_dir}/correlation_matrix.png")

print("✅ All visualizations created!")
print(f"Check directory: {viz_dir}")
