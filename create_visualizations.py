import sys
sys.path.append('/home/raviraja/projects/qagnn/src')
from analysis.visualization import Phase1Visualizer
import os

viz = Phase1Visualizer()
os.makedirs('results/figures/process/phase1', exist_ok=True)

# 1. Circuit schematic
viz.create_circuit_schematic()

# 2. Load and visualize batch 1
viz.load_batch_data('data/raw/batch_0000_0999.h5')
viz.plot_time_series_examples(n_examples=3)
viz.plot_parameter_distributions()

print("✅ Basic visualizations created")
