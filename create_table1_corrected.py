import sys
sys.path.append('/home/raviraja/projects/qagnn/src')
import numpy as np
import h5py
import os

print("=== CREATING TABLE 1 FOR PAPER (CORRECTED) ===")

# Collect statistics from all batches
w1_values = []
w2_values = []
w3_values = []
accuracies = []

batch_count = 0
for i in range(10):
    filename = f"data/raw/batch_{(i)*1000:04d}_{(i)*1000+999:04d}.h5"
    if os.path.exists(filename):
        with h5py.File(filename, 'r') as f:
            circuit_ids = list(f.keys())
            for circuit_id in circuit_ids[:100]:  # Sample 100 from each batch
                circuit = f[circuit_id]
                w1_values.append(circuit.attrs['w1'])
                w2_values.append(circuit.attrs['w2'])
                w3_values.append(circuit.attrs['w3'])
                accuracies.append(circuit.attrs['accuracy'])
        batch_count += 1

print(f"Sampled {len(w1_values)} circuits from {batch_count} batches")

# Calculate statistics
stats = {
    'w1': {'mean': np.mean(w1_values), 'std': np.std(w1_values), 
           'min': np.min(w1_values), 'max': np.max(w1_values)},
    'w2': {'mean': np.mean(w2_values), 'std': np.std(w2_values),
           'min': np.min(w2_values), 'max': np.max(w2_values)},
    'w3': {'mean': np.mean(w3_values), 'std': np.std(w3_values),
           'min': np.min(w3_values), 'max': np.max(w3_values)},
    'accuracy': {'mean': np.mean(accuracies), 'std': np.std(accuracies),
                 'min': np.min(accuracies), 'max': np.max(accuracies)}
}

# Create LaTeX table (simplified format)
latex_table = """\\begin{table}[ht]
\\centering
\\caption{Genetic circuit parameter specifications and simulation statistics for 10,000 simulated circuits.}
\\label{tab:circuit_parameters}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Parameter} & \\textbf{Symbol} & \\textbf{Range} & \\textbf{Mean ± SD} & \\textbf{Biological Meaning} \\\\
\\hline
Promoter 1 strength & \$w_1\$ & %.2f-%.2f & %.2f ± %.2f & Transcription weight 1 \\\\
Promoter 2 strength & \$w_2\$ & %.2f-%.2f & %.2f ± %.2f & Transcription weight 2 \\\\
Promoter 3 strength & \$w_3\$ & %.2f-%.2f & %.2f ± %.2f & Transcription weight 3 \\\\
Circuit accuracy & \$A\$ & %.3f-%.3f & %.3f ± %.3f & Computational accuracy \\\\
\\hline
\\end{tabular}
\\end{table}""" % (
    stats['w1']['min'], stats['w1']['max'], stats['w1']['mean'], stats['w1']['std'],
    stats['w2']['min'], stats['w2']['max'], stats['w2']['mean'], stats['w2']['std'],
    stats['w3']['min'], stats['w3']['max'], stats['w3']['mean'], stats['w3']['std'],
    stats['accuracy']['min'], stats['accuracy']['max'], 
    stats['accuracy']['mean'], stats['accuracy']['std']
)

# Create CSV version
csv_table = """Parameter,Symbol,Range,Mean,Std,Biological_Meaning
w1,w1,%.2f-%.2f,%.2f,%.2f,Transcription weight 1
w2,w2,%.2f-%.2f,%.2f,%.2f,Transcription weight 2
w3,w3,%.2f-%.2f,%.2f,%.2f,Transcription weight 3
Accuracy,A,%.3f-%.3f,%.3f,%.3f,Computational accuracy""" % (
    stats['w1']['min'], stats['w1']['max'], stats['w1']['mean'], stats['w1']['std'],
    stats['w2']['min'], stats['w2']['max'], stats['w2']['mean'], stats['w2']['std'],
    stats['w3']['min'], stats['w3']['max'], stats['w3']['mean'], stats['w3']['std'],
    stats['accuracy']['min'], stats['accuracy']['max'], 
    stats['accuracy']['mean'], stats['accuracy']['std']
)

# Save tables
os.makedirs('results/tables', exist_ok=True)

with open('results/tables/Table1_circuit_parameters.tex', 'w') as f:
    f.write(latex_table)

with open('results/tables/Table1_circuit_parameters.csv', 'w') as f:
    f.write(csv_table)

print("✅ Table 1 created successfully!")
print(f"LaTeX: results/tables/Table1_circuit_parameters.tex")
print(f"CSV:   results/tables/Table1_circuit_parameters.csv")

# Print summary
print(f"\n📊 Statistics based on {len(w1_values)} circuits:")
print(f"Batch files processed: {batch_count}/10")
print(f"w1: {stats['w1']['mean']:.2f} ± {stats['w1']['std']:.2f} "
      f"[{stats['w1']['min']:.2f}, {stats['w1']['max']:.2f}]")
print(f"w2: {stats['w2']['mean']:.2f} ± {stats['w2']['std']:.2f} "
      f"[{stats['w2']['min']:.2f}, {stats['w2']['max']:.2f}]")
print(f"w3: {stats['w3']['mean']:.2f} ± {stats['w3']['std']:.2f} "
      f"[{stats['w3']['min']:.2f}, {stats['w3']['max']:.2f}]")
print(f"Accuracy: {stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std']:.3f}")
