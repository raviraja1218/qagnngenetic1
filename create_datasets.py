import sys
sys.path.append('/home/raviraja/projects/qagnn/src')
import h5py
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split

print("=== CREATING DATASET SPLITS ===")

# Check if all batches are complete
batch_files = []
for i in range(10):
    filename = f"data/raw/batch_{(i)*1000:04d}_{(i)*1000+999:04d}.h5"
    if os.path.exists(filename):
        batch_files.append(filename)
    else:
        print(f"❌ Missing: {filename}")

print(f"Found {len(batch_files)} batch files")

if len(batch_files) < 10:
    print("⚠️ Not all batches complete. Waiting...")
    # We'll run this later
    sys.exit(0)

# Load all data
print("Loading all circuit data...")
all_features = []
all_labels = []

for batch_file in batch_files:
    with h5py.File(batch_file, 'r') as f:
        circuit_ids = list(f.keys())
        for circuit_id in circuit_ids:
            circuit = f[circuit_id]
            
            # Extract features: weights + some dynamics
            w1 = circuit.attrs['w1']
            w2 = circuit.attrs['w2']
            w3 = circuit.attrs['w3']
            accuracy = circuit.attrs['accuracy']
            
            # Get time series output (simplified: take every 10th point)
            if 'output' in circuit:
                output = circuit['output'][:]
                # Sample 30 points from the time series
                sampled_output = output[::10]  # 30 points (0, 10, 20, ..., 290)
            else:
                sampled_output = np.zeros(30)
            
            # Create feature vector: [w1, w2, w3] + sampled_output
            features = np.concatenate([[w1, w2, w3], sampled_output])
            all_features.append(features)
            all_labels.append(accuracy)

all_features = np.array(all_features)
all_labels = np.array(all_labels)

print(f"Total circuits loaded: {len(all_features)}")
print(f"Feature shape: {all_features.shape}")
print(f"Labels shape: {all_labels.shape}")

# Split into train/val/test
print("\nSplitting data...")
X_temp, X_test, y_temp, y_test = train_test_split(
    all_features, all_labels, test_size=0.1, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42  # 0.111 * 0.9 = 0.1
)

print(f"Train: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# Save datasets
print("\nSaving datasets...")
os.makedirs('data/processed', exist_ok=True)

with h5py.File('data/processed/train_dataset.h5', 'w') as f:
    f.create_dataset('X', data=X_train)
    f.create_dataset('y', data=y_train)
    f.attrs['description'] = 'Training dataset for QAGNN'
    f.attrs['num_samples'] = X_train.shape[0]
    f.attrs['feature_dim'] = X_train.shape[1]

with h5py.File('data/processed/val_dataset.h5', 'w') as f:
    f.create_dataset('X', data=X_val)
    f.create_dataset('y', data=y_val)
    f.attrs['description'] = 'Validation dataset for QAGNN'

with h5py.File('data/processed/test_dataset.h5', 'w') as f:
    f.create_dataset('X', data=X_test)
    f.create_dataset('y', data=y_test)
    f.attrs['description'] = 'Test dataset for QAGNN'

print("✅ Datasets saved!")
print(f"  Train: data/processed/train_dataset.h5")
print(f"  Val:   data/processed/val_dataset.h5")
print(f"  Test:  data/processed/test_dataset.h5")
