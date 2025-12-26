import h5py
import numpy as np

print("Fixing data normalization...")

for name in ['train', 'val', 'test']:
    path = f'data/processed/{name}_dataset.h5'
    
    with h5py.File(path, 'r') as f:
        X = f['X'][:]
        y = f['y'][:]
    
    # Normalize X to [0,1]
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0  # Avoid division by zero
    X_norm = (X - X_min) / X_range
    
    # Ensure y is in [0,1]
    y_norm = np.clip(y, 0, 1)
    
    print(f"{name}: X [{X_norm.min():.3f}, {X_norm.max():.3f}], "
          f"y [{y_norm.min():.3f}, {y_norm.max():.3f}]")
    
    # Save fixed data
    with h5py.File(f'data/processed/{name}_dataset_FIXED.h5', 'w') as f:
        f.create_dataset('X', data=X_norm.astype(np.float32))
        f.create_dataset('y', data=y_norm.astype(np.float32))

print("✅ Data fixed!")
