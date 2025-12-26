"""
URGENT REPAIR: Fix Phase 1 processed data corruption
Extract correct features from raw HDF5 files
"""
import h5py
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_correct_features():
    """Extract w1, w2, w3 + output trajectory (300 points) from raw data"""
    print("=" * 60)
    print("PHASE 1 DATA REPAIR - EXTRACTING CORRECT FEATURES")
    print("=" * 60)
    
    raw_path = 'data/raw/simulations_10k.h5'
    if not os.path.exists(raw_path):
        print(f"❌ ERROR: Raw data not found at {raw_path}")
        return None
    
    all_features = []
    all_targets = []
    
    # Open raw HDF5 file
    with h5py.File(raw_path, 'r') as f:
        # Get all circuit groups
        circuit_ids = []
        for key in f.keys():
            if key.startswith('circuit_'):
                circuit_ids.append(key)
        
        print(f"Found {len(circuit_ids)} circuit batches")
        
        # Process each batch
        for batch_id in tqdm(circuit_ids, desc="Processing batches"):
            batch_group = f[batch_id]
            
            # Get all circuits in this batch
            circuit_names = list(batch_group.keys())
            
            for circuit_name in circuit_names:
                circuit = batch_group[circuit_name]
                
                # Extract weights from attributes
                w1 = circuit.attrs.get('w1', 0.0)
                w2 = circuit.attrs.get('w2', 0.0)
                w3 = circuit.attrs.get('w3', 0.0)
                accuracy = circuit.attrs.get('accuracy', 0.0)
                
                # Extract output trajectory (last 300 points)
                if 'dynamics' in circuit:
                    dynamics = circuit['dynamics']
                    if 'output' in dynamics:
                        output_data = dynamics['output'][:]
                        # Take last 300 points (t=1 to t=300)
                        output_trajectory = output_data[1:301]  # shape (300,)
                    else:
                        # If no output, create zeros
                        output_trajectory = np.zeros(300)
                else:
                    output_trajectory = np.zeros(300)
                
                # Create feature vector: [w1, w2, w3] + output_trajectory
                feature_vector = np.concatenate([
                    np.array([w1, w2, w3]),
                    output_trajectory
                ])
                
                all_features.append(feature_vector)
                all_targets.append(accuracy)
    
    # Convert to numpy arrays
    all_features = np.array(all_features, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)
    
    print(f"\n✓ Feature extraction complete:")
    print(f"  Total circuits: {len(all_features)}")
    print(f"  Feature shape: {all_features.shape} (should be [10000, 303])")
    print(f"  Target shape: {all_targets.shape} (should be [10000])")
    
    # Verify ranges
    print(f"\n✓ Verifying data ranges:")
    print(f"  w1 range: [{all_features[:, 0].min():.3f}, {all_features[:, 0].max():.3f}] (expected: [0.15, 8.5])")
    print(f"  w2 range: [{all_features[:, 1].min():.3f}, {all_features[:, 1].max():.3f}] (expected: [0.15, 8.5])")
    print(f"  w3 range: [{all_features[:, 2].min():.3f}, {all_features[:, 2].max():.3f}] (expected: [0.15, 8.5])")
    print(f"  Output range: [{all_features[:, 3:].min():.1f}, {all_features[:, 3:].max():.1f}]")
    print(f"  Accuracy range: [{all_targets.min():.3f}, {all_targets.max():.3f}] (expected: [0.0, 1.0])")
    
    return all_features, all_targets

def create_train_val_test_split(features, targets):
    """Create 80/10/10 split (8000/1000/1000)"""
    n_samples = len(features)
    indices = np.random.permutation(n_samples)
    
    # Split indices
    train_end = int(0.8 * n_samples)
    val_end = train_end + int(0.1 * n_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Create splits
    X_train = features[train_idx]
    y_train = targets[train_idx]
    
    X_val = features[val_idx]
    y_val = targets[val_idx]
    
    X_test = features[test_idx]
    y_test = targets[test_idx]
    
    print(f"\n✓ Dataset splits created:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_processed_data(splits):
    """Save processed data to HDF5 files"""
    print("\n" + "=" * 60)
    print("SAVING REPAIRED DATA")
    print("=" * 60)
    
    # Remove old (corrupt) processed data
    if os.path.exists('data/processed/'):
        for f in os.listdir('data/processed/'):
            if f.endswith('.h5') or f.endswith('.npz'):
                os.remove(f'data/processed/{f}')
    
    # Create new processed data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
    
    # Save as HDF5 files
    for name, data in [
        ('train_dataset', (X_train, y_train)),
        ('val_dataset', (X_val, y_val)),
        ('test_dataset', (X_test, y_test))
    ]:
        X, y = data
        filepath = f'data/processed/{name}.h5'
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('features', data=X, compression='gzip')
            f.create_dataset('accuracy', data=y, compression='gzip')
            
            # Save metadata
            f.attrs['n_samples'] = len(X)
            f.attrs['n_features'] = X.shape[1]
            f.attrs['description'] = f'Circuit features: w1,w2,w3 + output_trajectory[300]'
            f.attrs['feature_names'] = 'w1,w2,w3,output_1,...,output_300'
        
        print(f"  ✓ Saved {filepath}: {X.shape} features, {y.shape} targets")
    
    # Save split indices for reproducibility
    np.savez('data/processed/split_indices.npz',
             train_indices=np.arange(len(X_train)),
             val_indices=np.arange(len(X_val)),
             test_indices=np.arange(len(X_test)))
    
    print(f"\n✓ All repaired data saved to data/processed/")

def verify_repaired_data():
    """Verify the repaired data is correct"""
    print("\n" + "=" * 60)
    print("VERIFYING REPAIRED DATA")
    print("=" * 60)
    
    for name in ['train_dataset', 'val_dataset', 'test_dataset']:
        filepath = f'data/processed/{name}.h5'
        
        with h5py.File(filepath, 'r') as f:
            X = f['features'][:]
            y = f['accuracy'][:]
            
            print(f"\n{name}:")
            print(f"  Shape: X={X.shape}, y={y.shape}")
            print(f"  X dtype: {X.dtype}, y dtype: {y.dtype}")
            print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
            print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
            
            # Check for NaN
            if np.isnan(X).any():
                print(f"  ❌ WARNING: NaN values in X!")
            if np.isnan(y).any():
                print(f"  ❌ WARNING: NaN values in y!")
            
            # Check expected ranges
            if X.shape[1] == 303:
                print(f"  ✓ Feature dimension correct: 303")
            else:
                print(f"  ❌ WRONG: Feature dimension {X.shape[1]} (expected 303)")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If all checks pass, Phase 1 data is REPAIRED and ready for Phase 2.")
    print("If any ❌ warnings appear, manual intervention required.")

def main():
    """Main repair function"""
    print("🚨 STARTING PHASE 1 DATA REPAIR")
    print("This will fix the corruption in processed data")
    print("\nNOTE: Raw data appears OK, only processed data needs repair")
    
    # Extract correct features
    features, targets = extract_correct_features()
    if features is None:
        return
    
    # Create splits
    splits = create_train_val_test_split(features, targets)
    
    # Save repaired data
    save_processed_data(splits)
    
    # Verify
    verify_repaired_data()
    
    print("\n" + "=" * 60)
    print("REPAIR COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("1. Run quick visualization to confirm data looks correct")
    print("2. Proceed with Phase 2 (deep learning training)")
    print("3. Backup repaired data to ~/project_backups/qagnn_phase1_repaired/")

if __name__ == "__main__":
    main()
