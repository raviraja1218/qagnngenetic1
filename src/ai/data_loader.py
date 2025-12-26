"""
Data loading and normalization for Phase 2
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class CircuitDataset(Dataset):
    """PyTorch Dataset for circuit data"""
    def __init__(self, h5_path, normalize=True):
        super().__init__()
        self.h5_path = h5_path
        
        # Load data
        with h5py.File(h5_path, 'r') as f:
            self.features = f['features'][:]
            self.labels = f['accuracy'][:]
            
        # Convert to float32
        self.features = self.features.astype(np.float32)
        self.labels = self.labels.astype(np.float32).reshape(-1, 1)
        
        # Normalize if needed
        self.normalize = normalize
        if normalize:
            self._normalize_features()
            
    def _normalize_features(self):
        """Normalize features to [0, 1] range"""
        self.feature_min = self.features.min(axis=0)
        self.feature_max = self.features.max(axis=0)
        self.features = (self.features - self.feature_min) / (self.feature_max - self.feature_min + 1e-8)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])
    
    def save_normalization(self, path):
        """Save normalization parameters"""
        if self.normalize:
            np.savez(path, 
                    feature_min=self.feature_min,
                    feature_max=self.feature_max)
    
    @classmethod
    def load_normalization(cls, path):
        """Load normalization parameters"""
        data = np.load(path)
        return data['feature_min'], data['feature_max']

def create_dataloaders(batch_size=32):
    """Create train, val, test dataloaders"""
    print("Creating dataloaders...")
    
    # Create datasets
    train_dataset = CircuitDataset('data/processed/train_dataset.h5', normalize=True)
    val_dataset = CircuitDataset('data/processed/val_dataset.h5', normalize=False)
    test_dataset = CircuitDataset('data/processed/test_dataset.h5', normalize=False)
    
    # Apply same normalization to val and test
    val_dataset.feature_min = train_dataset.feature_min
    val_dataset.feature_max = train_dataset.feature_max
    val_dataset.features = (val_dataset.features - val_dataset.feature_min) / (val_dataset.feature_max - val_dataset.feature_min + 1e-8)
    
    test_dataset.feature_min = train_dataset.feature_min
    test_dataset.feature_max = train_dataset.feature_max
    test_dataset.features = (test_dataset.features - test_dataset.feature_min) / (test_dataset.feature_max - test_dataset.feature_min + 1e-8)
    
    # Save normalization parameters
    os.makedirs('models/final', exist_ok=True)
    train_dataset.save_normalization('models/final/normalization_params.npz')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Print statistics
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    print(f"✓ Test: {len(test_dataset)} samples")
    print(f"✓ Feature shape: {train_dataset.features.shape[1]} (303 expected)")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the data loader
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=32)
    
    # Test one batch
    for features, labels in train_loader:
        print(f"\n✓ Batch features shape: {features.shape}")
        print(f"✓ Batch labels shape: {labels.shape}")
        print(f"✓ Features range: [{features.min():.3f}, {features.max():.3f}] (should be ~[0,1])")
        print(f"✓ Labels range: [{labels.min():.3f}, {labels.max():.3f}] (should be ~[0,1])")
        break
