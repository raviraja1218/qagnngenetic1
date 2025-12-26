"""
QAGNN Phase 2: Data loading utilities
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CircuitDataset(Dataset):
    """PyTorch Dataset for circuit training data"""
    
    def __init__(self, dataset_type='train'):
        """
        Args:
            dataset_type: 'train', 'val', or 'test'
        """
        self.path = f'data/processed/{dataset_type}_dataset.h5'
        with h5py.File(self.path, 'r') as f:
            self.X = f['X'][:].astype(np.float32)
            self.y = f['y'][:].astype(np.float32)
        
        print(f'📊 Loaded {dataset_type}: {len(self.X)} samples')
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def load_datasets(batch_size=32):
    """
    Load all datasets with DataLoader for training
    
    Returns:
        train_loader, val_loader, test_loader, (X_test, y_test)
    """
    train_dataset = CircuitDataset('train')
    val_dataset = CircuitDataset('val')
    test_dataset = CircuitDataset('test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Get raw test data for evaluation
    X_test, y_test = test_dataset.X, test_dataset.y
    
    print(f'🎯 Data loaded:')
    print(f'   Train batches: {len(train_loader)}')
    print(f'   Val batches: {len(val_loader)}')
    print(f'   Test samples: {len(test_dataset)}')
    
    return train_loader, val_loader, test_loader, (X_test, y_test)

if __name__ == '__main__':
    # Test the data loader
    train_loader, val_loader, test_loader, (X_test, y_test) = load_datasets()
    for X_batch, y_batch in train_loader:
        print(f'Batch shape: X={X_batch.shape}, y={y_batch.shape}')
        break
