"""
FIXED Data Loader using robust-scaled data
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FixedCircuitDataset(Dataset):
    """Fixed dataset using robust-scaled data"""
    
    def __init__(self, dataset_type='train'):
        if dataset_type == 'train':
            path = 'data/processed/train_dataset_ROBUST.h5'
        elif dataset_type == 'val':
            path = 'data/processed/val_dataset_ROBUST.h5'
        else:  # test
            path = 'data/processed/test_dataset_ROBUST.h5'
        
        with h5py.File(path, 'r') as f:
            self.X = f['X'][:]
            self.y = f['y'][:]
        
        print(f'📊 Loaded {dataset_type}: {len(self.X)} samples')
        print(f'   X range: {self.X.min():.3f} to {self.X.max():.3f}')
        print(f'   y range: {self.y.min():.3f} to {self.y.max():.3f}')
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32)

def load_fixed_datasets(batch_size=32):
    """Load fixed datasets"""
    train_dataset = FixedCircuitDataset('train')
    val_dataset = FixedCircuitDataset('val')
    test_dataset = FixedCircuitDataset('test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Get raw test data
    X_test, y_test = test_dataset.X, test_dataset.y
    
    return train_loader, val_loader, test_loader, (X_test, y_test)

if __name__ == '__main__':
    # Test
    train_loader, val_loader, test_loader, _ = load_fixed_datasets()
    print(f'✅ Data loaders created')
    print(f'   Train batches: {len(train_loader)}')
    print(f'   Val batches: {len(val_loader)}')
