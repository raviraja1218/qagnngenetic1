"""
QAGNN Phase 2: FIXED Training Script
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from pathlib import Path
from src.ai.model import CircuitPredictor
from src.ai.data_loader import load_datasets

class FixedTrainer:
    """Fixed trainer with proper R² calculation and data normalization"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'🎮 Using: {self.device}')
        
        # Create directories
        Path('models/final').mkdir(exist_ok=True)
        Path('results/logs/phase2').mkdir(exist_ok=True)
        
    def compute_r2_fixed(self, y_true, y_pred):
        """PROPER R² calculation"""
        y_true_mean = torch.mean(y_true)
        ss_total = torch.sum((y_true - y_true_mean) ** 2)
        ss_residual = torch.sum((y_true - y_pred) ** 2)
        
        # Avoid division by zero
        if ss_total == 0:
            return 0.0
        
        r2 = 1 - (ss_residual / ss_total)
        return r2.item()
    
    def normalize_data(self, X_train, X_val, X_test):
        """Normalize data to [0,1] range"""
        # Find min and max from training data only
        X_min = X_train.min(axis=0, keepdims=True)
        X_max = X_train.max(axis=0, keepdims=True)
        
        # Avoid division by zero
        range_vals = X_max - X_min
        range_vals[range_vals == 0] = 1.0
        
        # Normalize
        X_train_norm = (X_train - X_min) / range_vals
        X_val_norm = (X_val - X_min) / range_vals
        X_test_norm = (X_test - X_min) / range_vals
        
        print(f'📊 Normalization:')
        print(f'   X_train: [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]')
        print(f'   X_val: [{X_val_norm.min():.3f}, {X_val_norm.max():.3f}]')
        
        return X_train_norm, X_val_norm, X_test_norm
    
    def create_fixed_datasets(self, batch_size=32):
        """Create fixed datasets with normalization"""
        print('📦 Loading and fixing datasets...')
        
        from src.ai.data_loader import CircuitDataset
        import numpy as np
        
        # Load raw data
        train_dataset = CircuitDataset('train')
        val_dataset = CircuitDataset('val')
        test_dataset = CircuitDataset('test')
        
        X_train_raw = train_dataset.X
        y_train = train_dataset.y
        X_val_raw = val_dataset.X
        y_val = val_dataset.y
        X_test_raw = test_dataset.X
        y_test = test_dataset.y
        
        # Normalize
        X_train, X_val, X_test = self.normalize_data(X_train_raw, X_val_raw, X_test_raw)
        
        # Convert to PyTorch datasets
        class NormalizedDataset(torch.utils.data.Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32)
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        
        train_dataset_norm = NormalizedDataset(X_train, y_train)
        val_dataset_norm = NormalizedDataset(X_val, y_val)
        test_dataset_norm = NormalizedDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset_norm, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset_norm, batch_size=batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset_norm, batch_size=batch_size, shuffle=False
        )
        
        print(f'✅ Datasets fixed:')
        print(f'   Train: {len(train_dataset_norm)} samples')
        print(f'   Val: {len(val_dataset_norm)} samples')
        print(f'   Test: {len(test_dataset_norm)} samples')
        
        return train_loader, val_loader, test_loader
    
    def train_simple(self, epochs=20, learning_rate=0.001):
        """Simple training loop to verify everything works"""
        print('=' * 60)
        print('🚀 QAGNN Phase 2: FIXED Training')
        print('=' * 60)
        
        # Load fixed data
        train_loader, val_loader, _ = self.create_fixed_datasets()
        
        # Create simple model
        model = CircuitPredictor(dropout_rate=0.0)  # No dropout for now
        model = model.to(self.device)
        
        print(f'📐 Model parameters: {sum(p.numel() for p in model.parameters()):,}')
        
        # Use simpler optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        print('⚙️  Starting training...')
        print('=' * 60)
        
        history = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_r2 = 0
            num_batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_r2 += self.compute_r2_fixed(y_batch, predictions)
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            avg_train_r2 = train_r2 / num_batches
            
            # Validation
            model.eval()
            val_loss = 0
            val_r2 = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    predictions = model(X_batch).squeeze()
                    
                    val_loss += criterion(predictions, y_batch).item()
                    val_r2 += self.compute_r2_fixed(y_batch, predictions)
                    num_val_batches += 1
            
            avg_val_loss = val_loss / num_val_batches
            avg_val_r2 = val_r2 / num_val_batches
            
            # Save history
            history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_r2': avg_train_r2,
                'val_r2': avg_val_r2
            })
            
            print(f'Epoch {epoch+1:03d}/{epochs:03d} | '
                  f'Train Loss: {avg_train_loss:.4f}, R²: {avg_train_r2:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f}, R²: {avg_val_r2:.4f}')
            
            # Early success check
            if avg_val_r2 > 0.5:  # If we get decent R²
                print(f'✅ SUCCESS! R² > 0.5 achieved at epoch {epoch+1}')
                break
        
        # Save model if good
        if avg_val_r2 > 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_r2': avg_val_r2,
                'history': history
            }, 'models/final/circuit_predictor_fixed.pt')
            print(f'💾 Model saved: R² = {avg_val_r2:.4f}')
        
        # Save history
        import json
        with open('results/logs/phase2/training_history_fixed.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print('=' * 60)
        print(f'✅ Training completed')
        print(f'🏆 Final Validation R²: {avg_val_r2:.4f}')
        print('=' * 60)
        
        return history, avg_val_r2

def main():
    """Run fixed training"""
    trainer = FixedTrainer()
    history, final_r2 = trainer.train_simple(epochs=20)
    
    print('\n🎯 NEXT STEPS:')
    if final_r2 > 0.8:
        print('✅ Excellent! Proceed with full training')
    elif final_r2 > 0.5:
        print('⚠️  Decent. Try tuning hyperparameters')
    elif final_r2 > 0:
        print('❌ Poor. Check data quality')
    else:
        print('🚨 CRITICAL: Still negative R². Major issue with data.')

if __name__ == '__main__':
    main()
