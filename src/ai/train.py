"""
QAGNN Phase 2: Training script for circuit predictor
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path

from src.ai.model import CircuitPredictor
from src.ai.data_loader import load_datasets

class Trainer:
    """Handles model training, validation, and checkpointing"""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.device = self._setup_device()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
        # Create directories
        self.checkpoint_dir = Path('models/checkpoints')
        self.final_model_dir = Path('models/final')
        self.log_dir = Path('results/logs/phase2')
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.final_model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': [],
            'learning_rate': []
        }
        
    def get_default_config(self):
        """Default training configuration"""
        return {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'dropout_rate': 0.2,
            'patience': 10,  # For learning rate reduction
            'early_stopping_patience': 15,
            'checkpoint_frequency': 10,  # Save every N epochs
        }
    
    def _setup_device(self):
        """Setup GPU or CPU device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f'🎮 Using GPU: {torch.cuda.get_device_name(0)}')
            print(f'   Memory: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB free')
        else:
            device = torch.device('cpu')
            print('⚠️  Using CPU (GPU not available)')
        return device
    
    def compute_r2(self, y_true, y_pred):
        """Calculate R² score"""
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_r2 = 0
        num_batches = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch).squeeze()
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_r2 += self.compute_r2(y_batch, predictions).item()
            num_batches += 1
            
            # Print progress every 10% of batches
            if batch_idx % max(1, len(train_loader) // 10) == 0:
                print(f'   Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}')
        
        return epoch_loss / num_batches, epoch_r2 / num_batches
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        val_r2 = 0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = self.model(X_batch).squeeze()
                loss = self.criterion(predictions, y_batch)
                
                val_loss += loss.item()
                val_r2 += self.compute_r2(y_batch, predictions).item()
                num_batches += 1
        
        return val_loss / num_batches, val_r2 / num_batches
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': self.history['train_loss'][-1],
            'val_loss': self.history['val_loss'][-1],
            'val_r2': self.history['val_r2'][-1],
            'config': self.config
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = self.final_model_dir / 'circuit_predictor_best.pt'
            torch.save(checkpoint, best_path)
        
        # Final model (always save latest)
        final_path = self.final_model_dir / 'circuit_predictor_latest.pt'
        torch.save(checkpoint, final_path)
        
        return checkpoint_path
    
    def save_history(self):
        """Save training history to CSV"""
        import pandas as pd
        
        history_df = pd.DataFrame({
            'epoch': list(range(1, len(self.history['train_loss']) + 1)),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'train_r2': self.history['train_r2'],
            'val_r2': self.history['val_r2'],
            'learning_rate': self.history['learning_rate']
        })
        
        history_path = self.log_dir / 'training_history.csv'
        history_df.to_csv(history_path, index=False)
        print(f'📊 Training history saved to {history_path}')
        
        # Also save as JSON for quick loading
        json_path = self.log_dir / 'training_history.json'
        with open(json_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train(self):
        """Main training loop"""
        print('=' * 60)
        print('🚀 QAGNN Phase 2: Deep Learning Training')
        print('=' * 60)
        
        # Load data
        print('📦 Loading datasets...')
        train_loader, val_loader, test_loader, test_data = load_datasets(
            batch_size=self.config['batch_size']
        )
        
        # Initialize model
        print('🏗️  Initializing model...')
        self.model = CircuitPredictor(dropout_rate=self.config['dropout_rate'])
        self.model = self.model.to(self.device)
        
        print(f'📐 Model parameters: {self.model.get_num_parameters():,}')
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config['patience'],
            verbose=True
        )
        
        print(f'⚙️  Training config:')
        for key, value in self.config.items():
            print(f'   {key}: {value}')
        
        print('\\n🎯 Starting training...')
        print('=' * 60)
        
        best_val_r2 = -np.inf
        early_stopping_counter = 0
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            
            # Training
            train_loss, train_r2 = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_r2 = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_r2'].append(train_r2)
            self.history['val_r2'].append(val_r2)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch:03d}/{self.config["epochs"]:03d} | '
                  f'Time: {epoch_time:.1f}s | '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'  Train: Loss={train_loss:.4f}, R²={train_r2:.4f}')
            print(f'  Val:   Loss={val_loss:.4f}, R²={val_r2:.4f}')
            print('-' * 40)
            
            # Check for best model
            is_best = False
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                is_best = True
                early_stopping_counter = 0
                print(f'🏆 New best model! R²={val_r2:.4f}')
            else:
                early_stopping_counter += 1
            
            # Save checkpoint
            if epoch % self.config['checkpoint_frequency'] == 0 or is_best:
                checkpoint_path = self.save_checkpoint(epoch, is_best)
                print(f'💾 Checkpoint saved: {checkpoint_path}')
            
            # Early stopping
            if early_stopping_counter >= self.config['early_stopping_patience']:
                print(f'⏹️  Early stopping triggered after {epoch} epochs')
                break
        
        # Training complete
        total_time = time.time() - start_time
        print('=' * 60)
        print(f'✅ Training completed in {total_time/60:.1f} minutes')
        print(f'🏆 Best validation R²: {best_val_r2:.4f}')
        
        # Save final model and history
        self.save_checkpoint(epoch, is_best=False)
        self.save_history()
        
        # Save final config
        config_path = self.final_model_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f'📁 Models saved in: {self.final_model_dir}')
        print(f'📊 History saved in: {self.log_dir}')
        
        return best_val_r2

def main():
    """Main training execution"""
    trainer = Trainer()
    best_r2 = trainer.train()
    
    # Final message
    print('\\n' + '=' * 60)
    print('🎯 PHASE 2 COMPLETION CHECKLIST:')
    print('=' * 60)
    print(f'✅ Model trained: R²={best_r2:.4f} (Target: >0.92)')
    print('✅ Check if R² > 0.92: ', 'YES' if best_r2 > 0.92 else 'NO - NEEDS IMPROVEMENT')
    print('✅ Models saved in models/final/')
    print('✅ Training history saved in results/logs/phase2/')
    print('\\nNext steps:')
    print('1. Analyze model performance')
    print('2. Generate visualizations')
    print('3. Test inference speed')
    print('4. Create novel circuit designs')

if __name__ == '__main__':
    main()
