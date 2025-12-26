"""
FINAL Fixed Trainer
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import json
from pathlib import Path

from src.ai.model import CircuitPredictor
from src.ai.data_loader_fixed import load_fixed_datasets

class FinalTrainer:
    """Final working trainer"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'🎮 Using: {self.device}')
        
        # Config
        self.config = {
            'epochs': 30,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'dropout_rate': 0.2,
        }
        
        # Create dirs
        Path('models/final').mkdir(exist_ok=True)
        Path('results/logs/phase2').mkdir(exist_ok=True)
        
        self.history = []
    
    def compute_r2(self, y_true, y_pred):
        """Proper R² calculation"""
        y_true_mean = torch.mean(y_true)
        ss_total = torch.sum((y_true - y_true_mean) ** 2)
        ss_residual = torch.sum((y_true - y_pred) ** 2)
        
        if ss_total == 0:
            return 0.0
        
        return (1 - (ss_residual / ss_total)).item()
    
    def train(self):
        """Main training"""
        print('=' * 60)
        print('🚀 QAGNN Phase 2: FINAL Training')
        print('=' * 60)
        
        # Load data
        print('📦 Loading fixed datasets...')
        train_loader, val_loader, _, _ = load_fixed_datasets(
            batch_size=self.config['batch_size']
        )
        
        # Model
        print('🏗️  Creating model...')
        model = CircuitPredictor(dropout_rate=self.config['dropout_rate'])
        model = model.to(self.device)
        print(f'📐 Parameters: {sum(p.numel() for p in model.parameters()):,}')
        
        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        criterion = nn.MSELoss()
        
        print('⚙️  Starting training...')
        print('=' * 60)
        
        best_r2 = -np.inf
        
        for epoch in range(self.config['epochs']):
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
                train_r2 += self.compute_r2(y_batch, predictions)
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
                    val_r2 += self.compute_r2(y_batch, predictions)
                    num_val_batches += 1
            
            avg_val_loss = val_loss / num_val_batches
            avg_val_r2 = val_r2 / num_val_batches
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            # Track history
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_r2': avg_train_r2,
                'val_r2': avg_val_r2,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Print
            print(f'Epoch {epoch+1:03d}/{self.config["epochs"]:03d} | '
                  f'Train: Loss={avg_train_loss:.4f}, R²={avg_train_r2:.4f} | '
                  f'Val: Loss={avg_val_loss:.4f}, R²={avg_val_r2:.4f}')
            
            # Save best
            if avg_val_r2 > best_r2:
                best_r2 = avg_val_r2
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_r2': avg_val_r2,
                    'config': self.config
                }, 'models/final/circuit_predictor_best.pt')
                print(f'  💾 New best: R²={avg_val_r2:.4f}')
        
        # Save final
        torch.save({
            'epoch': self.config['epochs'],
            'model_state_dict': model.state_dict(),
            'val_r2': avg_val_r2,
            'config': self.config
        }, 'models/final/circuit_predictor_final.pt')
        
        # Save history
        import pandas as pd
        df = pd.DataFrame(self.history)
        df.to_csv('results/logs/phase2/training_history_final.csv', index=False)
        
        print('=' * 60)
        print(f'✅ TRAINING COMPLETE')
        print(f'🏆 Best Validation R²: {best_r2:.4f}')
        print(f'🎯 Target: >0.92')
        
        if best_r2 > 0.92:
            print('   🎉 TARGET ACHIEVED!')
        elif best_r2 > 0:
            print(f'   ⚠️  Progress: {best_r2:.4f} (need improvement)')
        else:
            print('   🚨 STILL NEGATIVE - CRITICAL PROBLEM')
        
        print('=' * 60)
        
        return best_r2

def main():
    trainer = FinalTrainer()
    best_r2 = trainer.train()
    
    # Quick evaluation
    print('\n🔍 Quick Evaluation:')
    if best_r2 > 0:
        print(f'✅ Model learned something (R² = {best_r2:.4f})')
        if best_r2 > 0.5:
            print('✅ Good starting point for tuning')
        else:
            print('⚠️  Needs hyperparameter tuning')
    else:
        print('❌ Model failed to learn')

if __name__ == '__main__':
    main()
