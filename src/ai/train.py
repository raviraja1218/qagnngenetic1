"""
Training script for circuit predictor model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ai.data_loader import create_dataloaders
from src.ai.model import CircuitPredictor, count_parameters

class Trainer:
    """Trainer class for model training"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs('models/checkpoints', exist_ok=True)
        os.makedirs('results/logs/phase2', exist_ok=True)
        
        # Initialize model - NOTE: 303 features, not 903!
        self.model = CircuitPredictor(
            input_dim=303,  # Our data has 303 features (3 weights + 300 outputs)
            dropout_rate=config['dropout_rate']
        ).to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.criterion = nn.MSELoss()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
    def calculate_r2(self, predictions, targets):
        """Calculate R² score"""
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return r2.item()
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_r2 = 0
        num_batches = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            r2 = self.calculate_r2(predictions, targets)
            
            total_loss += loss.item()
            total_r2 += r2
            num_batches += 1
            
            # Log progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}, R²={r2:.4f}")
        
        return total_loss / num_batches, total_r2 / num_batches
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_r2 = 0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                predictions = self.model(features)
                
                loss = self.criterion(predictions, targets)
                r2 = self.calculate_r2(predictions, targets)
                
                total_loss += loss.item()
                total_r2 += r2
                num_batches += 1
        
        return total_loss / num_batches, total_r2 / num_batches
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs")
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss, train_r2 = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_r2 = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_r2'].append(train_r2)
            self.history['val_r2'].append(val_r2)
            self.history['learning_rates'].append(current_lr)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, R²: {train_r2:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, R²: {val_r2:.4f}")
            print(f"  LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"models/checkpoints/epoch_{epoch+1:03d}.pt"
                self.model.save(checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save('models/final/circuit_predictor.pt')
                print(f"  Best model saved (val_loss={val_loss:.4f})")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                
            # Early stopping
            if self.early_stop_counter >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save history after each epoch
            self.save_history()
        
        # Final save
        self.model.save('models/final/circuit_predictor_final.pt')
        
        # Print summary
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation R²: {max(self.history['val_r2']):.4f}")
        
        return self.history
    
    def save_history(self):
        """Save training history to file"""
        history_path = 'results/logs/phase2/training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save_config(self):
        """Save training configuration"""
        config_path = 'models/final/training_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

def main():
    """Main training function"""
    # Configuration - NOTE: Changed for 303 features
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 50,
        'dropout_rate': 0.2,
        'weight_decay': 1e-5,
        'early_stopping_patience': 10
    }
    
    # Create trainer
    trainer = Trainer(config)
    trainer.save_config()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=config['batch_size']
    )
    
    # Train model
    history = trainer.train(train_loader, val_loader)
    
    # Save final history
    trainer.save_history()
    
    print("\n✓ Training completed successfully!")
    print("Model saved to: models/final/circuit_predictor.pt")
    print("History saved to: results/logs/phase2/training_history.json")

if __name__ == "__main__":
    main()
