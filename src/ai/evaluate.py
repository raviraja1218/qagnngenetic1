"""
QAGNN Phase 2: Model evaluation and analysis
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
import time

from src.ai.model import CircuitPredictor
from src.ai.data_loader import load_datasets

class ModelEvaluator:
    """Evaluates trained model performance"""
    
    def __init__(self, model_path='models/final/circuit_predictor_latest.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.model = None
        self.config = None
        
        # Setup directories
        self.figures_dir = Path('results/figures/process/phase2')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_model(self):
        """Load trained model from checkpoint"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f'📂 Loading model from {self.model_path}')
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load config
        self.config = checkpoint.get('config', {})
        
        # Initialize model
        dropout_rate = self.config.get('dropout_rate', 0.2)
        self.model = CircuitPredictor(dropout_rate=dropout_rate)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f'✅ Model loaded:')
        print(f'   Epoch: {checkpoint.get("epoch", "unknown")}')
        print(f'   Validation R²: {checkpoint.get("val_r2", 0):.4f}')
        print(f'   Parameters: {self.model.get_num_parameters():,}')
        
        return checkpoint
    
    def evaluate_on_test(self):
        """Evaluate model on test set"""
        print('\\n🧪 Evaluating on test set...')
        
        # Load data
        _, _, test_loader, (X_test, y_test) = load_datasets()
        
        # Make predictions
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                batch_pred = self.model(X_batch).cpu().numpy().squeeze()
                
                predictions.append(batch_pred)
                actuals.append(y_batch.numpy())
        
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        # R² calculation
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Correlation
        correlation = np.corrcoef(actuals, predictions)[0, 1]
        
        print('📊 Test Set Metrics:')
        print(f'   R² Score: {r2:.4f}')
        print(f'   MSE: {mse:.6f}')
        print(f'   MAE: {mae:.6f}')
        print(f'   RMSE: {rmse:.6f}')
        print(f'   Correlation: {correlation:.4f}')
        print(f'   Predictions range: {predictions.min():.3f} - {predictions.max():.3f}')
        
        # Save predictions
        predictions_path = Path('data/processed/phase2/test_predictions.npy')
        np.save(predictions_path, {
            'predictions': predictions,
            'actuals': actuals,
            'metrics': {
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'correlation': correlation
            }
        })
        print(f'💾 Predictions saved to {predictions_path}')
        
        return predictions, actuals, r2
    
    def measure_inference_speed(self, num_runs=1000, batch_sizes=[1, 32, 1000]):
        """Measure inference speed for different batch sizes"""
        print('\\n⏱️  Measuring inference speed...')
        
        if not torch.cuda.is_available():
            print('⚠️  GPU not available, using CPU for inference')
        
        speed_results = {}
        
        for batch_size in batch_sizes:
            print(f'   Testing batch size: {batch_size}')
            
            # Create dummy data
            dummy_input = torch.randn(batch_size, 303).to(self.device)
            
            # Warm-up
            for _ in range(10):
                _ = self.model(dummy_input)
            
            # Time inference
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = self.model(dummy_input)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            # Calculate speed
            total_time = end_time - start_time
            time_per_batch = total_time / num_runs
            time_per_sample = time_per_batch / batch_size
            samples_per_second = batch_size / time_per_batch
            
            speed_results[batch_size] = {
                'time_per_sample_ms': time_per_sample * 1000,
                'samples_per_second': samples_per_second,
                'time_per_batch_ms': time_per_batch * 1000
            }
            
            print(f'     Time per sample: {time_per_sample*1000:.3f} ms')
            print(f'     Samples/second: {samples_per_second:,.0f}')
        
        # Save results
        speed_path = Path('results/logs/phase2/inference_speed.json')
        import json
        with open(speed_path, 'w') as f:
            json.dump(speed_results, f, indent=2)
        
        print(f'💾 Inference speed saved to {speed_path}')
        
        # Check target
        single_sample_time = speed_results[1]['time_per_sample_ms']
        print(f'\\n🎯 Target: <1 ms per prediction')
        print(f'   Achieved: {single_sample_time:.3f} ms')
        print(f'   Status: {"✅ PASS" if single_sample_time < 1 else "❌ FAIL"}')
        
        return speed_results
    
    def plot_predictions_vs_actuals(self, predictions, actuals, r2):
        """Create prediction vs actual scatter plot"""
        print('\\n🎨 Creating prediction scatter plot...')
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        scatter = plt.scatter(actuals, predictions, alpha=0.6, s=20, 
                             c=np.abs(predictions - actuals), cmap='viridis')
        
        # Perfect prediction line
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect prediction')
        
        # Statistics text
        stats_text = f'R² = {r2:.4f}\\nn = {len(predictions):,}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.colorbar(scatter, label='Absolute Error')
        plt.xlabel('Actual Accuracy', fontsize=12)
        plt.ylabel('Predicted Accuracy', fontsize=12)
        plt.title('Model Predictions vs Actual Values', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        plot_path = self.figures_dir / 'predictions_vs_actuals.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'💾 Plot saved to {plot_path}')
        
        # Also create histogram of residuals
        self.plot_residuals(predictions, actuals)
        
        return plot_path
    
    def plot_residuals(self, predictions, actuals):
        """Plot distribution of residuals"""
        residuals = predictions - actuals
        
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Residual (Predicted - Actual)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Residuals', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot
        plt.subplot(1, 2, 2)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Statistics
        stats_text = f'Mean: {residuals.mean():.6f}\\nStd: {residuals.std():.6f}\\nSkew: {stats.skew(residuals):.3f}'
        plt.figtext(0.15, 0.85, stats_text, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plot_path = self.figures_dir / 'residuals_analysis.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'💾 Residuals plot saved to {plot_path}')
        
        return plot_path
    
    def analyze_feature_importance(self):
        """Analyze which features are most important"""
        print('\\n🔍 Analyzing feature importance...')
        
        # Use gradient-based importance
        _, _, test_loader, _ = load_datasets(batch_size=100)
        
        # Get a batch of data
        X_batch, y_batch = next(iter(test_loader))
        X_batch = X_batch.to(self.device)
        X_batch.requires_grad = True
        
        # Forward pass
        predictions = self.model(X_batch)
        
        # Compute gradients
        predictions.sum().backward()
        
        # Calculate importance scores (average absolute gradient)
        importance = torch.mean(torch.abs(X_batch.grad), dim=0).cpu().numpy()
        
        # Sort features by importance
        feature_indices = np.argsort(importance)[::-1]  # Descending
        
        # Print top features
        print('Top 10 most important features:')
        for i, idx in enumerate(feature_indices[:10]):
            if idx < 3:
                feature_name = f'Weight w{idx+1}'
            else:
                time_point = idx - 3
                feature_name = f'Time point {time_point}'
            
            print(f'   {i+1:2d}. {feature_name}: {importance[idx]:.6f}')
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        
        # Top 30 features
        top_n = min(30, len(importance))
        top_indices = feature_indices[:top_n]
        top_importance = importance[top_indices]
        
        # Create feature names
        feature_names = []
        for idx in top_indices:
            if idx < 3:
                feature_names.append(f'w{idx+1}')
            else:
                feature_names.append(f't{idx-3}')
        
        bars = plt.bar(range(top_n), top_importance)
        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance Scores', fontsize=14)
        plt.xticks(range(top_n), feature_names, rotation=45, fontsize=8)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Color bars by feature type
        for i, idx in enumerate(top_indices):
            if idx < 3:
                bars[i].set_color('red')  # Weights
            else:
                bars[i].set_color('blue')  # Time points
        
        plot_path = self.figures_dir / 'feature_importance.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'💾 Feature importance plot saved to {plot_path}')
        
        # Save importance scores
        importance_path = Path('data/processed/phase2/feature_importance.npy')
        np.save(importance_path, {
            'importance_scores': importance,
            'sorted_indices': feature_indices,
            'top_features': feature_indices[:10]
        })
        
        return importance
    
    def generate_training_curves(self):
        """Generate training history plots"""
        print('\\n📈 Generating training curves...')
        
        history_path = Path('results/logs/phase2/training_history.csv')
        if not history_path.exists():
            print('⚠️  Training history not found')
            return
        
        history_df = pd.read_csv(history_path)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(history_df['epoch'], history_df['train_loss'], label='Train', linewidth=2)
        ax.plot(history_df['epoch'], history_df['val_loss'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # R² curves
        ax = axes[0, 1]
        ax.plot(history_df['epoch'], history_df['train_r2'], label='Train', linewidth=2)
        ax.plot(history_df['epoch'], history_df['val_r2'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Training and Validation R²', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 0]
        ax.plot(history_df['epoch'], history_df['learning_rate'], linewidth=2, color='green')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Loss vs R²
        ax = axes[1, 1]
        scatter = ax.scatter(history_df['val_loss'], history_df['val_r2'], 
                           c=history_df['epoch'], cmap='viridis', s=50)
        ax.set_xlabel('Validation Loss', fontsize=12)
        ax.set_ylabel('Validation R²', fontsize=12)
        ax.set_title('Loss vs R² (Colored by Epoch)', fontsize=14)
        plt.colorbar(scatter, ax=ax, label='Epoch')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.figures_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'💾 Training curves saved to {plot_path}')
        
        return plot_path

def main():
    """Main evaluation routine"""
    print('=' * 60)
    print('🔬 QAGNN Phase 2: Model Evaluation')
    print('=' * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load model
    evaluator.load_model()
    
    # Evaluate on test set
    predictions, actuals, r2 = evaluator.evaluate_on_test()
    
    # Generate plots
    evaluator.plot_predictions_vs_actuals(predictions, actuals, r2)
    evaluator.generate_training_curves()
    evaluator.analyze_feature_importance()
    
    # Measure speed
    speed_results = evaluator.measure_inference_speed()
    
    print('\\n' + '=' * 60)
    print('✅ EVALUATION COMPLETE')
    print('=' * 60)
    print(f'🎯 R² Score: {r2:.4f} (Target: >0.92)')
    
    single_sample_time = speed_results[1]['time_per_sample_ms']
    print(f'⏱️  Inference Speed: {single_sample_time:.3f} ms (Target: <1 ms)')
    
    print(f'📊 Plots saved in: {evaluator.figures_dir}')
    print('\\nNext: Create novel circuit designs with inverse_design.py')

if __name__ == '__main__':
    main()
