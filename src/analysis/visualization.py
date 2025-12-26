import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from typing import Dict, List
import seaborn as sns

class Phase1Visualizer:
    """Visualization tools for Phase 1 results."""
    
    def __init__(self, style: str = 'nature'):
        """
        Initialize visualizer.
        
        Args:
            style: Plot style ('nature', 'default')
        """
        self.style = style
        self._setup_plotting()
        self.data = None
        
    def _setup_plotting(self):
        """Set up matplotlib configuration."""
        if self.style == 'nature':
            plt.rcParams.update({
                'font.size': 8,
                'axes.titlesize': 9,
                'axes.labelsize': 8,
                'xtick.labelsize': 7,
                'ytick.labelsize': 7,
                'legend.fontsize': 7,
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.format': 'png',
                'savefig.bbox': 'tight'
            })
        
        # Set color palette
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
    def create_circuit_schematic(self, save_path: str = None):
        """
        Create circuit schematic diagram.
        
        Args:
            save_path: Path to save figure (optional)
        """
        if save_path is None:
            save_path = "results/figures/process/phase1/circuit_schematic.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Simple schematic drawing
        # Input blocks
        ax.text(0.1, 0.7, "Biomarker A\n[0-1000 nM]", 
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        
        ax.text(0.1, 0.5, "Biomarker B\n[0-1000 nM]", 
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        
        ax.text(0.1, 0.3, "Biomarker C\n[0-1000 nM]", 
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        
        # Weight blocks
        ax.text(0.3, 0.7, "w₁ = 0.15-8.5", 
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        ax.text(0.3, 0.5, "w₂ = 0.15-8.5", 
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        ax.text(0.3, 0.3, "w₃ = 0.15-8.5", 
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        # Summation
        ax.text(0.5, 0.5, "Σ\nWeighted\nSum", 
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.5", facecolor='yellow'))
        
        # Activation
        ax.text(0.7, 0.5, "ReLU\nf(x)=max(0,x)", 
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='orange'))
        
        # Output
        ax.text(0.9, 0.5, "Output\nGFP Fluorescence", 
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='red'))
        
        # Arrows
        ax.annotate('', xy=(0.2, 0.7), xytext=(0.28, 0.7),
                   arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.annotate('', xy=(0.2, 0.5), xytext=(0.28, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.annotate('', xy=(0.2, 0.3), xytext=(0.28, 0.3),
                   arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.annotate('', xy=(0.32, 0.7), xytext=(0.45, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.annotate('', xy=(0.32, 0.5), xytext=(0.45, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.annotate('', xy=(0.32, 0.3), xytext=(0.45, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.annotate('', xy=(0.55, 0.5), xytext=(0.65, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.annotate('', xy=(0.75, 0.5), xytext=(0.85, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Genetic Circuit Architecture: 3-Input Weighted Summation', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Circuit schematic saved to: {save_path}")
    
    def load_batch_data(self, batch_file: str):
        """
        Load batch data from HDF5 file.
        
        Args:
            batch_file: Path to HDF5 file
        """
        print(f"Loading data from {batch_file}")
        
        self.data = {}
        with h5py.File(batch_file, 'r') as f:
            circuit_ids = list(f.keys())
            self.data['circuit_ids'] = circuit_ids
            
            # Load parameters
            w1_values = []
            w2_values = []
            w3_values = []
            accuracies = []
            
            for circuit_id in circuit_ids[:100]:  # Load first 100 for speed
                grp = f[circuit_id]
                w1_values.append(grp.attrs['w1'])
                w2_values.append(grp.attrs['w2'])
                w3_values.append(grp.attrs['w3'])
                accuracies.append(grp.attrs['accuracy'])
            
            self.data['w1'] = np.array(w1_values)
            self.data['w2'] = np.array(w2_values)
            self.data['w3'] = np.array(w3_values)
            self.data['accuracy'] = np.array(accuracies)
            
            # Load example time series
            example_circuit = circuit_ids[0]
            grp = f[example_circuit]
            self.data['example'] = {
                'time': grp['time'][:],
                'output': grp['output'][:],
                'w1': grp.attrs['w1'],
                'w2': grp.attrs['w2'],
                'w3': grp.attrs['w3']
            }
        
        print(f"Loaded {len(circuit_ids)} circuits")
    
    def plot_time_series_examples(self, n_examples: int = 3, save_path: str = None):
        """
        Plot time series examples.
        
        Args:
            n_examples: Number of examples to plot
            save_path: Path to save figure (optional)
        """
        if save_path is None:
            save_path = "results/figures/process/phase1/ode_time_series_examples.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create mock data if no real data loaded
        if self.data is None:
            self._create_mock_data()
        
        fig, axes = plt.subplots(n_examples, 1, figsize=(8, 3*n_examples))
        
        if n_examples == 1:
            axes = [axes]
        
        for i in range(n_examples):
            ax = axes[i]
            
            # Plot output over time
            time = self.data['example']['time']
            output = self.data['example']['output'] + i * 0.1  # Offset for visualization
            
            ax.plot(time, output, linewidth=1.5, color=self.colors[i])
            ax.set_xlabel('Time (seconds)', fontsize=8)
            ax.set_ylabel('Output Protein', fontsize=8)
            
            title = f'Circuit {i+1}: w₁={self.data["example"]["w1"]:.2f}, '
            title += f'w₂={self.data["example"]["w2"]:.2f}, '
            title += f'w₃={self.data["example"]["w3"]:.2f}'
            ax.set_title(title, fontsize=9)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 300)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Time series examples saved to: {save_path}")
    
    def plot_parameter_distributions(self, save_path: str = None):
        """
        Plot parameter distributions.
        
        Args:
            save_path: Path to save figure (optional)
        """
        if save_path is None:
            save_path = "results/figures/process/phase1/parameter_distributions.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create mock data if no real data loaded
        if self.data is None:
            self._create_mock_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        
        # Plot w1 distribution
        ax = axes[0, 0]
        ax.hist(self.data['w1'], bins=30, color=self.colors[0], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Weight w₁', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.set_title('Distribution of w₁ (0.15-8.5)', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot w2 distribution
        ax = axes[0, 1]
        ax.hist(self.data['w2'], bins=30, color=self.colors[1], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Weight w₂', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.set_title('Distribution of w₂ (0.15-8.5)', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot w3 distribution
        ax = axes[1, 0]
        ax.hist(self.data['w3'], bins=30, color=self.colors[2], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Weight w₃', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.set_title('Distribution of w₃ (0.15-8.5)', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot accuracy distribution
        ax = axes[1, 1]
        ax.hist(self.data['accuracy'], bins=30, color=self.colors[3], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Accuracy', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.set_title('Distribution of Circuit Accuracies', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Parameter distributions saved to: {save_path}")
    
    def _create_mock_data(self):
        """Create mock data for testing."""
        print("Creating mock data for visualization...")
        
        np.random.seed(42)
        n_circuits = 1000
        
        self.data = {
            'w1': np.random.uniform(0.15, 8.5, n_circuits),
            'w2': np.random.uniform(0.15, 8.5, n_circuits),
            'w3': np.random.uniform(0.15, 8.5, n_circuits),
            'accuracy': np.random.uniform(0.3, 1.0, n_circuits),
            'example': {
                'time': np.linspace(0, 300, 301),
                'output': 50 * (1 + np.sin(np.linspace(0, 4*np.pi, 301))),
                'w1': 5.2,
                'w2': 3.1,
                'w3': 1.8
            }
        }
    
    def create_batch_summary(self) -> str:
        """
        Create text summary of batch data.
        
        Returns:
            Summary string
        """
        if self.data is None:
            return "No data loaded."
        
        summary = []
        summary.append("=== BATCH DATA SUMMARY ===")
        summary.append(f"Number of circuits: {len(self.data['w1'])}")
        summary.append(f"w₁ range: [{self.data['w1'].min():.2f}, {self.data['w1'].max():.2f}]")
        summary.append(f"w₂ range: [{self.data['w2'].min():.2f}, {self.data['w2'].max():.2f}]")
        summary.append(f"w₃ range: [{self.data['w3'].min():.2f}, {self.data['w3'].max():.2f}]")
        summary.append(f"Accuracy range: [{self.data['accuracy'].min():.3f}, {self.data['accuracy'].max():.3f}]")
        summary.append(f"Mean accuracy: {self.data['accuracy'].mean():.3f} ± {self.data['accuracy'].std():.3f}")
        
        return "\n".join(summary)
