"""
Quick verification of repaired data
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def verify_and_visualize():
    """Create quick verification plots"""
    print("Creating verification plots for repaired data...")
    
    # Load data
    with h5py.File('data/processed/train_dataset.h5', 'r') as f:
        X = f['features'][:]
        y = f['accuracy'][:]
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Feature dimension: {X.shape[1]} (should be 303)")
    
    # Extract weights and outputs
    weights = X[:, :3]  # w1, w2, w3
    outputs = X[:, 3:]  # output trajectory (300 points)
    
    # Create verification plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Weight distributions
    for i in range(3):
        ax = axes[0, i]
        ax.hist(weights[:, i], bins=50, alpha=0.7)
        ax.set_title(f'Weight w{i+1} Distribution')
        ax.set_xlabel(f'w{i+1} value')
        ax.set_ylabel('Frequency')
        ax.axvline(weights[:, i].mean(), color='red', linestyle='--', label=f'Mean: {weights[:, i].mean():.2f}')
        ax.legend()
    
    # Plot 2: Accuracy distribution
    ax = axes[1, 0]
    ax.hist(y, bins=50, alpha=0.7)
    ax.set_title('Accuracy Distribution (Repaired)')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Frequency')
    ax.axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.3f}')
    ax.legend()
    
    # Plot 3: Sample output trajectories
    ax = axes[1, 1]
    for i in range(5):  # Show 5 sample circuits
        ax.plot(outputs[i, :50], alpha=0.7, label=f'Circuit {i+1}')
    ax.set_title('Sample Output Trajectories (first 50 points)')
    ax.set_xlabel('Time point')
    ax.set_ylabel('Output value')
    ax.legend()
    
    # Plot 4: Correlation: weights vs accuracy
    ax = axes[1, 2]
    colors = ['red', 'green', 'blue']
    for i in range(3):
        ax.scatter(weights[:, i], y, alpha=0.3, s=10, color=colors[i], label=f'w{i+1}')
    ax.set_title('Weights vs Accuracy')
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Accuracy')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/figures/process/phase1_repaired_verification.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/process/phase1_repaired_verification.pdf')
    plt.close()
    
    print("\n✓ Verification plots saved:")
    print("  - results/figures/process/phase1_repaired_verification.png")
    print("  - results/figures/process/phase1_repaired_verification.pdf")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("REPAIRED DATA STATISTICS")
    print("=" * 60)
    print(f"Weight ranges:")
    for i in range(3):
        print(f"  w{i+1}: [{weights[:, i].min():.3f}, {weights[:, i].max():.3f}]")
    
    print(f"\nOutput trajectory ranges:")
    print(f"  Min: {outputs.min():.1f}, Max: {outputs.max():.1f}")
    print(f"  Mean: {outputs.mean():.1f}, Std: {outputs.std():.1f}")
    
    print(f"\nAccuracy statistics:")
    print(f"  Range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Mean: {y.mean():.3f}, Std: {y.std():.3f}")
    print(f"  25th percentile: {np.percentile(y, 25):.3f}")
    print(f"  50th percentile: {np.percentile(y, 50):.3f}")
    print(f"  75th percentile: {np.percentile(y, 75):.3f}")
    
    # Check for issues
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)
    
    issues = []
    
    # Check 1: Weight ranges
    if weights[:, 0].min() < 0.1 or weights[:, 0].max() > 9.0:
        issues.append("Weight w1 outside expected range [0.15, 8.5]")
    
    # Check 2: NaN values
    if np.isnan(X).any():
        issues.append("NaN values in features")
    if np.isnan(y).any():
        issues.append("NaN values in targets")
    
    # Check 3: Feature dimension
    if X.shape[1] != 303:
        issues.append(f"Wrong feature dimension: {X.shape[1]} (expected 303)")
    
    # Check 4: Output range reasonable
    if outputs.max() > 1e9:  # Should be in millions, not billions
        issues.append(f"Output values too large: max={outputs.max():.1e}")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ ALL CHECKS PASSED")
        print("Data is ready for Phase 2 training!")
        return True

if __name__ == "__main__":
    verify_and_visualize()
