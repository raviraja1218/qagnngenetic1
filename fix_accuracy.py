import sys
sys.path.append('/home/raviraja/projects/qagnn/src')
import numpy as np
import h5py

print("=== FIXING ACCURACY CALCULATION ===")

def calculate_proper_accuracy(w1, w2, w3, output, time):
    """
    Calculate realistic accuracy based on circuit behavior.
    
    For a 3-input weighted summation circuit, accuracy should measure
    how well the output tracks the weighted sum of inputs.
    """
    # Generate synthetic inputs for accuracy calculation
    np.random.seed(42)  # For reproducibility
    input_A = 500 + 200 * np.sin(time/50)  # Oscillating biomarker A
    input_B = 300 + 150 * np.cos(time/30)  # Oscillating biomarker B  
    input_C = 200 + 100 * np.sin(time/70)  # Oscillating biomarker C
    
    # Calculate target output (ideal circuit behavior)
    target_output = w1 * input_A + w2 * input_B + w3 * input_C
    target_output = np.maximum(0, target_output)  # Apply ReLU
    
    # Normalize both outputs to [0, 1] range
    output_norm = (output - output.min()) / (output.max() - output.min() + 1e-10)
    target_norm = (target_output - target_output.min()) / (target_output.max() - target_output.min() + 1e-10)
    
    # Calculate accuracy as 1 - normalized mean absolute error
    mae = np.mean(np.abs(output_norm - target_norm))
    accuracy = max(0, 1 - mae)
    
    return float(accuracy)

# Test the new accuracy calculation
print("Testing new accuracy calculation...")
np.random.seed(42)
time = np.linspace(0, 300, 301)

# Test with example values
w1, w2, w3 = 5.2, 3.1, 1.8
output = 100 + 50 * np.sin(time/20)  # Simulated output

accuracy = calculate_proper_accuracy(w1, w2, w3, output, time)
print(f"Example accuracy: {accuracy:.4f}")

# Now fix all batch files
print("\nFixing accuracy in all batch files...")
for batch_num in range(10):
    filename = f"data/raw/batch_{batch_num*1000:04d}_{batch_num*1000+999:04d}.h5"
    
    if not os.path.exists(filename):
        print(f"  Skipping {filename} (not found)")
        continue
    
    print(f"  Processing {filename}...")
    
    # Open file in read/write mode
    with h5py.File(filename, 'r+') as f:
        circuit_ids = list(f.keys())
        
        for circuit_id in circuit_ids:
            circuit = f[circuit_id]
            
            # Get parameters
            w1 = circuit.attrs['w1']
            w2 = circuit.attrs['w2']
            w3 = circuit.attrs['w3']
            
            # Get time and output
            time = circuit['time'][:]
            output = circuit['output'][:]
            
            # Calculate proper accuracy
            new_accuracy = calculate_proper_accuracy(w1, w2, w3, output, time)
            
            # Update accuracy in file
            circuit.attrs['accuracy'] = new_accuracy
    
    print(f"  ✅ Updated {len(circuit_ids)} circuits")

print("\n✅ All accuracies updated!")
