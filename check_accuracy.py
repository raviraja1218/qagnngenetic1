import h5py
import numpy as np

print("=== CHECKING ACCURACY ISSUE ===")

# Check first few circuits
with h5py.File('data/raw/batch_0000_0999.h5', 'r') as f:
    circuit_ids = list(f.keys())[:5]
    
    for circuit_id in circuit_ids:
        circuit = f[circuit_id]
        print(f"\nCircuit: {circuit_id}")
        print(f"  w1: {circuit.attrs['w1']:.3f}")
        print(f"  w2: {circuit.attrs['w2']:.3f}")
        print(f"  w3: {circuit.attrs['w3']:.3f}")
        print(f"  Accuracy: {circuit.attrs.get('accuracy', 'NOT FOUND')}")
        
        if 'output' in circuit:
            output = circuit['output'][:]
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"  Output mean: {output.mean():.3f}")
            
            # Check if output is all zeros
            if np.all(output == 0):
                print("  ⚠️ WARNING: Output is all zeros!")
            elif np.all(np.isnan(output)):
                print("  ⚠️ WARNING: Output is all NaN!")
