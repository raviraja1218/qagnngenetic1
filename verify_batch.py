#!/usr/bin/env python
"""Verify batch files."""

import h5py
import numpy as np
import sys
import os

def verify_batch(batch_file):
    print(f"Verifying: {batch_file}")
    
    try:
        with h5py.File(batch_file, 'r') as f:
            # Count circuits
            circuit_keys = list(f.keys())
            circuit_count = len(circuit_keys)
            print(f"  Circuits found: {circuit_count}")
            
            if circuit_count == 0:
                print("  ❌ ERROR: No circuits in file")
                return False
            
            # Check first circuit
            first_key = circuit_keys[0]
            first_circuit = f[first_key]
            
            print(f"  First circuit: {first_key}")
            
            # Check parameters
            params_ok = True
            for param in ['w1', 'w2', 'w3']:
                if param in first_circuit.attrs:
                    value = first_circuit.attrs[param]
                    print(f"    {param}: {value:.4f}")
                else:
                    print(f"    ❌ Missing parameter: {param}")
                    params_ok = False
            
            # Check for required datasets
            required_dsets = ['time', 'output']
            dset_ok = True
            for ds in required_dsets:
                if ds in first_circuit:
                    data = first_circuit[ds][:]
                    print(f"    {ds}: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
                    
                    # Check for NaN
                    if np.isnan(data).any():
                        print(f"    ⚠️ WARNING: NaN values found in {ds}!")
                        dset_ok = False
                else:
                    print(f"    ❌ Missing dataset: {ds}")
                    dset_ok = False
            
            # Quick check all circuits for NaN
            print(f"  Checking all circuits for NaN...")
            nan_count = 0
            checked = 0
            for circuit_name in circuit_keys[:20]:  # Check first 20 only
                circuit = f[circuit_name]
                if 'output' in circuit:
                    output = circuit['output'][:]
                    if np.isnan(output).any():
                        nan_count += 1
                checked += 1
            
            print(f"  Circuits checked: {checked}")
            print(f"  Circuits with NaN: {nan_count}")
            
            if nan_count == 0 and params_ok and dset_ok:
                print(f"  ✅ VERIFICATION PASSED: {batch_file}")
                return True
            else:
                print(f"  ❌ VERIFICATION FAILED")
                return False
                
    except Exception as e:
        print(f"  ❌ ERROR opening file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        batch_file = sys.argv[1]
    else:
        # Find any batch file
        batch_files = [f for f in os.listdir('data/raw') if f.startswith('batch_') and f.endswith('.h5')]
        if batch_files:
            batch_file = os.path.join('data/raw', batch_files[0])
        else:
            print("No batch files found in data/raw/")
            sys.exit(1)
    
    success = verify_batch(batch_file)
    sys.exit(0 if success else 1)
