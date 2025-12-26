import sys
sys.path.append('/home/raviraja/projects/qagnn/src')

try:
    from circuits.ode_solver import CircuitSimulator
    from circuits.batch_processor import BatchProcessor
    import h5py
    import numpy as np
    import time
    import yaml
except ImportError as e:
    print(f"Import error: {e}")
    print("Please check the src/circuits/ directory structure")
    sys.exit(1)

print("Creating test simulation of 10 circuits...")

# Create configuration
config = {
    'circuits': {
        'parameters': {
            'w1_range': [0.15, 8.5],
            'w2_range': [0.15, 8.5],
            'w3_range': [0.15, 8.5]
        }
    }
}

# Initialize simulator
simulator = CircuitSimulator()

# Generate 10 test circuits
test_params = []
for i in range(10):
    params = {
        'w1': np.random.uniform(0.15, 8.5),
        'w2': np.random.uniform(0.15, 8.5),
        'w3': np.random.uniform(0.15, 8.5),
        'circuit_id': f'test_circuit_{i:04d}'
    }
    test_params.append(params)

# Simulate
print(f"Simulating {len(test_params)} test circuits...")
start_time = time.time()
results = simulator.batch_simulate(test_params, num_circuits=10)
end_time = time.time()

print(f"Test simulation completed in {end_time - start_time:.2f} seconds")
print(f"Average time per circuit: {(end_time - start_time)/10:.4f} seconds")

# Save test results
with h5py.File('data/raw/test_simulations.h5', 'w') as f:
    for i, result in enumerate(results):
        grp = f.create_group(f'circuit_{i:04d}')
        
        # Save arrays
        for key in ['time', 'output', 'mRNA1', 'mRNA2', 'mRNA3', 
                   'protein1', 'protein2', 'protein3']:
            if key in result:
                grp.create_dataset(key, data=result[key])
        
        # Save attributes
        for key in ['w1', 'w2', 'w3', 'accuracy', 'final_output', 'circuit_id']:
            if key in result:
                grp.attrs[key] = result[key]

print("Test simulations saved to: data/raw/test_simulations.h5")

# Verify the file
print("\nVerifying saved file...")
with h5py.File('data/raw/test_simulations.h5', 'r') as f:
    circuit_count = len(list(f.keys()))
    print(f"Circuits in file: {circuit_count}")
    
    # Check first circuit
    first_key = list(f.keys())[0]
    first_circuit = f[first_key]
    print(f"First circuit: {first_key}")
    print(f"  w1={first_circuit.attrs.get('w1', 'N/A')}")
    print(f"  w2={first_circuit.attrs.get('w2', 'N/A')}")
    print(f"  w3={first_circuit.attrs.get('w3', 'N/A')}")
    print(f"  accuracy={first_circuit.attrs.get('accuracy', 'N/A')}")
    
    if 'output' in first_circuit:
        output = first_circuit['output'][:]
        print(f"  output shape: {output.shape}")
        print(f"  output range: [{output.min():.2f}, {output.max():.2f}]")

print("\nSUCCESS: Test simulation completed!")
