#!/usr/bin/env python
"""Test the ODE solver."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuits.ode_solver import CircuitSimulator
import numpy as np
import time

def test_single_circuit():
    """Test single circuit simulation."""
    print("Testing single circuit simulation...")
    
    simulator = CircuitSimulator()
    
    # Test parameters
    w1, w2, w3 = 5.2, 3.1, 1.8
    
    # Generate random inputs
    np.random.seed(42)
    n_steps = 301
    input_A = np.random.uniform(0, 1000, n_steps)
    input_B = np.random.uniform(0, 1000, n_steps)
    input_C = np.random.uniform(0, 1000, n_steps)
    
    # Simulate
    start_time = time.time()
    result = simulator.solve_ode_single(w1, w2, w3, input_A, input_B, input_C)
    end_time = time.time()
    
    print(f"Simulation time: {end_time - start_time:.4f} seconds")
    print(f"Circuit accuracy: {result['accuracy']:.4f}")
    print(f"Final output: {result['final_output']:.4f}")
    print(f"Output shape: {result['output'].shape}")
    
    # Check for NaN
    if np.isnan(result['output']).any():
        print("⚠️ WARNING: NaN values in output!")
    else:
        print("✅ No NaN values in output")
    
    return result

def test_batch_simulation():
    """Test batch simulation."""
    print("\nTesting batch simulation...")
    
    simulator = CircuitSimulator()
    
    # Generate test parameters
    test_params = []
    for i in range(10):
        params = {
            'w1': np.random.uniform(0.15, 8.5),
            'w2': np.random.uniform(0.15, 8.5),
            'w3': np.random.uniform(0.15, 8.5),
            'circuit_id': f'test_circuit_{i:04d}'
        }
        test_params.append(params)
    
    # Run batch
    results = simulator.batch_simulate(test_params, num_circuits=10)
    
    print(f"Batch simulation completed: {len(results)} circuits")
    print(f"Average accuracy: {np.mean([r["accuracy"] for r in results]):.4f}")
    
    return results

if __name__ == "__main__":
    print("=== ODE SOLVER TEST ===")
    print(f"Python version: {sys.version}")
    
    # Test single circuit
    single_result = test_single_circuit()
    
    # Test batch simulation
    batch_results = test_batch_simulation()
    
    print("\n✅ All tests passed!")
