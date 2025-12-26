import numpy as np
import h5py
import yaml
import time
from typing import List, Dict
from .ode_solver import CircuitSimulator

class BatchProcessor:
    """Process batches of circuit simulations."""
    
    def __init__(self, config: Dict):
        """
        Initialize batch processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.simulator = CircuitSimulator()
        
    def generate_parameters(self, batch_num: int, start_idx: int, end_idx: int) -> List[Dict]:
        """
        Generate parameters for a batch of circuits.
        
        Args:
            batch_num: Batch number (1-based)
            start_idx: Starting circuit index
            end_idx: Ending circuit index
            
        Returns:
            List of parameter dictionaries
        """
        params_list = []
        n_circuits = end_idx - start_idx + 1
        
        print(f"Generating parameters for batch {batch_num} (circuits {start_idx:04d}-{end_idx:04d})")
        
        # Get parameter ranges from config
        w1_range = self.config['circuits']['parameters']['w1_range']
        w2_range = self.config['circuits']['parameters']['w2_range']
        w3_range = self.config['circuits']['parameters']['w3_range']
        
        for i in range(n_circuits):
            circuit_idx = start_idx + i
            
            # Generate random weights within specified ranges
            w1 = np.random.uniform(w1_range[0], w1_range[1])
            w2 = np.random.uniform(w2_range[0], w2_range[1])
            w3 = np.random.uniform(w3_range[0], w3_range[1])
            
            params = {
                'circuit_id': f'circuit_{circuit_idx:04d}',
                'w1': float(w1),
                'w2': float(w2),
                'w3': float(w3),
                'batch_num': batch_num,
                'circuit_index': circuit_idx
            }
            params_list.append(params)
        
        print(f"Generated {len(params_list)} parameter sets")
        return params_list
    
    def run_batch(self, params_list: List[Dict], batch_num: int) -> List[Dict]:
        """
        Run simulation for a batch of circuits.
        
        Args:
            params_list: List of parameter dictionaries
            batch_num: Batch number
            
        Returns:
            List of simulation results
        """
        print(f"Running simulations for batch {batch_num}...")
        start_time = time.time()
        
        # Run simulations
        results = self.simulator.batch_simulate(params_list, num_circuits=len(params_list))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Batch {batch_num} completed in {total_time:.2f} seconds")
        print(f"Average time per circuit: {total_time/len(params_list):.4f} seconds")
        
        return results
    
    def save_batch(self, results: List[Dict], batch_num: int):
        """
        Save batch results to HDF5 file.
        
        Args:
            results: List of simulation results
            batch_num: Batch number
        """
        # Determine file name
        start_idx = (batch_num - 1) * 1000
        end_idx = start_idx + len(results) - 1
        filename = f"data/raw/batch_{start_idx:04d}_{end_idx:04d}.h5"
        
        print(f"Saving batch {batch_num} to {filename}")
        
        with h5py.File(filename, 'w') as f:
            # Save each circuit as a group
            for i, result in enumerate(results):
                circuit_id = result['circuit_id']
                grp = f.create_group(circuit_id)
                
                # Save arrays
                for key in ['time', 'mRNA1', 'mRNA2', 'mRNA3', 
                           'protein1', 'protein2', 'protein3', 'output']:
                    if key in result:
                        grp.create_dataset(key, data=result[key])
                
                # Save scalar attributes
                for key in ['w1', 'w2', 'w3', 'accuracy', 'final_output', 'circuit_id']:
                    if key in result:
                        grp.attrs[key] = result[key]
                
                # Save batch metadata
                grp.attrs['batch_num'] = batch_num
                grp.attrs['circuit_index'] = start_idx + i
            
            # Save batch metadata
            f.attrs['batch_num'] = batch_num
            f.attrs['start_idx'] = start_idx
            f.attrs['end_idx'] = end_idx
            f.attrs['num_circuits'] = len(results)
            f.attrs['creation_time'] = time.ctime()
            f.attrs['total_simulation_time'] = sum(
                [r.get('simulation_time', 0) for r in results]
            )
        
        print(f"Saved {len(results)} circuits to {filename}")
    
    def merge_batches(self, output_file: str = "data/raw/simulations_10k.h5"):
        """
        Merge all batch files into a single HDF5 file.
        
        Args:
            output_file: Output HDF5 file path
        """
        print(f"Merging batches into {output_file}")
        
        with h5py.File(output_file, 'w') as output_f:
            total_circuits = 0
            
            # Find all batch files
            import glob
            batch_files = sorted(glob.glob("data/raw/batch_*.h5"))
            
            for batch_file in batch_files:
                print(f"  Processing {batch_file}")
                
                with h5py.File(batch_file, 'r') as input_f:
                    # Copy all groups
                    for circuit_id in input_f.keys():
                        if circuit_id not in output_f:
                            input_f.copy(circuit_id, output_f)
                            total_circuits += 1
            
            # Add metadata
            output_f.attrs['total_circuits'] = total_circuits
            output_f.attrs['merged_time'] = time.ctime()
            output_f.attrs['num_batches'] = len(batch_files)
        
        print(f"✅ Merged {total_circuits} circuits into {output_file}")
