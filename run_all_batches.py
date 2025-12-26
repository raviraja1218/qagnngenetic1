import sys
sys.path.append('/home/raviraja/projects/qagnn/src')
import yaml
import time
from circuits.batch_processor import BatchProcessor

def run_batch(batch_num):
    """Run a single batch."""
    print(f"\n{'='*50}")
    print(f"BATCH {batch_num} EXECUTION")
    print(f"{'='*50}")
    print(f"Start time: {time.ctime()}")
    
    config = {
        'circuits': {
            'parameters': {
                'w1_range': [0.15, 8.5],
                'w2_range': [0.15, 8.5],
                'w3_range': [0.15, 8.5]
            }
        }
    }
    
    processor = BatchProcessor(config)
    start_idx = (batch_num - 1) * 1000
    end_idx = start_idx + 999
    
    print(f"Generating parameters for circuits {start_idx:04d}-{end_idx:04d}...")
    params = processor.generate_parameters(
        batch_num=batch_num,
        start_idx=start_idx,
        end_idx=end_idx
    )
    
    print(f"Running {len(params)} simulations...")
    start_time = time.time()
    results = processor.run_batch(params, batch_num=batch_num)
    end_time = time.time()
    
    processor.save_batch(results, batch_num=batch_num)
    
    total_time = end_time - start_time
    print(f"✅ Batch {batch_num} completed in {total_time:.2f} seconds")
    print(f"Average: {total_time/1000:.4f} seconds per circuit")
    print(f"End time: {time.ctime()}")
    
    return total_time

def main():
    """Run all remaining batches."""
    print("=== PHASE 1 BATCH EXECUTION ===")
    print(f"Overall start: {time.ctime()}")
    
    # Check which batches are already done
    import os
    completed_batches = []
    for i in range(1, 11):
        if os.path.exists(f"data/raw/batch_{(i-1)*1000:04d}_{(i-1)*1000+999:04d}.h5"):
            completed_batches.append(i)
    
    print(f"Already completed batches: {completed_batches}")
    
    # Run remaining batches
    total_time = 0
    for batch_num in range(2, 11):  # Batches 2-10
        if batch_num not in completed_batches:
            batch_time = run_batch(batch_num)
            total_time += batch_time
        else:
            print(f"\nBatch {batch_num} already exists, skipping...")
    
    print(f"\n{'='*50}")
    print("ALL BATCHES COMPLETED!")
    print(f"Total simulation time: {total_time:.2f} seconds")
    print(f"Estimated total circuits: 10,000")
    print(f"Overall end: {time.ctime()}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
