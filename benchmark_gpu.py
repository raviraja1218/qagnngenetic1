import torch
import time
import numpy as np

print("GPU Performance Benchmark")

# Test 1: Matrix multiplication
if torch.cuda.is_available():
    device = torch.device('cuda')
    
    # Small matrix test
    size = 4096
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    torch.cuda.synchronize()
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    gflops = 2 * size**3 / (elapsed * 1e9)
    print(f"Matrix {size}x{size} multiplication: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")
    
    # Test 2: Memory bandwidth
    size_mb = 1000  # MB
    data = torch.randn(size_mb * 1024 * 1024 // 4, dtype=torch.float32, device=device)
    
    torch.cuda.synchronize()
    start = time.time()
    result = data * 2.0 + 1.0
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    bandwidth = (size_mb * 2) / elapsed  # MB/s (read + write)
    print(f"Memory bandwidth: {bandwidth:.0f} MB/s")
    
    # Test 3: Check memory
    free, total = torch.cuda.mem_get_info()
    print(f"GPU Memory: {free/1e9:.1f} GB free, {total/1e9:.1f} GB total")
else:
    print("GPU not available")
