import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
from whisper_cuda.gelu import CUDAGELU

def test_correctness(size=1000000, eps=1e-3):
    """Test the correctness of the CUDA GELU implementation against PyTorch's native implementation"""
    print(f"Testing correctness with tensor size={size}")
    
    # Create input tensor
    x = torch.randn(size, dtype=torch.float16, device="cuda")
    
    # Create CUDA GELU
    cuda_gelu = CUDAGELU().to("cuda")
    
    # Forward pass
    with torch.no_grad():
        torch_output = F.gelu(x)
        cuda_output = cuda_gelu(x)
    
    # Compare outputs
    max_diff = torch.max(torch.abs(torch_output - cuda_output)).item()
    avg_diff = torch.mean(torch.abs(torch_output - cuda_output)).item()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Avg difference: {avg_diff:.6f}")
    
    # Check if outputs are close enough
    passed = max_diff < eps
    if passed:
        print("✓ Test PASSED!")
    else:
        print("✗ Test FAILED!")
    
    return passed

def benchmark(size=1000000, num_iters=1000, warmup=100):
    """Benchmark the CUDA GELU implementation against PyTorch's native implementation"""
    print(f"Benchmarking with tensor size={size}")
    
    # Create input tensor
    x = torch.randn(size, dtype=torch.float16, device="cuda")
    
    # Create CUDA GELU
    cuda_gelu = CUDAGELU().to("cuda")
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = F.gelu(x)
            _ = cuda_gelu(x)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Benchmark PyTorch implementation
    start_time = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = F.gelu(x)
    torch.cuda.synchronize()
    torch_time = time.time() - start_time
    
    # Benchmark our CUDA implementation
    start_time = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = cuda_gelu(x)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    
    # Calculate throughput (elements per second)
    torch_throughput = num_iters * size / torch_time
    cuda_throughput = num_iters * size / cuda_time
    speedup = torch_time / cuda_time
    
    print(f"PyTorch GELU: {torch_time:.4f} seconds, {torch_throughput:.2e} elements/sec")
    print(f"CUDA GELU:    {cuda_time:.4f} seconds, {cuda_throughput:.2e} elements/sec")
    print(f"Speedup:      {speedup:.2f}x")
    
    return speedup

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and benchmark CUDA GELU")
    parser.add_argument("--test", action="store_true", help="Run correctness test")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--size", type=int, default=1000000, help="Tensor size")
    parser.add_argument("--iters", type=int, default=1000, help="Number of iterations for benchmark")
    args = parser.parse_args()
    
    # Default to running both tests if neither is specified
    if not args.test and not args.bench:
        args.test = True
        args.bench = True
    
    if args.test:
        test_correctness(args.size)
        print()
    
    if args.bench:
        benchmark(args.size, args.iters) 