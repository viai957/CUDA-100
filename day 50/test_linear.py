import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from whisper_cuda.linear import CUDALinear

def test_correctness(batch_size=32, in_features=512, out_features=512, eps=1e-3):
    """Test the correctness of the CUDA Linear implementation against PyTorch's native implementation"""
    print(f"Testing correctness with batch_size={batch_size}, in_features={in_features}, out_features={out_features}")
    
    # Create input tensor
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device="cuda")
    
    # Create PyTorch linear layer
    torch_linear = nn.Linear(in_features, out_features, bias=True).to("cuda").half()
    
    # Create our custom CUDA linear layer with the same weights
    cuda_linear = CUDALinear(in_features, out_features, bias=True)
    cuda_linear.weight.data.copy_(torch_linear.weight.data)
    cuda_linear.bias.data.copy_(torch_linear.bias.data)
    cuda_linear = cuda_linear.to("cuda")
    
    # Forward pass
    with torch.no_grad():
        torch_output = torch_linear(x)
        cuda_output = cuda_linear(x)
    
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

def benchmark(batch_size=32, in_features=512, out_features=512, num_iters=1000, warmup=100):
    """Benchmark the CUDA Linear implementation against PyTorch's native implementation"""
    print(f"Benchmarking with batch_size={batch_size}, in_features={in_features}, out_features={out_features}")
    
    # Create input tensor
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device="cuda")
    
    # Create PyTorch linear layer
    torch_linear = nn.Linear(in_features, out_features, bias=True).to("cuda").half()
    
    # Create our custom CUDA linear layer
    cuda_linear = CUDALinear(in_features, out_features, bias=True).to("cuda")
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = torch_linear(x)
            _ = cuda_linear(x)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Benchmark PyTorch implementation
    start_time = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = torch_linear(x)
    torch.cuda.synchronize()
    torch_time = time.time() - start_time
    
    # Benchmark our CUDA implementation
    start_time = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = cuda_linear(x)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    
    # Calculate throughput
    torch_throughput = num_iters * batch_size / torch_time
    cuda_throughput = num_iters * batch_size / cuda_time
    speedup = torch_time / cuda_time
    
    print(f"PyTorch Linear: {torch_time:.4f} seconds, {torch_throughput:.2f} samples/sec")
    print(f"CUDA Linear:    {cuda_time:.4f} seconds, {cuda_throughput:.2f} samples/sec")
    print(f"Speedup:        {speedup:.2f}x")
    
    return speedup

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and benchmark CUDA Linear layer")
    parser.add_argument("--test", action="store_true", help="Run correctness test")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--in-features", type=int, default=512, help="Input features")
    parser.add_argument("--out-features", type=int, default=512, help="Output features")
    parser.add_argument("--iters", type=int, default=1000, help="Number of iterations for benchmark")
    args = parser.parse_args()
    
    # Default to running both tests if neither is specified
    if not args.test and not args.bench:
        args.test = True
        args.bench = True
    
    if args.test:
        test_correctness(args.batch_size, args.in_features, args.out_features)
        print()
    
    if args.bench:
        benchmark(args.batch_size, args.in_features, args.out_features, args.iters) 