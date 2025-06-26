import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from whisper_cuda.attention import CUDAMultiHeadAttention

def test_correctness(batch_size=2, seq_len=32, embed_dim=512, num_heads=8, head_dim=64, eps=1e-2):
    """Test the correctness of the CUDA MultiHeadAttention implementation against PyTorch's native implementation"""
    print(f"Testing correctness with batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, num_heads={num_heads}")
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")
    
    # Create PyTorch multihead attention
    torch_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to("cuda").half()
    
    # Create our custom CUDA multihead attention
    cuda_mha = CUDAMultiHeadAttention(embed_dim, num_heads, head_dim).to("cuda")
    
    # Copy weights from PyTorch to our implementation for fair comparison
    with torch.no_grad():
        # PyTorch packs q, k, v weights into a single tensor, so we need to unpack them
        # This is a simplified version, actual implementation may need more careful weight copying
        qkv_weight = torch_mha.in_proj_weight
        q_weight, k_weight, v_weight = qkv_weight.chunk(3)
        
        qkv_bias = torch_mha.in_proj_bias
        q_bias, k_bias, v_bias = qkv_bias.chunk(3)
        
        cuda_mha.q_proj.weight.copy_(q_weight)
        cuda_mha.k_proj.weight.copy_(k_weight)
        cuda_mha.v_proj.weight.copy_(v_weight)
        
        cuda_mha.q_proj.bias.copy_(q_bias)
        cuda_mha.k_proj.bias.copy_(k_bias)
        cuda_mha.v_proj.bias.copy_(v_bias)
        
        cuda_mha.out_proj.weight.copy_(torch_mha.out_proj.weight)
        cuda_mha.out_proj.bias.copy_(torch_mha.out_proj.bias)
    
    # Forward pass
    with torch.no_grad():
        # PyTorch MultiHeadAttention expects different input format
        torch_output, _ = torch_mha(x, x, x)
        cuda_output, _ = cuda_mha(x)
    
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

def benchmark(batch_size=2, seq_len=32, embed_dim=512, num_heads=8, head_dim=64, num_iters=100, warmup=10):
    """Benchmark the CUDA MultiHeadAttention implementation against PyTorch's native implementation"""
    print(f"Benchmarking with batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, num_heads={num_heads}")
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")
    
    # Create PyTorch multihead attention
    torch_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to("cuda").half()
    
    # Create our custom CUDA multihead attention
    cuda_mha = CUDAMultiHeadAttention(embed_dim, num_heads, head_dim).to("cuda")
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _, _ = torch_mha(x, x, x)
            _, _ = cuda_mha(x)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Benchmark PyTorch implementation
    start_time = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _, _ = torch_mha(x, x, x)
    torch.cuda.synchronize()
    torch_time = time.time() - start_time
    
    # Benchmark our CUDA implementation
    start_time = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _, _ = cuda_mha(x)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    
    # Calculate throughput (sequences per second)
    torch_throughput = num_iters * batch_size / torch_time
    cuda_throughput = num_iters * batch_size / cuda_time
    speedup = torch_time / cuda_time
    
    print(f"PyTorch MHA: {torch_time:.4f} seconds, {torch_throughput:.2f} sequences/sec")
    print(f"CUDA MHA:    {cuda_time:.4f} seconds, {cuda_throughput:.2f} sequences/sec")
    print(f"Speedup:     {speedup:.2f}x")
    
    return speedup

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and benchmark CUDA MultiHeadAttention")
    parser.add_argument("--test", action="store_true", help="Run correctness test")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--embed-dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Dimension of each attention head")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations for benchmark")
    args = parser.parse_args()
    
    # Default to running both tests if neither is specified
    if not args.test and not args.bench:
        args.test = True
        args.bench = True
    
    if args.test:
        test_correctness(args.batch_size, args.seq_len, args.embed_dim, args.num_heads, args.head_dim)
        print()
    
    if args.bench:
        benchmark(args.batch_size, args.seq_len, args.embed_dim, args.num_heads, args.head_dim, args.iters) 