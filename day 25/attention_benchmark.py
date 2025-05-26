import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
from attention import MultiHeadAttentionBlock

# There are two ways to import our custom CUDA extension:
# 1. JIT compilation (slower first time, but more convenient for development)
try:
    import attention_cuda
except ImportError:
    import torch.utils.cpp_extension
    print("JIT compiling attention_cuda extension...")
    attention_cuda = torch.utils.cpp_extension.load(
        name="attention_cuda",
        sources=["attention_cuda.cpp", "attention_cuda_kernel.cu"],
        verbose=True
    )
    print("Compilation finished!")

class CudaMultiHeadAttention(nn.Module):
    """PyTorch module that wraps our custom CUDA kernel"""
    
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout_prob = dropout
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.w_q(q)  # (batch_size, seq_len, d_model)
        k = self.w_k(k)  # (batch_size, seq_len, d_model)
        v = self.w_v(v)  # (batch_size, seq_len, d_model)
        
        # Call our custom CUDA kernel
        output = attention_cuda.multi_head_attention(
            q, k, v, 
            self.h, 
            self.dropout_prob
        )
        
        # Apply output projection
        return self.w_o(output)

def run_pytorch_attention(pytorch_model, q, k, v, mask, repeats=100):
    """Run PyTorch's attention implementation and time it"""
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(repeats):
        with torch.no_grad():
            output_pytorch = pytorch_model(q, k, v, mask)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    return output_pytorch, (end_time - start_time) * 1000 / repeats  # ms per iteration

def run_cuda_attention(cuda_model, q, k, v, mask, repeats=100):
    """Run our custom CUDA attention implementation and time it"""
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(repeats):
        with torch.no_grad():
            output_cuda = cuda_model(q, k, v, mask)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    return output_cuda, (end_time - start_time) * 1000 / repeats  # ms per iteration

def benchmark_sequence_length():
    """Benchmark attention implementations with varying sequence lengths"""
    batch_size = 2
    d_model = 512
    h = 8
    dropout = 0.1
    
    # Sequence lengths to test
    seq_lengths = [32, 64, 128, 256, 512, 1024]
    pytorch_times = []
    cuda_times = []
    
    for seq_len in seq_lengths:
        print(f"Testing sequence length: {seq_len}")
        
        # Create inputs
        q = torch.randn(batch_size, seq_len, d_model, device="cuda")
        k = torch.randn(batch_size, seq_len, d_model, device="cuda")
        v = torch.randn(batch_size, seq_len, d_model, device="cuda")
        mask = None  # No mask for this test
        
        # Initialize models
        pytorch_attention = MultiHeadAttentionBlock(d_model, h, dropout).cuda()
        cuda_attention = CudaMultiHeadAttention(d_model, h, dropout).cuda()
        
        # Copy weights to ensure fair comparison
        cuda_attention.w_q.weight.data.copy_(pytorch_attention.w_q.weight.data)
        cuda_attention.w_k.weight.data.copy_(pytorch_attention.w_k.weight.data)
        cuda_attention.w_v.weight.data.copy_(pytorch_attention.w_v.weight.data)
        cuda_attention.w_o.weight.data.copy_(pytorch_attention.w_o.weight.data)
        
        # Warmup
        _ = pytorch_attention(q, k, v, mask)
        _ = cuda_attention(q, k, v, mask)
        
        # Benchmark
        _, pytorch_time = run_pytorch_attention(pytorch_attention, q, k, v, mask, repeats=10)
        _, cuda_time = run_cuda_attention(cuda_attention, q, k, v, mask, repeats=10)
        
        pytorch_times.append(pytorch_time)
        cuda_times.append(cuda_time)
        
        print(f"  PyTorch: {pytorch_time:.2f} ms, CUDA: {cuda_time:.2f} ms, Speedup: {pytorch_time/cuda_time:.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, pytorch_times, 'o-', label='PyTorch')
    plt.plot(seq_lengths, cuda_times, 'o-', label='CUDA Custom')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Attention Performance vs Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.savefig('attention_seq_len_benchmark.png')
    
def benchmark_head_count():
    """Benchmark attention implementations with varying head counts"""
    batch_size = 2
    seq_len = 256
    d_model = 512
    dropout = 0.1
    
    # Head counts to test (must be divisors of d_model)
    head_counts = [1, 2, 4, 8, 16]
    pytorch_times = []
    cuda_times = []
    
    for h in head_counts:
        print(f"Testing head count: {h}")
        
        # Create inputs
        q = torch.randn(batch_size, seq_len, d_model, device="cuda")
        k = torch.randn(batch_size, seq_len, d_model, device="cuda")
        v = torch.randn(batch_size, seq_len, d_model, device="cuda")
        mask = None  # No mask for this test
        
        # Initialize models
        pytorch_attention = MultiHeadAttentionBlock(d_model, h, dropout).cuda()
        cuda_attention = CudaMultiHeadAttention(d_model, h, dropout).cuda()
        
        # Copy weights to ensure fair comparison
        cuda_attention.w_q.weight.data.copy_(pytorch_attention.w_q.weight.data)
        cuda_attention.w_k.weight.data.copy_(pytorch_attention.w_k.weight.data)
        cuda_attention.w_v.weight.data.copy_(pytorch_attention.w_v.weight.data)
        cuda_attention.w_o.weight.data.copy_(pytorch_attention.w_o.weight.data)
        
        # Warmup
        _ = pytorch_attention(q, k, v, mask)
        _ = cuda_attention(q, k, v, mask)
        
        # Benchmark
        _, pytorch_time = run_pytorch_attention(pytorch_attention, q, k, v, mask, repeats=10)
        _, cuda_time = run_cuda_attention(cuda_attention, q, k, v, mask, repeats=10)
        
        pytorch_times.append(pytorch_time)
        cuda_times.append(cuda_time)
        
        print(f"  PyTorch: {pytorch_time:.2f} ms, CUDA: {cuda_time:.2f} ms, Speedup: {pytorch_time/cuda_time:.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(head_counts, pytorch_times, 'o-', label='PyTorch')
    plt.plot(head_counts, cuda_times, 'o-', label='CUDA Custom')
    plt.xlabel('Number of Attention Heads')
    plt.ylabel('Time (ms)')
    plt.title('Attention Performance vs Number of Heads')
    plt.legend()
    plt.grid(True)
    plt.savefig('attention_head_count_benchmark.png')

def verify_implementation():
    """Verify that our CUDA implementation matches PyTorch's"""
    # Model configuration
    batch_size = 2
    seq_len = 128
    d_model = 512
    h = 8
    dropout = 0.0  # Disable dropout for deterministic comparison
    
    # Create inputs
    torch.manual_seed(42)  # For reproducibility
    q = torch.randn(batch_size, seq_len, d_model, device="cuda")
    k = torch.randn(batch_size, seq_len, d_model, device="cuda")
    v = torch.randn(batch_size, seq_len, d_model, device="cuda")
    mask = None  # No mask for this test
    
    # Initialize models
    pytorch_attention = MultiHeadAttentionBlock(d_model, h, dropout).cuda()
    cuda_attention = CudaMultiHeadAttention(d_model, h, dropout).cuda()
    
    # Use identical weights
    cuda_attention.w_q.weight.data.copy_(pytorch_attention.w_q.weight.data)
    cuda_attention.w_k.weight.data.copy_(pytorch_attention.w_k.weight.data)
    cuda_attention.w_v.weight.data.copy_(pytorch_attention.w_v.weight.data)
    cuda_attention.w_o.weight.data.copy_(pytorch_attention.w_o.weight.data)
    
    # Forward pass through both models
    with torch.no_grad():
        output_pytorch = pytorch_attention(q, k, v, mask)
        output_cuda = cuda_attention(q, k, v, mask)
    
    # Compare outputs
    diff = (output_pytorch - output_cuda).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Verification Results:")
    print(f"  Maximum absolute difference: {max_diff}")
    print(f"  Mean absolute difference: {mean_diff}")
    print(f"  Outputs match: {max_diff < 1e-3}")
    
    # Visualize a single example of the attention outputs
    sample_idx = (0, 0)  # First element of batch, first position in sequence
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("PyTorch Output")
    plt.imshow(output_pytorch[sample_idx[0], sample_idx[1]].cpu().numpy().reshape(8, -1))
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("CUDA Output")
    plt.imshow(output_cuda[sample_idx[0], sample_idx[1]].cpu().numpy().reshape(8, -1))
    plt.colorbar()
    
    plt.savefig('attention_output_comparison.png')

if __name__ == "__main__":
    print("Verifying implementation correctness...")
    verify_implementation()
    
    print("\nBenchmarking sequence length...")
    benchmark_sequence_length()
    
    print("\nBenchmarking head count...")
    benchmark_head_count()
    
    print("\nAll benchmarks complete! Results saved as PNG files.") 