import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import math
from attention import MultiHeadAttentionBlock

class PytorchMultiHeadAttention(nn.Module):
    """PyTorch-only implementation of multi-head attention"""
    
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
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q_proj = self.w_q(q)  # (batch_size, seq_len, d_model)
        k_proj = self.w_k(k)  # (batch_size, seq_len, d_model)
        v_proj = self.w_v(v)  # (batch_size, seq_len, d_model)
        
        # Split into h heads
        q_proj = q_proj.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        k_proj = k_proj.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        v_proj = v_proj.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        
        # Apply attention
        output, _ = self.attention(q_proj, k_proj, v_proj, mask)
        
        # Concatenate heads and pass through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        return self.w_o(output)
    
    def attention(self, query, key, value, mask=None):
        d_k = query.shape[-1]
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) Apply softmax
        attention_scores = self.dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

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

def run_optimized_attention(optimized_model, q, k, v, mask, repeats=100):
    """Run our optimized attention implementation and time it"""
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(repeats):
        with torch.no_grad():
            output_optimized = optimized_model(q, k, v, mask)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    return output_optimized, (end_time - start_time) * 1000 / repeats  # ms per iteration

def benchmark_sequence_length():
    """Benchmark attention implementations with varying sequence lengths"""
    batch_size = 2
    d_model = 512
    h = 8
    dropout = 0.1
    
    # Sequence lengths to test
    seq_lengths = [32, 64, 128, 256, 512, 1024]
    pytorch_times = []
    optimized_times = []
    
    for seq_len in seq_lengths:
        print(f"Testing sequence length: {seq_len}")
        
        # Create inputs
        q = torch.randn(batch_size, seq_len, d_model, device="cuda")
        k = torch.randn(batch_size, seq_len, d_model, device="cuda")
        v = torch.randn(batch_size, seq_len, d_model, device="cuda")
        mask = None  # No mask for this test
        
        # Initialize models
        pytorch_attention = MultiHeadAttentionBlock(d_model, h, dropout).cuda()
        optimized_attention = PytorchMultiHeadAttention(d_model, h, dropout).cuda()
        
        # Copy weights to ensure fair comparison
        optimized_attention.w_q.weight.data.copy_(pytorch_attention.w_q.weight.data)
        optimized_attention.w_k.weight.data.copy_(pytorch_attention.w_k.weight.data)
        optimized_attention.w_v.weight.data.copy_(pytorch_attention.w_v.weight.data)
        optimized_attention.w_o.weight.data.copy_(pytorch_attention.w_o.weight.data)
        
        # Warmup
        _ = pytorch_attention(q, k, v, mask)
        _ = optimized_attention(q, k, v, mask)
        
        # Benchmark
        _, pytorch_time = run_pytorch_attention(pytorch_attention, q, k, v, mask, repeats=10)
        _, optimized_time = run_optimized_attention(optimized_attention, q, k, v, mask, repeats=10)
        
        pytorch_times.append(pytorch_time)
        optimized_times.append(optimized_time)
        
        print(f"  Original: {pytorch_time:.2f} ms, Optimized: {optimized_time:.2f} ms, Ratio: {pytorch_time/optimized_time:.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, pytorch_times, 'o-', label='Original')
    plt.plot(seq_lengths, optimized_times, 'o-', label='Optimized')
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
    optimized_times = []
    
    for h in head_counts:
        print(f"Testing head count: {h}")
        
        # Create inputs
        q = torch.randn(batch_size, seq_len, d_model, device="cuda")
        k = torch.randn(batch_size, seq_len, d_model, device="cuda")
        v = torch.randn(batch_size, seq_len, d_model, device="cuda")
        mask = None  # No mask for this test
        
        # Initialize models
        pytorch_attention = MultiHeadAttentionBlock(d_model, h, dropout).cuda()
        optimized_attention = PytorchMultiHeadAttention(d_model, h, dropout).cuda()
        
        # Copy weights to ensure fair comparison
        optimized_attention.w_q.weight.data.copy_(pytorch_attention.w_q.weight.data)
        optimized_attention.w_k.weight.data.copy_(pytorch_attention.w_k.weight.data)
        optimized_attention.w_v.weight.data.copy_(pytorch_attention.w_v.weight.data)
        optimized_attention.w_o.weight.data.copy_(pytorch_attention.w_o.weight.data)
        
        # Warmup
        _ = pytorch_attention(q, k, v, mask)
        _ = optimized_attention(q, k, v, mask)
        
        # Benchmark
        _, pytorch_time = run_pytorch_attention(pytorch_attention, q, k, v, mask, repeats=10)
        _, optimized_time = run_optimized_attention(optimized_attention, q, k, v, mask, repeats=10)
        
        pytorch_times.append(pytorch_time)
        optimized_times.append(optimized_time)
        
        print(f"  Original: {pytorch_time:.2f} ms, Optimized: {optimized_time:.2f} ms, Ratio: {pytorch_time/optimized_time:.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(head_counts, pytorch_times, 'o-', label='Original')
    plt.plot(head_counts, optimized_times, 'o-', label='Optimized')
    plt.xlabel('Number of Attention Heads')
    plt.ylabel('Time (ms)')
    plt.title('Attention Performance vs Number of Heads')
    plt.legend()
    plt.grid(True)
    plt.savefig('attention_head_count_benchmark.png')

def verify_implementation():
    """Verify that our optimized implementation matches the original"""
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
    optimized_attention = PytorchMultiHeadAttention(d_model, h, dropout).cuda()
    
    # Use identical weights
    optimized_attention.w_q.weight.data.copy_(pytorch_attention.w_q.weight.data)
    optimized_attention.w_k.weight.data.copy_(pytorch_attention.w_k.weight.data)
    optimized_attention.w_v.weight.data.copy_(pytorch_attention.w_v.weight.data)
    optimized_attention.w_o.weight.data.copy_(pytorch_attention.w_o.weight.data)
    
    # Forward pass through both models
    with torch.no_grad():
        output_pytorch = pytorch_attention(q, k, v, mask)
        output_optimized = optimized_attention(q, k, v, mask)
    
    # Compare outputs
    diff = (output_pytorch - output_optimized).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Verification Results:")
    print(f"  Maximum absolute difference: {max_diff}")
    print(f"  Mean absolute difference: {mean_diff}")
    print(f"  Outputs match closely: {max_diff < 1e-3}")
    
    # Visualize a single example of the attention outputs
    sample_idx = (0, 0)  # First element of batch, first position in sequence
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Output")
    plt.imshow(output_pytorch[sample_idx[0], sample_idx[1]].cpu().numpy().reshape(8, -1))
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Optimized Output")
    plt.imshow(output_optimized[sample_idx[0], sample_idx[1]].cpu().numpy().reshape(8, -1))
    plt.colorbar()
    
    plt.savefig('attention_output_comparison.png')

if __name__ == "__main__":
    print("Running verification...")
    verify_implementation()
    
    print("\nBenchmarking sequence length...")
    benchmark_sequence_length()
    
    print("\nBenchmarking head count...")
    benchmark_head_count()
    
    print("\nAll benchmarks complete! Results saved as PNG files.") 