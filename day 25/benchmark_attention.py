import torch
import torch.nn as nn
import time
import math
from attention import MultiHeadAttentionBlock

def benchmark_attention():
    """Benchmark PyTorch's built-in MultiheadAttention vs our implementation"""
    
    # Model configuration
    batch_size = 2
    seq_len = 128
    d_model = 512
    h = 8
    dropout = 0.1
    
    # Create inputs
    q = torch.randn(batch_size, seq_len, d_model, device="cuda")
    k = torch.randn(batch_size, seq_len, d_model, device="cuda")
    v = torch.randn(batch_size, seq_len, d_model, device="cuda")
    mask = None  # No mask for this test
    
    # Initialize models
    pytorch_builtin = nn.MultiheadAttention(d_model, h, dropout=dropout, batch_first=True).cuda()
    our_implementation = MultiHeadAttentionBlock(d_model, h, dropout).cuda()
    
    # Warmup
    for _ in range(5):
        _ = pytorch_builtin(q, k, v)
        _ = our_implementation(q, k, v)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch built-in implementation
    start_time = time.time()
    for _ in range(100):
        output_builtin, _ = pytorch_builtin(q, k, v)
    torch.cuda.synchronize()
    builtin_time = (time.time() - start_time) * 1000 / 100  # ms per iteration
    
    # Benchmark our implementation
    start_time = time.time()
    for _ in range(100):
        output_our = our_implementation(q, k, v)
    torch.cuda.synchronize()
    our_time = (time.time() - start_time) * 1000 / 100  # ms per iteration
    
    # Compare outputs (there will be differences due to different implementations)
    diff = (output_builtin - output_our).abs().max().item()
    
    print(f"PyTorch built-in MultiheadAttention: {builtin_time:.2f} ms per iteration")
    print(f"Our implementation: {our_time:.2f} ms per iteration")
    print(f"Relative performance: {builtin_time / our_time:.2f}x")
    print(f"Maximum absolute difference: {diff:.6f}")
    
    return output_builtin, output_our

if __name__ == "__main__":
    benchmark_attention() 