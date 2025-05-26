import torch
import torch.nn as nn
import torch.utils.cpp_extension
import time
import math
import numpy as np
from attention import MultiHeadAttentionBlock

# Compile CUDA extension (this would typically be done with a setup.py file)
# For this example, we'll use JIT compilation
attention_cuda = torch.utils.cpp_extension.load(
    name="attention_cuda",
    sources=["day 24/attention_cuda.cpp", "day 24/attention_cuda_kernel.cu"],
    verbose=True
)

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

def test_attention():
    """Test and compare PyTorch and CUDA attention implementations"""
    
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
    
    # Initialize both models
    pytorch_attention = MultiHeadAttentionBlock(d_model, h, dropout).cuda()
    cuda_attention = CudaMultiHeadAttention(d_model, h, dropout).cuda()
    
    # Copy weights to ensure fair comparison
    cuda_attention.w_q.weight.data.copy_(pytorch_attention.w_q.weight.data)
    cuda_attention.w_k.weight.data.copy_(pytorch_attention.w_k.weight.data)
    cuda_attention.w_v.weight.data.copy_(pytorch_attention.w_v.weight.data)
    cuda_attention.w_o.weight.data.copy_(pytorch_attention.w_o.weight.data)
    
    # Warmup
    for _ in range(5):
        _ = pytorch_attention(q, k, v, mask)
        _ = cuda_attention(q, k, v, mask)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch implementation
    start_time = time.time()
    for _ in range(100):
        output_pytorch, _ = pytorch_attention.attention(
            pytorch_attention.w_q(q).view(batch_size, seq_len, h, -1).transpose(1, 2),
            pytorch_attention.w_k(k).view(batch_size, seq_len, h, -1).transpose(1, 2),
            pytorch_attention.w_v(v).view(batch_size, seq_len, h, -1).transpose(1, 2),
            mask,
            pytorch_attention.dropout
        )
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) * 1000 / 100  # ms per iteration
    
    # Benchmark CUDA implementation
    start_time = time.time()
    for _ in range(100):
        output_cuda = cuda_attention(q, k, v, mask)
    torch.cuda.synchronize()
    cuda_time = (time.time() - start_time) * 1000 / 100  # ms per iteration
    
    # Compare outputs
    output_pytorch = pytorch_attention(q, k, v, mask)
    output_cuda = cuda_attention(q, k, v, mask)
    
    diff = (output_pytorch - output_cuda).abs().max().item()
    
    print(f"PyTorch implementation: {pytorch_time:.2f} ms per iteration")
    print(f"CUDA implementation: {cuda_time:.2f} ms per iteration")
    print(f"Speedup: {pytorch_time / cuda_time:.2f}x")
    print(f"Maximum absolute difference: {diff}")
    print(f"Outputs match: {diff < 1e-3}")
    
    return output_pytorch, output_cuda

if __name__ == "__main__":
    test_attention() 