import torch
import torch.nn as nn
import time
import math
import numpy as np
import os
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

def test_attention():
    """Test and compare PyTorch implementations"""
    
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
    pytorch_only_attention = PytorchMultiHeadAttention(d_model, h, dropout).cuda()
    
    # Copy weights to ensure fair comparison
    pytorch_only_attention.w_q.weight.data.copy_(pytorch_attention.w_q.weight.data)
    pytorch_only_attention.w_k.weight.data.copy_(pytorch_attention.w_k.weight.data)
    pytorch_only_attention.w_v.weight.data.copy_(pytorch_attention.w_v.weight.data)
    pytorch_only_attention.w_o.weight.data.copy_(pytorch_attention.w_o.weight.data)
    
    # Set same random seed for dropout
    torch.manual_seed(42)
    
    # Warmup
    for _ in range(5):
        _ = pytorch_attention(q, k, v, mask)
        _ = pytorch_only_attention(q, k, v, mask)
    
    torch.cuda.synchronize()
    
    # Benchmark first PyTorch implementation
    start_time = time.time()
    for _ in range(100):
        output_pytorch = pytorch_attention(q, k, v, mask)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) * 1000 / 100  # ms per iteration
    
    # Benchmark second PyTorch implementation
    start_time = time.time()
    for _ in range(100):
        output_pytorch_only = pytorch_only_attention(q, k, v, mask)
    torch.cuda.synchronize()
    pytorch_only_time = (time.time() - start_time) * 1000 / 100  # ms per iteration
    
    # Compare outputs
    diff = (output_pytorch - output_pytorch_only).abs().max().item()
    
    print(f"PyTorch implementation 1: {pytorch_time:.2f} ms per iteration")
    print(f"PyTorch implementation 2: {pytorch_only_time:.2f} ms per iteration")
    print(f"Relative performance: {pytorch_time / pytorch_only_time:.2f}x")
    print(f"Maximum absolute difference: {diff}")
    print(f"Outputs match: {diff < 1e-3}")
    
    return output_pytorch, output_pytorch_only

if __name__ == "__main__":
    test_attention() 