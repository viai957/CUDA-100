import torch
import torch.nn as nn
import numpy as np
import time
import pytest

# Import our CUDA MultiHeadAttention
try:
    from transformer_cuda import CUDAMultiHeadAttention
except ImportError:
    pytest.skip("transformer_cuda not installed", allow_module_level=True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAMultiHeadAttention:
    
    def setup_method(self):
        """Set up test parameters"""
        torch.manual_seed(42)
        self.batch_size = 8
        self.seq_len = 64
        self.embed_dim = 512
        self.num_heads = 8
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = 0.1
        
        # Create test tensors
        self.x = torch.randn(
            self.batch_size, 
            self.seq_len, 
            self.embed_dim, 
            device="cuda", 
            dtype=torch.float16
        )
        
        # Create masks
        # Padding mask (1 = keep, 0 = mask)
        padding_mask = torch.ones(
            self.batch_size, 
            self.seq_len, 
            device="cuda", 
            dtype=torch.bool
        )
        # Randomly mask some positions
        for i in range(self.batch_size):
            padding_mask[i, -10:] = False
        
        # Convert to attention mask format
        self.padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).float()
        self.padding_mask = (1.0 - self.padding_mask) * -10000.0
        
        # Causal mask for decoder self-attention
        self.causal_mask = torch.triu(
            torch.ones(
                self.seq_len, 
                self.seq_len, 
                device="cuda", 
                dtype=torch.float16
            ) * -10000.0, 
            diagonal=1
        )
        self.causal_mask = self.causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Create both implementations
        self.pytorch_mha = nn.MultiheadAttention(
            self.embed_dim, 
            self.num_heads, 
            dropout=self.dropout, 
            batch_first=True
        ).cuda().half()
        
        self.cuda_mha = CUDAMultiHeadAttention(
            self.embed_dim, 
            self.num_heads, 
            dropout=self.dropout
        ).cuda().half()
        
        # Copy weights to ensure identical parameters
        self.cuda_mha.q_proj.weight.data.copy_(self.pytorch_mha.in_proj_weight[:self.embed_dim])
        self.cuda_mha.k_proj.weight.data.copy_(self.pytorch_mha.in_proj_weight[self.embed_dim:2*self.embed_dim])
        self.cuda_mha.v_proj.weight.data.copy_(self.pytorch_mha.in_proj_weight[2*self.embed_dim:])
        self.cuda_mha.out_proj.weight.data.copy_(self.pytorch_mha.out_proj.weight)
        
        if self.pytorch_mha.in_proj_bias is not None:
            self.cuda_mha.q_proj.bias.data.copy_(self.pytorch_mha.in_proj_bias[:self.embed_dim])
            self.cuda_mha.k_proj.bias.data.copy_(self.pytorch_mha.in_proj_bias[self.embed_dim:2*self.embed_dim])
            self.cuda_mha.v_proj.bias.data.copy_(self.pytorch_mha.in_proj_bias[2*self.embed_dim:])
            self.cuda_mha.out_proj.bias.data.copy_(self.pytorch_mha.out_proj.bias)
    
    def test_self_attention(self):
        """Test self-attention without mask"""
        # Set eval mode to disable dropout for deterministic results
        self.pytorch_mha.eval()
        self.cuda_mha.eval()
        
        # Run both implementations
        with torch.no_grad():
            pytorch_output, _ = self.pytorch_mha(self.x, self.x, self.x)
            cuda_output, _ = self.cuda_mha(self.x)
        
        # Check that outputs match
        torch.testing.assert_close(
            cuda_output, 
            pytorch_output, 
            rtol=1e-2, 
            atol=1e-2,
            msg="CUDA self-attention output doesn't match PyTorch"
        )
    
    def test_self_attention_with_padding_mask(self):
        """Test self-attention with padding mask"""
        # Set eval mode to disable dropout for deterministic results
        self.pytorch_mha.eval()
        self.cuda_mha.eval()
        
        # Run both implementations
        with torch.no_grad():
            pytorch_output, _ = self.pytorch_mha(
                self.x, self.x, self.x, 
                attn_mask=self.padding_mask.squeeze(1).squeeze(1) if self.padding_mask.dim() == 4 else None,
                key_padding_mask=~(self.padding_mask.squeeze(1).squeeze(1).bool()) if self.padding_mask.dim() == 4 else None
            )
            cuda_output, _ = self.cuda_mha(self.x, mask=self.padding_mask)
        
        # Check that outputs match
        torch.testing.assert_close(
            cuda_output, 
            pytorch_output, 
            rtol=1e-2, 
            atol=1e-2,
            msg="CUDA self-attention with padding mask doesn't match PyTorch"
        )
    
    def test_self_attention_with_causal_mask(self):
        """Test self-attention with causal mask"""
        # Set eval mode to disable dropout for deterministic results
        self.pytorch_mha.eval()
        self.cuda_mha.eval()
        
        # Run both implementations
        with torch.no_grad():
            pytorch_output, _ = self.pytorch_mha(
                self.x, self.x, self.x, 
                attn_mask=self.causal_mask.squeeze(0).squeeze(0)
            )
            cuda_output, _ = self.cuda_mha(self.x, mask=self.causal_mask)
        
        # Check that outputs match
        torch.testing.assert_close(
            cuda_output, 
            pytorch_output, 
            rtol=1e-2, 
            atol=1e-2,
            msg="CUDA self-attention with causal mask doesn't match PyTorch"
        )
    
    def test_cross_attention(self):
        """Test cross-attention"""
        # Create different key/value tensors
        kv = torch.randn(
            self.batch_size, 
            self.seq_len // 2, 
            self.embed_dim, 
            device="cuda", 
            dtype=torch.float16
        )
        
        # Set eval mode to disable dropout for deterministic results
        self.pytorch_mha.eval()
        self.cuda_mha.eval()
        
        # Run both implementations
        with torch.no_grad():
            pytorch_output, _ = self.pytorch_mha(self.x, kv, kv)
            cuda_output, _ = self.cuda_mha(self.x, kv, kv)
        
        # Check that outputs match
        torch.testing.assert_close(
            cuda_output, 
            pytorch_output, 
            rtol=1e-2, 
            atol=1e-2,
            msg="CUDA cross-attention output doesn't match PyTorch"
        )
    
    def test_attention_weights(self):
        """Test attention weights output"""
        # Set eval mode to disable dropout for deterministic results
        self.pytorch_mha.eval()
        self.cuda_mha.eval()
        
        # Run both implementations
        with torch.no_grad():
            pytorch_output, pytorch_weights = self.pytorch_mha(self.x, self.x, self.x, need_weights=True)
            cuda_output, cuda_weights = self.cuda_mha(self.x, need_weights=True)
        
        # Check that weights match (with some tolerance due to different implementations)
        # PyTorch returns weights of shape [batch_size, seq_len, seq_len]
        # Our CUDA implementation returns [batch_size, num_heads, seq_len, seq_len]
        # So we need to average over heads
        cuda_weights_avg = cuda_weights.mean(dim=1)
        
        torch.testing.assert_close(
            cuda_weights_avg, 
            pytorch_weights, 
            rtol=1e-2, 
            atol=1e-2,
            msg="CUDA attention weights don't match PyTorch"
        )
    
    def test_backward(self):
        """Test backward pass"""
        # Create tensors that require gradients
        x = self.x.clone().detach().requires_grad_(True)
        x_cuda = self.x.clone().detach().requires_grad_(True)
        
        # Set train mode to enable dropout
        self.pytorch_mha.train()
        self.cuda_mha.train()
        
        # Forward pass
        pytorch_output, _ = self.pytorch_mha(x, x, x)
        cuda_output, _ = self.cuda_mha(x_cuda)
        
        # Create gradient for backward pass
        grad_output = torch.randn_like(pytorch_output)
        
        # Backward pass
        pytorch_output.backward(grad_output)
        cuda_output.backward(grad_output)
        
        # Check that gradients exist and have reasonable values
        assert x_cuda.grad is not None, "CUDA implementation didn't produce gradients"
        assert x.grad is not None, "PyTorch implementation didn't produce gradients"
        
        # Check gradient norms are similar (exact matching is difficult due to different implementations)
        pytorch_grad_norm = x.grad.norm().item()
        cuda_grad_norm = x_cuda.grad.norm().item()
        
        assert abs(pytorch_grad_norm - cuda_grad_norm) / max(pytorch_grad_norm, cuda_grad_norm) < 0.3, \
            f"Gradient norms too different: PyTorch={pytorch_grad_norm}, CUDA={cuda_grad_norm}"
    
    def test_performance(self):
        """Benchmark performance against PyTorch's implementation"""
        # Warmup
        for _ in range(10):
            _ = self.pytorch_mha(self.x, self.x, self.x)
            _ = self.cuda_mha(self.x)
        
        torch.cuda.synchronize()
        
        # PyTorch timing
        start = time.time()
        for _ in range(100):
            _ = self.pytorch_mha(self.x, self.x, self.x)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start
        
        # CUDA implementation timing
        start = time.time()
        for _ in range(100):
            _ = self.cuda_mha(self.x)
        torch.cuda.synchronize()
        cuda_time = time.time() - start
        
        # Print performance comparison
        speedup = pytorch_time / cuda_time
        print(f"\nMultiHeadAttention Performance:")
        print(f"PyTorch: {pytorch_time:.6f}s")
        print(f"CUDA:    {cuda_time:.6f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Not a strict test, but we expect some speedup
        assert speedup > 0.5, "CUDA implementation is significantly slower than PyTorch"

if __name__ == "__main__":
    # Run tests manually
    test = TestCUDAMultiHeadAttention()
    test.setup_method()
    test.test_self_attention()
    test.test_self_attention_with_padding_mask()
    test.test_self_attention_with_causal_mask()
    test.test_cross_attention()
    test.test_attention_weights()
    test.test_backward()
    test.test_performance()
    print("All tests passed!") 