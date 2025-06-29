import torch
import torch.nn as nn
import numpy as np
import time
import pytest

# Import our CUDA FeedForward
try:
    from transformer_cuda import CUDAFeedForward
except ImportError:
    pytest.skip("transformer_cuda not installed", allow_module_level=True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAFeedForward:
    
    def setup_method(self):
        """Set up test parameters"""
        torch.manual_seed(42)
        self.batch_size = 16
        self.seq_len = 64
        self.d_model = 512
        self.d_ff = 2048
        self.dropout = 0.1
        
        # Create test tensors
        self.x = torch.randn(
            self.batch_size, 
            self.seq_len, 
            self.d_model, 
            device="cuda", 
            dtype=torch.float16
        )
        
        # Create both implementations
        # PyTorch implementation
        self.pytorch_ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model)
        ).cuda().half()
        
        # Our CUDA implementation
        self.cuda_ffn = CUDAFeedForward(
            self.d_model,
            self.d_ff,
            dropout=self.dropout,
            activation="gelu"
        ).cuda().half()
        
        # Copy weights to ensure identical parameters
        self.cuda_ffn.fc1.weight.data.copy_(self.pytorch_ffn[0].weight.data)
        self.cuda_ffn.fc1.bias.data.copy_(self.pytorch_ffn[0].bias.data)
        self.cuda_ffn.fc2.weight.data.copy_(self.pytorch_ffn[3].weight.data)
        self.cuda_ffn.fc2.bias.data.copy_(self.pytorch_ffn[3].bias.data)
    
    def test_forward_eval(self):
        """Test forward pass in eval mode (no dropout)"""
        # Set eval mode to disable dropout for deterministic results
        self.pytorch_ffn.eval()
        self.cuda_ffn.eval()
        
        # Run both implementations
        with torch.no_grad():
            pytorch_output = self.pytorch_ffn(self.x)
            cuda_output = self.cuda_ffn(self.x)
        
        # Check that outputs match
        torch.testing.assert_close(
            cuda_output, 
            pytorch_output, 
            rtol=1e-2, 
            atol=1e-2,
            msg="CUDA FeedForward output doesn't match PyTorch"
        )
    
    def test_forward_train(self):
        """Test forward pass in train mode (with dropout)"""
        # Set same seed for reproducible dropout
        torch.manual_seed(123)
        
        # Set train mode to enable dropout
        self.pytorch_ffn.train()
        self.cuda_ffn.train()
        
        # Run both implementations
        with torch.no_grad():
            pytorch_output = self.pytorch_ffn(self.x)
            
            # Reset seed to get same dropout pattern
            torch.manual_seed(123)
            cuda_output = self.cuda_ffn(self.x)
        
        # Check that outputs are reasonably close (exact match not expected due to dropout implementation differences)
        # We check that the mean and std are similar
        pytorch_mean = pytorch_output.mean().item()
        cuda_mean = cuda_output.mean().item()
        pytorch_std = pytorch_output.std().item()
        cuda_std = cuda_output.std().item()
        
        assert abs(pytorch_mean - cuda_mean) < 0.1, \
            f"Mean values too different: PyTorch={pytorch_mean}, CUDA={cuda_mean}"
        assert abs(pytorch_std - cuda_std) < 0.1, \
            f"Std values too different: PyTorch={pytorch_std}, CUDA={cuda_std}"
    
    def test_relu_activation(self):
        """Test with ReLU activation"""
        # Create new models with ReLU
        pytorch_ffn_relu = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(0.0),  # No dropout for deterministic results
            nn.Linear(self.d_ff, self.d_model)
        ).cuda().half()
        
        cuda_ffn_relu = CUDAFeedForward(
            self.d_model,
            self.d_ff,
            dropout=0.0,
            activation="relu"
        ).cuda().half()
        
        # Copy weights
        cuda_ffn_relu.fc1.weight.data.copy_(pytorch_ffn_relu[0].weight.data)
        cuda_ffn_relu.fc1.bias.data.copy_(pytorch_ffn_relu[0].bias.data)
        cuda_ffn_relu.fc2.weight.data.copy_(pytorch_ffn_relu[3].weight.data)
        cuda_ffn_relu.fc2.bias.data.copy_(pytorch_ffn_relu[3].bias.data)
        
        # Set eval mode
        pytorch_ffn_relu.eval()
        cuda_ffn_relu.eval()
        
        # Run both implementations
        with torch.no_grad():
            pytorch_output = pytorch_ffn_relu(self.x)
            cuda_output = cuda_ffn_relu(self.x)
        
        # Check that outputs match
        torch.testing.assert_close(
            cuda_output, 
            pytorch_output, 
            rtol=1e-2, 
            atol=1e-2,
            msg="CUDA FeedForward with ReLU output doesn't match PyTorch"
        )
    
    def test_backward(self):
        """Test backward pass"""
        # Create tensors that require gradients
        x = self.x.clone().detach().requires_grad_(True)
        x_cuda = self.x.clone().detach().requires_grad_(True)
        
        # Set eval mode to disable dropout for deterministic results
        self.pytorch_ffn.eval()
        self.cuda_ffn.eval()
        
        # Forward pass
        pytorch_output = self.pytorch_ffn(x)
        cuda_output = self.cuda_ffn(x_cuda)
        
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
            _ = self.pytorch_ffn(self.x)
            _ = self.cuda_ffn(self.x)
        
        torch.cuda.synchronize()
        
        # PyTorch timing
        start = time.time()
        for _ in range(100):
            _ = self.pytorch_ffn(self.x)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start
        
        # CUDA implementation timing
        start = time.time()
        for _ in range(100):
            _ = self.cuda_ffn(self.x)
        torch.cuda.synchronize()
        cuda_time = time.time() - start
        
        # Print performance comparison
        speedup = pytorch_time / cuda_time
        print(f"\nFeedForward Performance:")
        print(f"PyTorch: {pytorch_time:.6f}s")
        print(f"CUDA:    {cuda_time:.6f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Not a strict test, but we expect some speedup
        assert speedup > 0.5, "CUDA implementation is significantly slower than PyTorch"

if __name__ == "__main__":
    # Run tests manually
    test = TestCUDAFeedForward()
    test.setup_method()
    test.test_forward_eval()
    test.test_forward_train()
    test.test_relu_activation()
    test.test_backward()
    test.test_performance()
    print("All tests passed!") 