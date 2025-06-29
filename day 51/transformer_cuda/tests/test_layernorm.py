import torch
import torch.nn as nn
import numpy as np
import time
import pytest

# Import our CUDA LayerNorm
try:
    from transformer_cuda import CUDALayerNorm
except ImportError:
    pytest.skip("transformer_cuda not installed", allow_module_level=True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDALayerNorm:
    
    def setup_method(self):
        """Set up test parameters"""
        torch.manual_seed(42)
        self.batch_size = 32
        self.seq_len = 128
        self.hidden_size = 512
        self.eps = 1e-5
        
        # Create test tensors
        self.x_2d = torch.randn(
            self.batch_size * self.seq_len, 
            self.hidden_size, 
            device="cuda", 
            dtype=torch.float16
        )
        self.x_3d = torch.randn(
            self.batch_size, 
            self.seq_len, 
            self.hidden_size, 
            device="cuda", 
            dtype=torch.float16
        )
        
        # Create both implementations
        self.pytorch_ln = nn.LayerNorm(self.hidden_size, eps=self.eps).cuda().half()
        self.cuda_ln = CUDALayerNorm(self.hidden_size, eps=self.eps).cuda().half()
        
        # Copy weights to ensure identical parameters
        self.cuda_ln.weight.data.copy_(self.pytorch_ln.weight.data)
        self.cuda_ln.bias.data.copy_(self.pytorch_ln.bias.data)
    
    def test_forward_2d(self):
        """Test forward pass with 2D input"""
        # Run both implementations
        with torch.no_grad():
            pytorch_output = self.pytorch_ln(self.x_2d)
            cuda_output = self.cuda_ln(self.x_2d)
        
        # Check that outputs match
        torch.testing.assert_close(
            cuda_output, 
            pytorch_output, 
            rtol=1e-3, 
            atol=1e-3,
            msg="CUDA LayerNorm 2D output doesn't match PyTorch"
        )
    
    def test_forward_3d(self):
        """Test forward pass with 3D input"""
        # Run both implementations
        with torch.no_grad():
            pytorch_output = self.pytorch_ln(self.x_3d)
            cuda_output = self.cuda_ln(self.x_3d)
        
        # Check that outputs match
        torch.testing.assert_close(
            cuda_output, 
            pytorch_output, 
            rtol=1e-3, 
            atol=1e-3,
            msg="CUDA LayerNorm 3D output doesn't match PyTorch"
        )
    
    def test_backward(self):
        """Test backward pass"""
        # Create tensors that require gradients
        x_2d = self.x_2d.clone().detach().requires_grad_(True)
        x_2d_cuda = self.x_2d.clone().detach().requires_grad_(True)
        
        # Forward pass
        pytorch_output = self.pytorch_ln(x_2d)
        cuda_output = self.cuda_ln(x_2d_cuda)
        
        # Create gradient for backward pass
        grad_output = torch.randn_like(pytorch_output)
        
        # Backward pass
        pytorch_output.backward(grad_output)
        cuda_output.backward(grad_output)
        
        # Check that gradients match
        torch.testing.assert_close(
            x_2d_cuda.grad, 
            x_2d.grad, 
            rtol=1e-3, 
            atol=1e-3,
            msg="CUDA LayerNorm backward gradients don't match PyTorch"
        )
    
    def test_performance(self):
        """Benchmark performance against PyTorch's implementation"""
        # Warmup
        for _ in range(10):
            _ = self.pytorch_ln(self.x_3d)
            _ = self.cuda_ln(self.x_3d)
        
        torch.cuda.synchronize()
        
        # PyTorch timing
        start = time.time()
        for _ in range(100):
            _ = self.pytorch_ln(self.x_3d)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start
        
        # CUDA implementation timing
        start = time.time()
        for _ in range(100):
            _ = self.cuda_ln(self.x_3d)
        torch.cuda.synchronize()
        cuda_time = time.time() - start
        
        # Print performance comparison
        speedup = pytorch_time / cuda_time
        print(f"\nLayerNorm Performance:")
        print(f"PyTorch: {pytorch_time:.6f}s")
        print(f"CUDA:    {cuda_time:.6f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Not a strict test, but we expect some speedup
        assert speedup > 0.5, "CUDA implementation is significantly slower than PyTorch"

if __name__ == "__main__":
    # Run tests manually
    test = TestCUDALayerNorm()
    test.setup_method()
    test.test_forward_2d()
    test.test_forward_3d()
    test.test_backward()
    test.test_performance()
    print("All tests passed!") 