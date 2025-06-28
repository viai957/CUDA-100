import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gelu_cuda
except ImportError:
    raise ImportError(
        "CUDA GELU extension not found. Please build the extension with: "
        "python setup.py install"
    )

class CUDAGELU(nn.Module):
    """
    Custom CUDA implementation of GELU activation function for Whisper model.
    
    This implementation uses our optimized CUDA kernel for the forward pass,
    with half precision (FP16) computation.
    """
    
    def __init__(self):
        """Initialize the CUDAGELU activation function."""
        super().__init__()
    
    def forward(self, input):
        """
        Forward pass using custom CUDA kernel.
        
        Args:
            input: Input tensor of any shape
            
        Returns:
            Output tensor of the same shape as input
        """
        # Ensure input is in half precision
        if input.dtype != torch.float16:
            input = input.half()
        
        # Use PyTorch's native implementation as fallback if not on CUDA
        if not input.is_cuda:
            return F.gelu(input)
        
        # Call our custom CUDA kernel
        return gelu_cuda.gelu_forward(input)
    
    def extra_repr(self):
        """String representation of the activation function"""
        return "GELU(cuda)" 