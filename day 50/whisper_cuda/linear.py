import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import linear_cuda
except ImportError:
    raise ImportError(
        "CUDA Linear extension not found. Please build the extension with: "
        "python setup.py install"
    )

class CUDALinear(nn.Module):
    """
    Custom CUDA implementation of Linear layer for Whisper model.
    
    This implementation uses our optimized CUDA kernel for the forward pass,
    with half precision (FP16) computation.
    """
    
    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize the CUDALinear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If set to False, the layer will not learn an additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        """
        Forward pass using custom CUDA kernel.
        
        Args:
            input: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Ensure input is in half precision
        if input.dtype != torch.float16:
            input = input.half()
        
        # Use PyTorch's native implementation as fallback if not on CUDA
        if not input.is_cuda:
            return F.linear(input, self.weight, self.bias)
        
        # Call our custom CUDA kernel
        return linear_cuda.linear_forward(input, self.weight, self.bias if self.bias is not None else torch.tensor([]))
    
    def extra_repr(self):
        """String representation of the layer"""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}' 