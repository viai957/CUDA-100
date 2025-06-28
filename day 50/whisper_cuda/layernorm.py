import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import layernorm_cuda
except ImportError:
    raise ImportError(
        "CUDA LayerNorm extension not found. Please build the extension with: "
        "python setup.py install"
    )

class CUDALayerNorm(nn.Module):
    """
    Custom CUDA implementation of LayerNorm for Whisper model.
    
    This implementation uses our optimized CUDA kernel for the forward pass,
    with half precision (FP16) computation.
    """
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        Initialize the CUDALayerNorm layer.
        
        Args:
            normalized_shape: Size of the features to be normalized
            eps: Small constant for numerical stability
            elementwise_affine: If True, learn elementwise affine parameters
        """
        super().__init__()
        
        if isinstance(normalized_shape, int):
            self.normalized_shape = normalized_shape
        else:
            self.normalized_shape = normalized_shape[0]
        
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, dtype=torch.float16))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape, dtype=torch.float16))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, input):
        """
        Forward pass using custom CUDA kernel.
        
        Args:
            input: Input tensor of shape (batch_size, normalized_shape)
            
        Returns:
            Output tensor of shape (batch_size, normalized_shape)
        """
        # Ensure input is in half precision
        if input.dtype != torch.float16:
            input = input.half()
        
        # Use PyTorch's native implementation as fallback if not on CUDA
        if not input.is_cuda:
            return F.layer_norm(
                input, 
                [self.normalized_shape], 
                self.weight, 
                self.bias, 
                self.eps
            )
        
        # Prepare weight and bias
        weight = self.weight if self.elementwise_affine else torch.ones(
            self.normalized_shape, dtype=torch.float16, device=input.device
        )
        bias = self.bias if self.elementwise_affine else torch.tensor([], device=input.device)
        
        # Call our custom CUDA kernel
        output, _, _ = layernorm_cuda.layernorm_forward(
            input, weight, bias, self.eps
        )
        
        return output
    
    def extra_repr(self):
        """String representation of the layer"""
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}' 