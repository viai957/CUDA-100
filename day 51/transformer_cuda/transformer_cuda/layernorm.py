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
    CUDA-accelerated implementation of LayerNorm for Transformer models.
    
    This implementation uses an optimized CUDA kernel with Welford's algorithm
    for numerical stability and supports half precision (FP16/BF16) computation.
    """
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        Initialize the CUDALayerNorm layer.
        
        Args:
            normalized_shape: Size of the features to be normalized (hidden_size)
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
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, input):
        """
        Forward pass using custom CUDA kernel.
        
        Args:
            input: Input tensor of shape (batch_size, seq_len, hidden_size)
                  or (batch_size, hidden_size)
            
        Returns:
            Output tensor of same shape as input
        """
        # Use PyTorch's native implementation as fallback if not on CUDA
        if not input.is_cuda:
            return F.layer_norm(
                input, 
                [self.normalized_shape], 
                self.weight, 
                self.bias, 
                self.eps
            )
        
        # Handle different input shapes
        orig_shape = input.shape
        if len(orig_shape) > 2:
            # For (batch_size, seq_len, hidden_size), reshape to (batch_size*seq_len, hidden_size)
            input = input.reshape(-1, self.normalized_shape)
        
        # Ensure input is in correct precision
        input_dtype = input.dtype
        if input_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            input = input.float()
        
        # Prepare weight and bias
        weight = self.weight if self.elementwise_affine else torch.ones(
            self.normalized_shape, dtype=input.dtype, device=input.device
        )
        bias = self.bias if self.elementwise_affine else torch.zeros(
            self.normalized_shape, dtype=input.dtype, device=input.device
        )
        
        # Call our custom CUDA kernel
        output, mean, inv_std = layernorm_cuda.layernorm_forward(
            input, weight, bias, self.eps
        )
        
        # Restore original shape if needed
        if len(orig_shape) > 2:
            output = output.reshape(orig_shape)
        
        # Restore original dtype if needed
        if output.dtype != input_dtype:
            output = output.to(input_dtype)
        
        return output
    
    def extra_repr(self):
        """String representation of the layer"""
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}' 