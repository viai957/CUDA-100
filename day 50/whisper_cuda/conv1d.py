import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
from torch import Tensor
from typing import Optional, Tuple

import whisper_cuda_conv1d


class Conv1d(nn.Module):
    """
    Optimized CUDA implementation of 1D convolution for the Whisper model.
    
    This implementation uses half-precision (FP16) computation for maximum throughput
    with good numerical stability through internal FP32 accumulation.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution (default: 1)
        padding (int): Zero-padding added to both sides of the input (default: 0)
        bias (bool): If True, adds a learnable bias to the output (default: True)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels, kernel_size, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    @custom_fwd
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of Conv1d operation.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, in_channels, width]
                        Must be in half precision (float16)
        
        Returns:
            Tensor: Output tensor of shape [batch_size, out_channels, new_width]
        """
        # Cast inputs to fp16 if necessary
        if x.dtype != torch.float16:
            x = x.to(dtype=torch.float16)
        
        # Check if we're on a supported GPU
        if not x.is_cuda:
            # Fall back to PyTorch's implementation on CPU
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding
            )
        
        # Check for unsupported kernel sizes and fall back to PyTorch if necessary
        if self.kernel_size not in [3, 5, 7]:
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding
            )
        
        # Use our custom CUDA implementation
        return whisper_cuda_conv1d.forward(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.Tensor(),
            self.stride,
            self.padding
        )
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}, '
            f'bias={self.bias is not None}'
        )
    
    @staticmethod
    def from_pytorch(module: nn.Conv1d) -> 'Conv1d':
        """
        Convert a PyTorch Conv1d module to our optimized version.
        
        Args:
            module (nn.Conv1d): PyTorch Conv1d module to convert
            
        Returns:
            Conv1d: Our optimized Conv1d module with the same parameters
        """
        cuda_module = Conv1d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            bias=module.bias is not None
        )
        
        # Copy parameters
        with torch.no_grad():
            cuda_module.weight.copy_(module.weight.to(dtype=torch.float16))
            if module.bias is not None and cuda_module.bias is not None:
                cuda_module.bias.copy_(module.bias.to(dtype=torch.float16))
        
        return cuda_module


def replace_with_cuda_conv1d(model: nn.Module) -> nn.Module:
    """
    Replace all nn.Conv1d modules in the model with our optimized CUDA Conv1d.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        nn.Module: Model with replaced Conv1d modules
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Conv1d):
            setattr(model, name, Conv1d.from_pytorch(module))
        else:
            replace_with_cuda_conv1d(module)
    return model 