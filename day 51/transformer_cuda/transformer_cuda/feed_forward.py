import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import feed_forward_cuda
except ImportError:
    # We'll use a fallback implementation with our individual components
    pass

class CUDAFeedForward(nn.Module):
    """
    CUDA-accelerated implementation of the Feed Forward Network for Transformer models.
    
    This implementation uses optimized CUDA kernels for the forward and backward passes,
    with support for fused operations (Linear → GELU → Linear) for better performance.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "gelu"):
        """
        Initialize the CUDAFeedForward module.
        
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension
            dropout: Dropout probability
            activation: Activation function ("gelu" or "relu")
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        
        # Initialize linear layers
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Flag to indicate if we should use the fused kernel
        self.use_fused_kernel = False
        try:
            # Check if the CUDA extension is available
            import feed_forward_cuda
            self.use_fused_kernel = True
        except ImportError:
            # Fall back to PyTorch implementation
            pass
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with Glorot uniform initialization"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0.)
            nn.init.constant_(self.fc2.bias, 0.)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using custom CUDA kernels when available.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Use fused kernel if available and on CUDA
        if self.use_fused_kernel and x.is_cuda:
            return feed_forward_cuda.feed_forward_forward(
                x,
                self.fc1.weight,
                self.fc1.bias,
                self.fc2.weight,
                self.fc2.bias,
                self.activation,
                self.dropout if self.training else 0.0
            )
        
        # Otherwise, use PyTorch implementation
        if self.activation == "gelu":
            return self.fc2(self.dropout_layer(F.gelu(self.fc1(x))))
        else:  # relu
            return self.fc2(self.dropout_layer(F.relu(self.fc1(x))))
    
    def extra_repr(self):
        """String representation of the module"""
        return f'd_model={self.d_model}, d_ff={self.d_ff}, dropout={self.dropout}, activation={self.activation}' 