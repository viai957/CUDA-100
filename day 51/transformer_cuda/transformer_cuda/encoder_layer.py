import torch
import torch.nn as nn
from typing import Optional

from transformer_cuda.layernorm import CUDALayerNorm
from transformer_cuda.attention import CUDAMultiHeadAttention
from transformer_cuda.feed_forward import CUDAFeedForward

class CUDATransformerEncoderLayer(nn.Module):
    """
    CUDA-accelerated implementation of a Transformer Encoder Layer.
    
    This implementation uses optimized CUDA kernels for the individual components
    (self-attention, feed-forward network, and layer normalization).
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5
    ):
        """
        Initialize the CUDATransformerEncoderLayer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            activation: Activation function ("gelu" or "relu")
            norm_first: If True, layer norm is done before attention and feedforward
                        operations, otherwise after
            layer_norm_eps: The eps value in layer normalization
        """
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        
        # Self-attention layer
        self.self_attn = CUDAMultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        
        # Feed-forward network
        self.feed_forward = CUDAFeedForward(
            d_model=d_model,
            d_ff=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        # Layer normalization
        self.norm1 = CUDALayerNorm(
            normalized_shape=d_model,
            eps=layer_norm_eps
        )
        self.norm2 = CUDALayerNorm(
            normalized_shape=d_model,
            eps=layer_norm_eps
        )
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def _sa_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Self-attention block"""
        x_attn, _ = self.self_attn(x, mask=attn_mask)
        return self.dropout1(x_attn)
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward block"""
        return self.dropout2(self.feed_forward(x))
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the encoder layer.
        
        Args:
            src: Input tensor of shape (batch_size, seq_len, d_model)
            src_mask: Attention mask for self-attention
            src_key_padding_mask: Mask for padding tokens
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Convert key padding mask to attention mask if provided
        if src_key_padding_mask is not None:
            # src_key_padding_mask: (batch_size, seq_len)
            # Convert to attention mask: (batch_size, 1, 1, seq_len)
            mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2).float()
            mask = (1.0 - mask) * -10000.0
            
            # Combine with src_mask if provided
            if src_mask is not None:
                mask = mask + src_mask
        else:
            mask = src_mask
        
        # Apply layer norm first if specified
        if self.norm_first:
            # Self-attention block
            src = src + self._sa_block(self.norm1(src), mask)
            # Feed-forward block
            src = src + self._ff_block(self.norm2(src))
        else:
            # Self-attention block
            src = self.norm1(src + self._sa_block(src, mask))
            # Feed-forward block
            src = self.norm2(src + self._ff_block(src))
        
        return src
    
    def extra_repr(self) -> str:
        """String representation of the module"""
        return (f"d_model={self.d_model}, nhead={self.nhead}, "
                f"dim_feedforward={self.dim_feedforward}, dropout={self.dropout}, "
                f"activation={self.activation}, norm_first={self.norm_first}") 