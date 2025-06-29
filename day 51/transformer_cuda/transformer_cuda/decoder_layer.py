import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from transformer_cuda.layernorm import CUDALayerNorm
from transformer_cuda.attention import CUDAMultiHeadAttention
from transformer_cuda.feed_forward import CUDAFeedForward

class CUDATransformerDecoderLayer(nn.Module):
    """
    CUDA-accelerated implementation of a Transformer Decoder Layer.
    
    This implementation uses optimized CUDA kernels for the individual components
    (self-attention, cross-attention, feed-forward network, and layer normalization).
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
        Initialize the CUDATransformerDecoderLayer.
        
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
        
        # Cross-attention layer
        self.cross_attn = CUDAMultiHeadAttention(
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
        self.norm3 = CUDALayerNorm(
            normalized_shape=d_model,
            eps=layer_norm_eps
        )
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def _sa_block(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[nn.Linear, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Self-attention block"""
        x_attn, _ = self.self_attn(x, mask=attn_mask, kv_cache=kv_cache)
        return self.dropout1(x_attn)
    
    def _ca_block(
        self, 
        x: torch.Tensor, 
        memory: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[nn.Linear, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Cross-attention block"""
        x_attn, _ = self.cross_attn(x, memory, memory, mask=attn_mask, kv_cache=kv_cache)
        return self.dropout2(x_attn)
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward block"""
        return self.dropout3(self.feed_forward(x))
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, Dict[nn.Linear, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Dict[nn.Linear, torch.Tensor]]]]:
        """
        Forward pass of the decoder layer.
        
        Args:
            tgt: Target tensor of shape (batch_size, tgt_seq_len, d_model)
            memory: Memory tensor from encoder of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Attention mask for self-attention
            memory_mask: Attention mask for cross-attention
            tgt_key_padding_mask: Mask for padding tokens in target sequence
            memory_key_padding_mask: Mask for padding tokens in source sequence
            kv_cache: Optional key-value cache for efficient autoregressive generation
            
        Returns:
            output: Output tensor of shape (batch_size, tgt_seq_len, d_model)
            updated_kv_cache: Updated key-value cache
        """
        # Initialize or get caches
        self_kv_cache = None
        cross_kv_cache = None
        if kv_cache is not None:
            if "self" not in kv_cache:
                kv_cache["self"] = {}
            if "cross" not in kv_cache:
                kv_cache["cross"] = {}
            self_kv_cache = kv_cache["self"]
            cross_kv_cache = kv_cache["cross"]
        
        # Convert key padding masks to attention masks if provided
        if tgt_key_padding_mask is not None:
            # tgt_key_padding_mask: (batch_size, tgt_seq_len)
            # Convert to attention mask: (batch_size, 1, 1, tgt_seq_len)
            tgt_pad_mask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(2).float()
            tgt_pad_mask = (1.0 - tgt_pad_mask) * -10000.0
            
            # Combine with tgt_mask if provided
            if tgt_mask is not None:
                tgt_mask = tgt_pad_mask + tgt_mask
            else:
                tgt_mask = tgt_pad_mask
        
        if memory_key_padding_mask is not None:
            # memory_key_padding_mask: (batch_size, src_seq_len)
            # Convert to attention mask: (batch_size, 1, 1, src_seq_len)
            memory_pad_mask = memory_key_padding_mask.unsqueeze(1).unsqueeze(2).float()
            memory_pad_mask = (1.0 - memory_pad_mask) * -10000.0
            
            # Combine with memory_mask if provided
            if memory_mask is not None:
                memory_mask = memory_pad_mask + memory_mask
            else:
                memory_mask = memory_pad_mask
        
        # Apply layer norm first if specified
        if self.norm_first:
            # Self-attention block
            tgt2 = self.norm1(tgt)
            tgt = tgt + self._sa_block(tgt2, tgt_mask, self_kv_cache)
            
            # Cross-attention block
            tgt2 = self.norm2(tgt)
            tgt = tgt + self._ca_block(tgt2, memory, memory_mask, cross_kv_cache)
            
            # Feed-forward block
            tgt2 = self.norm3(tgt)
            tgt = tgt + self._ff_block(tgt2)
        else:
            # Self-attention block
            tgt2 = self._sa_block(tgt, tgt_mask, self_kv_cache)
            tgt = self.norm1(tgt + tgt2)
            
            # Cross-attention block
            tgt2 = self._ca_block(tgt, memory, memory_mask, cross_kv_cache)
            tgt = self.norm2(tgt + tgt2)
            
            # Feed-forward block
            tgt2 = self._ff_block(tgt)
            tgt = self.norm3(tgt + tgt2)
        
        return tgt, kv_cache
    
    def extra_repr(self) -> str:
        """String representation of the module"""
        return (f"d_model={self.d_model}, nhead={self.nhead}, "
                f"dim_feedforward={self.dim_feedforward}, dropout={self.dropout}, "
                f"activation={self.activation}, norm_first={self.norm_first}") 