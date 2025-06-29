import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

try:
    import attention_cuda
except ImportError:
    raise ImportError(
        "CUDA Attention extension not found. Please build the extension with: "
        "python setup.py install"
    )

class CUDAMultiHeadAttention(nn.Module):
    """
    CUDA-accelerated implementation of MultiHeadAttention for Transformer models.
    
    This implementation uses optimized CUDA kernels for the forward and backward passes,
    with support for both self-attention and cross-attention, causal masking,
    and key-value caching for efficient autoregressive generation.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        """
        Initialize the CUDAMultiHeadAttention module.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of parallel attention heads
            dropout: Dropout probability on attention weights
            bias: Whether to include bias in linear projections
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Initialize query, key, value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with Glorot uniform initialization"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: Optional[torch.Tensor] = None, 
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[nn.Linear, torch.Tensor]] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using custom CUDA kernels.
        
        Args:
            query: Query tensor of shape (batch_size, tgt_len, embed_dim)
            key: Key tensor of shape (batch_size, src_len, embed_dim), or None for self-attention
            value: Value tensor of shape (batch_size, src_len, embed_dim), or None for self-attention
            mask: Optional attention mask of shape (batch_size, 1, tgt_len, src_len) or (batch_size, 1, 1, src_len)
            kv_cache: Optional key-value cache for efficient autoregressive generation
            need_weights: Whether to return attention weights
            
        Returns:
            output: Output tensor of shape (batch_size, tgt_len, embed_dim)
            attention_weights: Optional attention weights if need_weights is True
        """
        # Use key/value if provided, otherwise use query for self-attention
        key = query if key is None else key
        value = query if value is None else value
        
        # Use PyTorch's native implementation as fallback if not on CUDA
        if not query.is_cuda or not key.is_cuda or not value.is_cuda:
            return self._pytorch_forward(query, key, value, mask, kv_cache, need_weights)
        
        # Get dimensions
        batch_size, tgt_len, _ = query.shape
        _, src_len, _ = key.shape
        
        # Project query, key, value
        if kv_cache is not None and self.k_proj in kv_cache and self.v_proj in kv_cache:
            # Use cached keys and values
            q = self.q_proj(query)
            k = kv_cache[self.k_proj]
            v = kv_cache[self.v_proj]
            
            # Append new keys and values to cache
            if key is not None:
                new_k = self.k_proj(key)
                new_v = self.v_proj(value)
                k = torch.cat([k, new_k], dim=1)
                v = torch.cat([v, new_v], dim=1)
                kv_cache[self.k_proj] = k
                kv_cache[self.v_proj] = v
        else:
            # Compute fresh projections
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
            
            # Update cache if provided
            if kv_cache is not None:
                kv_cache[self.k_proj] = k
                kv_cache[self.v_proj] = v
        
        # Reshape for multi-head attention
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Determine if we need causal masking
        is_causal = mask is not None and mask.size(-1) > 1 and mask.size(-2) > 1
        
        # Call our custom CUDA kernel
        output, attn_weights = attention_cuda.attention_forward(
            q, k, v, mask, is_causal, self.dropout if self.training else 0.0, need_weights
        )
        
        # Reshape output
        output = output.reshape(batch_size, tgt_len, self.embed_dim)
        
        # Apply output projection
        output = self.out_proj(output)
        
        return output, attn_weights if need_weights else None
    
    def _pytorch_forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[nn.Linear, torch.Tensor]] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        PyTorch fallback implementation for CPU tensors.
        """
        # Get dimensions
        batch_size, tgt_len, _ = query.shape
        _, src_len, _ = key.shape
        
        # Project query, key, value
        if kv_cache is not None and self.k_proj in kv_cache and self.v_proj in kv_cache:
            # Use cached keys and values
            q = self.q_proj(query)
            k = kv_cache[self.k_proj]
            v = kv_cache[self.v_proj]
            
            # Append new keys and values to cache
            if key is not None:
                new_k = self.k_proj(key)
                new_v = self.v_proj(value)
                k = torch.cat([k, new_k], dim=1)
                v = torch.cat([v, new_v], dim=1)
                kv_cache[self.k_proj] = k
                kv_cache[self.v_proj] = v
        else:
            # Compute fresh projections
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
            
            # Update cache if provided
            if kv_cache is not None:
                kv_cache[self.k_proj] = k
                kv_cache[self.v_proj] = v
        
        # Reshape for multi-head attention
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights + mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply dropout if training
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention
        output = torch.matmul(attn_weights, v)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        
        # Apply output projection
        output = self.out_proj(output)
        
        return output, attn_weights if need_weights else None
    
    def extra_repr(self):
        """String representation of the module"""
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout}' 