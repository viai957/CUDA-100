import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import attention_cuda
except ImportError:
    raise ImportError(
        "CUDA Attention extension not found. Please build the extension with: "
        "python setup.py install"
    )

class CUDAMultiHeadAttention(nn.Module):
    """
    Custom CUDA implementation of MultiHeadAttention for Whisper model.
    
    This implementation uses our optimized CUDA kernels for the forward pass,
    with half precision (FP16) computation.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int = 64):
        """
        Initialize the CUDAMultiHeadAttention module.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of parallel attention heads
            head_dim: Dimension of each attention head (default: 64)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Initialize query, key, value projections
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=True)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with Glorot uniform initialization"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(
        self, 
        x: torch.Tensor, 
        xa: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using custom CUDA kernels.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            xa: Optional cross-attention input tensor
            mask: Optional attention mask
            kv_cache: Optional key/value cache for efficient decoding
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, embed_dim)
            attention_weights: Optional attention weights for visualization
        """
        # Ensure inputs are in half precision
        if x.dtype != torch.float16:
            x = x.half()
        if xa is not None and xa.dtype != torch.float16:
            xa = xa.half()
        
        # Use PyTorch's native implementation as fallback if not on CUDA
        if not x.is_cuda:
            # Implement fallback using PyTorch's native implementation
            return self._pytorch_forward(x, xa, mask, kv_cache)
        
        # Handle cross-attention
        if xa is not None:
            # For cross-attention, use x for queries and xa for keys/values
            input_q = x
            input_kv = xa
        else:
            # For self-attention, use x for queries, keys, and values
            input_q = input_kv = x
        
        batch_size, seq_len, _ = input_q.shape
        
        # Project queries, keys, values
        if kv_cache is None or xa is None or self.k_proj not in kv_cache:
            # Compute Q, K, V projections using our custom CUDA kernel
            q = self.q_proj(input_q)
            k = self.k_proj(input_kv)
            v = self.v_proj(input_kv)
            
            # Reshape for attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim)
        else:
            # Use cached keys and values for efficient decoding
            q = self.q_proj(input_q).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = kv_cache[self.k_proj]
            v = kv_cache[self.v_proj]
        
        # Determine if we need causal masking
        causal_mask = mask is not None and seq_len > 1
        
        # Compute attention using our custom CUDA kernel
        attn_output = attention_cuda.attention_forward(q, k, v, mask, causal_mask)
        
        # Reshape attention output
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Apply output projection
        output = self.out_proj(attn_output)
        
        # We don't return attention weights for efficiency, but could be added if needed
        return output, None
    
    def _pytorch_forward(
        self, 
        x: torch.Tensor, 
        xa: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        PyTorch fallback implementation for CPU tensors.
        """
        # Handle cross-attention
        if xa is not None:
            # For cross-attention, use x for queries and xa for keys/values
            input_q = x
            input_kv = xa
        else:
            # For self-attention, use x for queries, keys, and values
            input_q = input_kv = x
        
        batch_size, seq_len, _ = input_q.shape
        
        # Project queries, keys, values
        if kv_cache is None or xa is None or self.k_proj not in kv_cache:
            q = self.q_proj(input_q)
            k = self.k_proj(input_kv)
            v = self.v_proj(input_kv)
        else:
            q = self.q_proj(input_q)
            k = kv_cache[self.k_proj]
            v = kv_cache[self.v_proj]
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = (self.head_dim) ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        
        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights + mask.unsqueeze(1)
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention
        attn_output = attn_weights @ v
        
        # Reshape attention output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Apply output projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights
    
    def extra_repr(self):
        """String representation of the module"""
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}' 