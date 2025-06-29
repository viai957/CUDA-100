import torch
import torch.nn as nn
import math
from typing import Optional

try:
    import embedding_cuda
except ImportError:
    # We'll use a fallback implementation with PyTorch's native embedding
    pass

class CUDAEmbedding(nn.Module):
    """
    CUDA-accelerated implementation of Embedding with optional positional encoding.
    
    This implementation supports both learned and fixed sinusoidal positional encodings,
    and can optionally use a CUDA kernel for the embedding lookup operation.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        positional_encoding: str = "sinusoidal",  # "sinusoidal", "learned", or "none"
        scale_embeddings: bool = True
    ):
        """
        Initialize the CUDAEmbedding module.
        
        Args:
            num_embeddings: Size of the dictionary of embeddings
            embedding_dim: The size of each embedding vector
            padding_idx: If specified, the entries at padding_idx do not contribute to the gradient
            max_seq_len: Maximum sequence length for positional encodings
            dropout: Dropout probability applied to the embeddings
            positional_encoding: Type of positional encoding to use
            scale_embeddings: Whether to scale embeddings by sqrt(embedding_dim)
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_seq_len = max_seq_len
        self.scale_embeddings = scale_embeddings
        self.positional_encoding = positional_encoding
        
        # Create the embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # Scaling factor
        self.scale = math.sqrt(embedding_dim) if scale_embeddings else 1.0
        
        # Positional encoding
        if positional_encoding == "sinusoidal":
            self.positional_encoding_table = self._create_sinusoidal_encoding(max_seq_len, embedding_dim)
            self.register_buffer("pos_encoding", self.positional_encoding_table)
        elif positional_encoding == "learned":
            self.positional_encoding_table = nn.Parameter(torch.zeros(max_seq_len, embedding_dim))
            nn.init.normal_(self.positional_encoding_table, mean=0, std=0.02)
        else:  # "none"
            self.positional_encoding_table = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Flag to indicate if we should use the CUDA kernel
        self.use_cuda_kernel = False
        try:
            # Check if the CUDA extension is available
            import embedding_cuda
            self.use_cuda_kernel = True
        except ImportError:
            # Fall back to PyTorch implementation
            pass
    
    def _create_sinusoidal_encoding(self, max_seq_len: int, embedding_dim: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding table.
        
        Args:
            max_seq_len: Maximum sequence length
            embedding_dim: Embedding dimension
            
        Returns:
            Positional encoding table of shape (max_seq_len, embedding_dim)
        """
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * 
            (-math.log(10000.0) / embedding_dim)
        )
        
        pos_encoding = torch.zeros(max_seq_len, embedding_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the embedding layer.
        
        Args:
            x: Input tensor of token indices of shape (batch_size, seq_len)
            positions: Optional tensor of position indices of shape (batch_size, seq_len)
                      If None, positions are assumed to be [0, 1, 2, ...]
            
        Returns:
            Embedded tensor of shape (batch_size, seq_len, embedding_dim)
        """
        seq_len = x.size(1)
        
        # Use CUDA kernel if available and on CUDA
        if self.use_cuda_kernel and x.is_cuda:
            if self.positional_encoding != "none":
                if positions is None:
                    # Default positions: [0, 1, 2, ...]
                    positions = torch.arange(seq_len, device=x.device).expand_as(x)
                
                return embedding_cuda.embedding_forward(
                    x,
                    self.embedding.weight,
                    positions,
                    self.positional_encoding_table,
                    self.scale,
                    self.padding_idx,
                    self.dropout.p if self.training else 0.0
                )
            else:
                return embedding_cuda.embedding_forward(
                    x,
                    self.embedding.weight,
                    None,
                    None,
                    self.scale,
                    self.padding_idx,
                    self.dropout.p if self.training else 0.0
                )
        
        # Otherwise, use PyTorch implementation
        embeddings = self.embedding(x) * self.scale
        
        # Add positional encoding if specified
        if self.positional_encoding != "none":
            if positions is None:
                # Default positions: [0, 1, 2, ...]
                positions = torch.arange(seq_len, device=x.device).expand_as(x)
            
            # Ensure positions are within bounds
            positions = positions.clamp(0, self.max_seq_len - 1)
            
            # Get positional encodings for the given positions
            pos_encodings = self.positional_encoding_table[positions]
            
            # Add positional encodings to the embeddings
            embeddings = embeddings + pos_encodings
        
        # Apply dropout
        return self.dropout(embeddings)
    
    def extra_repr(self) -> str:
        """String representation of the module"""
        return (f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
                f"padding_idx={self.padding_idx}, positional_encoding={self.positional_encoding}, "
                f"scale_embeddings={self.scale_embeddings}") 