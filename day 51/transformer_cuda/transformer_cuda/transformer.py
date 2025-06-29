import torch
import torch.nn as nn
import copy
from typing import Optional, Dict, List, Tuple

from transformer_cuda.embedding import CUDAEmbedding
from transformer_cuda.encoder_layer import CUDATransformerEncoderLayer
from transformer_cuda.decoder_layer import CUDATransformerDecoderLayer
from transformer_cuda.layernorm import CUDALayerNorm

class CUDATransformerEncoder(nn.Module):
    """
    CUDA-accelerated implementation of a Transformer Encoder.
    
    This implementation stacks multiple encoder layers and applies layer normalization
    to the output.
    """
    
    def __init__(
        self,
        encoder_layer: CUDATransformerEncoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None
    ):
        """
        Initialize the CUDATransformerEncoder.
        
        Args:
            encoder_layer: An instance of CUDATransformerEncoderLayer
            num_layers: Number of encoder layers to stack
            norm: Optional normalization layer
        """
        super().__init__()
        
        # Create a stack of encoder layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the encoder.
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len, d_model)
            mask: Attention mask for self-attention
            src_key_padding_mask: Mask for padding tokens
            
        Returns:
            Output tensor of shape (batch_size, src_seq_len, d_model)
        """
        output = src
        
        # Process through each encoder layer
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        # Apply final normalization if provided
        if self.norm is not None:
            output = self.norm(output)
        
        return output

class CUDATransformerDecoder(nn.Module):
    """
    CUDA-accelerated implementation of a Transformer Decoder.
    
    This implementation stacks multiple decoder layers and applies layer normalization
    to the output.
    """
    
    def __init__(
        self,
        decoder_layer: CUDATransformerDecoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None
    ):
        """
        Initialize the CUDATransformerDecoder.
        
        Args:
            decoder_layer: An instance of CUDATransformerDecoderLayer
            num_layers: Number of decoder layers to stack
            norm: Optional normalization layer
        """
        super().__init__()
        
        # Create a stack of decoder layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Dict[str, Dict[nn.Linear, torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, Dict[nn.Linear, torch.Tensor]]]]]:
        """
        Forward pass of the decoder.
        
        Args:
            tgt: Target tensor of shape (batch_size, tgt_seq_len, d_model)
            memory: Memory tensor from encoder of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Attention mask for self-attention
            memory_mask: Attention mask for cross-attention
            tgt_key_padding_mask: Mask for padding tokens in target sequence
            memory_key_padding_mask: Mask for padding tokens in source sequence
            kv_cache: Optional list of key-value caches for each decoder layer
            
        Returns:
            output: Output tensor of shape (batch_size, tgt_seq_len, d_model)
            updated_kv_cache: Updated key-value caches
        """
        output = tgt
        
        # Initialize kv_cache if not provided
        if kv_cache is None:
            kv_cache = [None] * self.num_layers
        
        # Process through each decoder layer
        for idx, layer in enumerate(self.layers):
            output, kv_cache[idx] = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                kv_cache=kv_cache[idx]
            )
        
        # Apply final normalization if provided
        if self.norm is not None:
            output = self.norm(output)
        
        return output, kv_cache

class CUDATransformer(nn.Module):
    """
    CUDA-accelerated implementation of a Transformer model.
    
    This implementation includes an encoder, a decoder, and embedding layers for
    both source and target sequences.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = False,
        src_vocab_size: int = 10000,
        tgt_vocab_size: int = 10000,
        max_seq_len: int = 5000,
        src_padding_idx: Optional[int] = None,
        tgt_padding_idx: Optional[int] = None,
        positional_encoding: str = "sinusoidal"
    ):
        """
        Initialize the CUDATransformer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            activation: Activation function ("gelu" or "relu")
            norm_first: If True, layer norm is done before attention and feedforward
            src_vocab_size: Size of the source vocabulary
            tgt_vocab_size: Size of the target vocabulary
            max_seq_len: Maximum sequence length for positional encodings
            src_padding_idx: Padding index for source sequences
            tgt_padding_idx: Padding index for target sequences
            positional_encoding: Type of positional encoding ("sinusoidal", "learned", or "none")
        """
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        # Source and target embeddings
        self.src_embedding = CUDAEmbedding(
            num_embeddings=src_vocab_size,
            embedding_dim=d_model,
            padding_idx=src_padding_idx,
            max_seq_len=max_seq_len,
            dropout=dropout,
            positional_encoding=positional_encoding
        )
        
        self.tgt_embedding = CUDAEmbedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=d_model,
            padding_idx=tgt_padding_idx,
            max_seq_len=max_seq_len,
            dropout=dropout,
            positional_encoding=positional_encoding
        )
        
        # Create encoder layer
        encoder_layer = CUDATransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first
        )
        
        # Create encoder
        encoder_norm = CUDALayerNorm(d_model)
        self.encoder = CUDATransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm
        )
        
        # Create decoder layer
        decoder_layer = CUDATransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first
        )
        
        # Create decoder
        decoder_norm = CUDALayerNorm(d_model)
        self.decoder = CUDATransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=decoder_norm
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with Glorot uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode the source sequence.
        
        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            src_mask: Attention mask for self-attention
            src_key_padding_mask: Mask for padding tokens
            
        Returns:
            Memory tensor of shape (batch_size, src_seq_len, d_model)
        """
        # Embed source tokens
        src_emb = self.src_embedding(src)
        
        # Encode
        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        return memory
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Dict[str, Dict[nn.Linear, torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, Dict[nn.Linear, torch.Tensor]]]]]:
        """
        Decode the target sequence.
        
        Args:
            tgt: Target token indices of shape (batch_size, tgt_seq_len)
            memory: Memory tensor from encoder of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Attention mask for self-attention
            memory_mask: Attention mask for cross-attention
            tgt_key_padding_mask: Mask for padding tokens in target sequence
            memory_key_padding_mask: Mask for padding tokens in source sequence
            kv_cache: Optional list of key-value caches for each decoder layer
            
        Returns:
            output: Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
            updated_kv_cache: Updated key-value caches
        """
        # Embed target tokens
        tgt_emb = self.tgt_embedding(tgt)
        
        # Decode
        output, kv_cache = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            kv_cache=kv_cache
        )
        
        # Project to vocabulary
        output = self.output_projection(output)
        
        return output, kv_cache
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the transformer.
        
        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            tgt: Target token indices of shape (batch_size, tgt_seq_len)
            src_mask: Attention mask for encoder self-attention
            tgt_mask: Attention mask for decoder self-attention
            memory_mask: Attention mask for decoder cross-attention
            src_key_padding_mask: Mask for padding tokens in source sequence
            tgt_key_padding_mask: Mask for padding tokens in target sequence
            memory_key_padding_mask: Mask for padding tokens in source sequence (for cross-attention)
            
        Returns:
            Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Create causal mask for decoder self-attention if not provided
        if tgt_mask is None:
            tgt_seq_len = tgt.size(1)
            device = tgt.device
            tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)
        
        # Encode
        memory = self.encode(src, src_mask, src_key_padding_mask)
        
        # If memory_key_padding_mask is not provided, use src_key_padding_mask
        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask
        
        # Decode
        output, _ = self.decode(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask
        )
        
        return output
    
    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        bos_token_id: int,
        eos_token_id: int,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate a sequence autoregressively.
        
        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            max_len: Maximum length of the generated sequence
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            src_mask: Attention mask for encoder self-attention
            src_key_padding_mask: Mask for padding tokens in source sequence
            temperature: Sampling temperature (1.0 means no change, lower means more deterministic)
            top_k: If specified, only sample from the top-k most likely tokens
            top_p: If specified, sample from the top tokens with cumulative probability >= top_p
            
        Returns:
            Generated token indices of shape (batch_size, seq_len)
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode the source sequence
        memory = self.encode(src, src_mask, src_key_padding_mask)
        
        # Initialize the target sequence with BOS token
        tgt = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        # Initialize key-value cache
        kv_cache = None
        
        # Generate tokens autoregressively
        for _ in range(max_len - 1):
            # Create causal mask for the current target length
            tgt_seq_len = tgt.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)
            
            # Decode
            output, kv_cache = self.decode(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
                kv_cache=kv_cache
            )
            
            # Get the next token probabilities
            next_token_logits = output[:, -1, :] / temperature
            
            # Apply top-k sampling
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the new token
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Check if all sequences have generated EOS
            if (tgt == eos_token_id).any(dim=1).all():
                break
        
        return tgt
    
    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate a square causal mask for the sequence.
        
        Args:
            sz: Sequence length
            device: Device to create the mask on
            
        Returns:
            Mask tensor of shape (sz, sz) where mask[i, j] = -inf if i < j else 0
        """
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def extra_repr(self) -> str:
        """String representation of the module"""
        return (f"d_model={self.d_model}, nhead={self.nhead}, "
                f"encoder_layers={self.encoder.num_layers}, decoder_layers={self.decoder.num_layers}") 