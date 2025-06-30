import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union, List

import numpy as np
from whisper.model import ModelDimensions

# Import our custom CUDA layers
from whisper_cuda import CUDALayerNorm, CUDALinear, CUDAGELU, CUDAMultiHeadAttention, Conv1d


class CUDAResidualAttentionBlock(nn.Module):
    """
    Residual Attention Block using CUDA optimized layers
    """
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        # Self-attention using our CUDA implementation
        self.attn = CUDAMultiHeadAttention(
            embed_dim=n_state,
            num_heads=n_head,
            head_dim=n_state // n_head
        )
        self.attn_ln = CUDALayerNorm(n_state)

        # Cross-attention (optional)
        self.cross_attn = (
            CUDAMultiHeadAttention(
                embed_dim=n_state,
                num_heads=n_head,
                head_dim=n_state // n_head
            ) if cross_attention else None
        )
        self.cross_attn_ln = CUDALayerNorm(n_state) if cross_attention else None

        # Feed-forward network
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            CUDALinear(n_state, n_mlp),
            CUDAGELU(),
            CUDALinear(n_mlp, n_state)
        )
        self.mlp_ln = CUDALayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        xa: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        # Self-attention with residual connection
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        
        # Cross-attention (if applicable)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        
        # Feed-forward network with residual connection
        x = x + self.mlp(self.mlp_ln(x))
        return x


class CUDAAudioEncoder(nn.Module):
    """
    Audio Encoder using CUDA optimized layers
    """
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", self._sinusoids(n_ctx, n_state))

        self.blocks: Iterable[CUDAResidualAttentionBlock] = nn.ModuleList(
            [CUDAResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = CUDALayerNorm(n_state)

    def _sinusoids(self, length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).half()

    def forward(self, x: torch.Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        # Convert input to half precision
        x = x.half()
        
        # Apply 1D convolutions
        x = self.conv1(x)
        x = torch.nn.functional.gelu(x)  # Replace with CUDAGELU if needed
        x = self.conv2(x)
        x = torch.nn.functional.gelu(x)  # Replace with CUDAGELU if needed
        
        # Reshape for transformer blocks
        x = x.permute(0, 2, 1)

        # Check dimensions
        assert x.shape[1:] == self.positional_embedding.shape, f"incorrect audio shape: {x.shape[1:]} vs {self.positional_embedding.shape}"
        
        # Add positional embeddings
        x = (x + self.positional_embedding).to(x.dtype)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer normalization
        x = self.ln_post(x)
        return x


class CUDATextDecoder(nn.Module):
    """
    Text Decoder using CUDA optimized layers
    """
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        # Token embedding layer (kept as nn.Embedding for compatibility)
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        # Transformer blocks with cross-attention
        self.blocks: Iterable[CUDAResidualAttentionBlock] = nn.ModuleList(
            [
                CUDAResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = CUDALayerNorm(n_state)

        # Causal mask for autoregressive decoding
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        # Convert tensors to half precision
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        
        # Token + positional embedding
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        # Final layer normalization
        x = self.ln(x)
        
        # Project to vocabulary
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class CUDAWhisper(nn.Module):
    """
    Whisper model with CUDA-optimized layers for maximum inference speed
    """
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        
        # Initialize encoder and decoder with CUDA layers
        self.encoder = CUDAAudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = CUDATextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        
        # Set up alignment heads as in the original model
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        """Set alignment heads from a compressed byte array"""
        import base64
        import gzip
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        """Embed audio features using the encoder"""
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        """Get logits from tokens and audio features"""
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: encode audio, then decode with text"""
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        """Get the device the model is on"""
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        """Check if the model is multilingual"""
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        """Get the number of supported languages"""
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        Install hooks for key-value caching to speed up decoding
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, CUDAMultiHeadAttention):
                hooks.append(layer.q_proj.register_forward_hook(save_to_cache))
                hooks.append(layer.k_proj.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks


def load_cuda_whisper_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> CUDAWhisper:
    """
    Load a CUDA-optimized Whisper model from a standard Whisper checkpoint.
    
    Args:
        checkpoint_path: Path to the Whisper checkpoint
        device: Device to load the model on ("cuda" or "cpu")
        
    Returns:
        CUDAWhisper: CUDA-optimized Whisper model
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model dimensions
    dims = ModelDimensions(**checkpoint["dims"])
    
    # Create the CUDA model
    model = CUDAWhisper(dims)
    
    # Load state dict with careful handling of potential mismatches
    model_state = model.state_dict()
    pretrained_state = checkpoint["model_state_dict"]
    
    # Filter and potentially adapt keys
    for k, v in pretrained_state.items():
        if k in model_state:
            if model_state[k].shape != v.shape:
                print(f"Shape mismatch for {k}: model has {model_state[k].shape}, checkpoint has {v.shape}")
                continue
            model_state[k] = v
    
    # Load the filtered state dict
    model.load_state_dict(model_state, strict=False)
    
    # Set alignment heads if available
    if "alignment_heads" in checkpoint:
        model.set_alignment_heads(checkpoint["alignment_heads"])
    
    return model.to(device)


def convert_from_original_whisper(original_model) -> CUDAWhisper:
    """
    Convert a standard Whisper model to our CUDA-optimized version.
    
    Args:
        original_model: Standard Whisper model
        
    Returns:
        CUDAWhisper: CUDA-optimized Whisper model
    """
    # Create CUDA model with same dimensions
    cuda_model = CUDAWhisper(original_model.dims)
    
    # Copy weights from original model with careful handling
    # Encoder: convolutional layers
    cuda_model.encoder.conv1.weight.data.copy_(original_model.encoder.conv1.weight.half())
    if original_model.encoder.conv1.bias is not None:
        cuda_model.encoder.conv1.bias.data.copy_(original_model.encoder.conv1.bias.half())
    
    cuda_model.encoder.conv2.weight.data.copy_(original_model.encoder.conv2.weight.half())
    if original_model.encoder.conv2.bias is not None:
        cuda_model.encoder.conv2.bias.data.copy_(original_model.encoder.conv2.bias.half())
    
    # Encoder: positional embeddings
    cuda_model.encoder.positional_embedding.copy_(original_model.encoder.positional_embedding.half())
    
    # Encoder: transformer blocks
    for i, (src_block, dst_block) in enumerate(zip(original_model.encoder.blocks, cuda_model.encoder.blocks)):
        # Self-attention
        dst_block.attn.q_proj.weight.data.copy_(src_block.attn.query.weight.half())
        dst_block.attn.q_proj.bias.data.copy_(src_block.attn.query.bias.half())
        
        dst_block.attn.k_proj.weight.data.copy_(src_block.attn.key.weight.half())
        dst_block.attn.k_proj.bias.data.copy_(src_block.attn.key.bias.half())
        
        dst_block.attn.v_proj.weight.data.copy_(src_block.attn.value.weight.half())
        dst_block.attn.v_proj.bias.data.copy_(src_block.attn.value.bias.half())
        
        dst_block.attn.out_proj.weight.data.copy_(src_block.attn.out.weight.half())
        dst_block.attn.out_proj.bias.data.copy_(src_block.attn.out.bias.half())
        
        # Layer Norm
        dst_block.attn_ln.weight.data.copy_(src_block.attn_ln.weight.half())
        dst_block.attn_ln.bias.data.copy_(src_block.attn_ln.bias.half())
        
        # MLP
        dst_block.mlp[0].weight.data.copy_(src_block.mlp[0].weight.half())
        dst_block.mlp[0].bias.data.copy_(src_block.mlp[0].bias.half())
        
        dst_block.mlp[2].weight.data.copy_(src_block.mlp[2].weight.half())
        dst_block.mlp[2].bias.data.copy_(src_block.mlp[2].bias.half())
        
        dst_block.mlp_ln.weight.data.copy_(src_block.mlp_ln.weight.half())
        dst_block.mlp_ln.bias.data.copy_(src_block.mlp_ln.bias.half())
    
    # Encoder: output layer norm
    cuda_model.encoder.ln_post.weight.data.copy_(original_model.encoder.ln_post.weight.half())
    cuda_model.encoder.ln_post.bias.data.copy_(original_model.encoder.ln_post.bias.half())
    
    # Decoder: token embedding
    cuda_model.decoder.token_embedding.weight.data.copy_(original_model.decoder.token_embedding.weight.data)
    cuda_model.decoder.positional_embedding.data.copy_(original_model.decoder.positional_embedding.data.half())
    
    # Decoder: transformer blocks
    for i, (src_block, dst_block) in enumerate(zip(original_model.decoder.blocks, cuda_model.decoder.blocks)):
        # Self-attention
        dst_block.attn.q_proj.weight.data.copy_(src_block.attn.query.weight.half())
        dst_block.attn.q_proj.bias.data.copy_(src_block.attn.query.bias.half())
        
        dst_block.attn.k_proj.weight.data.copy_(src_block.attn.key.weight.half())
        dst_block.attn.k_proj.bias.data.copy_(src_block.attn.key.bias.half())
        
        dst_block.attn.v_proj.weight.data.copy_(src_block.attn.value.weight.half())
        dst_block.attn.v_proj.bias.data.copy_(src_block.attn.value.bias.half())
        
        dst_block.attn.out_proj.weight.data.copy_(src_block.attn.out.weight.half())
        dst_block.attn.out_proj.bias.data.copy_(src_block.attn.out.bias.half())
        
        # Self-attention layer norm
        dst_block.attn_ln.weight.data.copy_(src_block.attn_ln.weight.half())
        dst_block.attn_ln.bias.data.copy_(src_block.attn_ln.bias.half())
        
        # Cross-attention
        dst_block.cross_attn.q_proj.weight.data.copy_(src_block.cross_attn.query.weight.half())
        dst_block.cross_attn.q_proj.bias.data.copy_(src_block.cross_attn.query.bias.half())
        
        dst_block.cross_attn.k_proj.weight.data.copy_(src_block.cross_attn.key.weight.half())
        dst_block.cross_attn.k_proj.bias.data.copy_(src_block.cross_attn.key.bias.half())
        
        dst_block.cross_attn.v_proj.weight.data.copy_(src_block.cross_attn.value.weight.half())
        dst_block.cross_attn.v_proj.bias.data.copy_(src_block.cross_attn.value.bias.half())
        
        dst_block.cross_attn.out_proj.weight.data.copy_(src_block.cross_attn.out.weight.half())
        dst_block.cross_attn.out_proj.bias.data.copy_(src_block.cross_attn.out.bias.half())
        
        # Cross-attention layer norm
        dst_block.cross_attn_ln.weight.data.copy_(src_block.cross_attn_ln.weight.half())
        dst_block.cross_attn_ln.bias.data.copy_(src_block.cross_attn_ln.bias.half())
        
        # MLP
        dst_block.mlp[0].weight.data.copy_(src_block.mlp[0].weight.half())
        dst_block.mlp[0].bias.data.copy_(src_block.mlp[0].bias.half())
        
        dst_block.mlp[2].weight.data.copy_(src_block.mlp[2].weight.half())
        dst_block.mlp[2].bias.data.copy_(src_block.mlp[2].bias.half())
        
        dst_block.mlp_ln.weight.data.copy_(src_block.mlp_ln.weight.half())
        dst_block.mlp_ln.bias.data.copy_(src_block.mlp_ln.bias.half())
    
    # Decoder: final layer norm
    cuda_model.decoder.ln.weight.data.copy_(original_model.decoder.ln.weight.half())
    cuda_model.decoder.ln.bias.data.copy_(original_model.decoder.ln.bias.half())
    
    # Alignment heads
    if hasattr(original_model, "alignment_heads"):
        cuda_model.alignment_heads = original_model.alignment_heads
    
    return cuda_model


if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Test Whisper CUDA model")
    parser.add_argument("--model", type=str, default="tiny", help="Whisper model size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=3000, help="Audio sequence length")
    args = parser.parse_args()
    
    # Load standard Whisper model
    standard_model = whisper.load_model(args.model).cuda().half()
    
    # Convert to CUDA model
    cuda_model = convert_from_original_whisper(standard_model).cuda()
    
    # Create random input
    mel = torch.randn(args.batch_size, standard_model.dims.n_mels, args.seq_len, dtype=torch.float16, device="cuda")
    tokens = torch.ones(args.batch_size, 1, dtype=torch.long, device="cuda")
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = standard_model(mel, tokens)
            _ = cuda_model(mel, tokens)
    
    torch.cuda.synchronize()
    
    # Benchmark standard model
    start_time = time.time()
    with torch.no_grad():
        standard_output = standard_model(mel, tokens)
    torch.cuda.synchronize()
    standard_time = time.time() - start_time
    
    # Benchmark CUDA model
    start_time = time.time()
    with torch.no_grad():
        cuda_output = cuda_model(mel, tokens)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    
    # Compare outputs
    max_diff = torch.max(torch.abs(standard_output - cuda_output)).item()
    
    # Print results
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Standard model time: {standard_time:.4f} seconds")
    print(f"CUDA model time: {cuda_time:.4f} seconds")
    print(f"Speedup: {standard_time / cuda_time:.2f}x")
    print(f"Max difference: {max_diff:.6f}")
    
    if max_diff < 0.1:
        print("✓ Test PASSED!")
    else:
        print("✗ Test FAILED! Outputs differ significantly.") 