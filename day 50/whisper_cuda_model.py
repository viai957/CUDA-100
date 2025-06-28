import torch
import torch.nn as nn
from typing import Dict, Iterable, Optional, Tuple
import whisper
from whisper.model import ModelDimensions, Whisper, AudioEncoder, TextDecoder, ResidualAttentionBlock

from whisper_cuda import CUDALayerNorm, CUDALinear, CUDAGELU, CUDAMultiHeadAttention, Conv1d

class CUDAResidualAttentionBlock(nn.Module):
    """
    A ResidualAttentionBlock using our custom CUDA implementations.
    """
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        
        # Use our custom CUDA implementations
        self.attn = CUDAMultiHeadAttention(n_state, n_head, head_dim=n_state // n_head)
        self.attn_ln = CUDALayerNorm(n_state)
        
        self.cross_attn = (
            CUDAMultiHeadAttention(n_state, n_head, head_dim=n_state // n_head) if cross_attention else None
        )
        self.cross_attn_ln = CUDALayerNorm(n_state) if cross_attention else None
        
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
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x

class CUDAAudioEncoder(nn.Module):
    """
    An AudioEncoder using our custom CUDA implementations.
    """
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        # Use our custom Conv1d implementation instead of nn.Conv1d
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", whisper.model.sinusoids(n_ctx, n_state))
        
        # Use our custom CUDA implementations for the blocks
        self.blocks: Iterable[CUDAResidualAttentionBlock] = nn.ModuleList(
            [CUDAResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = CUDALayerNorm(n_state)
        
        # GELU activation for the convolutional layers
        self.gelu = CUDAGELU()
    
    def forward(self, x: torch.Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        # No need to manually apply GELU as it's handled in our Conv1d forward
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_post(x)
        return x

class CUDATextDecoder(nn.Module):
    """
    A TextDecoder using our custom CUDA implementations.
    """
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        
        # Use our custom CUDA implementations for the blocks
        self.blocks: Iterable[CUDAResidualAttentionBlock] = nn.ModuleList(
            [
                CUDAResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = CUDALayerNorm(n_state)
        
        mask = torch.empty(n_ctx, n_ctx).fill_(-torch.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
    
    def forward(self, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)
        
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        
        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        
        return logits

class CUDAWhisper(nn.Module):
    """
    A Whisper model using our custom CUDA implementations.
    """
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
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
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)
    
    def set_alignment_heads(self, dump: bytes):
        """Set alignment heads from a dump"""
        self.register_buffer("alignment_heads", whisper.model.Whisper.set_alignment_heads(self, dump), persistent=False)
    
    def embed_audio(self, mel: torch.Tensor):
        """Embed audio using the encoder"""
        return self.encoder(mel)
    
    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        """Get logits from tokens and audio features"""
        return self.decoder(tokens, audio_features)
    
    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        return self.decoder(tokens, self.encoder(mel))
    
    @property
    def device(self):
        """Get device"""
        return next(self.parameters()).device
    
    @property
    def is_multilingual(self):
        """Check if model is multilingual"""
        return self.dims.n_vocab >= 51865
    
    @property
    def num_languages(self):
        """Get number of languages"""
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)
    
    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """Install key-value cache hooks for efficient decoding"""
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
                hooks.append(layer.v_proj.register_forward_hook(save_to_cache))
        
        self.decoder.apply(install_hooks)
        return cache, hooks

def convert_whisper_to_cuda(model: Whisper) -> CUDAWhisper:
    """
    Convert a standard Whisper model to use our custom CUDA implementations.
    
    Args:
        model: A standard Whisper model
        
    Returns:
        A CUDAWhisper model with the same weights
    """
    # Create a new CUDAWhisper model with the same dimensions
    cuda_model = CUDAWhisper(model.dims)
    
    # Copy weights from the standard model to our CUDA model
    # This is a simplified version, actual implementation may need more careful weight copying
    
    # Copy encoder weights
    # Conv layers - using from_pytorch for our custom Conv1d
    cuda_model.encoder.conv1 = Conv1d.from_pytorch(model.encoder.conv1)
    cuda_model.encoder.conv2 = Conv1d.from_pytorch(model.encoder.conv2)
    
    # Positional embedding
    cuda_model.encoder.positional_embedding.copy_(model.encoder.positional_embedding)
    
    # Encoder blocks
    for i, (src_block, dst_block) in enumerate(zip(model.encoder.blocks, cuda_model.encoder.blocks)):
        # Self-attention
        dst_block.attn.q_proj.weight.data.copy_(src_block.attn.query.weight.data)
        dst_block.attn.k_proj.weight.data.copy_(src_block.attn.key.weight.data)
        dst_block.attn.v_proj.weight.data.copy_(src_block.attn.value.weight.data)
        dst_block.attn.out_proj.weight.data.copy_(src_block.attn.out.weight.data)
        
        dst_block.attn.q_proj.bias.data.copy_(src_block.attn.query.bias.data)
        dst_block.attn.k_proj.bias.data.copy_(src_block.attn.key.bias.data)
        dst_block.attn.v_proj.bias.data.copy_(src_block.attn.value.bias.data)
        dst_block.attn.out_proj.bias.data.copy_(src_block.attn.out.bias.data)
        
        # Layer norms
        dst_block.attn_ln.weight.data.copy_(src_block.attn_ln.weight.data)
        dst_block.attn_ln.bias.data.copy_(src_block.attn_ln.bias.data)
        dst_block.mlp_ln.weight.data.copy_(src_block.mlp_ln.weight.data)
        dst_block.mlp_ln.bias.data.copy_(src_block.mlp_ln.bias.data)
        
        # MLP
        dst_block.mlp[0].weight.data.copy_(src_block.mlp[0].weight.data)
        dst_block.mlp[0].bias.data.copy_(src_block.mlp[0].bias.data)
        dst_block.mlp[2].weight.data.copy_(src_block.mlp[2].weight.data)
        dst_block.mlp[2].bias.data.copy_(src_block.mlp[2].bias.data)
    
    # Encoder final layer norm
    cuda_model.encoder.ln_post.weight.data.copy_(model.encoder.ln_post.weight.data)
    cuda_model.encoder.ln_post.bias.data.copy_(model.encoder.ln_post.bias.data)
    
    # Copy decoder weights
    # Token and positional embeddings
    cuda_model.decoder.token_embedding.weight.data.copy_(model.decoder.token_embedding.weight.data)
    cuda_model.decoder.positional_embedding.data.copy_(model.decoder.positional_embedding.data)
    
    # Decoder blocks
    for i, (src_block, dst_block) in enumerate(zip(model.decoder.blocks, cuda_model.decoder.blocks)):
        # Self-attention
        dst_block.attn.q_proj.weight.data.copy_(src_block.attn.query.weight.data)
        dst_block.attn.k_proj.weight.data.copy_(src_block.attn.key.weight.data)
        dst_block.attn.v_proj.weight.data.copy_(src_block.attn.value.weight.data)
        dst_block.attn.out_proj.weight.data.copy_(src_block.attn.out.weight.data)
        
        dst_block.attn.q_proj.bias.data.copy_(src_block.attn.query.bias.data)
        dst_block.attn.k_proj.bias.data.copy_(src_block.attn.key.bias.data)
        dst_block.attn.v_proj.bias.data.copy_(src_block.attn.value.bias.data)
        dst_block.attn.out_proj.bias.data.copy_(src_block.attn.out.bias.data)
        
        # Cross-attention
        dst_block.cross_attn.q_proj.weight.data.copy_(src_block.cross_attn.query.weight.data)
        dst_block.cross_attn.k_proj.weight.data.copy_(src_block.cross_attn.key.weight.data)
        dst_block.cross_attn.v_proj.weight.data.copy_(src_block.cross_attn.value.weight.data)
        dst_block.cross_attn.out_proj.weight.data.copy_(src_block.cross_attn.out.weight.data)
        
        dst_block.cross_attn.q_proj.bias.data.copy_(src_block.cross_attn.query.bias.data)
        dst_block.cross_attn.k_proj.bias.data.copy_(src_block.cross_attn.key.bias.data)
        dst_block.cross_attn.v_proj.bias.data.copy_(src_block.cross_attn.value.bias.data)
        dst_block.cross_attn.out_proj.bias.data.copy_(src_block.cross_attn.out.bias.data)
        
        # Layer norms
        dst_block.attn_ln.weight.data.copy_(src_block.attn_ln.weight.data)
        dst_block.attn_ln.bias.data.copy_(src_block.attn_ln.bias.data)
        dst_block.cross_attn_ln.weight.data.copy_(src_block.cross_attn_ln.weight.data)
        dst_block.cross_attn_ln.bias.data.copy_(src_block.cross_attn_ln.bias.data)
        dst_block.mlp_ln.weight.data.copy_(src_block.mlp_ln.weight.data)
        dst_block.mlp_ln.bias.data.copy_(src_block.mlp_ln.bias.data)
        
        # MLP
        dst_block.mlp[0].weight.data.copy_(src_block.mlp[0].weight.data)
        dst_block.mlp[0].bias.data.copy_(src_block.mlp[0].bias.data)
        dst_block.mlp[2].weight.data.copy_(src_block.mlp[2].weight.data)
        dst_block.mlp[2].bias.data.copy_(src_block.mlp[2].bias.data)
    
    # Decoder final layer norm
    cuda_model.decoder.ln.weight.data.copy_(model.decoder.ln.weight.data)
    cuda_model.decoder.ln.bias.data.copy_(model.decoder.ln.bias.data)
    
    # Copy alignment heads
    cuda_model.alignment_heads = model.alignment_heads
    
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
    cuda_model = convert_whisper_to_cuda(standard_model).cuda()
    
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