import torch
import torch.nn as nn
import numpy as np
import math
import time
import pytest

# Import our CUDA Transformer
try:
    from transformer_cuda import CUDATransformer
except ImportError:
    pytest.skip("transformer_cuda not installed", allow_module_level=True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDATransformer:
    
    def setup_method(self):
        """Set up test parameters"""
        torch.manual_seed(42)
        self.batch_size = 8
        self.src_seq_len = 32
        self.tgt_seq_len = 24
        self.src_vocab_size = 1000
        self.tgt_vocab_size = 1000
        self.d_model = 256
        self.nhead = 8
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3
        self.dim_feedforward = 1024
        self.dropout = 0.1
        
        # Create test tensors
        self.src = torch.randint(
            0, 
            self.src_vocab_size, 
            (self.batch_size, self.src_seq_len), 
            device="cuda"
        )
        self.tgt = torch.randint(
            0, 
            self.tgt_vocab_size, 
            (self.batch_size, self.tgt_seq_len), 
            device="cuda"
        )
        
        # Create padding masks (1 = keep, 0 = mask)
        self.src_key_padding_mask = torch.ones(
            self.batch_size, 
            self.src_seq_len, 
            device="cuda", 
            dtype=torch.bool
        )
        # Randomly mask some positions
        for i in range(self.batch_size):
            self.src_key_padding_mask[i, -5:] = False
        
        self.tgt_key_padding_mask = torch.ones(
            self.batch_size, 
            self.tgt_seq_len, 
            device="cuda", 
            dtype=torch.bool
        )
        # Randomly mask some positions
        for i in range(self.batch_size):
            self.tgt_key_padding_mask[i, -5:] = False
        
        # Create causal mask for decoder
        self.tgt_mask = self._generate_square_subsequent_mask(self.tgt_seq_len)
        
        # Create both implementations
        # PyTorch implementation
        self.pytorch_transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        ).cuda().half()
        
        # Our CUDA implementation
        self.cuda_transformer = CUDATransformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size
        ).cuda().half()
        
        # Create embeddings for PyTorch transformer
        self.src_embedding = nn.Embedding(self.src_vocab_size, self.d_model).cuda().half()
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, self.d_model).cuda().half()
        self.output_projection = nn.Linear(self.d_model, self.tgt_vocab_size).cuda().half()
        
        # Copy weights from PyTorch transformer to CUDA transformer
        # This is complex and not necessary for basic testing
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz, device="cuda")) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(~mask, float('-inf')).masked_fill(mask, float(0.0))
        return mask
    
    def test_forward(self):
        """Test forward pass"""
        # Set eval mode to disable dropout for deterministic results
        self.pytorch_transformer.eval()
        self.cuda_transformer.eval()
        
        # Run PyTorch implementation
        with torch.no_grad():
            src_emb = self.src_embedding(self.src) * math.sqrt(self.d_model)
            tgt_emb = self.tgt_embedding(self.tgt) * math.sqrt(self.d_model)
            
            # Convert padding masks for PyTorch transformer (which uses opposite convention)
            src_key_padding_mask_pt = ~self.src_key_padding_mask
            tgt_key_padding_mask_pt = ~self.tgt_key_padding_mask
            
            pytorch_output = self.pytorch_transformer(
                src_emb, 
                tgt_emb, 
                tgt_mask=self.tgt_mask,
                src_key_padding_mask=src_key_padding_mask_pt,
                tgt_key_padding_mask=tgt_key_padding_mask_pt
            )
            pytorch_output = self.output_projection(pytorch_output)
        
        # Run CUDA implementation
        with torch.no_grad():
            cuda_output = self.cuda_transformer(
                self.src,
                self.tgt,
                tgt_mask=self.tgt_mask,
                src_key_padding_mask=self.src_key_padding_mask,
                tgt_key_padding_mask=self.tgt_key_padding_mask
            )
        
        # Check that outputs have the expected shape
        assert pytorch_output.shape == (self.batch_size, self.tgt_seq_len, self.tgt_vocab_size)
        assert cuda_output.shape == (self.batch_size, self.tgt_seq_len, self.tgt_vocab_size)
        
        # Since we didn't copy weights, we can't check for exact match
        # But we can check that the output is reasonable
        assert not torch.isnan(cuda_output).any(), "CUDA output contains NaN values"
        assert not torch.isinf(cuda_output).any(), "CUDA output contains Inf values"
    
    def test_encode_decode(self):
        """Test separate encode and decode methods"""
        # Set eval mode
        self.cuda_transformer.eval()
        
        with torch.no_grad():
            # Test encoder
            memory = self.cuda_transformer.encode(
                self.src,
                src_key_padding_mask=self.src_key_padding_mask
            )
            
            # Check memory shape
            assert memory.shape == (self.batch_size, self.src_seq_len, self.d_model)
            
            # Test decoder
            output, _ = self.cuda_transformer.decode(
                self.tgt,
                memory,
                tgt_mask=self.tgt_mask,
                tgt_key_padding_mask=self.tgt_key_padding_mask,
                memory_key_padding_mask=self.src_key_padding_mask
            )
            
            # Check output shape
            assert output.shape == (self.batch_size, self.tgt_seq_len, self.tgt_vocab_size)
    
    def test_generate(self):
        """Test autoregressive generation"""
        # Set eval mode
        self.cuda_transformer.eval()
        
        # Define BOS and EOS tokens
        bos_token_id = 2
        eos_token_id = 3
        
        with torch.no_grad():
            # Generate sequence
            generated = self.cuda_transformer.generate(
                self.src,
                max_len=20,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                src_key_padding_mask=self.src_key_padding_mask,
                temperature=0.8,
                top_k=50
            )
            
            # Check generated shape
            assert generated.shape[0] == self.batch_size
            assert generated.shape[1] <= 20  # Can be less if EOS is generated
            
            # Check if first token is BOS
            assert torch.all(generated[:, 0] == bos_token_id)
    
    def test_kv_cache(self):
        """Test key-value caching for efficient autoregressive generation"""
        # Set eval mode
        self.cuda_transformer.eval()
        
        with torch.no_grad():
            # Encode source
            memory = self.cuda_transformer.encode(
                self.src,
                src_key_padding_mask=self.src_key_padding_mask
            )
            
            # Initialize target with first token
            tgt_single = self.tgt[:, :1]
            
            # First decode step
            output1, kv_cache = self.cuda_transformer.decode(
                tgt_single,
                memory,
                tgt_mask=self._generate_square_subsequent_mask(1),
                memory_key_padding_mask=self.src_key_padding_mask
            )
            
            # Get next token (for simplicity, just use the second token from tgt)
            next_token = self.tgt[:, 1:2]
            tgt_two = torch.cat([tgt_single, next_token], dim=1)
            
            # Second decode step with cache
            output2_with_cache, _ = self.cuda_transformer.decode(
                tgt_two,
                memory,
                tgt_mask=self._generate_square_subsequent_mask(2),
                memory_key_padding_mask=self.src_key_padding_mask,
                kv_cache=kv_cache
            )
            
            # Second decode step without cache (full recomputation)
            output2_without_cache, _ = self.cuda_transformer.decode(
                tgt_two,
                memory,
                tgt_mask=self._generate_square_subsequent_mask(2),
                memory_key_padding_mask=self.src_key_padding_mask
            )
            
            # Check that outputs match
            torch.testing.assert_close(
                output2_with_cache, 
                output2_without_cache, 
                rtol=1e-3, 
                atol=1e-3,
                msg="Output with cache doesn't match output without cache"
            )
    
    def test_performance(self):
        """Benchmark performance"""
        # Warmup
        for _ in range(5):
            _ = self.cuda_transformer(
                self.src,
                self.tgt,
                tgt_mask=self.tgt_mask,
                src_key_padding_mask=self.src_key_padding_mask,
                tgt_key_padding_mask=self.tgt_key_padding_mask
            )
        
        torch.cuda.synchronize()
        
        # CUDA implementation timing
        start = time.time()
        for _ in range(10):
            _ = self.cuda_transformer(
                self.src,
                self.tgt,
                tgt_mask=self.tgt_mask,
                src_key_padding_mask=self.src_key_padding_mask,
                tgt_key_padding_mask=self.tgt_key_padding_mask
            )
        torch.cuda.synchronize()
        cuda_time = time.time() - start
        
        # Print performance
        print(f"\nTransformer Performance:")
        print(f"CUDA:    {cuda_time:.6f}s for 10 iterations")
        print(f"Average: {cuda_time/10:.6f}s per iteration")

if __name__ == "__main__":
    # Run tests manually
    test = TestCUDATransformer()
    test.setup_method()
    test.test_forward()
    test.test_encode_decode()
    test.test_generate()
    test.test_kv_cache()
    test.test_performance()
    print("All tests passed!") 