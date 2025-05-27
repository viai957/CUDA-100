import torch
import torch.nn as nn
import math


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimention of vector seen by each vector
        self.w_q = nn.Linear(d_model, d_model, bias=False) # W_q
        self.w_k = nn.Linear(d_model, d_model, bias=False) # W_k
        self.w_v = nn.Linear(d_model, d_model, bias=False) # W_v
        self.w_o = nn.Linear(d_model, d_model, bias=False) # W_o
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q_proj = self.w_q(q)  # (batch_size, seq_len, d_model)
        k_proj = self.w_k(k)  # (batch_size, seq_len, d_model)
        v_proj = self.w_v(v)  # (batch_size, seq_len, d_model)
        
        # Split into h heads
        q_proj = q_proj.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        k_proj = k_proj.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        v_proj = v_proj.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        
        # Apply attention
        output, _ = self.attention(q_proj, k_proj, v_proj, mask, self.dropout)
        
        # Concatenate heads and pass through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        return self.w_o(output)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores
    
