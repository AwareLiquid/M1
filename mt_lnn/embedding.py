import math
import torch
import torch.nn as nn
from .config import MTLNNConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE). Pre-computes sin/cos tables."""

    def __init__(self, d_head: int, max_seq_len: int):
        super().__init__()
        assert d_head % 2 == 0
        self.d_head = d_head
        # θ_i = 1 / 10000^(2i / d_head)  (kept to lazily extend the tables)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_tables(max_seq_len)

    def _build_tables(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)       # (seq_len, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)     # (seq_len, d_head)
        self.register_buffer("cos_table", emb.cos(), persistent=False)
        self.register_buffer("sin_table", emb.sin(), persistent=False)
        self._table_len = seq_len

    def _maybe_extend(self, needed: int) -> None:
        """Grow the sin/cos tables on demand so decoding past the configured
        max_seq_len extrapolates correctly instead of silently truncating."""
        if needed <= self._table_len:
            return
        # Grow geometrically to amortise the (rare) rebuild cost.
        new_len = max(needed, self._table_len * 2)
        self._build_tables(new_len)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """
        x: (B, n_heads, T, d_head)
        position_offset: absolute starting position of x[:, :, 0, :] in the sequence
                         (used during incremental decoding with KV cache)
        Returns x with RoPE applied at absolute positions [offset, offset+T).
        """
        T = x.shape[2]
        self._maybe_extend(position_offset + T)
        cos = self.cos_table[position_offset: position_offset + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_table[position_offset: position_offset + T].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin


class MTLNNEmbedding(nn.Module):
    """Token embedding + RoPE (shared across all attention layers)."""

    def __init__(self, config: MTLNNConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(config.d_head, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, T)
        Returns: (B, T, d_model)
        """
        x = self.token_embed(input_ids)   # (B, T, d_model)
        return self.dropout(x)
