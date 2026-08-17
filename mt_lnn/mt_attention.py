"""
mt_attention.py — Microtubule Attention with GQA + KV cache + SDPA backend.

Two microtubule-inspired modifications layered on top of GQA:

  1. Polarity bias  — each Q-head has a learned signed scalar polarity[h] ∈ [-1,1].
                       Bias[h, i_abs, j_abs] = polarity[h] * (j_abs - i_abs) / max_seq_len.
                       Computed in absolute positions so it is cache-consistent.

  2. GTP-cap log-bias — a learned per-head decay γ_h is folded in as an *additive*
                       log-bias to attention logits: -γ_h × (i_abs - j_abs).
                       This is mathematically equivalent to multiplying softmax
                       inputs by exp(-γ_h × distance) (the GTP-cap factor) but
                       composes cleanly with SDPA's attn_mask interface.

Implementation notes:
  - Uses torch.nn.functional.scaled_dot_product_attention (SDPA) so PyTorch can
    pick the Flash-Attention / memory-efficient kernel when available.
  - Causal masking is also expressed as -inf entries in the additive mask, so a
    single tensor carries causal + polarity + GTP + padding constraints.
  - Q has full head count; K/V have n_kv_heads → GQA-shrunk KV cache.

KV-cache contract:
  forward(x, past_kv=None, position_offset=0, use_cache=False)
    → returns (out, new_kv) where new_kv = (K_total, V_total).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MTLNNConfig
from .embedding import RotaryEmbedding


def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA helper: repeat each KV head n_rep times along the head dim."""
    if n_rep == 1:
        return kv
    B, H_kv, T, D = kv.shape
    return kv[:, :, None, :, :].expand(B, H_kv, n_rep, T, D).reshape(B, H_kv * n_rep, T, D)


class MicrotubuleAttention(nn.Module):
    def __init__(self, config: MTLNNConfig, rope: RotaryEmbedding):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.d_model, config.n_heads    * config.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        # MT parameters (per Q-head)
        self.polarity_direction = nn.Parameter(torch.zeros(config.n_heads))

        # Optional low-rank bilinear polarity bias.
        # bias = σ((x W_A) (x W_B)^T) — a content-aware, T×T mask that
        # generalises across sequence lengths (no fixed T parameter). Models
        # α/β-tubulin dimer pair interactions.
        self.polarity_mode = config.polarity_mode
        if config.polarity_mode == "low_rank":
            r = config.polarity_rank
            self.pol_W_A = nn.Linear(config.d_model, r, bias=False)
            self.pol_W_B = nn.Linear(config.d_model, r, bias=False)
            # Per-head mixing weight in [0, 1]: learns how much of the bilinear
            # mask to apply, vs. the scalar polarity bias. Init so bilinear is
            # initially disabled (sigmoid(-3) ≈ 0.05) — model can rely on the
            # scalar bias first.
            self.pol_bilinear_gate = nn.Parameter(torch.full((config.n_heads,), -3.0))

        # gtp_gamma stored in raw space; γ = softplus(gtp_gamma) > 0.
        # **ALiBi-style multi-scale init**: heads span a geometric sequence so
        # different heads see different effective receptive fields — some focus
        # locally (large γ), others span far context (small γ).
        # Geometric ratio chosen so head 0 has γ ≈ gamma_init * 8 (strongly local)
        # and head (H-1) has γ ≈ gamma_init / 8 (effectively global).
        gamma_targets = self._build_alibi_gamma(
            config.n_heads, config.gamma_init,
            n_global_heads=getattr(config, "n_global_heads", 0),
        )
        raw_init = torch.log(torch.expm1(gamma_targets.clamp(min=1e-4)))
        self.gtp_gamma = nn.Parameter(raw_init)

        self.rope = rope
        self.resid_dropout = nn.Dropout(config.dropout)

        # Position-free timing signal extractor (only used when use_position_free_attention=True)
        # Extracts timing information from h_prev (B, P, S, D) using tau-weighted aggregation.
        # WARNING: Must NOT use simple averaging - different τ scales have different importance.
        if config.use_position_free_attention:
            # Learnable tau weights: each time scale gets its own importance weight
            # Initialize based on time scales: faster scales (small τ) get higher initial weight
            # because short-term patterns are typically more important for position encoding.
            tau_freqs = torch.tensor(config.resonance_freqs, dtype=torch.float32)
            # Init as 1/sqrt(τ) so fast scales (τ=0.01) have ~10× weight of slow scales (τ=10)
            tau_init = 1.0 / torch.sqrt(tau_freqs)
            self.tau_weights = nn.Parameter(tau_init)

            # Project multi-scale h_prev to position signal
            # Input: (B, P*D) after tau-weighted pooling over S
            # Output: (B, d_model) timing signal to add to embeddings
            self.h_prev_position_proj = nn.Linear(
                config.n_protofilaments * config.d_proto,
                config.d_model,
                bias=False
            )

        # ------------------------------------------------------------------
        # Precomputed distance matrices (saved as buffers, not parameters).
        # _delta[i, j]   = i - j           (positive for past keys)
        # _causal[i, j]  = True            iff j ≤ i
        # We slice these by [position_offset:position_offset+T_new, :T_total]
        # in the forward path so no fresh arange / tensor allocations occur.
        # ------------------------------------------------------------------
        self._build_pos_buffers(config.max_seq_len)

    def _build_pos_buffers(self, seq_len: int) -> None:
        dev = self._delta.device if hasattr(self, "_delta") else None
        idx = torch.arange(seq_len, device=dev)
        delta = idx[:, None] - idx[None, :]                  # (L, L) signed int
        self.register_buffer("_delta", delta.float(), persistent=False)
        self.register_buffer("_causal", (delta >= 0), persistent=False)
        self._pos_len = seq_len

    def _maybe_extend_pos(self, needed: int) -> None:
        """Grow the precomputed distance/causal matrices on demand so decoding
        past max_seq_len stays correct instead of silently truncating the bias."""
        if needed <= self._pos_len:
            return
        self._build_pos_buffers(max(needed, self._pos_len * 2))

    #: γ init for reserved GLOBAL heads: at distance 100 the penalty is only
    #: 0.1 nats — effectively no decay within any practical context window.
    _GLOBAL_HEAD_GAMMA = 1e-3

    @staticmethod
    def _build_alibi_gamma(n_heads: int, base_gamma: float,
                           n_global_heads: int = 0) -> torch.Tensor:
        """
        Geometric γ schedule: γ_h = base_gamma * 2^(slope_h), where slope_h
        ranges from +3 (head 0 → very local) down to -3 (head H-1 → very global).
        For n_heads = 16 this gives γ ∈ [base/8, base*8] — a 64× spread.

        n_global_heads > 0 reserves that many TRULY global heads (γ≈1e-3,
        no effective distance decay) at the tail; the remaining heads keep the
        geometric biological-decay schedule. Motivation (M2 architecture
        principle #1, docs/ROADMAP_M2.md §4.5): the GTP decay init measurably
        blocked in-context relational lookup — 8-node pointer chase went from
        0.25 (stuck) to 1.00 (solved) once heads could see far. NOTE the P0
        evidence also shows ONE quasi-global head (the old H-1 slope, γ≈0.012
        at 4 heads) was NOT sufficient — induction circuits need several
        far-sighted heads composing across layers — so tune this with the
        `--n_global_heads` sweep in benchmarks/reasoning_depth.py rather than
        assuming 1 is enough. Default 0 keeps the historical init bit-exact.
        """
        n_global_heads = max(0, min(int(n_global_heads), n_heads))
        n_local = n_heads - n_global_heads
        if n_local == 0:
            return torch.full((n_heads,),
                              MicrotubuleAttention._GLOBAL_HEAD_GAMMA)
        if n_local == 1:
            local = torch.tensor([base_gamma])
        else:
            slopes = torch.linspace(3.0, -3.0, n_local)
            local = base_gamma * (2.0 ** slopes)
        if n_global_heads == 0:
            return local
        glob = torch.full((n_global_heads,),
                          MicrotubuleAttention._GLOBAL_HEAD_GAMMA)
        return torch.cat([local, glob])

    def _extract_h_prev_timing(self, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Extract timing signal from h_prev using tau-weighted aggregation.

        h_prev: (B, P, S, D) where P=protofilaments, S=time_scales, D=d_proto
        Returns: (B, d_model) timing signal

        Key insight from warning document:
        - Different τ time scales have different temporal resolution
        - Small τ (fast decay) → recent info, large τ (slow decay) → long-term
        - Must use learned weights, NOT simple averaging
        - Initial weight should be small (0.01) so position signal is weak
        """
        B, P, S, D = h_prev.shape

        # Tau-weighted pooling across time scales
        # tau_weights: (S,) learnable importance per time scale
        weights = F.softmax(self.tau_weights, dim=0)  # Normalize to sum=1
        # Weight each time scale: (B, P, S, D) × (1, 1, S, 1) → (B, P, S, D)
        weighted = h_prev * weights.view(1, 1, S, 1)
        # Sum over time scales: (B, P, D)
        temporal_pooled = weighted.sum(dim=2)

        # Flatten protofilaments: (B, P*D)
        flat = temporal_pooled.reshape(B, -1)

        # Project to d_model: (B, d_model)
        timing_signal = self.h_prev_position_proj(flat)

        # Scale by config weight (must be << 1.0 so content dominates)
        timing_signal = self.config.h_prev_position_weight * timing_signal

        return timing_signal

    # ------------------------------------------------------------------
    # Combined additive attention mask: causal + polarity + GTP + (padding)
    # ------------------------------------------------------------------

    def _build_attn_bias(
        self,
        x_q: torch.Tensor,                    # (B, T_q, d_model) — new tokens only
        x_kv: Optional[torch.Tensor],         # (B, T_k, d_model) or None (use x_q)
        q_start: int,
        k_len: int,
        pad_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Combined SDPA-compatible attention bias:

          bias[h,i,j] = polarity[h] * (j-i)/L                           (scalar polarity)
                     + α_h * σ((x_q W_A)(x_kv W_B)^T)                   (low-rank bilinear, if enabled)
                     - γ[h] * (i-j)                                     (GTP log-decay)
                     +∞-mask-out for j > i (causal) or pad.

        Note: low-rank mode is *content-aware* — depends on x. With KV caching
        we'd need the full (cached + new) x sequence to compute it. For now
        the bilinear bias is computed only over the *new* tokens × new tokens
        block; old positions get scalar polarity only. This is exact when
        prefilling and a mild approximation during incremental decode.
        """
        B, T_q, _ = x_q.shape
        H = self.n_heads
        L = float(self.max_seq_len)

        # Precomputed distance / causal slices (lazily extended past max_seq_len)
        self._maybe_extend_pos(max(q_start + T_q, k_len))
        delta  = self._delta[q_start: q_start + T_q, :k_len]          # (T_q, T_k)
        causal = self._causal[q_start: q_start + T_q, :k_len]         # (T_q, T_k) bool

        # Scalar polarity bias
        pol = self.polarity_direction.clamp(-1.0, 1.0)                # (H,)
        polarity_bias = -pol.view(H, 1, 1) * (delta / L).unsqueeze(0) # (H,T_q,T_k)

        # GTP log-bias
        gamma = F.softplus(self.gtp_gamma).clamp(min=1e-6)            # (H,)
        gtp_log_bias = -gamma.view(H, 1, 1) * delta.clamp(min=0.0).unsqueeze(0)

        bias = (polarity_bias + gtp_log_bias).to(dtype=dtype)         # (H,T_q,T_k)
        bias = bias.unsqueeze(0).expand(B, -1, -1, -1).clone()        # (B,H,T_q,T_k)

        # Low-rank content-aware polarity bias (opt-in)
        if self.polarity_mode == "low_rank":
            xq_A = self.pol_W_A(x_q)                                  # (B,T_q,r)
            xq_B = self.pol_W_B(x_q)                                  # (B,T_q,r)
            # Bilinear: M_qq[b,i,j] = σ(xq_A[b,i] · xq_B[b,j])
            M_qq = torch.sigmoid(xq_A @ xq_B.transpose(-2, -1))       # (B,T_q,T_q)

            # Place M_qq into the rightmost T_q×T_q sub-block of the (T_q,T_k)
            # bias matrix — corresponds to attention among the new tokens only.
            full = torch.zeros(B, T_q, k_len, device=x_q.device, dtype=dtype)
            full[:, :, k_len - T_q:] = M_qq.to(dtype)
            gate = torch.sigmoid(self.pol_bilinear_gate).view(1, H, 1, 1)  # (1,H,1,1)
            bias = bias + gate * full.unsqueeze(1)                    # broadcast over H

        # Causal: invalid positions get -inf
        neg_inf = torch.finfo(dtype).min
        invalid = (~causal).unsqueeze(0).unsqueeze(0)                 # (1,1,T_q,T_k)
        bias = bias.masked_fill(invalid, neg_inf)

        if pad_mask is not None:
            invalid_pad = (~pad_mask)[:, None, None, :]
            bias = bias.masked_fill(invalid_pad, neg_inf)

        return bias

    def _build_position_free_bias(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor],
        k_len: int,
        pad_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Position-free attention bias: computes bias WITHOUT absolute positions.

        Uses:
          1. Content-based polarity (low-rank bilinear)
          2. Relative distance GTP-cap (inferred from cache length)
          3. Causal mask (computed from relative positions)

        No RoPE, no absolute position encodings - position info comes from h_prev.
        """
        B, T_q, _ = x_q.shape
        H = self.n_heads

        # Compute relative distances dynamically (not from precomputed buffers)
        # q_indices: [0, 1, ..., T_q-1] (positions within new query tokens)
        # k_indices: [0, 1, ..., k_len-1] (positions in full KV cache)
        # For causal attention: q can attend to k if k_idx < k_len - T_q + q_idx
        q_offset = k_len - T_q  # Starting position of query tokens in the sequence

        # Relative distances: i - j where i is query pos, j is key pos
        q_pos = torch.arange(T_q, device=x_q.device).float() + q_offset
        k_pos = torch.arange(k_len, device=x_q.device).float()
        delta = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)  # (T_q, k_len)

        # Causal mask: valid when j <= i (key position <= query position)
        causal = (delta >= 0.0)  # (T_q, k_len) bool

        # Content-based polarity bias (always use low-rank in position-free mode)
        if x_kv is None:
            x_kv = x_q

        # Compute bilinear polarity across full KV sequence
        if self.polarity_mode == "low_rank":
            # Query side
            xq_A = self.pol_W_A(x_q)  # (B, T_q, r)
            xq_B = self.pol_W_B(x_q)  # (B, T_q, r)

            # For cached keys, we only have x_q (new tokens)
            # Approximate: apply polarity only to new×new block
            M_qq = torch.sigmoid(xq_A @ xq_B.transpose(-2, -1))  # (B, T_q, T_q)

            # Place in rightmost block
            polarity_bias = torch.zeros(B, T_q, k_len, device=x_q.device, dtype=dtype)
            polarity_bias[:, :, k_len - T_q:] = M_qq.to(dtype)

            gate = torch.sigmoid(self.pol_bilinear_gate).view(1, H, 1, 1)
            polarity_bias = gate * polarity_bias.unsqueeze(1)  # (B, H, T_q, k_len)
        else:
            # Scalar polarity fallback (though position-free should prefer content-based)
            pol = self.polarity_direction.clamp(-1.0, 1.0)
            L = float(self.max_seq_len)
            polarity_bias = -pol.view(1, H, 1, 1) * (delta / L).unsqueeze(0).unsqueeze(0).to(dtype)
            polarity_bias = polarity_bias.expand(B, -1, -1, -1)

        # GTP-cap relative distance bias (KEEP THIS - essential for causal structure)
        if self.config.keep_relative_bias:
            gamma = F.softplus(self.gtp_gamma).clamp(min=1e-6)  # (H,)
            gtp_log_bias = -gamma.view(1, H, 1, 1) * delta.clamp(min=0.0).unsqueeze(0).unsqueeze(0).to(dtype)
            bias = polarity_bias + gtp_log_bias
        else:
            bias = polarity_bias

        # Apply causal mask
        neg_inf = torch.finfo(dtype).min
        invalid = (~causal).unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, k_len)
        bias = bias.masked_fill(invalid, neg_inf)

        # Apply padding mask if provided
        if pad_mask is not None:
            invalid_pad = (~pad_mask)[:, None, None, :]
            bias = bias.masked_fill(invalid_pad, neg_inf)

        return bias

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                              # (B, T_new, d_model)
        pad_mask: Optional[torch.Tensor] = None,      # (B, T_total) bool
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        use_cache: bool = False,
        h_prev: Optional[torch.Tensor] = None,        # (B, P, S, D) for position-free mode
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        B, T_new, _ = x.shape
        H_q, H_kv, D = self.n_heads, self.n_kv_heads, self.d_head

        # Position-free mode: inject timing from h_prev into embeddings
        if self.config.use_position_free_attention and h_prev is not None:
            # Extract tau-weighted timing signal: (B, d_model)
            timing_signal = self._extract_h_prev_timing(h_prev)
            # Broadcast to all tokens: (B, 1, d_model) → (B, T_new, d_model)
            x = x + timing_signal.unsqueeze(1)

        # Project Q (n_heads), K/V (n_kv_heads) — GQA
        Q = self.q_proj(x).view(B, T_new, H_q,  D).transpose(1, 2)   # (B,H_q,T_new,D)
        K = self.k_proj(x).view(B, T_new, H_kv, D).transpose(1, 2)
        V = self.v_proj(x).view(B, T_new, H_kv, D).transpose(1, 2)

        # RoPE at absolute positions (ORIGINAL PATH ONLY)
        if not self.config.use_position_free_attention:
            Q = self.rope(Q, position_offset=position_offset)
            K = self.rope(K, position_offset=position_offset)
        # else: Position-free mode - skip RoPE, timing comes from h_prev

        # Concatenate KV cache
        if past_kv is not None:
            K_total = torch.cat([past_kv[0], K], dim=2)
            V_total = torch.cat([past_kv[1], V], dim=2)
        else:
            K_total, V_total = K, V

        T_total = K_total.shape[2]
        new_kv = (K_total, V_total) if use_cache else None

        # GQA: replicate K/V heads for the matmul
        K_rep = repeat_kv(K_total, self.n_rep)                        # (B,H_q,T_total,D)
        V_rep = repeat_kv(V_total, self.n_rep)

        # Build combined attention bias (dual-path: original vs position-free)
        if self.config.use_position_free_attention:
            # NEW PATH: Position-free bias without absolute positions
            attn_bias = self._build_position_free_bias(
                x_q=x, x_kv=None,
                k_len=T_total,
                pad_mask=pad_mask, dtype=Q.dtype,
            )
        else:
            # ORIGINAL PATH: Full position-aware bias with RoPE
            attn_bias = self._build_attn_bias(
                x_q=x, x_kv=None,
                q_start=position_offset, k_len=T_total,
                pad_mask=pad_mask, dtype=Q.dtype,
            )

        # Flash-Attention / memory-efficient SDPA
        out = F.scaled_dot_product_attention(
            Q, K_rep, V_rep,
            attn_mask=attn_bias,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,                     # causal already encoded in attn_bias
        )                                         # (B,H_q,T_new,D)

        # reshape, not contiguous().view(): identical semantics, but view() on
        # the transposed tensor fails under torch.export (the tracer sees the
        # non-contiguous strides and refuses), which blocks ONNX export and
        # therefore the browser/WebGPU deployment path.
        out = out.transpose(1, 2).reshape(B, T_new, H_q * D)
        return self.resid_dropout(self.out_proj(out)), new_kv
