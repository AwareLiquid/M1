"""
global_coherence.py — Global Coherence Layer (Orch-OR collapse).

Sparse top-k causal self-attention with a learned collapse gate.
Supports KV cache for streaming inference (`past_kv`).

Output: x + coherence_scale × gate × sparse_attn_out
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MTLNNConfig


class GlobalCoherenceLayer(nn.Module):
    def __init__(self, config: MTLNNConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.coherence_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.coherence_heads
        self.sparsity = config.coherence_sparsity
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.use_decay_wm = getattr(config, "use_decay_wm", False)
        if self.use_decay_wm:
            self.update_gate = nn.Linear(config.d_model, config.d_model)
            wm_decay_init = getattr(config, "wm_decay_rate_init", 0.99)
            self.decay_rate = nn.Parameter(torch.tensor(wm_decay_init))

        # Orch-OR collapse gate parameters
        self.collapse_threshold = nn.Parameter(torch.tensor(0.5))
        self.coherence_scale = nn.Parameter(torch.tensor(0.1))

        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Diagnostic buffer: last forward's collapse-gate activation in [0,1].
        # Not a parameter, not saved in state_dict — purely for monitoring.
        self.register_buffer("last_gate", torch.zeros(()), persistent=False)

    def _sparse_causal_scores(
        self, scores: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        scores: (B, H, T_q, T_k)
        Apply causal mask (using absolute positions) then keep only top-k entries per row.
        """
        # Causal mask: keep entries where k_pos[j] ≤ q_pos[i]
        # dtype-aware fill: a hardcoded -1e9 overflows fp16 (max ±65504) and
        # crashes under torch.autocast('cuda', float16); finfo.min is the same
        # "effectively -inf after softmax" for every dtype.
        neg = torch.finfo(scores.dtype).min
        causal = (k_pos[None, :] <= q_pos[:, None])                    # (T_q, T_k) bool
        scores = scores.masked_fill(~causal[None, None, :, :], neg)

        # Sparse top-k retention
        T_k = scores.shape[-1]
        k = max(1, int(T_k * self.sparsity))
        topk_vals, _ = torch.topk(scores, k=min(k, T_k), dim=-1)
        threshold = topk_vals[..., -1:].detach()
        scores = scores.masked_fill(scores < threshold, neg)
        return scores

    @staticmethod
    def _gate_energy(
        raw: torch.Tensor,                                    # (B, H, T_q, T_k)
        causal: torch.Tensor,                                 # (T_q, T_k) float
        key_mask: Optional[torch.Tensor],                     # (B, T_k) float or None
        query_mask: Optional[torch.Tensor],                   # (B, T_q) float or None
    ) -> torch.Tensor:
        """Per-sample mean attention energy over the valid (causal & non-pad)
        entries. Returns (B,) — one energy per sample, so sample i's collapse
        gate no longer depends on what else happens to share its batch."""
        valid = causal[None, None, :, :]                      # (1,1,T_q,T_k)
        if key_mask is not None:
            valid = valid * key_mask[:, None, None, :]
        if query_mask is not None:
            valid = valid * query_mask[:, None, :, None]
        H = raw.shape[1]
        # fp16 hardening. Two hazards live in this reduction and both are
        # invisible in fp32:
        #   1. `raw` can hold a non-finite entry under autocast; `raw * valid`
        #      then evaluates Inf * 0 = NaN at every masked position, which the
        #      sum spreads to the whole sample. Select-then-zero instead of
        #      multiply so masked entries contribute an exact 0.
        #   2. The sum runs over H*T_q*T_k (~1e6 entries at T=512), which
        #      overflows fp16's 65504 ceiling on the way back to storage.
        #      Accumulate in fp32 explicitly rather than relying on autocast.
        raw32 = raw.float()
        valid32 = valid.float()
        masked = torch.where(valid32 > 0, raw32, torch.zeros_like(raw32))
        energy = masked.sum(dim=(1, 2, 3))                    # (B,) fp32
        # `valid` has a broadcast head dim of size 1 → multiply the count by H.
        count = valid32.expand(raw.shape[0], 1, -1, -1).sum(dim=(1, 2, 3)) * H
        # A literal 1e-9 underflows to exactly 0 in fp16, so the guard it was
        # meant to provide silently disappears; clamp in fp32 instead.
        return energy / count.clamp_min(1e-6)

    def forward(
        self,
        x: torch.Tensor,                                      # (B, T_new, d_model)
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        use_cache: bool = False,
        pad_mask: Optional[torch.Tensor] = None,              # (B, T_total) bool; True = keep
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        B, T_new, _ = x.shape
        H, D = self.n_heads, self.d_head
        device = x.device

        Q = self.q_proj(x).view(B, T_new, H, D).transpose(1, 2)        # (B,H,T_new,D)
        K = self.k_proj(x).view(B, T_new, H, D).transpose(1, 2)
        V = self.v_proj(x).view(B, T_new, H, D).transpose(1, 2)

        # Current-chunk slice of the pad mask (queries). pad_mask covers the
        # full T_total key axis, same convention as MicrotubuleAttention.
        query_pad = pad_mask[:, -T_new:] if pad_mask is not None else None

        if not self.use_decay_wm:
            if past_kv is not None:
                K = torch.cat([past_kv[0], K], dim=2)
                V = torch.cat([past_kv[1], V], dim=2)

            T_total = K.shape[2]
            new_kv = (K, V) if use_cache else None

            scores = (Q / self.scale) @ K.transpose(-2, -1)                # (B,H,T_new,T_total)

            key_pad = pad_mask[:, :T_total] if pad_mask is not None else None
            if key_pad is not None:
                # Mask pad keys BEFORE the sparse top-k so pads cannot occupy
                # top-k slots. None → bit-identical to the original path.
                # (finfo.min, not -1e9: -1e9 overflows fp16 under autocast.)
                scores = scores.masked_fill(
                    ~key_pad[:, None, None, :].bool(), torch.finfo(scores.dtype).min
                )

            q_pos = torch.arange(position_offset, position_offset + T_new, device=device)
            k_pos = torch.arange(0, T_total, device=device)
            scores = self._sparse_causal_scores(scores, q_pos, k_pos)

            # Collapse gate based on raw (pre-sparse) energy mean — per sample.
            with torch.no_grad():
                raw = (Q / self.scale) @ K.transpose(-2, -1)
                causal = (k_pos[None, :] <= q_pos[:, None]).float()
                mean_energy = self._gate_energy(
                    raw, causal,
                    key_pad.float() if key_pad is not None else None,
                    query_pad.float() if query_pad is not None else None,
                )                                                          # (B,)
        else:
            # Working Memory Decay Mode: Constant O(1) space across sequence length.
            # We don't cat K, V over history. We just do self-attention on the current chunk.
            T_total = T_new
            scores = (Q / self.scale) @ K.transpose(-2, -1)

            key_pad = query_pad                                            # keys == current chunk
            if key_pad is not None:
                scores = scores.masked_fill(
                    ~key_pad[:, None, None, :].bool(), torch.finfo(scores.dtype).min
                )

            q_pos = torch.arange(position_offset, position_offset + T_new, device=device)
            k_pos = torch.arange(position_offset, position_offset + T_new, device=device)
            scores = self._sparse_causal_scores(scores, q_pos, k_pos)

            # Collapse gate on local chunk — per sample.
            with torch.no_grad():
                raw = (Q / self.scale) @ K.transpose(-2, -1)
                causal = (k_pos[None, :] <= q_pos[:, None]).float()
                mean_energy = self._gate_energy(
                    raw, causal,
                    key_pad.float() if key_pad is not None else None,
                    query_pad.float() if query_pad is not None else None,
                )                                                          # (B,)
        # Per-sample gate (B,) → broadcast (B,1,1) over (B, T_new, d_model).
        gate = torch.sigmoid((mean_energy - self.collapse_threshold) * 10.0)
        gate = gate.view(B, 1, 1)
        # Stash for diagnostics (scalar mean keeps the buffer shape stable)
        self.last_gate = gate.detach().mean()

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ V                                                  # (B,H,T_new,D)
        # reshape, not contiguous().view() — see mt_attention.py: view() on the
        # transposed tensor blocks torch.export/ONNX (browser deployment path).
        out = out.transpose(1, 2).reshape(B, T_new, self.d_model)
        out = self.out_proj(out)

        if not self.use_decay_wm:
            coherence_out = self.coherence_scale * gate * out
            return self.layer_norm(x + coherence_out), new_kv
        else:
            # Decay Working Memory Logic
            # past_kv actually holds (global_wm, global_wm_unused) 
            # where global_wm is (B, 1, d_model) - rolling running state.
            update_val = torch.sigmoid(self.update_gate(x))             # (B,T_new,d_model)
            
            global_wm = None
            if past_kv is not None:
                past_wm = past_kv[0][:, -1:, :]                         # take the last state (B,1,d_model)
            else:
                past_wm = torch.zeros(B, 1, self.d_model, device=device, dtype=x.dtype)

            # We process T_new chunk iteratively OR approximate chunk-level update.
            # To be sequence-exact during training (T_new > 1), we should unroll sequentially or parallel scan.
            # Since this is an exponential moving average (EMA) gate, it's parallelizable or 
            # we can approximate if T_new is large. For now, since out is (B,T_new,d_model),
            # let's do a fast sequential loop over T dimension for the global_wm. 
            # Usually inference T_new=1, so loop is length 1.
            wm_seq = []
            curr_wm = past_wm.squeeze(1)                                # (B, d_model)
            # decay_rate is an unconstrained raw Parameter (init 0.99). Clamp at
            # use time so the optimizer cannot drift it outside (0, 1) — outside
            # that range the EMA either explodes or flips sign. Clamping (rather
            # than a sigmoid reparam) keeps existing checkpoints loadable.
            decay = self.decay_rate.clamp(1e-4, 1.0 - 1e-4)
            
            for t in range(T_new):
                u_t = update_val[:, t, :]                               # (B, d_model)
                o_t = out[:, t, :]
                curr_wm = curr_wm * decay * (1.0 - u_t) + o_t * u_t
                wm_seq.append(curr_wm)
            
            wm_seq_t = torch.stack(wm_seq, dim=1)                       # (B,T_new,d_model)
            
            coherence_out = self.coherence_scale * gate * wm_seq_t
            
            # Repack the updated WM into the standard cache interface (so tests don't break)
            new_kv = (wm_seq_t, wm_seq_t) if use_cache else None
            return self.layer_norm(x + coherence_out), new_kv
