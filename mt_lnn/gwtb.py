"""
gwtb.py — Global Workspace Theory Bottleneck + Competitive extension.

GWTBLayer
---------
Implements the "compress → process → broadcast" pipeline from Baars (1988) /
Dehaene (2011) GWT. Information must squeeze through a narrow d_gw « d_model
bottleneck before it can influence later computation (capacity constraint =
consciousness bottleneck).

Pipeline per token (with KV cache for autoregressive decoding):

  1. Compression (Ignition)
       z_t = LayerNorm(W_compress · x_t)              (B, T, d_gw)

  2. Workspace processing
       z'_t = SelfAttention(z_<=t)                    causal, multi-head over d_gw
     Information from different time steps competes for bottleneck capacity.

  3. Broadcast (Global Ignition)
       Δh_t = W_broadcast · z'_t                      (B, T, d_model)
       out_t = x_t + γ · Δh_t                         gated residual

broadcast_gate γ is initialised small (default 0.01) — the layer starts as
near-identity and gradually learns to ignite globally.

CompetitiveGWTBLayer  (Phase A, 2026-06-06)
-------------------------------------------
Extends GWTBLayer with winner-take-all competition among K specialist bids.

The core GWT claim is that "conscious content" is decided by *competition*:
multiple specialist modules (sensory, memory, reasoning, …) simultaneously
bid for the limited workspace, and only the winner's representation is globally
broadcast. This layer operationalises that claim inside MT-LNN.

Competition pipeline:
  1. K BidProjectors extract K specialised views from x
       bid_i = x + δ_i(x)      (δ initialised to 0 → bid_i ≡ x at init)
  2. ScoreHead rates each bid
       score_i = MLP(bid_i)     → (B, T, 1)
  3. Normalise across K bids
       w_i = softmax(scores)    → (B, T, K)  [soft; hard argmax at inference if configured]
  4. Weighted combination
       x_ws = Σ w_i · bid_i     → (B, T, d_model)   enters the workspace
  5. GWTBLayer workspace SA + broadcast on x_ws
  6. Residual on the ORIGINAL x (not x_ws)
       out = x + γ · broadcast(x_ws)

Residual on original x is intentional:  competition decides what the workspace
*processes*, but the broadcast enriches the *full specialist stream*.

Invariants:
  • At init: all δ_i = 0 → all bids = x → uniform weights → x_ws = x →
    CompetitiveGWTBLayer behaves identically to GWTBLayer.
  • use_competitive_gwtb=False → model uses GWTBLayer, not this class → zero
    regression risk on all existing tests.
  • KV cache contract unchanged: forward(x, past_kv, position_offset, use_cache)
    → (out, new_kv).
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MTLNNConfig


# ---------------------------------------------------------------------------
# GWTBLayer
# ---------------------------------------------------------------------------

class GWTBLayer(nn.Module):
    def __init__(self, config: MTLNNConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_gw = config.d_model // config.gwtb_compression_ratio
        assert self.d_gw % config.gwtb_n_heads == 0, \
            f"gwtb d_gw={self.d_gw} not divisible by gwtb_n_heads={config.gwtb_n_heads}"

        self.n_heads = config.gwtb_n_heads
        self.d_head = self.d_gw // self.n_heads
        self.scale = math.sqrt(self.d_head)
        self.max_seq_len = config.max_seq_len
        # J-Space J1 (docs/JSPACE_DESIGN.md): weight-tied reverberation passes
        # of the workspace self-attention. Runtime attribute (no parameters)
        # so MTLNNModel.set_workspace_iterations can vary it per step.
        self.workspace_iterations = int(getattr(config, "workspace_iterations", 1))

        # 1. Compression projection (d_model → d_gw)
        self.compress = nn.Linear(self.d_model, self.d_gw, bias=False)
        self.compress_norm = nn.LayerNorm(self.d_gw)

        # 2. Workspace multi-head self-attention (causal)
        self.q_proj = nn.Linear(self.d_gw, self.d_gw, bias=False)
        self.k_proj = nn.Linear(self.d_gw, self.d_gw, bias=False)
        self.v_proj = nn.Linear(self.d_gw, self.d_gw, bias=False)
        self.attn_out = nn.Linear(self.d_gw, self.d_gw, bias=False)
        self.workspace_norm = nn.LayerNorm(self.d_gw)

        # 3. Broadcast projection (d_gw → d_model) + gated residual
        self.broadcast = nn.Linear(self.d_gw, self.d_model, bias=False)
        self.broadcast_gate = nn.Parameter(torch.tensor(float(config.gwtb_broadcast_init)))

        self.dropout = nn.Dropout(config.dropout)

        # ----------------------------------------------------------------
        # Dynamic workspace bandwidth (2026-06-15)
        # ----------------------------------------------------------------
        # A per-channel "arousal" gate over the d_gw bottleneck makes the
        # workspace capacity input-dependent instead of a fixed full width.
        # Default OFF → no parameters built → bit-identical to the original
        # fixed-bandwidth pipeline (zero regression). See MTLNNConfig for the
        # neuroscience rationale and the init contract.
        self.dynamic_bandwidth = bool(getattr(config, "gwtb_dynamic_bandwidth", False))
        if self.dynamic_bandwidth:
            # NOTE: implemented as explicit Parameters rather than an nn.Linear on
            # purpose. MTLNNModel runs a global init pass (utils.init_weights) that
            # re-initialises every nn.Linear (weight → N(0, 0.02), bias → 0). That
            # would silently clobber the mostly-open init below. Bare Parameters are
            # not touched by that pass, so the init contract holds uniformly whether
            # the layer is built standalone or inside the full model.
            #
            # g = sigmoid(z @ W_gᵀ + b_g). Zero-init weight → at init the gate is
            # input-INDEPENDENT and equals sigmoid(b_g) on every channel; large
            # positive b_g → "mostly open" → the workspace starts at near-full
            # bandwidth and *learns* to close redundant channels (avoids starting
            # sparse with dead units).
            self.bandwidth_gate_weight = nn.Parameter(torch.zeros(self.d_gw, self.d_gw))
            self.bandwidth_gate_bias = nn.Parameter(
                torch.full(
                    (self.d_gw,),
                    float(getattr(config, "gwtb_bandwidth_gate_bias", 4.0)),
                )
            )
            self.bandwidth_threshold = float(
                getattr(config, "gwtb_bandwidth_threshold", 0.1)
            )
            self.bandwidth_hard_mask = bool(
                getattr(config, "gwtb_bandwidth_hard_mask", False)
            )
            # Diagnostics (non-persistent): mean gate value and the effective
            # active bandwidth (fraction of channels above threshold) from the
            # last forward. Monitoring only — never feeds back into compute.
            self.register_buffer(
                "last_bandwidth_gate_mean", torch.ones(()), persistent=False
            )
            self.register_buffer(
                "last_active_bandwidth", torch.ones(()), persistent=False
            )
            # Neuromodulatory hook (acetylcholine → workspace bandwidth). An
            # external NeuromodulationController may push a scalar offset onto the
            # gate bias to *widen* (positive) or *narrow* (negative) how many
            # bottleneck channels ignite, mirroring how ACh tracks expected
            # uncertainty and broadens cortical sampling under volatile input.
            # Non-persistent, zero at init → the default path is bit-identical to
            # the un-modulated gate (zero regression). Never trained; set only via
            # set_bandwidth_bias_offset(), so it survives the global init pass.
            self.register_buffer(
                "_bandwidth_bias_offset", torch.zeros(()), persistent=False
            )

        self._build_causal(config.max_seq_len)

    def set_bandwidth_bias_offset(self, offset: float) -> None:
        """Push a neuromodulatory offset onto the dynamic-bandwidth gate bias.

        Called by an external controller (e.g. the acetylcholine gate of
        :class:`mt_lnn.neuromodulation.NeuromodulationController`). Positive
        offset widens the ignition bandwidth (more channels fire), negative
        narrows it. No-op unless dynamic bandwidth is enabled. Resetting to 0.0
        restores the exact un-modulated behaviour, keeping the feature rollback-
        able and zero-regression.
        """
        if not self.dynamic_bandwidth:
            return
        self._bandwidth_bias_offset = torch.as_tensor(
            float(offset),
            dtype=self.bandwidth_gate_bias.dtype,
            device=self.bandwidth_gate_bias.device,
        )

    # ------------------------------------------------------------------
    # Dynamic workspace bandwidth gate
    # ------------------------------------------------------------------

    def _apply_bandwidth_gate(self, z: torch.Tensor) -> torch.Tensor:
        """Gate the compressed bottleneck z by an input-dependent per-channel
        arousal signal. No-op (returns z unchanged) when dynamic bandwidth is
        disabled, so the default path is bit-identical to fixed bandwidth.

        z : (B, T, d_gw) — pre-compressed workspace activations.
        returns z_gated : (B, T, d_gw).
        """
        if not self.dynamic_bandwidth:
            return z

        # g ∈ (0,1)^d_gw — how strongly each bottleneck channel ignites for this
        # token. Input-dependent once W_g has learned (zero at init → constant).
        # The acetylcholine offset (0.0 unless an external controller set it)
        # shifts every channel's bias up/down together: high ACh → wider workspace.
        bias = self.bandwidth_gate_bias + self._bandwidth_bias_offset
        g = torch.sigmoid(F.linear(z, self.bandwidth_gate_weight, bias))

        if (not self.training) and self.bandwidth_hard_mask:
            # Inference compute-saving: channels below threshold are treated as
            # dormant and forced to exactly zero (a real compute-skip), while the
            # surviving channels keep their graded gate multiplier.
            keep = (g >= self.bandwidth_threshold).to(z.dtype)
            z_gated = z * g * keep
        else:
            # Soft path (always during training): differentiable attenuation so
            # gradients reach every channel and the gate can learn what to silence.
            z_gated = z * g

        # Diagnostics: report the gate mean and the effective active bandwidth
        # (fraction of channels above the dormancy threshold).
        with torch.no_grad():
            self.last_bandwidth_gate_mean = g.detach().mean()
            self.last_active_bandwidth = (
                (g >= self.bandwidth_threshold).to(z.dtype).mean().detach()
            )
        return z_gated

    def _build_causal(self, seq_len: int) -> None:
        dev = self._causal.device if hasattr(self, "_causal") else None
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=dev))
        self.register_buffer("_causal", causal, persistent=False)
        self._causal_len = seq_len

    def _maybe_extend_causal(self, needed: int) -> None:
        """Grow the causal mask on demand so decoding past max_seq_len stays
        correct instead of silently truncating the attention mask."""
        if needed <= self._causal_len:
            return
        self._build_causal(max(needed, self._causal_len * 2))

    # ------------------------------------------------------------------
    # Internal: workspace SA pipeline
    # Separated from forward() so CompetitiveGWTBLayer can reuse it
    # after substituting a different input into the bottleneck.
    # ------------------------------------------------------------------

    def _run_workspace_pipeline(
        self,
        z: torch.Tensor,                              # (B, T_new, d_gw) — pre-compressed
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        position_offset: int,
        use_cache: bool,
        pad_mask: Optional[torch.Tensor] = None,      # (B, T_total) bool; True = keep
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Workspace SA + broadcast.

        Returns (delta, new_kv) where delta: (B, T_new, d_model).
        The caller decides how to add delta to the residual stream.
        """
        B, T_new, _ = z.shape
        H, D = self.n_heads, self.d_head

        # Mask construction is z-independent, so it is hoisted out of the
        # reverberation loop below (J-Space J1, docs/JSPACE_DESIGN.md §2).
        past_len = past_kv[0].shape[2] if past_kv is not None else 0
        T_total = past_len + T_new
        self._maybe_extend_causal(max(position_offset + T_new, T_total))
        causal_mask = self._causal[
            position_offset: position_offset + T_new, :T_total
        ]
        if pad_mask is None:
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)      # original path, bit-identical
        else:
            # Key-padding mask: pad positions cannot be attended to. Use a
            # float bias with finfo.min (same convention as MicrotubuleAttention)
            # instead of a bool mask so a fully-masked row (a pad query) yields a
            # uniform softmax rather than NaN.
            keep = causal_mask.unsqueeze(0).unsqueeze(0) & pad_mask[:, None, None, :T_total].bool()
            attn_mask = None  # dtype needs Q; built lazily on first pass below

        # J1 reverberation: the workspace self-attention re-applies its own
        # (weight-tied) pass workspace_iterations times — content reverberates
        # on the stage before broadcast. Pass k re-derives Q/K/V from pass
        # k-1's normed output; the PAST side of the KV concat stays fixed
        # (those tokens contributed their own final-pass K/V when they were
        # current). The cache stores the FINAL pass's K/V for consistency.
        # workspace_iterations=1 is the exact pre-existing single-pass path.
        z_cur = z
        K = V = None
        for _ in range(self.workspace_iterations):
            Q = self.q_proj(z_cur).view(B, T_new, H, D).transpose(1, 2)  # (B,H,T_new,D)
            K = self.k_proj(z_cur).view(B, T_new, H, D).transpose(1, 2)
            V = self.v_proj(z_cur).view(B, T_new, H, D).transpose(1, 2)
            if past_kv is not None:
                K = torch.cat([past_kv[0], K], dim=2)
                V = torch.cat([past_kv[1], V], dim=2)
            if pad_mask is not None and attn_mask is None:
                attn_mask = torch.zeros(
                    keep.shape, dtype=Q.dtype, device=Q.device
                ).masked_fill(~keep, torch.finfo(Q.dtype).min)
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
            # reshape, not contiguous().view() — see mt_attention.py: view() on the
            # transposed tensor blocks torch.export/ONNX (browser deployment path).
            out = out.transpose(1, 2).reshape(B, T_new, self.d_gw)
            z_cur = self.workspace_norm(z_cur + self.attn_out(out))

        new_kv = (K, V) if use_cache else None
        delta = self.broadcast(z_cur)                                # (B,T_new,d_model)
        return delta, new_kv

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                                              # (B, T_new, d_model)
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        use_cache: bool = False,
        pad_mask: Optional[torch.Tensor] = None,                      # (B, T_total) bool
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        z = self.compress_norm(self.compress(x))                      # (B, T_new, d_gw)
        z = self._apply_bandwidth_gate(z)                             # dynamic bandwidth (no-op if off)
        delta, new_kv = self._run_workspace_pipeline(
            z, past_kv, position_offset, use_cache, pad_mask=pad_mask
        )
        return x + self.broadcast_gate * delta, new_kv


# ---------------------------------------------------------------------------
# CompetitiveGWTBLayer
# ---------------------------------------------------------------------------

class BidProjector(nn.Module):
    """
    Single specialist bid: a residual projection of x.

    bid = x + delta(x)

    delta is a 2-layer bottleneck MLP initialised to output zero.
    At init: bid ≡ x for all K projectors → competition starts uniform →
    CompetitiveGWTBLayer output equals GWTBLayer output.
    During training: each projector learns a specialised deviation from x.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Bottleneck MLP: d_model → d_model//4 → d_model
        hidden = max(d_model // 4, 1)
        self.fc1 = nn.Linear(d_model, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, d_model, bias=True)
        # Zero-init output layer → delta = 0 at init → bid = x
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual: bid = x + GELU(fc1(x)) @ fc2
        delta = self.fc2(F.gelu(self.fc1(x)))
        return x + delta


class CompetitiveGWTBLayer(GWTBLayer):
    """
    Winner-take-all competition for the Global Workspace bottleneck.

    K BidProjectors extract K specialist views from x. A shared ScoreHead
    rates each bid. The softmax-weighted combination enters the bottleneck
    compression → workspace SA → broadcast pipeline. The broadcast is added
    back to the *original* x, not to the competed combination.

    See module docstring for full design rationale.

    P3.1 multi-source GWT (2026-06-07)
    ----------------------------------
    forward() / _compete() optionally accept ``external_bids`` — a list of
    (B, T, d_model) tensors submitted by *other* modules (e.g. the predictive
    world model's expectation of the current state). They join the competition
    on equal footing with the internal specialists via the shared score_head.
    The invariant is preserved: no external bids → identical to internal-only.
    If a caller supplies an external bid that equals x (as a zero-init residual
    adapter does at init), the init-identity property still holds.
    """

    def __init__(self, config: MTLNNConfig):
        super().__init__(config)
        K = getattr(config, "n_competitive_bids", 3)
        self.n_bids = K
        self.hard_winner = getattr(config, "competitive_hard_winner", False)
        # Symmetry-breaking config (P0 fix, 2026-06-07):
        self.score_noise_scale = float(getattr(config, "competitive_score_noise", 0.5))
        self.ortho_penalty_weight = float(getattr(config, "competitive_ortho_weight", 0.01))

        # K specialist projectors (all start as identity)
        self.bid_projectors = nn.ModuleList(
            [BidProjector(config.d_model) for _ in range(K)]
        )

        # Shared score head: bid → scalar relevance score.
        # Zero-init output → all scores start at 0 → uniform softmax at init.
        # (Symmetry is broken by competitive_score_noise during training, not by init.)
        score_hidden = max(config.d_model // 4, 1)
        self.score_head = nn.Sequential(
            nn.Linear(config.d_model, score_hidden, bias=True),
            nn.GELU(),
            nn.Linear(score_hidden, 1, bias=True),
        )
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)
        nn.init.normal_(self.score_head[0].weight, std=0.02)
        nn.init.zeros_(self.score_head[0].bias)

        # P3.1: whether this layer accepts external module bids (the model only
        # passes them when configured). The mechanism is identity-preserving:
        # external bids absent → behaviour is bit-identical to before.
        self.accept_external_bids = bool(getattr(config, "gwtb_external_bids", False))

        # Diagnostics: mean bid weights and competition entropy from the last
        # forward call. Non-persistent buffers — for monitoring only.
        self.register_buffer(
            "last_winner_weights", torch.ones(K) / K, persistent=False
        )
        self.register_buffer(
            "last_competition_entropy", torch.zeros(()), persistent=False
        )
        # P3.1: total softmax mass placed on EXTERNAL bids in the last forward
        # (0.0 when no external bids participated). A health signal: how much
        # the workspace currently relies on other modules vs its own specialists.
        self.register_buffer(
            "last_external_weight", torch.zeros(()), persistent=False
        )

        # Orthogonality penalty (computed in _compete during training, picked up
        # by MTLNNModel.forward and added to total loss). None at eval.
        self._last_ortho_penalty: "Optional[torch.Tensor]" = None

    def reset_zero_init(self) -> None:
        """Re-apply the zero initialisations that guarantee the 'bid ≡ x /
        uniform competition at init' invariant.

        MTLNNModel.__init__ runs a GLOBAL ``self.apply(init_weights)`` pass that
        re-initialises every nn.Linear weight to N(0, 0.02) — silently clobbering
        the zero-init of each BidProjector.fc2 and of score_head[-1] set in the
        constructors above. The model calls this method AFTER that global pass so
        the invariant (CompetitiveGWTBLayer ≡ GWTBLayer at init) actually holds.
        """
        for proj in self.bid_projectors:
            nn.init.zeros_(proj.fc2.weight)
            nn.init.zeros_(proj.fc2.bias)
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def _compete(
        self,
        x: torch.Tensor,
        external_bids: "Optional[List[torch.Tensor]]" = None,
    ) -> torch.Tensor:
        """
        Extract K internal bids (+ any external bids), score them, return the
        weighted combination that enters the workspace bottleneck.

        x             : (B, T, d_model)
        external_bids : optional list of (B, T, d_model) tensors submitted by
                        *other* modules (P3.1 multi-source GWT). Each competes on
                        equal footing with the internal BidProjector bids via the
                        shared score_head. None / empty → behaviour is identical
                        to the internal-only path (zero regression).
        returns       : (B, T, d_model)  — weighted combination

        Symmetry-breaking (P0 fix, 2026-06-07):
        - Training: Gaussian noise is added to scores BEFORE softmax. Without
          this, K identical bid projectors → identical bids → identical scores
          → softmax gradient is zero by symmetry → bid_projectors never diverge.
          DeepSeekMoE-style noise injection guarantees gradients carry signal.
        - Orthogonality penalty on bid_projectors' output layers actively pushes
          specialist projectors apart in weight space. Stored in
          self._last_ortho_penalty for MTLNNModel.forward to add to total loss.
          (External bids carry no fc2 weights here, so the penalty is unchanged
          — it still only separates the internal specialists.)
        """
        B, T, D = x.shape
        K = self.n_bids

        # 1. K specialist bids, each (B, T, d_model)
        bids: List[torch.Tensor] = [proj(x) for proj in self.bid_projectors]
        n_internal = len(bids)

        # 1b. P3.1: append validated external bids (multi-source competition).
        n_external = 0
        if external_bids:
            for j, eb in enumerate(external_bids):
                if eb is None:
                    continue
                if eb.shape != x.shape:
                    raise ValueError(
                        f"external bid {j} has shape {tuple(eb.shape)}, expected "
                        f"{tuple(x.shape)} (B, T, d_model) to match x"
                    )
                bids.append(eb)
                n_external += 1

        # 2. Score each bid (shared score_head)
        scores = torch.cat(
            [self.score_head(bid) for bid in bids], dim=-1
        )   # (B, T, K + n_external)

        # 2b. Symmetry-breaking noise during training.
        # Even when all bids are identical at init, noisy scores → different
        # weights per step → gradient signal flows into different bids → over
        # many steps, bid_projectors diverge.
        if self.training and self.score_noise_scale > 0.0:
            scores = scores + torch.randn_like(scores) * self.score_noise_scale

        # 3. Normalise: soft (training) or hard (inference when configured)
        if self.hard_winner and not self.training:
            n_total = scores.shape[-1]                                  # K + n_external
            winner_idx = scores.argmax(dim=-1)                          # (B, T)
            weights = F.one_hot(winner_idx, n_total).to(x.dtype)       # (B, T, K+ext)
        else:
            weights = F.softmax(scores, dim=-1)                         # (B, T, K)

        # 4. Weighted combination: Σ_k w_k · bid_k → (B, T, d_model)
        bids_stack = torch.stack(bids, dim=2)                           # (B, T, K, D)
        combined = (weights.unsqueeze(-1) * bids_stack).sum(dim=2)     # (B, T, D)

        # 5. Orthogonality penalty (only during training, when weight > 0)
        # Pushes the K bid output-projection matrices apart in weight space.
        # Without this, bids could re-converge after noise-induced divergence.
        if self.training and self.ortho_penalty_weight > 0.0 and K > 1:
            # Stack fc2 weights from K bid projectors: each is (d_model, hidden)
            fc2_stack = torch.stack(
                [proj.fc2.weight for proj in self.bid_projectors], dim=0
            )   # (K, d_model, hidden)
            # Flatten and L2-normalise each projector's flattened weight
            W_flat = fc2_stack.flatten(1)                               # (K, d_model*hidden)
            W_norm = F.normalize(W_flat, dim=-1, eps=1e-8)              # (K, d_model*hidden)
            # Pairwise cosine similarity (K, K)
            sim_matrix = W_norm @ W_norm.t()
            # Penalise OFF-DIAGONAL (we want different projectors to be orthogonal)
            off_diag_mask = 1.0 - torch.eye(K, device=W_flat.device, dtype=W_flat.dtype)
            self._last_ortho_penalty = (sim_matrix.pow(2) * off_diag_mask).sum() / (K * (K - 1))
        else:
            self._last_ortho_penalty = None

        # 6. Diagnostics (no_grad: detached from compute graph)
        with torch.no_grad():
            mean_weights = weights.detach().mean(dim=(0, 1))            # (K + n_external,)
            # Keep last_winner_weights reporting the INTERNAL specialists so the
            # shape is stable (== K) for existing observability; the external
            # mass is reported separately below.
            self.last_winner_weights = mean_weights[:n_internal]
            ent = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            self.last_competition_entropy = ent.detach()
            if n_external > 0:
                self.last_external_weight = mean_weights[n_internal:].sum()
            else:
                self.last_external_weight = torch.zeros((), device=x.device)

        return combined

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        use_cache: bool = False,
        external_bids: "Optional[List[torch.Tensor]]" = None,
        pad_mask: Optional[torch.Tensor] = None,                        # (B, T_total) bool
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Competition: which specialist view enters the workspace bottleneck.
        # external_bids (P3.1) let other modules compete for the workspace; None
        # → internal-only, identical to the pre-P3.1 behaviour.
        x_workspace = self._compete(x, external_bids=external_bids)     # (B, T, d_model)

        # Compress the competed view into the bottleneck
        z = self.compress_norm(self.compress(x_workspace))              # (B, T, d_gw)
        z = self._apply_bandwidth_gate(z)                              # dynamic bandwidth (no-op if off)

        # Workspace SA + broadcast (inherited pipeline, reused without modification)
        delta, new_kv = self._run_workspace_pipeline(
            z, past_kv, position_offset, use_cache, pad_mask=pad_mask
        )

        # Broadcast added to ORIGINAL x — competition determines workspace input,
        # broadcast enriches the full specialist stream.
        return x + self.broadcast_gate * delta, new_kv
