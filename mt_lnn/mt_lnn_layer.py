"""
mt_lnn_layer.py — Microtubule-Enhanced Liquid Neural Network Layer.

Replaces the Transformer FFN with a P-protofilament parallel closed-form LTC
network with lateral coupling, GTP gating, MAP-protein gating, and multi-scale
resonance.

This implementation is **fully vectorised** over the protofilament dimension P:
no Python `for p in range(P)` loops in the forward path. Increasing P from
the biological 13 up to 32 / 64 / 128 is essentially free on GPU.

Components:
  - VectorizedMultiScaleResonance: P × S closed-form LTC banks computed in one
    einsum + decay update.
  - LateralCoupling: identity residual + RMC-style content-aware self-attention
    across the P slots.
  - VectorizedMAPGate: P MAP-protein MLPs computed via batched einsum.
  - MTLNNLayer: assembles the pipeline.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import MTLNNConfig
from .parallel_scan import pscan_constant_A


# ---------------------------------------------------------------------------
# VectorizedMultiScaleResonance
# ---------------------------------------------------------------------------

class VectorizedMultiScaleResonance(nn.Module):
    """
    All P protofilaments × S time-scales of closed-form LTC, run in one shot.

    Per (proto p, scale s):
        decay_{p,s} = exp(-dt / τ_{p,s})
        A_{b,t,p,s} = σ(W_in[p,s] x_{b,t,p} + b_in[p,s])
        h^{p,s}_{new} = h_prev_{b,t,p} * decay_{p,s}  +  A_{b,t,p,s} * (1 - decay_{p,s})

    Then blend across scales with softmax(blend_weights[p]):
        h_blended_{b,t,p} = Σ_s w_{p,s} * h^{p,s}_{new}
    """

    def __init__(self, config: MTLNNConfig):
        super().__init__()
        P = config.n_protofilaments
        S = config.n_time_scales
        D = config.d_proto
        self.P, self.S, self.D = P, S, D
        self.tau_min = config.tau_min
        self.tau_max = config.tau_max
        self.dt = config.dt
        self.dynamic_scale_gates = getattr(config, "dynamic_scale_gates", True)
        self.scale_gate_active_threshold = getattr(config, "scale_gate_active_threshold", 0.5)
        self.scale_gate_skip_threshold = getattr(config, "scale_gate_skip_threshold", 0.0)
        self.sparse_resonance_kernel = getattr(config, "sparse_resonance_kernel", False)
        self.sparse_resonance_top_k = getattr(config, "sparse_resonance_top_k", 1)

        # Shared weights: (P, S, D, D) — independent W_in per (proto, scale)
        self.W_in = nn.Parameter(torch.empty(P, S, D, D))
        self.b_in = nn.Parameter(torch.zeros(P, S, D))
        nn.init.normal_(self.W_in, mean=0.0, std=0.02)

        self.compute_skip_threshold = getattr(config, "compute_skip_threshold", 0.0)

        # log_tau in raw space (softplus-parameterised); shape (P, S).
        # Init from config.resonance_freqs so each scale starts at a different τ.
        log_tau_init = torch.empty(P, S)
        for s, tau_s in enumerate(config.resonance_freqs[:S]):
            v = math.log(math.expm1(max(tau_s - config.tau_min, 1e-6)))
            log_tau_init[:, s] = v
        self.log_tau = nn.Parameter(log_tau_init)

        # blend_weights: (P, S) — statically balances the default scale mixture
        self.blend_weights = nn.Parameter(torch.zeros(P, S))
        
        # Endogenous Dynamic Kappa Gate: dynamically activates tau channels
        # based on input. Initialized open so channels are not dead at startup.
        if self.dynamic_scale_gates:
            self.kappa_gate = nn.Linear(D, S)
            nn.init.constant_(
                self.kappa_gate.bias,
                getattr(config, "scale_gate_init_bias", 2.0),
            )
        self.register_buffer("last_scale_gate_mean", torch.ones(S), persistent=False)
        self.register_buffer("last_active_scale_ratio", torch.ones(()), persistent=False)
        self.register_buffer("last_nonzero_scale_ratio", torch.ones(()), persistent=False)
        self.register_buffer("last_sparse_scale_ratio", torch.ones(()), persistent=False)
        self.register_buffer("last_sparse_selected_scales", torch.ones(S), persistent=False)
        # Cross-layer scale-gate sharing side-channel.
        # MTLNNModel.forward() sets _forced_active_idx on follower layers before
        # each forward call; None means this layer is the leader and computes its own.
        self._forced_active_idx: Optional[torch.Tensor] = None
        # Leader layers write the computed index here so the model loop can read it.
        self._last_computed_active_idx: Optional[torch.Tensor] = None
        
        self.use_predictive_coding = getattr(config, "use_predictive_coding", False)
        if self.use_predictive_coding and S > 1:
            # Predict faster channel (s-1) from slower channel (s)
            self.W_pred = nn.Parameter(torch.empty(P, S-1, D, D))
            nn.init.normal_(self.W_pred, mean=0.0, std=0.02)
        self.register_buffer("last_pred_error", torch.zeros(()), persistent=False)

        # Rhythm gate (EEG-LAVI). Active only when use_rhythm=True in config.
        self.use_rhythm = getattr(config, "use_rhythm", False)
        if self.use_rhythm:
            # Fixed scale_preference: -1 = fastest τ scale, +1 = slowest τ scale.
            # Not a learnable parameter — it encodes which scales are "slow".
            self.register_buffer(
                "scale_preference",
                torch.linspace(-1.0, 1.0, S),
                persistent=False,
            )
            # Learnable influence strength; tanh(rhythm_scale) ∈ (-1, 1).
            rhythm_scale_val = getattr(config, "rhythm_scale_init", 0.1)
            self.rhythm_scale = nn.Parameter(torch.tensor(float(rhythm_scale_val)))
        self.register_buffer("last_lavi_mean", torch.zeros(()), persistent=False)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor,
                use_scan: bool = True, lavi: torch.Tensor = None):
        """
        x:        (B, T, P, D)
        h_prev:   real-mode:    (B, P, S, D)  — per-scale recurrent state
                  legacy mode:  (B, T, P, D)  — broadcast h_prev (no recurrence)
                  None:         treated as zeros

        Returns (out, h_last) where
          out    : (B, T, P, D)  — blended across scales
          h_last : (B, P, S, D)  — per-scale state at the last time step,
                                   suitable for caching to the next forward.

        When use_scan=True (default), runs a true parallel scan so that
        h_t = decay * h_{t-1} + (1 - decay) * A_t. This makes the "liquid"
        recurrence actually recurrent during training.

        When use_scan=False, falls back to the legacy parallel mode where
        h_prev is broadcast across all T positions. Kept only so ablations
        can isolate the contribution of real recurrence.
        """
        B, T, P, D = x.shape
        S = self.S

        dynamic_kappa = None
        sparse_scale_mask = None
        active_idx = torch.arange(S, device=x.device)
        if self.dynamic_scale_gates:
            dynamic_kappa = torch.sigmoid(self.kappa_gate(x))             # (B,T,P,S)

            if self.sparse_resonance_kernel:
                top_k = max(1, min(int(self.sparse_resonance_top_k), S))
                # Cross-layer gate sharing: follower layers reuse the leader's
                # active_idx (skip the topk call) but keep their own kappa blend.
                # This saves the topk + index propagation without changing quality.
                if self._forced_active_idx is not None:
                    active_idx = self._forced_active_idx.to(device=x.device)
                    self._last_computed_active_idx = None
                else:
                    gate_mean = dynamic_kappa.detach().mean(dim=(0, 1, 2))  # (S,)
                    active_idx = torch.topk(gate_mean, k=top_k).indices.sort().values
                    self._last_computed_active_idx = active_idx              # leader: export

                sparse_scale_mask = torch.zeros(S, device=x.device, dtype=torch.bool)
                sparse_scale_mask[active_idx] = True

                with torch.no_grad():
                    self.last_sparse_scale_ratio = torch.tensor(
                        float(top_k) / float(S), device=x.device, dtype=x.dtype
                    )
                    self.last_sparse_selected_scales = sparse_scale_mask.to(dtype=x.dtype)
            else:
                self._last_computed_active_idx = None
                with torch.no_grad():
                    self.last_sparse_scale_ratio = torch.ones((), device=x.device, dtype=x.dtype)
                    self.last_sparse_selected_scales = torch.ones(S, device=x.device, dtype=x.dtype)
        else:
            self._last_computed_active_idx = None
            with torch.no_grad():
                self.last_sparse_scale_ratio = torch.ones((), device=x.device, dtype=x.dtype)
                self.last_sparse_selected_scales = torch.ones(S, device=x.device, dtype=x.dtype)

        # 1. Per-(proto, scale) input projection: (B,T,P,D) × (P,S,D,D) → (B,T,P,S,D)
        W_in = self.W_in[:, active_idx, :, :]
        b_in = self.b_in[:, active_idx, :]
        A = torch.einsum("btpd,pkde->btpke", x, W_in)                 # (B,T,P,K,D)
        A = A + b_in                                                   # broadcast (P,K,D)
        A = torch.sigmoid(A)

        # 2. Decay per (proto, scale)
        tau = F.softplus(self.log_tau) + self.tau_min                 # (P,S)
        tau = tau.clamp(self.tau_min, self.tau_max)
        decay = torch.exp(-self.dt / tau)                              # (P,S)
        decay_active = decay[:, active_idx]                            # (P,K)
        K_active = active_idx.numel()

        if not use_scan:
            # Legacy parallel mode — h_prev: (B, T, P, D) broadcast across T
            decay_full = decay_active.view(1, 1, P, K_active, 1)
            if h_prev is None:
                h_prev_full = torch.zeros(B, T, P, S, D, device=x.device, dtype=x.dtype)
                h_prev_e = h_prev_full[:, :, :, active_idx, :]
            else:
                # Allow either (B,T,P,D) or (B,P,S,D) — broadcast to (B,T,P,S,D)
                if h_prev.dim() == 4 and h_prev.shape == (B, T, P, D):
                    h_prev_full = h_prev.unsqueeze(3).expand(B, T, P, S, D).clone()
                    h_prev_e = h_prev.unsqueeze(3).expand(B, T, P, K_active, D)
                else:
                    # Treat as (B, P, S, D) → broadcast across T
                    h_prev_full = h_prev.unsqueeze(1).expand(B, T, P, S, D).clone()
                    h_prev_e = h_prev[:, :, active_idx, :].unsqueeze(1).expand(B, T, P, K_active, D)
            h_active = h_prev_e * decay_full + A * (1.0 - decay_full)
            if K_active == S:
                h_per_scale = h_active
            else:
                h_per_scale = h_prev_full
                h_per_scale[:, :, :, active_idx, :] = h_active
        else:
            # Real recurrence via parallel scan.
            # pscan expects (..., T, D); permute (B,T,P,S,D) -> (B,P,S,T,D)
            A_perm = A.permute(0, 2, 3, 1, 4)                          # (B,P,K,T,D)
            X = (1.0 - decay_active).view(1, P, K_active, 1, 1) * A_perm # (B,P,K,T,D)
            decay_bps = decay_active.unsqueeze(0).expand(B, P, K_active) # (B,P,K)

            # Initial state: h_prev is the per-scale state (B, P, S, D).
            # If only (B, P, D) is passed, broadcast across scales (zero-init common case).
            if h_prev is None:
                h_init = torch.zeros(B, P, S, D, device=x.device, dtype=x.dtype)
            elif h_prev.dim() == 4 and h_prev.shape == (B, P, S, D):
                h_init = h_prev
            elif h_prev.dim() == 3 and h_prev.shape == (B, P, D):
                h_init = h_prev.unsqueeze(2).expand(B, P, S, D).contiguous()
            else:
                raise ValueError(
                    f"h_prev shape {tuple(h_prev.shape)} not compatible; "
                    f"expected (B,P,S,D)=({B},{P},{S},{D}) or (B,P,D)=({B},{P},{D})"
                )

            h_init_active = h_init[:, :, active_idx, :]
            H = pscan_constant_A(decay_bps, X, h_init=h_init_active)  # (B,P,K,T,D)
            h_active = H.permute(0, 3, 1, 2, 4)                       # (B,T,P,K,D)
            if K_active == S:
                h_per_scale = h_active
            else:
                h_per_scale = h_init.unsqueeze(1).expand(B, T, P, S, D).clone()
                h_per_scale[:, :, :, active_idx, :] = h_active

        # 3. Dynamic Channel Attention (Endogenous Anesthesia) + Blending
        w = F.softmax(self.blend_weights, dim=-1)                     # (P,S)
        if self.dynamic_scale_gates:
            if sparse_scale_mask is not None:
                dynamic_kappa = dynamic_kappa.masked_fill(
                    ~sparse_scale_mask.view(1, 1, 1, S), 0.0
                )

            # Optional hard masking. In dense mode this masks the blend only;
            # in sparse mode the upstream resonance has already skipped
            # unselected scales.
            skip_threshold = max(self.compute_skip_threshold, self.scale_gate_skip_threshold)
            if skip_threshold > 0.0:
                active = dynamic_kappa >= skip_threshold
                top_source = dynamic_kappa
                if sparse_scale_mask is not None:
                    top_source = top_source.masked_fill(
                        ~sparse_scale_mask.view(1, 1, 1, S), -1.0
                    )
                top_idx = top_source.argmax(dim=-1, keepdim=True)
                top_one_hot = torch.zeros_like(dynamic_kappa, dtype=torch.bool)
                top_one_hot.scatter_(-1, top_idx, True)
                active = active | top_one_hot
                dynamic_kappa = dynamic_kappa.masked_fill(~active, 0.0)

            gated_w = w.view(1, 1, P, S) * dynamic_kappa
            gated_w = gated_w / gated_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            with torch.no_grad():
                self.last_scale_gate_mean = dynamic_kappa.detach().mean(dim=(0, 1, 2))
                self.last_active_scale_ratio = (
                    dynamic_kappa.detach() > self.scale_gate_active_threshold
                ).float().mean()
                self.last_nonzero_scale_ratio = (
                    dynamic_kappa.detach() > 0.0
                ).float().mean()
        else:
            gated_w = w.view(1, 1, P, S).expand(B, T, P, S)
            with torch.no_grad():
                self.last_scale_gate_mean = torch.ones(S, device=x.device, dtype=x.dtype)
                self.last_active_scale_ratio = torch.ones((), device=x.device, dtype=x.dtype)
                self.last_nonzero_scale_ratio = torch.ones((), device=x.device, dtype=x.dtype)

        # LAVI rhythm modulation — runs only when use_rhythm=True and lavi provided.
        # High LAVI (persistent) → boost slow scales; Low LAVI (transient) → boost fast.
        # rhythm_bonus is zero when lavi=0.5 (neutral) and scales with rhythm_scale.
        if lavi is not None and self.use_rhythm:
            rhythm_bonus = (
                torch.tanh(self.rhythm_scale)
                * (lavi - 0.5)                                       # (B,T,P,1) centred
                * self.scale_preference.view(1, 1, 1, S)             # (1,1,1,S)
            )   # (B,T,P,S) — positive = boost slow, negative = boost fast
            gated_w = gated_w * (1.0 + rhythm_bonus)
            gated_w = gated_w / gated_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            with torch.no_grad():
                self.last_lavi_mean = lavi.detach().mean()
        else:
            with torch.no_grad():
                self.last_lavi_mean.zero_()

        out = (h_per_scale * gated_w.unsqueeze(-1)).sum(dim=3)        # (B,T,P,D)

        # 4. Multi-Scale Predictive Coding Loss
        if self.use_predictive_coding and self.S > 1:
            if self.training:
                # Predict faster channel (idx :-1) from slower channel (idx 1:)
                predictor = h_per_scale[:, :, :, 1:, :]                     # (B,T,P,S-1,D)
                target = h_per_scale[:, :, :, :-1, :].detach()              # (B,T,P,S-1,D)
                pred = torch.einsum("btpse,psed->btpsd", predictor, self.W_pred) 
                
                # mse loss of prediction vs target
                self.last_pred_error = F.mse_loss(pred, target)
            else:
                self.last_pred_error.zero_()

        # Per-scale last state for caching — preserves the scan's internal
        # state across calls without going through the blend.
        h_last_per_scale = h_per_scale[:, -1, :, :, :]                # (B,P,S,D)
        return out, h_last_per_scale


# ---------------------------------------------------------------------------
# LateralCoupling — RMC content-aware mixing across protofilaments
# ---------------------------------------------------------------------------

class LateralCoupling(nn.Module):
    """
    Three-way lateral coupling across protofilaments:

      1. **Identity/static residual** (W_lat, 13×13 learned) — analogue of a
         general inter-filament weighting.

      2. **Nearest-neighbor coupling** — each protofilament i exchanges a
         tanh-gated signal with its two neighbors (i-1, i+1) mod P. This is
         the actual biological topology: tubulin lattice B-bonds connect only
         adjacent protofilaments. Implemented branchlessly with torch.roll —
         10× faster than the per-protofilament `for i in range(13)` loop and
         truly *synchronous* (all P updates see the same snapshot, avoiding
         left-to-right propagation bias).

      3. **RMC self-attention** — content-aware all-to-all mixing over the P
         slots, gated by sigmoid(rmc_gate). Strict superset of (2) but starts
         at sigmoid(-3) ≈ 0.05 so the model can rely on the biological
         topology first.

    h: (B, T, P, d_proto)  → (B, T, P, d_proto)
    """

    def __init__(self, n_protofilaments: int, d_proto: int):
        super().__init__()
        self.n_protofilaments = n_protofilaments
        self.d_proto = d_proto

        # --- Nearest-neighbor coupling (biological topology) ---
        # Two linear maps: one for the left neighbor, one for the right.
        self.W_left = nn.Linear(d_proto, d_proto, bias=False)
        self.W_right = nn.Linear(d_proto, d_proto, bias=False)
        # eta: learnable scalar strength of nearest-neighbor coupling.
        # Init small (0.1) so early training is driven by the W_lat identity.
        self.nn_eta = nn.Parameter(torch.tensor(0.1))

        # --- RMC content-aware coupling ---
        self.q_proj = nn.Linear(d_proto, d_proto, bias=False)
        self.k_proj = nn.Linear(d_proto, d_proto, bias=False)
        self.v_proj = nn.Linear(d_proto, d_proto, bias=False)
        self.out_proj = nn.Linear(d_proto, d_proto, bias=False)

        # --- Static identity residual ---
        self.W_lat = nn.Parameter(torch.eye(n_protofilaments))
        self.rmc_gate = nn.Parameter(torch.zeros(()))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Identity / static-matrix residual
        residual = torch.einsum("btpd,pq->btqd", h, self.W_lat)

        # Nearest-neighbor coupling via torch.roll — fully synchronous, all P
        # updates simultaneously see the pre-update snapshot.
        h_left = torch.roll(h, shifts=1,  dims=2)    # neighbor at i-1
        h_right = torch.roll(h, shifts=-1, dims=2)   # neighbor at i+1
        nn_coupling = torch.tanh(self.W_left(h_left) + self.W_right(h_right))
        nearest = self.nn_eta * nn_coupling

        # RMC self-attention over the P "slots" (B*T as batch dim)
        B, T, P, D = h.shape
        h_flat = h.reshape(B * T, P, D)
        q = self.q_proj(h_flat).unsqueeze(1)                          # (B*T,1,P,D)
        k = self.k_proj(h_flat).unsqueeze(1)
        v = self.v_proj(h_flat).unsqueeze(1)
        attn = F.scaled_dot_product_attention(q, k, v).squeeze(1)     # (B*T,P,D)
        rmc = self.out_proj(attn).reshape(B, T, P, D)

        return residual + nearest + torch.sigmoid(self.rmc_gate) * rmc


# ---------------------------------------------------------------------------
# VectorizedMAPGate — all P MAP gates in one shot
# ---------------------------------------------------------------------------

class VectorizedMAPGate(nn.Module):
    """
    P parallel 2-layer MLPs producing per-protofilament stabilisation gates s ∈ [0,1].
        z_p = ReLU(W1_p [h_p; x_p] + b1_p)
        s_p = σ(W2_p z_p + b2_p)

    Implemented as two batched einsums, one weight tensor per layer with leading
    P dimension.
    """

    def __init__(self, n_protofilaments: int, d_proto: int, map_hidden_dim: int):
        super().__init__()
        P = n_protofilaments
        in_dim = 2 * d_proto
        H = map_hidden_dim
        self.fc1_weight = nn.Parameter(torch.empty(P, in_dim, H))
        self.fc1_bias = nn.Parameter(torch.zeros(P, H))
        self.fc2_weight = nn.Parameter(torch.empty(P, H, 1))
        # Initialise fc2_bias = +2 so sigmoid(2) ≈ 0.88 at step 0 — the MAP gates
        # start mostly *open* and the protofilament signal flows through nearly
        # unattenuated. Mirrors the GRU "forget-gate-near-1" trick: prevents
        # gradient starvation by letting information propagate before the gates
        # learn what to suppress.
        self.fc2_bias = nn.Parameter(torch.full((P, 1), 2.0))
        nn.init.normal_(self.fc1_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2_weight, mean=0.0, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        h, x: (B, T, P, D)
        Returns: (B, T, P, 1)
        """
        z = torch.cat([h, x], dim=-1)                                  # (B,T,P,2D)
        z = torch.einsum("btpi,pih->btph", z, self.fc1_weight) + self.fc1_bias
        z = F.relu(z)
        z = torch.einsum("btph,pho->btpo", z, self.fc2_weight) + self.fc2_bias
        return torch.sigmoid(z)


# ---------------------------------------------------------------------------
# MTLNNLayer — fully vectorized over P
# ---------------------------------------------------------------------------

class MTLNNLayer(nn.Module):
    """
    The full Microtubule LNN layer.

    Pipeline:
      1. in_proj: (B,T,d_model) → (B,T,d_proto_total) → reshape to (B,T,P,D)
      2. VectorizedMultiScaleResonance: (B,T,P,D) → (B,T,P,D)
      3. LateralCoupling (identity + RMC) with GTP temporal gate
      4. VectorizedMAPGate per protofilament
      5. out_proj: (B,T,P*D) → (B,T,d_model)

    Returns (output, h_last), where h_last is (B,P,D) — the recurrent state
    for the next call when streaming.
    """

    def __init__(self, config: MTLNNConfig):
        super().__init__()
        self.n_proto = config.n_protofilaments
        self.d_proto = config.d_proto
        self.d_proto_total = config.d_proto_total

        self.in_proj = nn.Linear(config.d_model, config.d_proto_total, bias=True)
        self.out_proj = nn.Linear(config.d_proto_total, config.d_model, bias=True)

        self.resonance = VectorizedMultiScaleResonance(config)
        self.lateral = LateralCoupling(config.n_protofilaments, config.d_proto)
        self.map_gates = VectorizedMAPGate(
            config.n_protofilaments, config.d_proto, config.map_hidden_dim
        )

        # GTP hydrolysis on lateral signal — temporal decay
        self.gtp_gamma = nn.Parameter(torch.tensor(config.gamma_init))
        # GTP cap renewal period: lateral coupling refreshes every `gtp_period`
        # tokens so long contexts don't silently kill mixing. Stored as a buffer
        # (non-learned, fixed at config time).
        self.register_buffer("gtp_period",
                             torch.tensor(float(config.gtp_period)),
                             persistent=False)

        self.dropout = nn.Dropout(config.dropout)

        # LAVI estimator — attached only when use_rhythm=True.
        # LAVIEstimator has no weight sharing with the resonance bank;
        # its only parameters are a per-protofilament bias vector.
        self.use_rhythm = getattr(config, "use_rhythm", False)
        if self.use_rhythm:
            from .rhythm import LAVIEstimator
            self.lavi_estimator = LAVIEstimator(config.d_proto, config.n_protofilaments)
        else:
            self.lavi_estimator = None

        # Hebbian signal accumulator (Phase D). When use_hebbian=True, the
        # forward pass writes self._hebb_signal as a scalar tensor (with grad)
        # for HebbianRegularizer.compute_loss() to collect.
        # Plain Python attribute (not register_buffer) so it stays in the graph.
        self.use_hebbian = getattr(config, "use_hebbian", False)
        self._hebb_signal: Optional[torch.Tensor] = None

        # Hebbian REFACTOR (Phase D', 2026-06-16). When use_hebbian_refactor=True
        # the forward stashes the block's input/output SEQUENCES (B,T,d_model,
        # in-graph) so HebbianPlasticity can build a within-sequence lagged
        # co-activation signal and its own LAVI gate. Gated so the flag-off path
        # is untouched (both attrs stay None).
        self.use_hebbian_refactor = getattr(config, "use_hebbian_refactor", False)
        self._hebb_ref_out: Optional[torch.Tensor] = None
        self._hebb_ref_in: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, d_model)
        h_prev: torch.Tensor = None,           # (B,P,S,D) [real] or (B,P,D) [legacy] or None
        position_offset: int = 0,
        use_scan: bool = True,
    ):
        """
        use_scan=True (default): real recurrence via parallel scan.
        use_scan=False         : legacy parallel mode (h_prev broadcast across T).

        Returns (out, h_last_per_scale) where h_last_per_scale: (B, P, S, D) is
        the per-scale recurrent state to cache for the next forward.
        """
        B, T, _ = x.shape
        P, D = self.n_proto, self.d_proto

        # 1. Project and split into protofilaments
        x_proto = self.in_proj(x)                                      # (B,T,d_proto_total)
        x_split = x_proto.view(B, T, P, D)                            # (B,T,P,D)

        # 1b. LAVI rhythmicity signal (optional, computed before resonance so it
        #     can gate the τ-scale blend inside the resonance bank).
        #     World-model integration (Phase C linkage): MTLNNModel.forward() sets
        #     self._last_wm_pred_error before the block loop when use_world_model=True.
        #     High pred_error → model was surprised → LAVI nudged toward transient.
        lavi = None
        if self.lavi_estimator is not None:
            wm_err = getattr(self, "_last_wm_pred_error", 0.0)
            lavi = self.lavi_estimator(h_prev, x_split, pred_error=wm_err)

        # 2. Run the resonance bank. It accepts h_prev in either form and
        # returns the per-scale state we need to cache.
        h_stack, h_last_per_scale = self.resonance(
            x_split, h_prev, use_scan=use_scan, lavi=lavi
        )                                                              # (B,T,P,D), (B,P,S,D)

        # 4. Lateral coupling with GTP temporal gate.
        # Originally used absolute position t_abs which makes exp(-γ·t_abs) → 0
        # for large contexts (e.g. t=4096) and lateral coupling silently
        # disappears. We now use a **periodic local clock**: t mod period.
        # This mimics microtubule GTP-cap renewal: a fresh cap forms every
        # `gtp_period` steps, lateral coupling refreshes, and the model never
        # loses lateral mixing in long sequences.
        period = self.gtp_period
        t_idx = torch.arange(position_offset, position_offset + T,
                             device=x.device, dtype=x.dtype)
        t_local = t_idx % period                                       # (T,)
        gtp_scale = torch.exp(-self.gtp_gamma.clamp(min=1e-4) * t_local)
        gtp_scale = gtp_scale.view(1, T, 1, 1)
        h_lateral = self.lateral(h_stack)                              # (B,T,P,D)
        h_coupled = h_stack + gtp_scale * (h_lateral - h_stack)

        # 5. ALL MAP gates in one shot
        s = self.map_gates(h_coupled, x_split)                         # (B,T,P,1)
        h_gated = h_coupled * s                                         # (B,T,P,D)

        # 6. Output projection
        h_flat = h_gated.reshape(B, T, P * D)
        out = self.dropout(self.out_proj(h_flat))                      # (B,T,d_model)

        # Phase D: Hebbian CENTERED-COVARIANCE signal (P0 fix, 2026-06-07).
        # The original mean(out ⊙ x) was always ≈ 0 on untrained nets because
        # random projections have zero mean correlation. The centered form
        # measures FLUCTUATION correlation, which is non-zero even at init.
        #
        # Mathematical equivalence:
        #   mean((out - mean(out)) ⊙ (x - mean(x)))
        #   = mean(out ⊙ x) - mean(out) · mean(x)
        #   ≈ cov(out, x) at the elementwise level
        #
        # This is the standard Hebbian-covariance form used in modern
        # neuroscience-inspired learning (Oja 1989, Földiák 1990).
        # Subtract over (batch, time) dims while keeping the feature dim.
        if self.use_hebbian:
            out_centered = out - out.mean(dim=(0, 1), keepdim=True)
            x_centered = x - x.mean(dim=(0, 1), keepdim=True)
            self._hebb_signal = (out_centered * x_centered).mean()
        else:
            self._hebb_signal = None

        # Phase D' (refactor): stash the in/out SEQUENCES (in-graph) for the
        # within-sequence Hebbian signal + dedicated LAVI gate. Gated so the
        # flag-off path stays bit-identical.
        if self.use_hebbian_refactor:
            self._hebb_ref_out = out
            self._hebb_ref_in = x
        else:
            self._hebb_ref_out = None
            self._hebb_ref_in = None

        # We cache the resonance bank's per-scale state (B, P, S, D) — that
        # is the actual recurrent state of the LNN. Caching h_gated would
        # break parity because lateral coupling and MAP gating are applied
        # after the scan and are not part of the recurrence.
        return out, h_last_per_scale


# ---------------------------------------------------------------------------
# Backward-compat aliases (so existing __init__ exports still resolve)
# ---------------------------------------------------------------------------

# Keep exporting these names for compatibility with the package __init__.
MultiScaleResonance = VectorizedMultiScaleResonance
MAPGate = VectorizedMAPGate


class ProtofilamentLTC(nn.Module):
    """Single-protofilament LTC — kept only for backward-compat tests / docs.

    Not used by MTLNNLayer anymore (replaced by VectorizedMultiScaleResonance).
    """

    def __init__(self, d_proto: int, tau_init: float, tau_min: float,
                 tau_max: float, dt: float):
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dt = dt
        self.W_in = nn.Linear(d_proto, d_proto, bias=True)
        log_tau_init = math.log(math.expm1(max(tau_init - tau_min, 1e-6)))
        self.log_tau = nn.Parameter(torch.full((), log_tau_init))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.log_tau) + self.tau_min
        tau = tau.clamp(self.tau_min, self.tau_max)
        decay = torch.exp(-self.dt / tau)
        A = torch.sigmoid(self.W_in(x))
        return h_prev * decay + A * (1.0 - decay)
