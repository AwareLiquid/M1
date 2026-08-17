"""
mt_lnn_v2.py — MT-LNN adapter, generation 2. Honest, lean, recall-capable.

WHAT V1 GOT RIGHT (kept, unchanged in spirit)
---------------------------------------------
  * Multi-timescale closed-form LTC recurrence trained with a real parallel
    scan (pscan_constant_A) — the "liquid" core.
  * Dynamic per-input timescale gating (kappa gate) + learned static blend.
  * Content-aware mixing across the P parallel state slots (RMC attention).

WHAT V1 GOT WRONG (fixed here, with receipts)
---------------------------------------------
Measured on TinyLlama-1.1B (D=2048, P=13, S=5, 6 adapters): v1 spends
62.8M trainable params, of which

    in_proj + out_proj (2048->2054->2048 dense)   80.4%
    W_in resonance (P*S dense 158x158 maps)       15.5%
    everything actually "liquid"                  <5%

i.e. 96% of the budget is plumbing, not dynamics — and the real trainable
fraction was 5.4%, not the advertised 0.196%. V2 fixes the plumbing:

  1. BOTTLENECK, FACTORIZED projections. d_proto is a free hyperparameter
     (default 64, Tensor-Core aligned) instead of d_model/P, and the two
     projections are low-rank factorized (default rank 128):
     2 x 4.21M  ->  2 x 0.37M per adapter.
  2. DIAGONAL per-scale input maps. One shared d x d mixing matrix per
     protofilament + per-(proto, scale) diagonal gain/bias replaces the dense
     (P, S, d, d) bank: 1.62M -> 0.06M. This is the S4D/Mamba lesson: dense
     per-scale input maps are not where quality comes from.
  3. ONE lateral path. V1 ran three couplings in parallel (static W_lat,
     nearest-neighbor roll, RMC attention) whose own docstring admits RMC is
     a strict superset of the roll path. V2 keeps RMC only.
  4. Per-DIMENSION output gating (SiLU, Mamba-style) replaces the
     per-protofilament scalar MAP gate: strictly more expressive, fewer params.
  5. FastWeightMemoryV2 ON by default — the content-addressable memory v1
     shipped disabled — reimplemented with a CHUNKED parallel scan
     (O(T/C) sequential steps instead of O(T) python-loop steps), so it is
     actually trainable at seq_len 512+ without 100x slowdown.

No Orch-OR / quantum / consciousness framing: the mechanism is a
multi-timescale gated linear recurrence with slot attention and fast-weight
associative memory. "Protofilament" survives only as the name of the P
parallel state slots.

Default budget (TinyLlama-1.1B, 6 adapters): ~8.4M trainable (~0.76%),
7.5x smaller than v1 while ADDING associative recall. Set proj_rank=64,
fast_weight_dim=32 for a ~4.4M (~0.4%) config when matching smaller budgets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_scan import pscan, pscan_constant_A


@dataclass
class MTAdapterV2Config:
    hidden_size: int
    n_protofilaments: int = 13
    d_proto: int = 64              # free hyperparameter now (v1 forced ceil(D/P))
    n_time_scales: int = 5
    proj_rank: int = 128           # factorization rank of in/out projections
    tau_min: float = 0.1
    tau_max: float = 100.0
    dt: float = 1.0
    # Initial timescales, one per scale (geometric ladder if fewer given).
    tau_init: Tuple[float, ...] = (0.5, 2.0, 8.0, 32.0, 90.0)
    init_scale: float = 1e-3       # residual gate init (same convention as v1)
    dropout: float = 0.0
    # Selective (input-dependent) decay — the Mamba lesson. The static ladder
    # gives every token the same persistence per (proto, scale); with
    # selectivity the CONTENT decides how long it is remembered:
    #     dt_t = softplus(W_dt u_t + b_dt)      (init b_dt so dt ~= 1)
    #     decay_t = exp(-dt_t / tau_ps)
    # At init this reproduces the static behaviour exactly (dt=1), so it is a
    # strict generalisation. ~4K params/adapter. Off by default so the v2
    # ablation baseline stays what was benchmarked; "v2s" configs turn it on.
    selective_decay: bool = False
    # Transition parameterisation for selective_decay (E5e, 2026-08-15).
    # "mamba" (default, historical): decay_t = exp(-dt/tau) ∈ (0,1) — the
    # input-dependent dt form; positive-only, CANNOT express parity (needs
    # signed flip, Sarrof Thm 2).
    # "exp": lam_t = 2·exp(-dt/tau) - 1 ∈ (-1,1) — signed, reaches ±1 exactly;
    # restores length extrapolation (E5d/E5e: 0.999 on the full layer).
    selective_decay_mode: str = "mamba"
    # Fast-weight associative memory (Ba et al. 2016 / gated linear attention).
    # V2 turns it ON by default: precise in-context recall is the single
    # biggest capability gap of pure recurrent state (the honest 0% needle
    # result), and a decaying fixed-size state cannot close it.
    use_fast_weight: bool = True
    fast_weight_dim: int = 64
    fast_weight_heads: int = 1
    fast_weight_init_decay: float = 0.95
    fast_weight_chunk: int = 64    # chunk length of the parallel scan
    fast_weight_rule: str = "outer"   # "outer" (default) or "delta" (gradient-as-memory)


class FastWeightMemoryV2(nn.Module):
    """Chunked-parallel causal fast-weight memory.

    Same math as v1's FastWeightMemory (write k->v outer products into a
    decaying matrix F, read associatively with q):

        F_t = lam * F_{t-1} + k_t v_t^T          z_t = lam * z_{t-1} + k_t
        r_t = (q_t F_t) / (q_t . z_t + eps)

    but evaluated chunk-by-chunk: within a chunk of length C the reads are a
    single masked matmul against decay powers lam^(i-j); only the chunk
    boundary state is carried sequentially. T=512, C=64 -> 8 sequential steps
    instead of 512. Equivalence with the sequential reference is asserted in
    tests (rtol 1e-4 in fp32).
    """

    def __init__(self, d_model: int, d_mem: int = 64, n_heads: int = 1,
                 init_decay: float = 0.95, chunk: int = 64,
                 write_rule: str = "outer"):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.chunk = chunk
        # "outer" = classic outer-product accumulation (chunk-parallel, the
        # default, proven at 0.56 cross-window). "delta" = gradient-as-memory:
        # a 1-step inner gradient on ||k^T F - v^T||^2 per position (DeltaNet /
        # Titans), which CORRECTS existing associations instead of only adding —
        # the candidate for encoding distributed context the outer product
        # can't (the out-of-window LM null). It is nonlinear in (k,v,F), so the
        # closed-form chunk scan does not apply — the delta path runs sequential.
        assert write_rule in ("outer", "delta")
        self.write_rule = write_rule
        inner = n_heads * d_mem
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)
        init_decay = min(max(init_decay, 1e-3), 1 - 1e-3)
        raw = math.log(init_decay / (1.0 - init_decay))
        self.decay_raw = nn.Parameter(torch.full((n_heads,), float(raw)))
        if write_rule == "delta":
            # learnable inner learning-rate eta per head; sigmoid(0)=0.5 init.
            self.eta_raw = nn.Parameter(torch.zeros(n_heads))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """x: (B, T, d_model) -> (out (B, T, d_model), (F, z)) carry state."""
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_mem
        C = min(self.chunk, T)

        # phi = elu+1 keeps keys/queries positive => denominator stays positive.
        k = (F.elu(self.W_k(x)) + 1.0).view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        q = (F.elu(self.W_q(x)) + 1.0).view(B, T, H, D).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, D).transpose(1, 2)

        lam = torch.sigmoid(self.decay_raw).to(x.dtype)                 # (H,)

        if state is None:
            Fmat = x.new_zeros(B, H, D, D)
            zvec = x.new_zeros(B, H, D)
        else:
            Fmat, zvec = state

        if self.write_rule == "delta":
            # Gradient-as-memory: F_t = lam*F_{t-1} - eta * k_t (k_t^T F_{t-1} - v_t^T).
            # Read r_t = q_t^T F_t (the delta rule self-normalises by regressing
            # v directly, so no q·z denominator). Sequential — F appears in its
            # own write, so the closed-form chunk scan does not hold. zvec is
            # carried unchanged (kept for the (F,z) snapshot/restore contract).
            #
            # STABILITY FIX (2026-07-12 — GPU decisive-experiment postmortem):
            # F_t = (lam*I - eta*k_t k_t^T) F_{t-1} + eta*k_t v_t^T, so the
            # per-step contraction eigenvalue along k_t is (lam - eta*||k_t||^2).
            # The shared elu(x)+1 feature map (kept for the OUTER branch's
            # positive q·z denominator) is UNNORMALISED — at d_mem=64 on real
            # TinyLlama activations ||k_t||^2 ~ 64 typically, giving eigenvalue
            # ~ -32: F blows up ~32x/step and hits fp32 overflow within ~20
            # sequential steps (verified numerically; the Kaggle run was NaN by
            # its first logged checkpoint). L2-normalising k_t to unit norm makes
            # the eigenvalue exactly (lam - eta), which is GUARANTEED in (-1, 1)
            # for any lam, eta in (0,1) (both sigmoid-parameterised) — bounded
            # regardless of input scale or d_mem. q_t is left unnormalised
            # (read-only; once F is bounded, q^T F is bounded too).
            k_hat = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            eta = torch.sigmoid(self.eta_raw).to(x.dtype).view(1, H, 1, 1)
            lam_d = lam.view(1, H, 1, 1)
            reads = []
            for t in range(T):
                kt, qt, vt = k_hat[:, :, t], q[:, :, t], v[:, :, t]   # (B,H,D)
                pred = torch.einsum("bhd,bhde->bhe", kt, Fmat)        # k^T F  (B,H,D)
                err = pred - vt                                       # (B,H,D)
                Fmat = lam_d * Fmat - eta * (kt.unsqueeze(-1) * err.unsqueeze(-2))
                reads.append(torch.einsum("bhd,bhde->bhe", qt, Fmat))  # q^T F
            r = torch.stack(reads, dim=2).transpose(1, 2).reshape(B, T, H * D)
            return self.W_o(r), (Fmat, zvec)

        # Precompute intra-chunk decay tables once (shapes (H,C,C) / (H,C)).
        t_idx = torch.arange(C, device=x.device, dtype=x.dtype)
        delta = t_idx.view(C, 1) - t_idx.view(1, C)                     # t - j
        causal = delta >= 0
        # lam^(t-j) for j<=t, 0 elsewhere. exp(log) form keeps it stable.
        log_lam = torch.log(lam.clamp_min(1e-6)).view(H, 1, 1)
        decay_tj = torch.where(
            causal.view(1, C, C), torch.exp(log_lam * delta.clamp(min=0).view(1, C, C)),
            x.new_zeros(1),
        )                                                               # (H,C,C)
        decay_t1 = torch.exp(log_lam.view(H, 1) * (t_idx + 1.0).view(1, C))  # lam^(t+1), (H,C)

        outs = []
        for s0 in range(0, T, C):
            s1 = min(s0 + C, T)
            L = s1 - s0
            kc, qc, vc = k[:, :, s0:s1], q[:, :, s0:s1], v[:, :, s0:s1]  # (B,H,L,D)
            d_tj = decay_tj[:, :L, :L]                                   # (H,L,L)
            d_t1 = decay_t1[:, :L]                                       # (H,L)

            # Intra-chunk contribution: sum_{j<=t} lam^(t-j) (q_t.k_j) v_j
            scores = torch.einsum("bhtd,bhjd->bhtj", qc, kc) * d_tj      # (B,H,L,L)
            num = torch.einsum("bhtj,bhjd->bhtd", scores, vc)            # (B,H,L,D)
            # Denominator mirrors the numerator with v_j -> 1: q_t . z_t where
            # z_t = lam^(t+1) z0 + sum_{j<=t} lam^(t-j) k_j
            den_intra = torch.einsum("bhtd,bhjd,htj->bht", qc, kc, d_tj)
            # Cross-chunk contribution from carried state (F0, z0):
            num = num + d_t1.view(1, H, L, 1) * torch.einsum("bhtd,bhde->bhte", qc, Fmat)
            den = den_intra + d_t1.view(1, H, L) * torch.einsum("bhtd,bhd->bht", qc, zvec)

            outs.append(num / den.clamp_min(1e-6).unsqueeze(-1))

            # Update carry state to end of chunk:
            #   F_end = lam^L F0 + sum_j lam^(L-1-j) k_j v_j^T
            w_end = torch.exp(log_lam.view(H, 1) * (L - 1.0 - t_idx[:L]).view(1, L))  # (H,L)
            Fmat = (
                torch.exp(log_lam.view(H, 1, 1) * L) * Fmat
                + torch.einsum("bhjd,bhje,hj->bhde", kc, vc, w_end)
            )
            zvec = (
                torch.exp(log_lam.view(H, 1) * L).view(1, H, 1) * zvec
                + torch.einsum("bhjd,hj->bhd", kc, w_end)
            )

        r = torch.cat(outs, dim=2).transpose(1, 2).reshape(B, T, H * D)
        return self.W_o(r), (Fmat, zvec)


class MTLNNLayerV2(nn.Module):
    """Multi-timescale gated linear-recurrence block over P state slots.

    Pipeline (x: (B, T, d_model)):
      1. u   = in_proj(x)            factorized D -> r -> P*d, view (B,T,P,d)
      2. a   = W_mix per-proto mixing (P, d, d), shared across scales
         A_ps= sigmoid(g_ps * a + b_ps)  per-scale DIAGONAL modulation
      3. scan h_ps,t = decay_ps * h_ps,t-1 + (1-decay_ps) * A_ps,t   (pscan)
      4. blend scales: softmax(blend) * kappa(x) dynamic gate, sum over S
      5. RMC slot attention across P (single lateral path), sigmoid-gated
      6. per-dim SiLU output gate from u (Mamba-style)
      7. y = out_proj(flatten)       factorized P*d -> r -> D
    Returns (y, h_last (B,P,S,d)) — same state contract as v1.
    """

    def __init__(self, cfg: MTAdapterV2Config):
        super().__init__()
        P, d, S, r = cfg.n_protofilaments, cfg.d_proto, cfg.n_time_scales, cfg.proj_rank
        self.P, self.d, self.S = P, d, S
        self.tau_min, self.tau_max, self.dt = cfg.tau_min, cfg.tau_max, cfg.dt
        dpt = P * d

        # 1/7. Factorized projections (bias-free; the residual carries identity)
        self.in_a = nn.Linear(cfg.hidden_size, r, bias=False)
        self.in_b = nn.Linear(r, dpt, bias=False)
        self.out_a = nn.Linear(dpt, r, bias=False)
        self.out_b = nn.Linear(r, cfg.hidden_size, bias=False)

        # 2. Shared per-proto mixing + per-(proto,scale) diagonal modulation
        self.W_mix = nn.Parameter(torch.empty(P, d, d))
        nn.init.normal_(self.W_mix, std=0.02)
        self.scale_gain = nn.Parameter(torch.ones(P, S, d))
        self.scale_bias = nn.Parameter(torch.zeros(P, S, d))

        # 3. Timescales: softplus-parameterised tau ladder, per (proto, scale)
        taus = list(cfg.tau_init)[:S]
        while len(taus) < S:                       # extend geometrically if short
            taus.append(min(taus[-1] * 4.0, cfg.tau_max * 0.9))
        log_tau = torch.empty(P, S)
        for s, t in enumerate(taus):
            log_tau[:, s] = math.log(math.expm1(max(t - cfg.tau_min, 1e-6)))
        self.log_tau = nn.Parameter(log_tau)

        # 3b. Selective decay (opt-in): per-token dt from the block input.
        # b_dt = softplus^-1(1) so dt starts at exactly 1 -> identical to the
        # static path at init; W_dt small so selectivity is learned, not noise.
        self.selective_decay = cfg.selective_decay
        self.sel_mode = getattr(cfg, "selective_decay_mode", "mamba")
        if cfg.selective_decay:
            self.W_dt = nn.Parameter(torch.empty(P, d, S))
            nn.init.normal_(self.W_dt, std=0.02)
            self.b_dt = nn.Parameter(torch.full((P, S), math.log(math.e - 1.0)))

        # 4. Scale blending: static + dynamic kappa gate (kept from v1 — cheap
        # and it IS the "input-dependent timescale selection" mechanism)
        self.blend = nn.Parameter(torch.zeros(P, S))
        self.kappa_gate = nn.Linear(d, S)
        nn.init.constant_(self.kappa_gate.bias, 2.0)   # start open (v1 trick)

        # 5. RMC slot attention (the one lateral path we keep).
        # NOTE the rmc_ prefix: PEFT's LoRA target_modules matches by module-name
        # SUFFIX, so naming these q_proj/k_proj/v_proj would get them silently
        # LoRA-wrapped when the host model applies LoRA to its attention. (v1 has
        # exactly this contamination: its LateralCoupling q/k/v_proj are wrapped
        # by PEFT in the shipped phase5b checkpoint.)
        self.rmc_q = nn.Linear(d, d, bias=False)
        self.rmc_k = nn.Linear(d, d, bias=False)
        self.rmc_v = nn.Linear(d, d, bias=False)
        self.rmc_o = nn.Linear(d, d, bias=False)
        self.rmc_gate = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1)≈0.27

        # 6. Per-dimension output gate (replaces the scalar MAP gate)
        self.W_gate = nn.Parameter(torch.empty(P, d, d))
        nn.init.normal_(self.W_gate, std=0.02)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,                           # (B, T, d_model)
        h_prev: Optional[torch.Tensor] = None,     # (B, P, S, d) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        P, d, S = self.P, self.d, self.S

        u = self.in_b(self.in_a(x)).view(B, T, P, d)                  # (B,T,P,d)

        # Input map: shared mixing then per-scale diagonal modulation
        a = torch.einsum("btpd,pde->btpe", u, self.W_mix)             # (B,T,P,d)
        A = torch.sigmoid(
            a.unsqueeze(3) * self.scale_gain + self.scale_bias
        )                                                              # (B,T,P,S,d)

        tau = F.softplus(self.log_tau) + self.tau_min
        tau = tau.clamp(self.tau_min, self.tau_max)

        # Parallel scan: h_t = decay_t*h_{t-1} + (1-decay_t)*A_t
        A_perm = A.permute(0, 2, 3, 1, 4)                              # (B,P,S,T,d)
        h_init = h_prev if h_prev is not None else None
        if self.selective_decay:
            # Input-dependent dt -> per-token decay (Mamba-style selectivity)
            dt = F.softplus(
                torch.einsum("btpd,pds->btps", u, self.W_dt) + self.b_dt
            )                                                          # (B,T,P,S)
            decay_t = torch.exp(-dt / tau.view(1, 1, P, S))            # (B,T,P,S)
            if self.sel_mode == "exp":
                # E5e: signed transition λ_t = 2·exp(-dt/τ) - 1 ∈ (-1,1).
                # Reaches ±1 exactly (dt→0 ⇒ +1, dt→∞ ⇒ −1) — the length-
                # extrapolation parameterisation (0.999 on the full layer).
                # Input coefficient stays (1 - decay_t) with the POSITIVE
                # decay_t, keeping the input path identical to mamba mode.
                lam_t = 2.0 * decay_t - 1.0                             # (B,T,P,S)
            else:
                lam_t = decay_t
            decay_t = decay_t.permute(0, 2, 3, 1)                      # (B,P,S,T)
            lam_t = lam_t.permute(0, 2, 3, 1)                          # (B,P,S,T)
            X = (1.0 - decay_t).unsqueeze(-1) * A_perm
            H = pscan(lam_t, X, h_init=h_init)                         # (B,P,S,T,d)
        else:
            decay = torch.exp(-self.dt / tau)                          # (P,S)
            X = (1.0 - decay).view(1, P, S, 1, 1) * A_perm
            decay_b = decay.unsqueeze(0).expand(B, P, S)
            H = pscan_constant_A(decay_b, X, h_init=h_init)            # (B,P,S,T,d)
        h_scales = H.permute(0, 3, 1, 2, 4)                            # (B,T,P,S,d)
        h_last = h_scales[:, -1]                                       # (B,P,S,d)

        # Blend across scales: static softmax * dynamic kappa, renormalised
        w = F.softmax(self.blend, dim=-1).view(1, 1, P, S)
        kappa = torch.sigmoid(self.kappa_gate(u))                      # (B,T,P,S)
        gw = w * kappa
        gw = gw / gw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        h = (h_scales * gw.unsqueeze(-1)).sum(dim=3)                   # (B,T,P,d)

        # RMC slot attention across P (batch = B*T)
        hf = h.reshape(B * T, P, d)
        attn = F.scaled_dot_product_attention(
            self.rmc_q(hf).unsqueeze(1),
            self.rmc_k(hf).unsqueeze(1),
            self.rmc_v(hf).unsqueeze(1),
        ).squeeze(1)
        h = h + torch.sigmoid(self.rmc_gate) * self.rmc_o(attn).view(B, T, P, d)

        # Per-dim SiLU gate computed from the block input u
        gate = torch.einsum("btpd,pde->btpe", u, self.W_gate)
        h = h * F.silu(gate)

        y = self.dropout(self.out_b(self.out_a(h.reshape(B, T, P * d))))
        return y, h_last


class MTResidualAdapterV2(nn.Module):
    """Pre-norm residual adapter: x + scale*MT(x) [+ fw_scale*FastWeight(x)].

    Streaming state contract matches v1's MTResidualAdapter (see its
    docstring): opt-in via set_adapter_streaming, inference-only, detached,
    reset at sequence start. Without it, KV-cached decode runs every T=1
    step from zero state — a train/serve mismatch.
    """

    def __init__(self, cfg: MTAdapterV2Config):
        super().__init__()
        self.config = cfg
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.mt_layer = MTLNNLayerV2(cfg)
        self.scale = nn.Parameter(torch.tensor(float(cfg.init_scale)))
        self.fast_weight: Optional[FastWeightMemoryV2] = None
        if cfg.use_fast_weight:
            self.fast_weight = FastWeightMemoryV2(
                cfg.hidden_size,
                d_mem=cfg.fast_weight_dim,
                n_heads=cfg.fast_weight_heads,
                init_decay=cfg.fast_weight_init_decay,
                chunk=cfg.fast_weight_chunk,
                write_rule=cfg.fast_weight_rule,
            )
            self.fw_scale = nn.Parameter(torch.tensor(float(cfg.init_scale)))
        # Streaming state — transient attributes, never in state_dict.
        self.stream_enabled: bool = False
        # See v1: cross-window trainers flip these to backprop through state.
        self.stream_in_training: bool = False
        self.stream_detach: bool = True
        self._stream_h: Optional[torch.Tensor] = None
        self._stream_fw: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._stream_pos: int = 0   # kept for API parity with v1 (no clock here)

    def reset_stream(self) -> None:
        self._stream_h = None
        self._stream_fw = None
        self._stream_pos = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B = hidden_states.shape[0]
        streaming = self.stream_enabled and (
            not self.training or self.stream_in_training
        )
        h_prev = None
        if streaming:
            if self._stream_h is not None and self._stream_h.shape[0] != B:
                self.reset_stream()
            elif self._stream_h is not None and (
                self._stream_h.device != hidden_states.device
                or self._stream_h.dtype != hidden_states.dtype
            ):
                # Module moved (.to()/.half()/.cuda()) after a stream was
                # started: residual state would crash on device mismatch or
                # silently mix dtypes. Migrate it to the input's device/dtype.
                self._stream_h = self._stream_h.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
            h_prev = self._stream_h

        normed = self.norm(hidden_states)
        mt_out, h_last = self.mt_layer(normed, h_prev=h_prev)
        out = hidden_states + self.scale * mt_out
        if self.fast_weight is not None:
            fw_state = self._stream_fw if streaming else None
            if fw_state is not None and fw_state[0].shape[0] != B:
                fw_state = None
            elif fw_state is not None and (
                fw_state[0].device != hidden_states.device
                or fw_state[0].dtype != hidden_states.dtype
            ):
                fw_state = tuple(
                    t.to(device=hidden_states.device, dtype=hidden_states.dtype)
                    for t in fw_state
                )
            fw_out, fw_state = self.fast_weight(normed, state=fw_state)
            out = out + self.fw_scale * fw_out
            if streaming:
                self._stream_fw = tuple(
                    (t.detach() if self.stream_detach else t) for t in fw_state
                )
        if streaming:
            self._stream_h = h_last.detach() if self.stream_detach else h_last
            self._stream_pos += hidden_states.shape[1]
        return out


def attach_mt_v2_adapters(
    model: nn.Module,
    hidden_size: Optional[int] = None,
    layer_indices: Optional[Iterable[int]] = None,
    every: int = 4,
    n_protofilaments: int = 13,
    d_proto: int = 64,
    n_time_scales: int = 5,
    proj_rank: int = 128,
    init_scale: float = 1e-3,
    dropout: float = 0.0,
    selective_decay: bool = False,
    selective_decay_mode: str = "mamba",
    use_fast_weight: bool = True,
    fast_weight_dim: int = 64,
    fast_weight_heads: int = 1,
    fast_weight_init_decay: float = 0.95,
    fast_weight_rule: str = "outer",
) -> List[int]:
    """Freeze `model`, wrap every Nth decoder layer with a V2 adapter.

    Same contract as v1's attach_mt_adapters (returns wrapped indices; reuses
    DecoderLayerWithMTAdapter so checkpoints keep the `mt_adapter` key prefix
    that save_adapter_checkpoint filters on).
    """
    from .llama_adapter import (
        DecoderLayerWithMTAdapter,
        find_decoder_layers,
        freeze_module,
        select_layer_indices,
    )

    freeze_module(model)
    layers = find_decoder_layers(model)
    if hidden_size is None:
        cfg = getattr(model, "config", None)
        hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if hidden_size is None:
        raise ValueError("hidden_size was not provided and could not be inferred.")

    chosen = list(layer_indices) if layer_indices is not None else select_layer_indices(
        len(layers), every=every
    )
    for idx in chosen:
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"layer index {idx} out of range for {len(layers)} layers")
        if isinstance(layers[idx], DecoderLayerWithMTAdapter):
            continue
        cfg = MTAdapterV2Config(
            hidden_size=hidden_size,
            n_protofilaments=n_protofilaments,
            d_proto=d_proto,
            n_time_scales=n_time_scales,
            proj_rank=proj_rank,
            init_scale=init_scale,
            dropout=dropout,
            selective_decay=selective_decay,
            selective_decay_mode=selective_decay_mode,
            use_fast_weight=use_fast_weight,
            fast_weight_dim=fast_weight_dim,
            fast_weight_heads=fast_weight_heads,
            fast_weight_init_decay=fast_weight_init_decay,
            fast_weight_rule=fast_weight_rule,
        )
        layers[idx] = DecoderLayerWithMTAdapter(
            layers[idx],
            MTResidualAdapterV2(cfg).to(getattr(model, "dtype", torch.float32)),
        )
    return chosen


def iter_mt_v2_adapter_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MTResidualAdapterV2):
            yield from module.parameters()
