"""kuramoto_coupling.py — Plugin A: Kuramoto phase coupling over protofilament
channels, as an inference-only diagnostic / optional bias.

Idea (see README.md §2-A)
-------------------------
Reshape a block's hidden state ``x (B,T,d_model)`` into ``P`` protofilament
channels ``(B,T,P,d_proto)``. Extract one PHASE per channel, run a few steps of
mean-field Kuramoto coupling, and read off the order parameter

    R = | (1/P) Σ_j exp(i·θ_j) |  ∈ [0,1]

as a cross-channel phase-synchrony / information-binding signal (the Orch-OR
"coherence" read). R≈0 → channels incoherent; R≈1 → fully phase-locked.

Why this respects the hard rules
--------------------------------
* **No training / param-free.** Since plugins may not train, the coupling is a
  FIXED mean-field model: a single scalar coupling ``kappa`` and (by default)
  zero natural frequencies. There are no learnable parameters, so nothing about
  the native weights changes.
* **CPU-cheap.** Mean-field Kuramoto is O(P) per step (compute R,ψ once, update
  all channels), not O(P²). Phases are derived by a fixed, deterministic 2-D
  projection (seeded), so the op is reproducible and allocation-light.
* **Linear-contraction default.** With ``linear=True`` the update is
  ``θ_i ← θ_i + a·(ψ − θ_i)`` (circular), a contraction toward the mean phase ψ
  with rate ``a = dt·kappa·R``; choosing ``a < 1`` guarantees settling in a
  fixed handful of steps (no adaptive ODE solver). The full ``sin`` coupling is
  available behind ``linear=False`` for ablation.

Modes
-----
* ``mode="observe"`` (default): compute R + settling diagnostics, return None →
  the block output is left untouched (bit-identical to native).
* ``mode="intervene"``: additionally rotate each channel's 2-D projection by its
  phase change Δθ_i and add a small ``alpha``-scaled residual back into x, so the
  synchronization biases the representation. Bounded and fully removable — for
  measuring the effect on PPL only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .hooks import PluginHook


@dataclass
class KuramotoCoupling(PluginHook):
    """Mean-field Kuramoto phase coupling over P protofilament channels.

    Parameters
    ----------
    n_protofilaments :
        P — number of channels the d_model axis is split into. Must divide
        d_model (the O1 default d_model=13·d_proto satisfies this with P=13).
    kappa :
        Fixed global coupling strength.
    n_steps :
        Number of fixed Euler steps per block (keep small: 2–4).
    dt :
        Euler step size.
    linear :
        True → linearized contraction update (sin(Δ)≈Δ). False → full sin.
    mode :
        "observe" (no output change) or "intervene" (apply alpha-scaled bias).
    alpha :
        Intervention strength (only used in intervene mode).
    layers :
        Optional whitelist of block indices to act on; None → all blocks.
    seed :
        RNG seed for the deterministic phase-projection directions.
    """

    n_protofilaments: int = 13
    kappa: float = 0.6
    n_steps: int = 4
    dt: float = 0.5
    linear: bool = True
    mode: str = "observe"
    alpha: float = 0.05
    layers: Optional[List[int]] = None
    seed: int = 0
    center: bool = False  # subtract cross-channel mean before phase extraction
    #                       (de-anisotropy control: tests whether a high order
    #                        parameter R reflects real cross-channel binding or
    #                        just a shared dominant direction in the hidden state)
    name: str = "kuramoto"

    # populated per forward
    last_diagnostics: Dict[str, float] = field(default_factory=dict, init=False)
    _proj: Dict[int, torch.Tensor] = field(default_factory=dict, init=False)
    _R_by_layer: Dict[int, float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.mode not in ("observe", "intervene"):
            raise ValueError(f"mode must be observe|intervene, got {self.mode}")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.last_diagnostics = {}
        self._R_by_layer = {}

    # ------------------------------------------------------------------ #
    def _projection(self, d_proto: int, device, dtype) -> torch.Tensor:
        """Two fixed orthonormal directions (d_proto, 2) for phase extraction.

        Deterministic in ``seed`` and ``d_proto``; cached per width.
        """
        key = d_proto
        cached = self._proj.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        g = torch.Generator(device="cpu").manual_seed(self.seed + d_proto)
        m = torch.randn(d_proto, 2, generator=g)
        # Gram-Schmidt → orthonormal columns u, v
        u = m[:, 0]
        u = u / (u.norm() + 1e-8)
        v = m[:, 1] - (m[:, 1] @ u) * u
        v = v / (v.norm() + 1e-8)
        proj = torch.stack([u, v], dim=1).to(device=device, dtype=dtype)  # (d_proto, 2)
        self._proj[key] = proj
        return proj

    # ------------------------------------------------------------------ #
    @staticmethod
    def _order_parameter(theta: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        """Mean-field order parameter. theta: (..., P) → (R, psi), each (...,)."""
        z = torch.exp(1j * theta.to(torch.float32)).mean(dim=-1)  # (...,) complex
        return z.abs(), z.angle()

    # ------------------------------------------------------------------ #
    def step(self, layer_idx: int, x: torch.Tensor) -> Optional[torch.Tensor]:
        if layer_idx == 0:
            # new forward begins — clear per-forward accumulators so R_mean
            # averages over this forward's layers, not across chunks.
            self._R_by_layer = {}

        if self.layers is not None and layer_idx not in self.layers:
            return None

        B, T, d_model = x.shape
        P = self.n_protofilaments
        if d_model % P != 0:
            # Channels don't divide evenly; skip rather than guess a split.
            return None
        d_proto = d_model // P

        h = x.reshape(B, T, P, d_proto).to(torch.float32)        # (B,T,P,dp)
        proj = self._projection(d_proto, x.device, torch.float32)  # (dp,2)

        # De-anisotropy control: remove the component shared by all P channels so
        # the extracted phases describe channel-SPECIFIC structure, not a common
        # dominant direction (which would inflate R as an artifact).
        h_phase = h - h.mean(dim=2, keepdim=True) if self.center else h

        coords = h_phase @ proj                                  # (B,T,P,2): [x_i, y_i]
        cx, cy = coords[..., 0], coords[..., 1]                  # (B,T,P)
        theta0 = torch.atan2(cy, cx)                             # (B,T,P) in (-pi,pi]

        # --- relax phases (fixed-step mean-field Kuramoto) --------------- #
        theta = theta0
        R_traj: List[float] = []
        for _ in range(self.n_steps):
            R, psi = self._order_parameter(theta)                # (B,T)
            R_traj.append(float(R.mean()))
            psi_e = psi.unsqueeze(-1)                            # (B,T,1)
            if self.linear:
                # circular difference wrapped to (-pi,pi], then linear pull
                dphi = torch.atan2(torch.sin(psi_e - theta), torch.cos(psi_e - theta))
                dtheta = self.kappa * R.unsqueeze(-1) * dphi
            else:
                dtheta = self.kappa * R.unsqueeze(-1) * torch.sin(psi_e - theta)
            theta = theta + self.dt * dtheta

        R_final, _ = self._order_parameter(theta)
        R_final_mean = float(R_final.mean())
        R_init_mean = float(self._order_parameter(theta0)[0].mean())

        # stash diagnostics (observe always reports these)
        self._R_by_layer[layer_idx] = R_final_mean
        self.last_diagnostics[f"R_init/L{layer_idx}"] = R_init_mean
        self.last_diagnostics[f"R_final/L{layer_idx}"] = R_final_mean
        self.last_diagnostics[f"R_gain/L{layer_idx}"] = R_final_mean - R_init_mean
        # global summary (mean over layers seen so far)
        seen = list(self._R_by_layer.values())
        self.last_diagnostics["R_mean"] = sum(seen) / len(seen)

        if self.mode == "observe":
            return None

        # --- intervene: rotate each channel's (u,v) projection by Δθ ------ #
        dtheta_tot = theta - theta0                              # (B,T,P)
        c, s = torch.cos(dtheta_tot), torch.sin(dtheta_tot)
        # rotate (cx,cy) -> (cx',cy')
        cx2 = c * cx - s * cy
        cy2 = s * cx + c * cy
        dcoords = torch.stack([cx2 - cx, cy2 - cy], dim=-1)      # (B,T,P,2)
        # lift back to d_proto via the orthonormal directions: delta = dcoords @ projᵀ
        delta = dcoords @ proj.transpose(0, 1)                   # (B,T,P,dp)
        x_new = h + self.alpha * delta
        x_new = x_new.reshape(B, T, d_model).to(x.dtype)
        return x_new
