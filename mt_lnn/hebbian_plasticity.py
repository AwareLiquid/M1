"""
mt_lnn/hebbian_plasticity.py — refactored Hebbian branch (mechanism A: loss-term).

WHY A SEPARATE MODULE
---------------------
The legacy `HebbianRegularizer` (plasticity.py) is kept untouched for back-compat
and as the A/B baseline. This module is the ground-up rebuild, fully isolated so
it can be (a) toggled with a single config flag, (b) rolled back by deleting this
file, and (c) developed in graded stages without ever touching the BP main path.

VERIFIED FAILURE MODES THIS FIXES (see experiments/report_ablation_hebbian_lr.md
and the code audit of mt_lnn_layer.py / rhythm.py):
  1. Legacy gate input was 0 -> sigmoid(0)=0.5, never modulating, because LAVI is
     only computed when use_rhythm=True. FIX (Stage 1): own a dedicated, tiny LAVI
     estimator so the gate has a live input independent of the rhythm flag.
  2. Gradient share was ~8e-5 even at hebbian_lr=1e-1 (term value ~1e-8). FIX
     (Stage 2): decouple base lr AND adaptively align the Hebbian gradient norm to
     the main gradient norm, capped at hebbian_grad_frac_cap (default 5%).
  3. Signal was same-timestep co-activation only. FIX (Stage 1+): add a
     WITHIN-SEQUENCE lagged term (h_t * h_{t-1}) along the sequence dim -- NOT
     across optimiser steps (shuffled batches are not temporally adjacent).

STAGED BUILD  (status corrected 2026-07-12 — see the architecture audit)
  Stage 0 (superseded): was a param-free scaffold with compute_loss()==None.
  Stage 1 (LANDED): dedicated LAVI estimator (gate_temp/b_lavi params) +
    non-trivial gate + within-seq lag term — compute_signal() below.
  Stage 2 (LANDED): compute_loss() now returns a REAL in-graph tensor
    (alpha_eff * raw_loss), and recalibrate() implements the gradient-fraction
    safety cap. ***compute_loss() NO LONGER returns None*** when a training
    forward stashes block sequences — the old "verified no-op" claim is STALE.
  Stage 3 (open): effectiveness validation. Current status is NEGATIVE, not
    neutral: experiments/report_ablation_hebbian_refactor.md shows val PPL rises
    monotonically with the Hebbian gradient share (+9~18 PPL); "max effective
    share is ~0". So this term currently HURTS.

  ⚠ SAFETY: recalibrate() (the grad-fraction cap) is invoked ONLY in
  experiments/ — NOT in train.py. If use_hebbian_refactor=True is set on the
  shipped training entrypoint, _scale stays 1.0 and the term runs UNCAPPED at
  base_lr. Do not enable on a real run without wiring recalibrate() into the
  loop first. compute_loss() emits a one-time warning if called un-recalibrated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .config import MTLNNConfig
    from .model import MTLNNModel


class HebbianPlasticity(nn.Module):
    """Refactored Hebbian co-activation regularizer (loss-term form).

    Stages 1+2 LANDED (status corrected 2026-07-12): holds the decoupled hyper-
    parameters plus the Stage-1 LAVI gate params (gate_temp, b_lavi), and
    compute_loss() now returns a REAL in-graph loss term (alpha_eff * raw), NOT
    None. The old "no-op until Stage 2" contract is STALE — use_hebbian_refactor=
    True is now an ACTIVE loss term. It is also NOT yet validated as helpful (the
    ablation shows it hurts) and its grad-fraction cap only engages if the
    training loop calls recalibrate(). See the module docstring safety note.
    """

    def __init__(self, config: "MTLNNConfig"):
        super().__init__()
        # Decoupled hyper-parameters (read once; no dependence on the main lr).
        self.base_lr: float = float(getattr(config, "hebbian_base_lr", 1e-2))
        self.window: int = int(getattr(config, "hebbian_window", 32))
        self.grad_frac_cap: float = float(getattr(config, "hebbian_grad_frac_cap", 0.05))
        self.mode: str = str(getattr(config, "hebbian_refactor_mode", "weak"))

        # Diagnostics surface (populated from Stage 1 onward). Kept as a plain
        # dict, not buffers, so it never enters a checkpoint or the autograd graph.
        self._last_stats: Dict[str, float] = {}

        # ---- Stage 1: dedicated lightweight LAVI gate ------------------------
        # The legacy gate died because it consumed the resonance module's LAVI,
        # which is only produced when use_rhythm=True (so the gate input was 0 ->
        # sigmoid(0)=0.5, never modulating). This branch owns its OWN LAVI proxy:
        # the within-sequence cosine similarity between adjacent-token block
        # outputs (history-based "how stable is the representation"), so the gate
        # has a live, per-token input regardless of use_rhythm.
        #
        # Both params are CONSTANT-init (torch.tensor, no RNG draw) so building
        # this module consumes none of the model's init RNG stream -> the
        # flag-on path stays bit-identical to flag-off at init (verified in
        # tests). gate_temp = learnable sharpness s; b_lavi = learnable bias that
        # shifts the sigmoid decision boundary (kills the death zone).
        self.gate_temp = nn.Parameter(torch.tensor(1.0))   # s
        self.b_lavi = nn.Parameter(torch.tensor(0.0))      # b_lavi

        # ---- Stage 2: gradient-alignment scale ------------------------------
        # The effective weight applied to the (already gated) co-activation term
        # is alpha_eff = base_lr * _scale, where _scale in [0,1] is recomputed by
        # recalibrate() from the measured gradient-norm ratio so the Hebbian
        # contribution can never exceed grad_frac_cap (default 5%) of the main
        # gradient norm -- the hard safety bound (cannot dominate the LM loss).
        # Plain float (not a buffer/param): it is an optimisation-loop control
        # knob, never learned and never checkpointed.
        self._scale: float = 1.0
        self._eps: float = 1e-12
        # Safety tracking: the grad-fraction cap only engages once the training
        # loop calls recalibrate(). If compute_loss() ever produces a real term
        # without that ever happening, the term runs UNCAPPED — warn once.
        self._recalibrated: bool = False
        self._warned_uncapped: bool = False

    # ---- Stage 1 internals ---------------------------------------------------

    @staticmethod
    def _within_seq_lavi(out_seq: torch.Tensor) -> torch.Tensor:
        """History-based LAVI proxy: cosine similarity between each token's block
        output and the previous token's. (B, T, D) -> (B, T-1), in [-1, 1].
        High => representation is persistent/stable across the step.

        The sequence is first centred over time (per batch/feature). Without this
        the raw block outputs are dominated by a shared DC component, so adjacent
        cosines are ~1.0 with ZERO variance (verified at init) -> the gate would
        be just as dead as the legacy one. Centring exposes the FLUCTUATION
        stability, which carries genuine per-token variance (std ~0.25 at init).
        This mirrors the Hebbian-covariance 'centre then correlate' philosophy."""
        centred = out_seq - out_seq.mean(dim=1, keepdim=True)
        cur = centred[:, 1:]
        prev = centred[:, :-1]
        return torch.nn.functional.cosine_similarity(cur, prev, dim=-1)

    def _gate(self, lavi: torch.Tensor) -> torch.Tensor:
        """Map a per-token LAVI signal to a dynamic gate in (0,1).

        lavi_norm = lavi - mean_t(lavi) + b_lavi   (per-sequence centring removes
        the death zone: the gate spans (0,1) with real per-token deviation rather
        than sitting at a constant 0.5). gate = sigmoid(s * lavi_norm)."""
        lavi_norm = lavi - lavi.mean(dim=1, keepdim=True) + self.b_lavi
        return torch.sigmoid(self.gate_temp * lavi_norm)

    def compute_signal(self, model: "MTLNNModel") -> Optional[torch.Tensor]:
        """Gated within-sequence Hebbian co-activation, aggregated over blocks.

        For each block we form the lagged covariance between the post-synaptic
        output at t and the pre-synaptic input at t-1 (the h_t * h_{t-1} temporal
        association the legacy same-timestep signal lacked), weighted per-token by
        the dedicated LAVI gate. Returns a scalar in-graph tensor, or None if no
        block stashed its sequences (e.g. use_hebbian_refactor off, or eval-only).
        Populates last_stats for monitoring. Does NOT add anything to the loss --
        Stage 2 consumes this to build the capped, gradient-aligned loss term.
        """
        sigs = []
        gate_means, gate_stds, lavi_means = [], [], []
        for block in model.blocks:
            out_seq = getattr(block.lnn, "_hebb_ref_out", None)
            in_seq = getattr(block.lnn, "_hebb_ref_in", None)
            if out_seq is None or in_seq is None or out_seq.shape[1] < 2:
                continue
            post = out_seq[:, 1:]                 # h_t       (B, T-1, D)
            pre = in_seq[:, :-1]                  # x_{t-1}   (B, T-1, D)
            post_c = post - post.mean(dim=(0, 1), keepdim=True)
            pre_c = pre - pre.mean(dim=(0, 1), keepdim=True)
            coact = (post_c * pre_c).mean(dim=-1)  # (B, T-1) per-position covariance
            lavi = self._within_seq_lavi(out_seq)  # (B, T-1)
            gate = self._gate(lavi)                # (B, T-1)
            sigs.append((gate * coact).mean())
            with torch.no_grad():
                gate_means.append(float(gate.mean()))
                gate_stds.append(float(gate.std()))
                lavi_means.append(float(lavi.mean()))

        if not sigs:
            self._last_stats = {}
            return None

        signal = torch.stack(sigs).mean()
        self._last_stats = {
            "signal": float(signal.detach()),
            "gate_mean": sum(gate_means) / len(gate_means),
            "gate_std": sum(gate_stds) / len(gate_stds),
            "lavi_mean": sum(lavi_means) / len(lavi_means),
            "n_blocks": float(len(sigs)),
        }
        return signal

    @property
    def last_stats(self) -> Dict[str, float]:
        """Read-only view of the most recent diagnostics (gate mean, grad
        fractions, ...). Empty until Stage 1 populates it."""
        return dict(self._last_stats)

    def compute_loss(self, model: "MTLNNModel") -> Optional[torch.Tensor]:
        """Return the Hebbian loss term to add to the total loss, or None.

        Stages 1+2 LANDED: returns a REAL in-graph term (alpha_eff * raw_loss)
        whenever a training forward stashed block sequences; returns None only
        when nothing was stashed (eval, or use_hebbian_refactor off). The old
        "Stage 0 always returns None" contract is STALE (corrected 2026-07-12).
        The signature still matches the legacy HebbianRegularizer.compute_loss so
        the model integration and the finiteness guard (`_aux_or_skip`) are
        unchanged.

        SAFETY: alpha_eff = base_lr * _scale, and _scale is only shrunk below 1.0
        by recalibrate(). If the training loop never calls recalibrate() the term
        runs UNCAPPED at base_lr — a one-time warning fires here in that case.
        """
        raw = self.raw_loss(model)
        if raw is None:
            return None
        if not self._recalibrated and not self._warned_uncapped:
            import warnings
            warnings.warn(
                "HebbianPlasticity.compute_loss() is producing an ACTIVE loss "
                "term but recalibrate() has never been called, so the "
                "grad-fraction cap is not engaged and the term runs UNCAPPED at "
                f"base_lr={self.base_lr}. Wire recalibrate() into the training "
                "loop (see experiments/) before trusting this path. NOTE the "
                "current ablation shows this term HURTS val PPL.",
                RuntimeWarning, stacklevel=2,
            )
            self._warned_uncapped = True
        alpha_eff = self.base_lr * self._scale
        self._last_stats["alpha_eff"] = alpha_eff
        return alpha_eff * raw

    def raw_loss(self, model: "MTLNNModel") -> Optional[torch.Tensor]:
        """Unscaled Hebbian loss = -signal (minimising it MAXIMISES co-activation,
        the Hebb rule). In-graph scalar, or None when no block stashed sequences.
        Carries no base_lr / _scale so its gradient norm is the 'raw' quantity
        recalibrate() aligns against the main gradient."""
        signal = self.compute_signal(model)
        if signal is None:
            return None
        return -signal

    @torch.no_grad()
    def recalibrate(self, g_main_norm: float, g_hebb_raw_norm: float) -> float:
        """Update _scale so that  alpha_eff * g_hebb_raw <= cap * g_main, i.e. the
        Hebbian gradient never exceeds grad_frac_cap of the main gradient norm.

            alpha_eff = base_lr * _scale ,  with
            _scale = min(1, cap * g_main / (base_lr * g_hebb_raw))

        Decoupled from the main BP lr (only base_lr appears) and self-limiting:
        if the raw Hebbian gradient is large relative to the main one, _scale
        shrinks; if it is already small, _scale stays at 1 (alpha_eff=base_lr).
        Called periodically by the training loop (it owns the extra backward
        passes that measure the two norms). Returns the new _scale.
        """
        denom = self.base_lr * g_hebb_raw_norm + self._eps
        scale = min(1.0, self.grad_frac_cap * g_main_norm / denom)
        self._scale = float(scale)
        self._recalibrated = True   # cap is now engaged (silences the warning)
        # projected post-scale gradient fraction (for monitoring / asserting cap)
        self._last_stats["g_main_norm"] = float(g_main_norm)
        self._last_stats["g_hebb_raw_norm"] = float(g_hebb_raw_norm)
        self._last_stats["scale"] = self._scale
        self._last_stats["grad_frac"] = (
            (self.base_lr * self._scale * g_hebb_raw_norm) / (g_main_norm + self._eps)
        )
        return self._scale
