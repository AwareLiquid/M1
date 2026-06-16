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

STAGED BUILD
  Stage 0 (this commit): scaffold only. Param-free; compute_loss() returns None;
    integration in MTLNNModel is gated so the whole thing is a verified no-op.
  Stage 1: dedicated LAVI estimator + non-trivial gate + within-seq lag term.
  Stage 2: equivalent Hebbian loss added to total loss, gradient alignment + cap.
  Stage 3: ablation/effectiveness validation.
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

    Stage 0 scaffold: holds the decoupled hyper-parameters from config and
    exposes the public surface (`compute_loss`, `last_stats`) that the model and
    the staged tests build against, but performs NO computation yet
    (`compute_loss` returns None). This guarantees that opting in via
    `use_hebbian_refactor=True` is a no-op until Stage 2 wires the real term in.
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

        Stage 0: always returns None (verified no-op). The signature matches the
        legacy HebbianRegularizer.compute_loss so the model integration and the
        finiteness guard (`_aux_or_skip`) work unchanged once Stage 2 lands.
        """
        # Stage 1 (DONE): compute_signal() builds the dedicated-LAVI-gated,
        # within-sequence lagged co-activation signal (see above).
        # Stage 2 (TODO): wrap that signal into the loss term, measure its raw
        # gradient norm vs the main gradient norm, and scale so the Hebbian share
        # <= grad_frac_cap before adding to the total loss.
        return None
