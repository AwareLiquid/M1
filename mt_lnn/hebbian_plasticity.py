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

        # NOTE: no nn.Parameters / submodules are created in Stage 0. The
        # dedicated LAVI estimator (with its learnable bias + temperature) and
        # the gradient-alignment state are introduced in Stage 1 / Stage 2. This
        # keeps the opt-in path param-free and forward-identical for now.

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
        # Stage 1 will: pull per-block hidden states, run the dedicated LAVI
        # estimator to get a non-trivial gate, and form the within-sequence
        # lagged co-activation signal.
        # Stage 2 will: build the equivalent loss, measure its raw gradient norm
        # vs the main gradient norm, and scale so the Hebbian share <= cap.
        return None
