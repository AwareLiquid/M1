"""
mt_lnn/predictive_coding.py — hierarchical predictive-coding loop for MT-LNN.

Why this module exists
----------------------
The brain is a layered prediction machine: each cortical level sends a
*top-down prediction* of the level below, the level below returns the
*prediction error* (what the prediction missed), and only that error propagates
upward. Higher levels predict, lower levels correct — a bidirectional loop, not
a feedforward stack (Rao & Ballard 1999; Friston 2010; Bastos et al. 2012).

MT-LNN already has a *single-head* next-state predictor
(:class:`mt_lnn.world_model.PredictiveStateHead`: ``h_t`` predicts ``h_{t+1}``
at the final layer). What was missing is the *hierarchical, across-depth*
version: every layer predicting the layer beneath it, producing a **per-layer**
error spectrum. That spectrum is exactly the signal the neuromodulatory hub
wants — large errors at a level mean "the world here is surprising", which
should raise acetylcholine / norepinephrine (see
:class:`mt_lnn.neuromodulation.NeuromodulationController`). This module closes
the predict → error → neuromodulate loop.

What it does (and does NOT) do
------------------------------
* It is an **observer/encoder over a stack of layer activations**: you hand it
  the per-layer hidden states (lowest → highest) and it returns top-down
  predictions, per-layer errors, a precision-weighted coding loss (a free-energy
  proxy a trainer can add), and a bounded aggregate surprise in ``[0, 1]``.
* It does **not** edit the model's forward pass or rewrite the layers'
  computation — that would discretise/centralise the liquid core and is exactly
  the kind of invasive change the project forbids. The per-step error feeds the
  *next* step's neuromodulation (the same way ``PredictiveStateHead`` already
  feeds the LAVI gate), so the loop is real without surgery on the hot path.

Design / coupling
-----------------
* **Zero model coupling.** Imports only ``torch``. It never imports ``model.py``;
  the activations are passed in by the caller (e.g. collected from
  ``MTLNNBlock`` outputs via forward hooks, or returned by a wrapper).
* **Lives outside MTLNNModel**, so the model's global init pass never touches its
  trainable top-down predictors — they keep their own initialisation.
* **Duck-typed surprise bridge.** It exposes ``last_pred_error`` ∈ [0, 1] with
  the same name/semantics as ``PredictiveStateHead``, so it drops straight into
  :func:`mt_lnn.salience_events.world_model_surprise` and the neuromodulation
  hub's ``surprise`` input with no glue.
* **Default-off / zero-regression.** Nothing in MT-LNN constructs it; if you
  never build it, the model is unchanged.

Pinned in tests/test_predictive_coding.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PredictiveCodingResult",
    "HierarchicalPredictiveCoder",
]


@dataclass
class PredictiveCodingResult:
    """Output of one hierarchical predictive-coding pass.

    Attributes
    ----------
    predictions : list of top-down predictions, ``predictions[l]`` predicts level
        ``l`` from level ``l+1`` (length ``n_levels-1``).
    errors : list of residual tensors ``activations[l] - predictions[l]``.
    per_level_error : 1-D tensor of bounded per-level surprise ∈ [0, 1]
        (``(1 - cos)/2`` between each level and its top-down prediction).
    coding_loss : scalar precision-weighted sum of squared errors (free-energy
        proxy; add this to the training loss to train the loop).
    surprise : scalar bounded aggregate surprise ∈ [0, 1] (mean of
        ``per_level_error``); mirrors ``PredictiveStateHead.last_pred_error``.
    """

    predictions: List[torch.Tensor]
    errors: List[torch.Tensor]
    per_level_error: torch.Tensor
    coding_loss: torch.Tensor
    surprise: torch.Tensor


class HierarchicalPredictiveCoder(nn.Module):
    """Top-down prediction + bottom-up error across a stack of layer activations.

    .. note::
        **研究实验组件 / Research prototype — NOT wired into the main path.**
        This class is exported as a standalone research component and is never
        instantiated by ``MTLNNModel`` (training or inference). The live model's
        predictive-coding signal comes directly from each block's resonance
        ``last_pred_error`` (see ``model.py``), bypassing this facade. Treat this
        as a code prototype for offline experiments; it does not run in the
        training or serving loop.


    For ``n_levels`` representations ordered low→high, level ``l+1`` predicts
    level ``l`` through a trainable linear top-down projector; the residual is
    the prediction error. Errors are weighted by a learnable per-level
    *precision* (inverse-variance), the predictive-coding mechanism that lets the
    loop trust reliable levels more — yielding the precision-weighted coding loss
    (a free-energy proxy) and a bounded aggregate surprise for neuromodulation.

    Parameters
    ----------
    dims : per-level hidden widths, lowest → highest. Length ``n_levels >= 2``.
        Pass ``[d_model] * n_layers`` for a uniform stack.
    learn_precision : if True (default) the per-level log-precision is a trainable
        parameter; if False it is a fixed buffer (uniform precision).
    precision_init : initial log-precision value (softplus-parameterised, so the
        effective precision is ``softplus(precision_init)``; 0.0 → ~0.69).
    detach_target : if True (default) the predicted level is detached when forming
        the error/loss, so the coding loss trains the *predictor* to match the
        representations rather than dragging the representations toward the
        predictor (the standard stop-grad-on-target predictive-coding choice).
    """

    def __init__(
        self,
        dims: Sequence[int],
        *,
        learn_precision: bool = True,
        precision_init: float = 0.0,
        detach_target: bool = True,
    ):
        super().__init__()
        dims = list(int(d) for d in dims)
        if len(dims) < 2:
            raise ValueError(f"need at least 2 levels, got dims={dims}")
        if any(d < 1 for d in dims):
            raise ValueError(f"all dims must be >= 1, got {dims}")
        self.dims = dims
        self.n_levels = len(dims)
        self.detach_target = bool(detach_target)

        # Top-down predictors: level l+1 → level l. Small init so the loop starts
        # near-quiet (predictions ~0) and learns structure, not noise.
        self.predictors = nn.ModuleList(
            [nn.Linear(dims[l + 1], dims[l]) for l in range(self.n_levels - 1)]
        )
        for lin in self.predictors:
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

        # Per-level log-precision (n_levels-1 error levels).
        logp = torch.full((self.n_levels - 1,), float(precision_init))
        if learn_precision:
            self.log_precision = nn.Parameter(logp)
        else:
            self.register_buffer("log_precision", logp, persistent=True)

        # Diagnostics (non-persistent): bounded aggregate + per-level surprise.
        self.register_buffer("last_pred_error", torch.zeros(()), persistent=False)
        self.register_buffer(
            "last_per_level_error", torch.zeros(self.n_levels - 1), persistent=False
        )

    def _check(self, activations: Sequence[torch.Tensor]) -> None:
        if len(activations) != self.n_levels:
            raise ValueError(
                f"expected {self.n_levels} activation levels, got {len(activations)}"
            )
        for l, a in enumerate(activations):
            if a.shape[-1] != self.dims[l]:
                raise ValueError(
                    f"level {l} last dim {a.shape[-1]} != dims[{l}]={self.dims[l]}"
                )
            if a.shape[:-1] != activations[0].shape[:-1]:
                raise ValueError(
                    "all levels must share leading (batch, …) dims; "
                    f"level {l} has {tuple(a.shape[:-1])} vs "
                    f"{tuple(activations[0].shape[:-1])}"
                )

    def forward(self, activations: Sequence[torch.Tensor]) -> PredictiveCodingResult:
        """Run one predictive-coding pass over the activation stack (low→high)."""
        self._check(activations)

        predictions: List[torch.Tensor] = []
        errors: List[torch.Tensor] = []
        per_level: List[torch.Tensor] = []
        precision = F.softplus(self.log_precision)             # (n_levels-1,)

        weighted_sq = activations[0].new_zeros(())
        for l in range(self.n_levels - 1):
            target = activations[l]
            top_down = activations[l + 1]
            if self.detach_target:
                target = target.detach()
            pred = self.predictors[l](top_down)                # predict level l
            err = target - pred
            predictions.append(pred)
            errors.append(err)

            # Precision-weighted squared error → free-energy proxy term.
            weighted_sq = weighted_sq + precision[l] * err.pow(2).mean()

            # Bounded per-level surprise = (1 - cos(level, prediction)) / 2 ∈ [0,1].
            tn = F.normalize(target, dim=-1, eps=1e-8)
            pn = F.normalize(pred, dim=-1, eps=1e-8)
            cos = (tn * pn).sum(dim=-1).mean()                 # ∈ [-1, 1]
            per_level.append(((1.0 - cos) * 0.5).clamp(0.0, 1.0))

        per_level_t = torch.stack(per_level)                   # (n_levels-1,)
        surprise = per_level_t.mean()

        with torch.no_grad():
            self.last_per_level_error = per_level_t.detach()
            self.last_pred_error = surprise.detach()

        return PredictiveCodingResult(
            predictions=predictions,
            errors=errors,
            per_level_error=per_level_t,
            coding_loss=weighted_sq,
            surprise=surprise,
        )

    def coding_loss(self, activations: Sequence[torch.Tensor]) -> torch.Tensor:
        """Convenience: the precision-weighted coding loss (free-energy proxy).

        Add to the training objective alongside the LM loss to train the
        top-down predictors (analogous to ``HebbianRegularizer.compute_loss`` /
        the world-model loss). Leaves the model's own forward untouched.
        """
        return self.forward(activations).coding_loss

    def __repr__(self) -> str:
        return (
            f"HierarchicalPredictiveCoder(dims={self.dims}, "
            f"detach_target={self.detach_target})"
        )
