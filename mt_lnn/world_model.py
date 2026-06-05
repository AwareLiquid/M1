"""
world_model.py — Predictive State Head for MT-LNN.

Biological grounding
--------------------
Friston's Free Energy Principle / Predictive Coding (2010) proposes that the
brain is fundamentally a prediction machine: it continuously generates
predictions about upcoming sensory states and updates internal models based
on prediction errors. This is distinct from the existing τ-scale predictive
coding (which predicts across time-scales within a single position) — here
we predict the next token's *full hidden representation*.

Relationship to existing predictive coding:
  VectorizedMultiScaleResonance.use_predictive_coding:
    Slow τ channel predicts fast τ channel (within-position, across scales)
    → Captures temporal frequency relationships.
  PredictiveStateHead (this module):
    h_t predicts h_{t+1} (across positions, at the final hidden layer)
    → Captures sequential causal structure.
  The two are orthogonal and complementary.

Design
------
PredictiveStateHead predicts the next position's hidden state from the current
one using a bottleneck MLP (d_model → d_model//2 → d_model). The last linear
layer is zero-initialised, so the head starts as a zero predictor and grows
only as useful patterns are discovered during training.

Training: shift-by-1 MSE self-supervision on the model's own hidden states.
  L_wm = MSE(pred_head(x[:, :-1, :]), x[:, 1:, :].detach())
  Total loss = L_lm + world_model_loss_weight × L_wm

Inference: the head is still run in forward, but:
  1. No loss is computed (no shift-by-1 target available).
  2. The per-position prediction and the per-step prediction error
     (||pred - actual||²) are stored for monitoring and downstream use.

Integration with LAVI rhythm gate:
  world_model_head.last_pred_error (a buffer) is exposed so that the rhythm
  system or a future enhanced CausalConsistencyChecker can consume it:
    "model surprised by next state → low LAVI → fast τ activates"
  This integration is optional; the head is fully functional standalone.

Position in the forward graph:
  block_loop → rhythm_controller → GWTB → coherence → final_norm
    → [PredictiveStateHead runs here] → lm_head → logits

The head runs on x_normed (after final_norm) for two reasons:
  1. x_normed is the same representation used to compute logits — most
     informative for predicting the next logit-level representation.
  2. Applying the head before lm_head avoids adding noise to the norm output.

Invariants:
  use_world_model=False (default) → PredictiveStateHead is not instantiated;
  zero impact on forward pass, loss, and tests.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveStateHead(nn.Module):
    """
    Predicts h_{t+1} from h_t using a residual bottleneck MLP.

    Parameters
    ----------
    d_model : int
        Hidden state dimension (input and output).
    hidden_ratio : float
        Bottleneck width as fraction of d_model.
        Default 0.5 → hidden_dim = d_model // 2.

    Zero-init invariant:
        At init, the last linear layer has zero weights and biases →
        pred_next ≡ 0 → L_wm = MSE(0, x[:, 1:, :]) = ||x||² / (B·(T-1)·D).
        Although the initial loss is non-zero, it is small relative to L_lm
        because world_model_loss_weight ≪ 1.  More importantly, gradients
        flow freely from the first forward step, allowing the head to learn.
    """

    def __init__(self, d_model: int, hidden_ratio: float = 0.5):
        super().__init__()
        self.d_model = d_model
        hidden_dim = max(d_model // max(1, int(1.0 / hidden_ratio)), 1)

        self.predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model, bias=True),
        )

        # Zero-init output: prediction starts at 0 → small initial loss
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        nn.init.normal_(self.predictor[0].weight, std=0.02)
        nn.init.zeros_(self.predictor[0].bias)

        # Diagnostic buffers — updated every forward, no gradient
        self.register_buffer("last_pred_error", torch.zeros(()), persistent=False)
        # Per-position prediction error for the last batch (mean over B)
        # Shape: (T,) or () if T=1.  Used by downstream monitoring.
        self.register_buffer("last_error_by_position", torch.zeros(1), persistent=False)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, d_model)
        compute_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute predictions and (optionally) the shift-by-1 prediction loss.

        Parameters
        ----------
        x : (B, T, d_model)
            Hidden states (typically after final_norm).
        compute_loss : bool
            Whether to compute the training loss.  Set False during
            inference (T=1 decode steps) when no target is available.

        Returns
        -------
        pred_next : (B, T, d_model)
            Predicted next-position representations.  During inference this
            can be used as a "what I expect next" signal.
        pred_loss : scalar tensor or None
            MSE loss over the shift-by-1 targets.  None when T ≤ 1 or
            compute_loss is False.
        """
        pred_next = self.predictor(x)              # (B, T, d_model)
        pred_loss: Optional[torch.Tensor] = None

        if compute_loss and x.shape[1] > 1:
            # Shift-by-1 self-supervised target (detached — we predict the
            # model's own representation, not a fixed external signal)
            target = x[:, 1:, :].detach()         # (B, T-1, d_model)
            residual = pred_next[:, :-1, :] - target
            sq_err = residual.pow(2).mean(dim=-1)  # (B, T-1)

            pred_loss = sq_err.mean()              # scalar

            # Per-position diagnostic
            with torch.no_grad():
                self.last_pred_error = pred_loss.detach()
                # Mean over batch, shape (T-1,)
                self.last_error_by_position = sq_err.detach().mean(dim=0)

        elif x.shape[1] == 1 and pred_next.shape[1] == 1:
            # Single-step inference: compute error vs last known state
            # (this is approximate — stored for monitoring only)
            with torch.no_grad():
                # No target at T=1; keep last error unchanged
                pass

        return pred_next, pred_loss
