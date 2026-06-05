"""
rhythm.py — EEG-inspired rhythmicity gate for MT-LNN.

Biological grounding
--------------------
EEG research identifies two complementary oscillatory modes in cortical circuits:
  Persistent mode  (high rhythmicity) — sustained oscillations, stable τ, context
                                         maintenance. Analogous to theta/alpha bands.
  Transient mode   (low rhythmicity)  — burst dynamics, fast τ, rapid context
                                         switching. Analogous to gamma bursts.

The LAVI (Lag Angle Vector Index) approximation used here measures how much
the current input resembles the previous recurrent state. High cosine similarity
→ we are in a persistent regime; low similarity → novel/transient regime.

This is complementary to the existing κ-gate (dynamic_scale_gates), which is
content-based ("what is the input now?"). LAVI is history-based ("how stable
has the state been?").

Integration points
------------------
  LAVIEstimator         — attached per MTLNNLayer when use_rhythm=True
  GlobalRhythmController — attached to MTLNNModel when global_rhythm=True

Both are zero-impact by default (use_rhythm=False in MTLNNConfig). The rhythm
influence starts at 0 and the model learns its strength during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LAVIEstimator(nn.Module):
    """Per-layer rhythmicity estimator (LAVI proxy via cosine similarity).

    Computes, for each protofilament at each position, how much the current
    input matches the previous recurrent state. This is a single-step
    approximation of the autocorrelation-based LAVI index.

    High output (→ 1): input resembles prior state → persistent/stable mode
    Low output  (→ 0): input is novel relative to state → transient/burst mode

    Shape contract
    --------------
    h_prev  : (B, P, S, D) recurrent state from the previous step, or None
    x_split : (B, T, P, D) current input projected into protofilament space
    output  : (B, T, P, 1) ∈ [0, 1], high = persistent, low = transient
    """

    def __init__(self, d_proto: int, n_protofilaments: int):
        super().__init__()
        self.P = n_protofilaments
        # Per-protofilament bias — shifts the sigmoid decision boundary.
        # Initialised to 0: LAVI starts at 0.5 (neutral) for cosine_sim = 0.
        self.bias = nn.Parameter(torch.zeros(n_protofilaments))

        # World-model integration gate (Phase C linkage).
        # When PredictiveStateHead.last_pred_error is passed as pred_error,
        # this scalar scales its influence on LAVI.
        # Init = 0.0 → no effect at start; learned to be useful if world_model
        # is also enabled.  Always zero when pred_error=0.0 (default).
        self.pred_error_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        h_prev: torch.Tensor,
        x_split: torch.Tensor,
        pred_error: float = 0.0,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_prev     : (B, P, S, D) or None
        x_split    : (B, T, P, D)
        pred_error : float — world model prediction error from the previous step.
                     High error → model surprised → LAVI nudged toward transient
                     (lower). Requires use_world_model=True in config; defaults
                     to 0.0 (no effect) otherwise.
        """
        B, T, P, D = x_split.shape

        if h_prev is None:
            # No prior state: neutral rhythmicity, no bias toward either mode.
            return torch.full(
                (B, T, P, 1), 0.5, device=x_split.device, dtype=x_split.dtype
            )

        # Collapse scale dimension (B, P, S, D) → (B, P, D)
        if h_prev.dim() == 4:
            h_ref = h_prev.mean(dim=2)
        else:
            h_ref = h_prev  # legacy (B, P, D)

        # Build reference sequence aligned to current T positions:
        #   t=0       → uses h_ref (state from before this chunk)
        #   t=1..T-1  → uses x_split[:,t-1,:,:] (within-chunk autoregression)
        prev = torch.cat(
            [h_ref.unsqueeze(1), x_split[:, :-1, :, :]],
            dim=1,
        )  # (B, T, P, D)

        sim = F.cosine_similarity(prev, x_split, dim=-1)  # (B, T, P)

        # World-model correction: high pred_error nudges LAVI downward (transient).
        # tanh(pred_error_scale) ∈ (-1,1); pred_error ≥ 0 → correction ≥ 0.
        # When pred_error=0 or pred_error_scale=0 → no effect.
        wm_correction = torch.tanh(self.pred_error_scale) * float(pred_error)

        lavi = torch.sigmoid(sim + self.bias.view(1, 1, P) - wm_correction)
        return lavi.unsqueeze(-1)  # (B, T, P, 1)


class GlobalRhythmController(nn.Module):
    """Cross-layer global rhythm controller.

    After all MTLNNBlocks have run, reads the per-layer mean LAVI values stored
    in MTLNNLayer.last_lavi_mean buffers, aggregates them into a global
    rhythmicity index, and applies a small learned correction to the residual
    stream before GWTB/coherence layers.

    The correction starts at 0 (self.scale is initialised to 0) so the layer
    begins as identity and only grows when the rhythm signal proves useful
    during training.

    Positioning in MTLNNModel.forward():
        block loop → [GlobalRhythmController here] → GWTB → coherence → lm_head
    """

    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        self.n_layers = n_layers

        # Learnable softmax weights for aggregating per-layer LAVI
        self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)

        # Project global LAVI scalar → d_model correction vector.
        # Small hidden dim (d_model//4) keeps parameter count negligible.
        self.bias_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, d_model),
        )

        # Scale gate: tanh(scale) ∈ (-1, 1).
        # Initialised 0 → correction starts at 0 (identity at step 0).
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        layer_lavi_means: torch.Tensor,   # (n_layers,)
        x: torch.Tensor,                   # (B, T, d_model)
    ):
        """
        Returns
        -------
        x_corrected : (B, T, d_model)
        global_lavi : scalar tensor in [0, 1]
        """
        weights = F.softmax(self.layer_weights, dim=0)
        global_lavi = (weights * layer_lavi_means).sum()   # scalar in [0, 1]

        # Map scalar → d_model correction, gated by learned scale
        correction = self.bias_proj(global_lavi.view(1, 1))   # (1, 1, d_model)
        x_corrected = x + torch.tanh(self.scale) * correction
        return x_corrected, global_lavi
