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

Design (v2.1 — EMA target encoder, BYOL / V-JEPA style)
-------------------------------------------------------
The naive version predicted the model's own *raw* next state with MSE:
    L = MSE(predictor(x[:, :-1]), x[:, 1:].detach())
This is prone to *representational collapse*: the cheapest way to lower the
loss is for the encoder to make all hidden states similar, so the predictor
trivially "predicts" a constant. BYOL (arxiv 2006.07733) and V-JEPA
(arxiv 2404.08471) solve this with an asymmetric online/target pair:

  online path :  z_pred = predictor( online_proj( x_t ) )
  target path :  z_tgt  = target_proj( x_{t+1} ).detach()        (stop-grad)

  target_proj is NOT trained by backprop — it is an exponential moving
  average (EMA) of online_proj:
        target_proj ← decay · target_proj + (1 - decay) · online_proj

The predictor (present only on the online side) breaks the symmetry, and the
slow-moving stop-grad target prevents the trivial constant solution. This is
the precise asymmetry the v2.0 audit (V2_REVIEW.md) found missing.

Loss is computed in a normalised latent space so it is scale-free:
    z_pred, z_tgt are L2-normalised
    L_wm        = mean ||z_pred - z_tgt||²  ∈ [0, 4]
    pred_error  = (1 - cos(z_pred, z_tgt)) / 2  ∈ [0, 1]   ← surprise signal

The normalised `pred_error` (NOT the raw MSE) is what the LAVI rhythm gate
consumes, so the world-model → LAVI coupling is bounded and stable regardless
of representation magnitude.

Inference: the head still runs in forward, but no loss is computed (no
shift-by-1 target at T=1 decode steps). The per-step normalised prediction
error is still updated for monitoring and the LAVI linkage.

Integration with LAVI rhythm gate:
  last_pred_error (a buffer ∈ [0,1]) is exposed so the rhythm system can read:
    "model surprised by next state → high pred_error → LAVI nudged transient"

Position in the forward graph:
  block_loop → rhythm_controller → GWTB → coherence → final_norm
    → [PredictiveStateHead runs here] → lm_head → logits

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
    Predicts the next position's latent representation from the current one
    using an asymmetric online-predictor / EMA-target pair (BYOL / V-JEPA).

    Parameters
    ----------
    d_model : int
        Hidden state dimension (the head's input).
    hidden_ratio : float
        Predictor bottleneck width as a fraction of the projection dim.
        Default 0.5 → predictor hidden = proj_dim // 2.
    proj_ratio : float
        Latent projection width as a fraction of d_model.
        Default 0.5 → proj_dim = d_model // 2 (cheaper, lower consumption).
    ema_decay : float
        EMA momentum for the target projector. Higher = slower target.
        Default 0.99 (standard BYOL range 0.99–0.999).
    use_ema_target : bool
        If True (default), use the EMA target encoder (recommended — prevents
        collapse). If False, fall back to a *trainable* target projector with
        stop-grad on its output (a weaker baseline, kept for ablation).
    warmup_steps : int
        Number of initial EMA updates that use a gentler decay (0.9) before
        switching to `ema_decay`. Lets the target track the fast-moving online
        projector early in training. Set 0 to disable warm-up.

    Notes
    -----
    The online and target projectors are L2-normalised before the loss, so
    the loss is invariant to representation scale.  The exposed
    ``last_pred_error`` buffer is the *normalised* surprise ∈ [0, 1] — this is
    what the LAVI rhythm gate consumes.  ``last_pred_error_raw`` keeps the raw
    latent MSE for diagnostics.
    """

    def __init__(
        self,
        d_model: int,
        hidden_ratio: float = 0.5,
        proj_ratio: float = 0.5,
        ema_decay: float = 0.99,
        use_ema_target: bool = True,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.ema_decay = float(ema_decay)
        self.use_ema_target = bool(use_ema_target)

        proj_dim = max(int(d_model * proj_ratio), 1)
        hidden_dim = max(int(proj_dim * hidden_ratio), 1)
        self.proj_dim = proj_dim

        # Online projector: x_t → latent.  Trained by backprop.
        # Small-scale init (NOT zero — a zero latent has an undefined direction
        # after L2 normalisation).
        self.online_proj = nn.Linear(d_model, proj_dim, bias=False)
        nn.init.normal_(self.online_proj.weight, std=0.02)

        # Predictor: online latent → predicted *next* latent.  This online-only
        # head is the asymmetry that breaks symmetry and prevents collapse.
        # Output layer kept small so the initial prediction is near the
        # projector output (not exactly zero — see online_proj note above).
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim, bias=True),
        )
        nn.init.normal_(self.predictor[0].weight, std=0.02)
        nn.init.zeros_(self.predictor[0].bias)
        nn.init.normal_(self.predictor[-1].weight, std=0.02)
        nn.init.zeros_(self.predictor[-1].bias)

        # Target projector: x_{t+1} → target latent.  A frozen EMA copy of
        # online_proj (no gradient flows through it). Created AFTER online_proj
        # is initialised so the two are guaranteed identical at step 0.
        self.target_proj = nn.Linear(d_model, proj_dim, bias=False)
        self.target_proj.weight.data.copy_(self.online_proj.weight.data)
        for p in self.target_proj.parameters():
            p.requires_grad = False

        # EMA warm-up: early in training online_proj moves fast, so a fixed
        # high decay leaves the target lagging too far. Use a gentler decay
        # (0.9) for the first `warmup_steps` updates, then the configured decay.
        self.warmup_steps = int(warmup_steps)
        self._ema_step = 0

        # Diagnostic buffers — updated on TRAINING passes only (T>1 with
        # compute_loss), no gradient.
        #   last_pred_error      : normalised surprise ∈ [0, 1]  (LAVI input)
        #   last_pred_error_raw  : raw latent MSE magnitude       (diagnostics)
        # persistent=True: with persistent=False a checkpointed model spawned in
        # a fresh INFERENCE process fed the LAVI gate a hardcoded 0 forever (the
        # buffer is never written at inference). Old checkpoints without these
        # keys still load under strict=True via _load_from_state_dict below.
        self.register_buffer("last_pred_error", torch.zeros(()), persistent=True)
        self.register_buffer("last_pred_error_raw", torch.zeros(()), persistent=True)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Backward compat: checkpoints saved when these buffers were
        # persistent=False have no entries for them — seed the defaults so
        # strict=True loading keeps working.
        for name in ("last_pred_error", "last_pred_error_raw"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def _update_target(self) -> None:
        """EMA update: target_proj ← decay·target_proj + (1-decay)·online_proj.

        Uses a gentler decay (0.9) during the warm-up window so the target can
        keep up with the fast-moving online projector early in training.
        """
        self._ema_step += 1
        d = self.ema_decay if self._ema_step > self.warmup_steps else min(self.ema_decay, 0.9)
        for tgt, src in zip(self.target_proj.parameters(), self.online_proj.parameters()):
            tgt.mul_(d).add_(src.detach(), alpha=1.0 - d)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, d_model)
        compute_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the predicted next-state latent and (optionally) the
        shift-by-1 BYOL-style prediction loss.

        Parameters
        ----------
        x : (B, T, d_model)
            Hidden states (typically after final_norm).
        compute_loss : bool
            Whether to compute the training loss.  Set False during inference
            (T=1 decode steps) when no shift-by-1 target is available.

        Returns
        -------
        z_pred : (B, T, proj_dim)
            Predicted next-position latent (online path). During inference this
            is a "what I expect next" signal.
        pred_loss : scalar tensor or None
            Normalised latent MSE over the shift-by-1 targets.  None when
            T ≤ 1 or compute_loss is False.
        """
        # Online path: project then predict the next latent.
        z_pred = self.predictor(self.online_proj(x))   # (B, T, proj_dim)
        pred_loss: Optional[torch.Tensor] = None

        if compute_loss and x.shape[1] > 1:
            # Refresh the EMA target *before* encoding the target so the
            # target reflects the latest online weights (standard BYOL order
            # is either acceptable; we update pre-encode for determinism).
            if self.use_ema_target:
                self._update_target()
                with torch.no_grad():
                    z_tgt_all = self.target_proj(x)             # (B, T, proj_dim)
            else:
                # Ablation: trainable projector but stop-grad on its output.
                z_tgt_all = self.online_proj(x)

            # Shift-by-1: predict t → t+1.  Target is always detached.
            online = z_pred[:, :-1, :]                          # (B, T-1, P)
            target = z_tgt_all[:, 1:, :].detach()               # (B, T-1, P)

            online_n = F.normalize(online, dim=-1, eps=1e-8)
            target_n = F.normalize(target, dim=-1, eps=1e-8)

            # Normalised MSE = 2·(1 - cos).  Scale-free, in [0, 4].
            residual = online_n - target_n
            pred_loss = residual.pow(2).sum(dim=-1).mean()      # scalar

            with torch.no_grad():
                # Normalised surprise ∈ [0, 1] for the LAVI gate.
                cos = (online_n * target_n).sum(dim=-1).mean()  # ∈ [-1, 1]
                self.last_pred_error = ((1.0 - cos) * 0.5).clamp(0.0, 1.0)
                self.last_pred_error_raw = pred_loss.detach()

        return z_pred, pred_loss
