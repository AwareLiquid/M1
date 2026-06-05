"""
causality.py — Causal Consistency Checker for AwareLiquid.

Biological grounding
--------------------
The prefrontal cortex performs continuous error monitoring: when the brain's
predicted state diverges sharply from the actual incoming state, it generates
an error signal that triggers re-evaluation (Friston's "precision-weighted
prediction error"). This module operationalises that mechanism for MT-LNN.

Design
------
CausalConsistencyChecker is a *stateful, pure-Python* object (not nn.Module).
No learnable parameters — it is a signal detector that runs at inference time
alongside the model's forward pass.

Algorithm:
  1. At each step, receive h_prev (the LNN recurrent state, any shape).
  2. Mean-pool h_prev to a 1D representative vector per batch.
  3. Compare current vector to the mean of the recent history window via
     cosine similarity.
  4. Map cosine similarity from [-1, 1] → [0, 1]  (0 = anti-correlated,
     0.5 = orthogonal, 1 = perfectly aligned).
  5. Apply exponential moving average (EMA) smoothing to filter single-step
     noise and surface sustained trajectory breaks.
  6. Expose score ∈ [0, 1]:  1 = consistent/smooth, 0 = abrupt break.

Integration
-----------
The checker runs alongside DeliberationRouter.decide():

    checker = CausalConsistencyChecker()

    # In the generation loop, after each step:
    checker.update(cache.layers[i][1])   # h_prev from last LNN layer
    score = checker.consistency_score()

    decision = router.decide(
        logits,
        query=query,
        evidence_log=evidence_log,
        consistency_signal=score,         # NEW optional param
    )

When consistency_signal < RouterThresholds.consistency_floor, the router
forces SELF_CRITIQUE regardless of token entropy. This catches cases where
the model's internal state has "drifted" even while producing low-entropy
(confident-looking) tokens — a known pattern in hallucination.

Deliberation.py changes (backward-compatible):
  - RouteDecision: new optional field  causal_consistency
  - RouterThresholds: new optional field  consistency_floor = 0.3
  - DeliberationRouter.decide(): new optional kwarg  consistency_signal = None

Not adopted: Prolog/CaRing symbolic reasoning engine integration. Reason:
10–100 ms query latency violates the <50 ms/token invariant (I2). This
lightweight trajectory checker achieves a similar goal in O(window × D) time,
which is microseconds on CPU.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F


class CausalConsistencyChecker:
    """
    Monitors the smoothness of the LNN recurrent-state trajectory.

    A high score (→ 1) means the model's internal state is evolving
    smoothly — consistent with continuous causal reasoning.

    A low score (→ 0) means the trajectory has made an abrupt jump —
    a potential causal inconsistency or hallucination boundary.

    Parameters
    ----------
    window : int
        Number of recent h_prev snapshots to keep in the history buffer.
        Larger window → more stable reference mean, slower to adapt.
    ema_alpha : float
        Weight of the current step's raw similarity in the EMA update.
        Higher → more reactive to sudden jumps.
    threshold : float
        Score below which `is_consistent` returns False.
        Should match RouterThresholds.consistency_floor.
    """

    def __init__(
        self,
        window: int = 8,
        ema_alpha: float = 0.4,
        threshold: float = 0.3,
    ):
        self.window = max(1, window)
        self.ema_alpha = float(ema_alpha)
        self.threshold = float(threshold)

        # Circular buffer of recent (normalised) h vectors — deque for O(1) append/pop
        self._history: deque = deque(maxlen=self.window)
        self._ema_score: float = 1.0   # start fully consistent
        self._step: int = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, h: torch.Tensor) -> float:
        """
        Ingest one step's hidden state and return the updated consistency score.

        h : any shape — mean-pooled over all dimensions to a 1-D vector.
            Typically (B, P, S, D) from MTLNNLayer.h_last_per_scale or
            (B, d_model) from a flat residual stream.

        Returns float ∈ [0, 1].
        """
        # Flatten and mean over batch/spatial dims → (D,)
        h_flat = h.detach().float().reshape(-1).clone()
        if h_flat.numel() == 0:
            return self._ema_score

        h_norm = F.normalize(h_flat.unsqueeze(0), dim=-1, eps=1e-8).squeeze(0)

        if len(self._history) > 0:
            # Reference: mean of the recent window (raw, then normalised)
            hist_stack = torch.stack(list(self._history), dim=0)   # (N, D)
            mean_ref = hist_stack.mean(dim=0)
            mean_norm = F.normalize(mean_ref.unsqueeze(0), dim=-1, eps=1e-8).squeeze(0)

            # Cosine similarity ∈ [-1, 1] → mapped to [0, 1]
            raw_sim = torch.dot(h_norm, mean_norm).item()
            sim_01 = (raw_sim + 1.0) / 2.0        # 0 = anti-correlated, 1 = aligned
            sim_01 = max(0.0, min(1.0, sim_01))   # numerical clamp

            # EMA smoothing: alpha weights the new observation
            self._ema_score = (
                self.ema_alpha * sim_01
                + (1.0 - self.ema_alpha) * self._ema_score
            )

        # Append raw (not normalised) vector to history for a stable mean
        self._history.append(h_flat.clone())
        self._step += 1
        return self._ema_score

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    def consistency_score(self) -> float:
        """Current EMA consistency score ∈ [0, 1]."""
        return self._ema_score

    @property
    def is_consistent(self) -> bool:
        """True when score ≥ threshold (trajectory is smooth)."""
        return self._ema_score >= self.threshold

    @property
    def steps_seen(self) -> int:
        """Number of update() calls since last reset."""
        return self._step

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear history and restore score to 1.0 (e.g. at session start)."""
        self._history.clear()
        self._ema_score = 1.0
        self._step = 0

    def __repr__(self) -> str:
        return (
            f"CausalConsistencyChecker("
            f"score={self._ema_score:.3f}, "
            f"steps={self._step}, "
            f"window={self.window})"
        )
