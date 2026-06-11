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

from .utils import unit_cosine_similarity


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
    method : str
        Trajectory-break detector:
          "cosine"   — cosine similarity of the current vector to the window
                       mean, mapped to [0,1]. Simple and cheap, but on
                       anisotropic hidden states (all vectors share a dominant
                       direction) the similarity saturates near 1 and rarely
                       fires on genuine breaks.
          "subspace" — project the current (centered) vector onto the principal
                       subspace of the recent window; the residual energy that
                       falls OUTSIDE that subspace is the novelty. Centering +
                       subspace removal cancels the shared anisotropic direction,
                       so only genuinely new directions register as breaks.
                       Recommended for real hidden states. O(window·D²) SVD on a
                       tiny window → microseconds on CPU.
    energy_keep : float
        (subspace method) Fraction of window variance the principal subspace
        must capture. Higher → stricter subspace, more sensitive to novelty.
    """

    def __init__(
        self,
        window: int = 8,
        ema_alpha: float = 0.4,
        threshold: float = 0.3,
        method: str = "cosine",
        energy_keep: float = 0.9,
    ):
        self.window = max(1, window)
        self.ema_alpha = float(ema_alpha)
        self.threshold = float(threshold)
        if method not in ("cosine", "subspace"):
            raise ValueError(f"method must be 'cosine' or 'subspace', got {method!r}")
        self.method = method
        self.energy_keep = float(energy_keep)

        # Circular buffer of recent (normalised) h vectors — deque for O(1) append/pop
        self._history: deque = deque(maxlen=self.window)
        self._ema_score: float = 1.0   # start fully consistent
        self._last_eff_rank: float = 1.0
        self._step: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config, **overrides) -> "CausalConsistencyChecker":
        """
        Build a checker from an MTLNNConfig's causal_check_* fields.

        Lets a single config drive the whole pipeline. Any keyword in
        `overrides` takes precedence over the config value (e.g. a longer
        window for a specific session).
        """
        kwargs = dict(
            window=getattr(config, "causal_check_window", 8),
            threshold=getattr(config, "causal_check_threshold", 0.3),
            method=getattr(config, "causal_check_method", "cosine"),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

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

        if len(self._history) > 0:
            if self.method == "subspace":
                sim_01 = self._subspace_similarity(h_flat)
            else:
                sim_01 = self._cosine_similarity(h_flat)

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
    # Detectors
    # ------------------------------------------------------------------

    def _cosine_similarity(self, h_flat: torch.Tensor) -> float:
        """Cosine of current vector to the window mean, mapped to [0, 1]."""
        hist_stack = torch.stack(list(self._history), dim=0)   # (N, D)
        mean_ref = hist_stack.mean(dim=0)
        return unit_cosine_similarity(h_flat, mean_ref)

    def principal_subspace(
        self, energy_keep: Optional[float] = None, *, exclude_last: bool = False
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Expose the principal subspace of the recent trajectory window.

        This is the *public, reusable* half of the subspace detector: it returns
        the orthonormal basis that spans the directions the recent hidden states
        actually move along — i.e. the "legal" causal subspace. Downstream
        consumers (notably :class:`~mt_lnn.causal_steering.CausalActivationSteerer`,
        which steers a drifting state back onto this subspace) reuse it instead
        of re-running their own SVD, so the detector and the steerer always agree
        on what "consistent" means.

        Parameters
        ----------
        energy_keep : optional override of the instance ``energy_keep`` — the
            fraction of window variance the returned basis must capture. Higher
            → larger (stricter) subspace.
        exclude_last : when True, drop the **most recently** ingested state from
            the window before computing the basis. This matters when the basis is
            used to *correct* that very state: the canonical
            ``update(h); steer(h)`` order has already appended ``h``, so including
            it would let the suspect break-direction become part of its own
            "legal" subspace (its off-subspace residual collapses toward zero and
            steering removes almost nothing). Excluding it defines the legal
            subspace as the directions the trajectory moved along *before* this
            step — the geometrically correct reference for a corrective
            projection. Needs ≥ 2 states remaining after the drop.

        Returns
        -------
        ``(Vk, mean)`` where ``Vk`` is ``(k, D)`` orthonormal rows spanning the
        principal subspace of the *mean-centered* window, and ``mean`` is the
        ``(D,)`` window mean (the centering offset). Returns ``None`` when there
        is insufficient (< 2) or degenerate history to define a subspace.

        Notes
        -----
        The basis is computed on the **centered** window, so ``mean`` must be
        subtracted before projecting and added back afterwards. This is what
        cancels the shared anisotropic direction that would otherwise dominate.
        """
        keep = self.energy_keep if energy_keep is None else float(energy_keep)
        history = list(self._history)
        if exclude_last:
            history = history[:-1]
        if len(history) < 2:
            return None

        hist_stack = torch.stack(history, dim=0).float()               # (N, D)
        mean = hist_stack.mean(dim=0)                                  # (D,)
        Mc = hist_stack - mean.unsqueeze(0)                            # (N, D)

        try:
            _, s, Vh = torch.linalg.svd(Mc, full_matrices=False)      # Vh: (r, D)
        except Exception:
            return None

        energy = (s ** 2)
        total = energy.sum()
        if total < 1e-12:
            # History collapsed to a point → no meaningful subspace.
            self._last_eff_rank = 1.0
            return None

        # Effective rank (participation ratio) — exposed as a diagnostic.
        self._last_eff_rank = (total ** 2 / (energy ** 2).sum()).item()

        # Keep the top-k directions capturing `keep` of the variance.
        cum = torch.cumsum(energy, dim=0) / total
        k = int((cum < keep).sum().item()) + 1
        k = max(1, min(k, Vh.shape[0]))
        return Vh[:k], mean                                          # (k, D), (D,)

    def _subspace_similarity(self, h_flat: torch.Tensor) -> float:
        """
        Novelty via principal-subspace residual (anisotropy-robust).

        1. Center the window and the current vector by the window mean
           (removes the shared/anisotropic direction).
        2. SVD the centered window; keep the top-k right singular vectors that
           capture `energy_keep` of the variance (via :meth:`principal_subspace`).
        3. Project the centered current vector onto that subspace; the residual
           energy outside it is the novelty ∈ [0, 1].
        4. consistency = 1 - novelty   (1 = lies in history subspace = smooth).

        Falls back to cosine when the window has < 2 samples or is degenerate.
        """
        if len(self._history) < 2:
            return self._cosine_similarity(h_flat)

        # Center on the window mean first. If the current vector *equals* that
        # mean, the trajectory hasn't moved at all → maximally consistent. This
        # must be checked BEFORE the subspace branch so a fully colinear /
        # collapsed window (no usable subspace) still reads as stable rather
        # than as a break.
        hist_stack = torch.stack(list(self._history), dim=0).float()   # (N, D)
        mean = hist_stack.mean(dim=0)                                  # (D,)
        hc = h_flat - mean                                            # (D,)
        hc_norm = hc.norm()
        if hc_norm < 1e-8:
            self._last_eff_rank = 1.0
            return 1.0

        basis = self.principal_subspace()
        if basis is None:
            # Window collapsed to a point but the current vector moved off it →
            # a genuinely novel direction with no legal subspace to support it.
            return 0.0
        Vk, _mean = basis

        # Reconstruct hc inside the subspace; residual is the novel component.
        coeffs = Vk @ hc                                             # (k,)
        recon = Vk.t() @ coeffs                                      # (D,)
        residual = hc - recon
        novelty = (residual.norm() / (hc_norm + 1e-8)).clamp(0.0, 1.0).item()
        return max(0.0, 1.0 - novelty)

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

    @property
    def effective_rank(self) -> float:
        """
        Participation ratio of the most recent window (subspace method only).

        ≈ 1 → trajectory collapsed to a line (highly correlated states).
        Higher → the recent window spans more independent directions, which
        rises sharply right after an abrupt topic switch introduces a new one.
        Always 1.0 under the cosine method (not computed).
        """
        return self._last_eff_rank

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear history and restore score to 1.0 (e.g. at session start)."""
        self._history.clear()
        self._ema_score = 1.0
        self._last_eff_rank = 1.0
        self._step = 0

    def __repr__(self) -> str:
        return (
            f"CausalConsistencyChecker("
            f"method={self.method}, "
            f"score={self._ema_score:.3f}, "
            f"steps={self._step}, "
            f"window={self.window})"
        )
