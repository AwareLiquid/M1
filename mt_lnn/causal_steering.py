"""
causal_steering.py — inference-time causal activation steering for AwareLiquid.

What it does
------------
A *corrective* counterpart to :class:`~mt_lnn.causality.CausalConsistencyChecker`.
The checker only *detects* when the LNN recurrent-state trajectory makes an
abrupt jump (a likely causal-inconsistency / hallucination boundary). This
module *acts* on that signal: when a break is detected it nudges the drifting
hidden state back onto the "legal" causal subspace — the directions the recent
trajectory was actually moving along — before the model continues generating.

Research grounding (STARS)
--------------------------
Inspired by STARS — "Exploring Diverse Generation Paths via Inference-time
Stiefel Activation Steering" (ICLR 2026 poster). STARS constrains activation
steering directions to a **Stiefel manifold** (orthonormal frames) so the edit
stays on a well-conditioned, low-distortion path rather than blowing up the
residual stream. We reuse exactly that geometry: the principal subspace basis
``Vk`` returned by :meth:`CausalConsistencyChecker.principal_subspace` is a set
of *orthonormal* rows — a point on the Stiefel manifold — and steering is the
**orthogonal projection** of the activation onto its span. The component of the
state that lies *outside* the legal subspace is the "illegal" novelty; removing
(a fraction of) it pulls the trajectory back onto consistent ground without
introducing any new, arbitrary direction.

Decoupling (注意耦合)
---------------------
* Pure-Python, **zero learnable parameters** — like the checker, this is an
  inference-time signal *actuator*, not a layer. It never imports the backbone
  (``model.py``), so the dependency graph stays one-directional.
* It does **not** re-run its own SVD. It asks the checker for the subspace via
  the public :meth:`principal_subspace` method, so detector and actuator can
  never disagree about what "consistent" means.
* Fully **opt-in**: nothing calls it unless explicitly wired (see
  ``SpatialReasoner(..., steerer=...)``). Default behaviour is unchanged.

Typical use (in a generation / reasoning loop)
----------------------------------------------
    checker = CausalConsistencyChecker(method="subspace")
    steerer = CausalActivationSteerer(strength=1.0, floor=0.3)

    h = last_lnn_state                      # (B, …, D) recurrent state
    checker.update(h)                       # detector ingests the step
    res = steerer.steer(h, checker)         # actuator (no-op if score ≥ floor)
    if res.applied:
        h = res.steered                     # feed the corrected state onward
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .causality import CausalConsistencyChecker


@dataclass
class SteerResult:
    """Outcome of a single :meth:`CausalActivationSteerer.steer` call.

    Attributes
    ----------
    steered : the (possibly unchanged) hidden state, same shape/dtype as input.
    applied : True iff a correction was actually applied this step.
    score : the consistency score that gated the decision (∈ [0, 1]).
    gain : the fraction of the off-subspace residual that was removed (∈ [0, 1]).
    correction_norm : L2 norm of the applied correction vector (0.0 if no-op).
    subspace_dim : dimensionality k of the legal subspace used (0 if no-op).
    reason : short human-readable explanation of the decision.
    """

    steered: torch.Tensor
    applied: bool
    score: float
    gain: float
    correction_norm: float
    subspace_dim: int
    reason: str


class CausalActivationSteerer:
    """Project a drifting hidden state back onto the legal causal subspace.

    Parameters
    ----------
    strength : float
        Maximum fraction of the off-subspace ("illegal") residual to remove.
        ``1.0`` fully projects the activation onto the legal subspace; ``0.0``
        disables steering. Values in between leave a controllable amount of
        novelty so steering corrects rather than flattens.
    floor : float
        Consistency-score gate. Steering only fires when the checker's score is
        **below** this floor (i.e. a break was detected). Should match the
        router's ``consistency_floor`` so detection, routing and steering share
        one threshold.
    adaptive : bool
        When True, the gain scales with how far the score fell below ``floor``:
        a shallow dip applies a gentle nudge, a deep break applies up to the
        full ``strength``. When False, a fixed ``strength`` is applied whenever
        the gate trips.
    energy_keep : Optional[float]
        Override for the subspace size requested from the checker (fraction of
        window variance the legal subspace must capture). ``None`` uses the
        checker's own setting, keeping detector and actuator in lock-step.

    Notes
    -----
    The steerer is stateless across calls (it holds no history of its own); all
    trajectory memory lives in the checker. This keeps the two objects cleanly
    separated and makes the steerer trivially safe to share or reset.
    """

    def __init__(
        self,
        strength: float = 1.0,
        floor: float = 0.3,
        adaptive: bool = True,
        energy_keep: Optional[float] = None,
    ):
        if not (0.0 <= strength <= 1.0):
            raise ValueError(f"strength must be in [0, 1], got {strength}")
        if not (0.0 <= floor <= 1.0):
            raise ValueError(f"floor must be in [0, 1], got {floor}")
        if energy_keep is not None and not (0.0 < energy_keep <= 1.0):
            raise ValueError(f"energy_keep must be in (0, 1], got {energy_keep}")
        self.strength = float(strength)
        self.floor = float(floor)
        self.adaptive = bool(adaptive)
        self.energy_keep = energy_keep

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config, **overrides) -> "CausalActivationSteerer":
        """Build a steerer from a config's ``causal_steer_*`` fields.

        Mirrors :meth:`CausalConsistencyChecker.from_config`. The ``floor``
        defaults to the same ``causal_check_threshold`` the checker/router use,
        so a single config keeps all three in agreement. Keys in ``overrides``
        win over config values.
        """
        kwargs = dict(
            strength=getattr(config, "causal_steer_strength", 1.0),
            floor=getattr(config, "causal_check_threshold", 0.3),
            adaptive=getattr(config, "causal_steer_adaptive", True),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def steer(
        self,
        h: torch.Tensor,
        checker: CausalConsistencyChecker,
        score: Optional[float] = None,
        *,
        exclude_last: bool = False,
    ) -> SteerResult:
        """Steer ``h`` toward the legal causal subspace if a break is detected.

        Parameters
        ----------
        h : the hidden state to (possibly) correct, any shape. It is flattened
            the same way the checker flattens states, so the subspace basis and
            the activation live in the same coordinates.
        checker : the *same* :class:`CausalConsistencyChecker` that has been
            ingesting the trajectory via ``update()``. Its history defines the
            legal subspace; its score gates the decision.
        score : optional explicit consistency score to gate on. Defaults to the
            checker's current EMA score. Pass this if you read the score once
            and want the steerer to reuse the exact same value.
        exclude_last : forwarded to :meth:`CausalConsistencyChecker.principal_subspace`.
            Set True in the canonical ``update(h); steer(h)`` order so the legal
            subspace is built from the trajectory *before* ``h`` — otherwise the
            just-ingested suspect state pollutes its own reference subspace and
            the off-subspace residual (hence the correction) collapses to ~0.

        Returns
        -------
        :class:`SteerResult` — always carries a usable ``steered`` tensor (the
        original when no correction was applied) plus diagnostics.
        """
        score = checker.consistency_score() if score is None else float(score)

        # Gate: only act on a detected break (score below the floor).
        if score >= self.floor:
            return SteerResult(
                steered=h, applied=False, score=score, gain=0.0,
                correction_norm=0.0, subspace_dim=0,
                reason=f"consistent (score {score:.3f} >= floor {self.floor:.3f})",
            )

        basis = checker.principal_subspace(self.energy_keep, exclude_last=exclude_last)
        if basis is None:
            return SteerResult(
                steered=h, applied=False, score=score, gain=0.0,
                correction_norm=0.0, subspace_dim=0,
                reason="no subspace (insufficient/degenerate history)",
            )
        Vk, mean = basis                                   # (k, D), (D,)

        orig_shape = h.shape
        orig_dtype = h.dtype
        h_flat = h.detach().float().reshape(-1)            # (D,)
        if Vk.shape[1] != h_flat.numel():
            # State dimensionality changed vs. the window → cannot project.
            return SteerResult(
                steered=h, applied=False, score=score, gain=0.0,
                correction_norm=0.0, subspace_dim=int(Vk.shape[0]),
                reason=(
                    f"shape mismatch (state D={h_flat.numel()} != "
                    f"subspace D={Vk.shape[1]})"
                ),
            )

        # Match device (history may live on a different device than h).
        Vk = Vk.to(h_flat.device)
        mean = mean.to(h_flat.device)

        hc = h_flat - mean                                 # center
        coeffs = Vk @ hc                                   # (k,) subspace coords
        recon = Vk.t() @ coeffs                            # in-subspace component
        residual = hc - recon                              # off-subspace (illegal)

        # Gain: how much of the illegal residual to remove this step.
        if self.adaptive:
            # Linearly ramp from 0 at the floor to `strength` at score 0.
            depth = 1.0 - (score / self.floor) if self.floor > 0 else 1.0
            gain = self.strength * max(0.0, min(1.0, depth))
        else:
            gain = self.strength

        correction = gain * residual                       # vector we subtract
        h_new_flat = h_flat - correction                   # = mean+recon+(1-gain)*residual
        steered = h_new_flat.reshape(orig_shape).to(orig_dtype)

        return SteerResult(
            steered=steered,
            applied=gain > 0.0,
            score=score,
            gain=float(gain),
            correction_norm=float(correction.norm().item()),
            subspace_dim=int(Vk.shape[0]),
            reason=(
                f"break (score {score:.3f} < floor {self.floor:.3f}); "
                f"removed {gain:.2f} of off-subspace residual"
            ),
        )

    def __repr__(self) -> str:
        return (
            f"CausalActivationSteerer("
            f"strength={self.strength}, floor={self.floor}, "
            f"adaptive={self.adaptive})"
        )
