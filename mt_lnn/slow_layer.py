"""
mt_lnn/slow_layer.py -- the slow half of the dual-speed engine.

Why this module exists
----------------------
A dual-speed system is only half-built if the slow layer is never actually run.
The hot loop (``DualSpeedSentry``) does a *one-step* predictive-coding forecast
every tick -- cheap, constant-time, runs forever. That single step is enough to
*notice* a surprise, but not to *reason* about it. When a salient event ignites,
the system should wake a more expensive, more deliberate computation that the hot
loop could never afford every tick. This module is that computation.

:class:`SlowThreatAssessor` is woken **only on ignition**. It rolls the target
forward over a whole horizon with the composable Newtonian operators
(:func:`physics_ops.rollout`) and asks, at every forecast step, "is it inside the
protected zone?" (:func:`spatial_ops.in_ball`). From that forecast it computes a
time-to-breach (ETA), the closest approach, and a discrete threat level with a
recommended posture. This is the deliberation the hot loop's single step cannot
do -- and, true to the dual-speed contract, it is paid for only when a real state
change has earned it.

Design / coupling
-----------------
* Pure composition of the public operator APIs; **zero trainable parameters**,
  not an ``nn.Module``, never imports ``model.py``. Deterministic.
* Stateless across calls: each :meth:`assess` is a function of its inputs.

Honest scope
------------
A ballistic forecast: it assumes the target keeps its current velocity (constant
acceleration field = 0), which is exactly the "what happens if nothing changes"
question -- the right one to ask the instant a manoeuvre is detected, before the
next manoeuvre. It is *not* an adversarial intent model. Pinned in
``tests/test_slow_layer.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .physics_ops import rollout
from .spatial_ops import in_ball

__all__ = ["ThreatAssessment", "SlowThreatAssessor"]


@dataclass(frozen=True)
class ThreatAssessment:
    """The slow layer's verdict on a forecast horizon.

    Attributes
    ----------
    woken_tick : int
        The hot-loop tick at which the slow layer was woken (for the audit trail).
    horizon : int
        Number of ballistic forecast steps examined.
    breaches : bool
        Whether the target enters the protected zone anywhere in the horizon
        (including right now).
    eta_breach : int or None
        Forecast steps until the *first* breach (0 = already inside this tick),
        or ``None`` if it never breaches within the horizon.
    min_range : float
        Closest the target gets to the zone centre over the horizon (m).
    closest_offset : int
        The forecast step offset at which ``min_range`` occurs.
    level : str
        Discrete threat level: ``"CLEAR"`` (no breach in horizon), ``"WATCH"``
        (will breach, but not imminently) or ``"ENGAGE"`` (breach imminent or
        already inside).
    recommendation : str
        A short, ASCII, human-readable posture for the operator / actuator layer.
    """

    woken_tick: int
    horizon: int
    breaches: bool
    eta_breach: Optional[int]
    min_range: float
    closest_offset: int
    level: str
    recommendation: str


class SlowThreatAssessor:
    """Multi-step threat deliberation, woken only on a salient ignition.

    Parameters
    ----------
    zone_center : sequence of float
        Centre of the protected zone (D-dimensional).
    danger_radius : float
        Radius of the protected zone; a forecast point within it is a breach.
    dt : float
        Forecast timestep (match the hot loop's tick for a consistent ETA).
    horizon : int
        Number of ballistic steps to look ahead (>= 1).
    engage_eta : int
        ETA (in steps) at or below which a forecast breach is escalated from
        ``"WATCH"`` to ``"ENGAGE"``.
    """

    def __init__(
        self,
        *,
        zone_center,
        danger_radius: float,
        dt: float = 1.0,
        horizon: int = 12,
        engage_eta: int = 3,
    ) -> None:
        if danger_radius <= 0.0:
            raise ValueError(f"danger_radius must be > 0, got {danger_radius}")
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if int(engage_eta) < 0:
            raise ValueError(f"engage_eta must be >= 0, got {engage_eta}")
        self.zone_center = torch.as_tensor(zone_center, dtype=torch.float32)
        self.danger_radius = float(danger_radius)
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.engage_eta = int(engage_eta)

    @property
    def n_parameters(self) -> int:
        """Trainable parameters: 0. Pure operator composition."""
        return 0

    def assess(self, pos, vel, *, tick: int = 0) -> ThreatAssessment:
        """Roll the target forward and grade the threat over the horizon.

        Parameters
        ----------
        pos, vel : sequence of float
            The target's current position and velocity (D-dimensional).
        tick : int
            The hot-loop tick this assessment was woken at (recorded only).
        """
        p0 = torch.as_tensor(pos, dtype=torch.float32).reshape(-1)
        v0 = torch.as_tensor(vel, dtype=torch.float32).reshape(-1)

        # Ballistic forecast: "what happens if nothing changes" -- one body, no
        # acceleration. positions: (horizon + 1, 1, D); strip the body axis.
        roll = rollout(p0.unsqueeze(0), v0.unsqueeze(0), steps=self.horizon, dt=self.dt)
        path = roll.positions[:, 0, :]                              # (horizon + 1, D)

        ranges = (path - self.zone_center).norm(dim=-1)            # (horizon + 1,)
        inside = in_ball(path, self.zone_center, self.danger_radius)  # (horizon + 1,)

        min_idx = int(torch.argmin(ranges).item())
        min_range = float(ranges[min_idx].item())

        breach_steps = torch.nonzero(inside, as_tuple=False).flatten()
        if breach_steps.numel() > 0:
            eta = int(breach_steps[0].item())
            breaches = True
        else:
            eta = None
            breaches = False

        if not breaches:
            level = "CLEAR"
            rec = "hold posture; continue passive track"
        elif eta is not None and eta <= self.engage_eta:
            level = "ENGAGE"
            rec = "slew to intercept bearing; arm response"
        else:
            level = "WATCH"
            rec = "escalate track; prime intercept"

        return ThreatAssessment(
            woken_tick=int(tick),
            horizon=self.horizon,
            breaches=breaches,
            eta_breach=eta,
            min_range=min_range,
            closest_offset=min_idx,
            level=level,
            recommendation=rec,
        )
