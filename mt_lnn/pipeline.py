"""
mt_lnn/pipeline.py -- the dual-speed sentry: one loop wiring the operator layers,
the world model, the failsafe actuators and the salience trigger together.

Why this module exists
----------------------
The individual layers (``spatial_ops``, ``physics_ops``, ``acoustic_ops``,
``salience_events``, ``imagination``, ``failsafe``) are each a single-purpose,
zero-coupling capability. A product is not a pile of capabilities -- it is the
*loop* that composes them. This module is that loop, written as the **dual-speed
engine** the architecture has described all along but never assembled:

    PERCEIVE   -- a binaural sensor localizes a moving target from the geometry
                  of sound (``acoustic_ops``: ITD -> azimuth, Doppler, range).
    REASON     -- spatial logic asks "is the target inside the protected zone?"
                  (``spatial_ops.in_ball``), recomputed every tick, not stored.
    PREDICT    -- a one-step ballistic forecast (``physics_ops.integrate``) gives
                  a predictive-coding *surprise*: how far the target departed from
                  "keep doing what it was doing". A manoeuvre spikes it.
    TRIGGER    -- that surprise drives a :class:`SalienceEventDetector`: the slow
                  / symbolic / cloud layer is *woken* (an ignition event) only on
                  a genuine state change, never polled on the hot loop. This is
                  the whole point of a dual-speed system.
    COAST      -- when the sensor feed stutters, a :class:`BlindRolloutGuard`
                  decides -- from a decaying trust budget -- whether to coast on
                  the world model (the target is dead-reckoned forward with
                  ``physics_ops``) or to declare itself blind and safe-stop.
    ACT SAFELY -- the aim command is pushed through a :class:`CircuitBreaker`, so
                  the actuator output is *always* finite, within its mechanical
                  limits and slew-rate, no matter what the upstream produced.

Design / coupling
-----------------
* A stateful **orchestrator**, not an ``nn.Module``; it adds **zero trainable
  parameters of its own**. The only parameter-bearing part is the optional
  world-model head the guard coasts on (a :class:`PredictiveStateHead`), which it
  will build a small default of if you do not pass one. Everything else --
  perception, reasoning, the surprise signal, the safety clamp -- is analytic and
  zero-parameter. It NEVER imports ``model.py``; it composes the public operator
  APIs and is duck-typed on the world-model head.
* Online: one :meth:`DualSpeedSentry.step` per tick, deterministic, resettable.

Honest scope
------------
A deterministic reference *integration* (an edge perimeter-sentry), not a trained
end-to-end agent. The salience signal is a real, training-free predictive-coding
residual (it genuinely fires on a manoeuvre and stays quiet on a steady
approach). The blind-rollout guard's confidence here is the imagination's
geometric trust-decay budget, which is meaningful *independent* of whether the
head is trained -- it controls "how long dare we fly blind", which is what the
sentry uses. ITD->azimuth uses the far-field model (place the scene in front of
the head). Pinned in ``tests/test_pipeline.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .acoustic_ops import (
    SPEED_OF_SOUND,
    interaural_time_difference,
    localize_azimuth,
    doppler_shift,
)
from .spatial_ops import in_ball
from .physics_ops import integrate
from .salience_events import SalienceEventDetector
from .failsafe import BlindRolloutGuard, CircuitBreaker
from .imagination import LatentImagination
from .world_model import PredictiveStateHead
from .slow_layer import SlowThreatAssessor, ThreatAssessment

__all__ = ["PerceptionEvent", "SentryTick", "DualSpeedSentry"]


@dataclass(frozen=True)
class PerceptionEvent:
    """A salient state change handed up to the slow / cloud layer.

    Emitted only when the predictive-coding surprise ignites the detector -- i.e.
    the target did something the steady-state model did not expect. Carries the
    perception snapshot as the broadcast payload (the slow layer never had to poll
    the hot loop to get it), plus the slow layer's :class:`ThreatAssessment` --
    the deliberate multi-step forecast the ignition actually paid for.
    """

    tick: int
    azimuth: float          # localized bearing (rad, right-positive)
    distance: float         # range to the head (m)
    inside_zone: bool       # was the target inside the protected radius?
    doppler: float          # observed frequency (Hz); > rest = approaching
    salience: float         # the z-score that ignited it (surprise in sigma)
    assessment: Optional["ThreatAssessment"] = None  # slow-layer verdict on wake


@dataclass(frozen=True)
class SentryTick:
    """One tick of :meth:`DualSpeedSentry.step` -- the full dual-speed readout.

    Attributes
    ----------
    tick : int
    azimuth : float
        Best current bearing estimate (rad, right-positive). While coasting this
        is the dead-reckoned bearing; while blind it holds the last good bearing.
    distance : float
        Best current range estimate (m).
    doppler : float
        Observed frequency (Hz); ``> freq`` approaching, ``< freq`` receding.
    inside_zone : bool
        Whether the (estimated) target is inside the protected radius this tick.
    surprise : float or None
        The predictive-coding residual fed to the salience detector this tick, or
        ``None`` on a tick with no real observation (blind / dark).
    source : str
        Perception source: "live" (sensor up), "imagined" (coasting on the world
        model through a dropout), or "dark" (blind and no longer trusted).
    confidence : float
        Trust in the perception this tick (1.0 live; the guard's decaying budget
        while coasting; 0.0 dark).
    blind_steps : int
        Consecutive sensor-dropout ticks (0 while live).
    command : float
        The safe actuator aim command actually emitted (rad) -- always within the
        mechanical limit and slew-rate, even through dropouts and garbage input.
    command_tripped : bool
        Whether the output circuit breaker is in its protective state.
    event : PerceptionEvent or None
        A wake-the-slow-layer event if one ignited this tick, else ``None``.
    """

    tick: int
    azimuth: float
    distance: float
    doppler: float
    inside_zone: bool
    surprise: Optional[float]
    source: str
    confidence: float
    blind_steps: int
    command: float
    command_tripped: bool
    event: Optional[PerceptionEvent]


class DualSpeedSentry:
    """Compose perception, prediction, salience and safety into one dual-speed loop.

    Parameters
    ----------
    left_ear, right_ear : sequence of float
        The two ear positions (2-D, on the interaural axis). Forward is the axis
        perpendicular to ``right_ear - left_ear``; azimuth is right-positive.
    zone_center : sequence of float
        Centre of the protected zone (2-D).
    danger_radius : float
        Radius of the protected zone; the target is "inside" within it.
    freq : float
        Rest frequency of the target's tone (Hz), for the Doppler readout.
    dt : float
        Tick duration used by the one-step ballistic forecast / dead-reckoning.
    speed : float
        Speed of sound (m/s) for the acoustic cues.
    aim_limit : float
        Mechanical actuator bound (rad); the aim command is clamped to
        ``[-aim_limit, aim_limit]``.
    max_slew : float
        Maximum aim change per tick (rad), the actuator slew-rate limit.
    ignite_z, release_z, refractory, warmup : salience-detector knobs
        Passed straight to :class:`SalienceEventDetector` (the surprise channel).
    imagination : LatentImagination, optional
        The world model the blind-rollout guard coasts on. If ``None``, a small
        default :class:`PredictiveStateHead` + :class:`LatentImagination` is built
        (untrained -- used for its trust-budget gating, see the module docstring).
    max_blind_steps : int
        How many consecutive dropout ticks the sentry may coast before going dark.
    confidence_floor : float
        Guard trust floor below which it goes dark mid-coast.
    feature_dim : int
        Width of the perceptual feature vector fed to the world-model head.
    slow_layer : object, optional
        The slow half of the dual-speed engine, woken only on a salient ignition.
        Any object exposing ``assess(pos, vel, *, tick) -> ThreatAssessment``
        (duck-typed). If ``None``, a default :class:`SlowThreatAssessor` is built
        over the same zone geometry.
    slow_horizon : int
        Forecast horizon (steps) for the default slow-layer threat assessment.
    engage_eta : int
        Forecast ETA (steps) at/below which the slow layer escalates to ENGAGE.
    """

    def __init__(
        self,
        *,
        left_ear,
        right_ear,
        zone_center,
        danger_radius: float,
        freq: float = 1000.0,
        dt: float = 1.0,
        speed: float = SPEED_OF_SOUND,
        aim_limit: float = math.pi / 2,
        max_slew: float = math.radians(20.0),
        ignite_z: float = 3.0,
        release_z: float = 1.0,
        refractory: int = 3,
        warmup: int = 8,
        imagination: Optional[LatentImagination] = None,
        max_blind_steps: int = 4,
        confidence_floor: float = 0.3,
        feature_dim: int = 6,
        slow_layer=None,
        slow_horizon: int = 12,
        engage_eta: int = 3,
    ) -> None:
        if danger_radius <= 0.0:
            raise ValueError(f"danger_radius must be > 0, got {danger_radius}")
        self.left_ear = torch.as_tensor(left_ear, dtype=torch.float32)
        self.right_ear = torch.as_tensor(right_ear, dtype=torch.float32)
        if self.left_ear.shape != self.right_ear.shape or self.left_ear.dim() != 1:
            raise ValueError("left_ear and right_ear must be equal-length 1-D coordinates")
        self.head_center = 0.5 * (self.left_ear + self.right_ear)
        self.head_width = float((self.right_ear - self.left_ear).norm().item())
        if self.head_width <= 0.0:
            raise ValueError("left_ear and right_ear must be distinct")
        self.zone_center = torch.as_tensor(zone_center, dtype=torch.float32)
        self.danger_radius = float(danger_radius)
        self.freq = float(freq)
        self.dt = float(dt)
        self.speed = float(speed)
        self.feature_dim = int(feature_dim)

        # -- the dual-speed trigger (predictive-coding surprise channel) -------
        self.detector = SalienceEventDetector(
            ignite_z=ignite_z, release_z=release_z, refractory=refractory, warmup=warmup
        )

        # -- the output safety actuator ---------------------------------------
        self.breaker = CircuitBreaker(
            lo=-abs(aim_limit), hi=abs(aim_limit), max_rate=abs(max_slew),
            fallback=None, trip_after=3, reset_after=3, initial=0.0,
        )

        # -- the world model the dropout guard coasts on ----------------------
        if imagination is None:
            head = PredictiveStateHead(d_model=self.feature_dim)
            head.eval()
            imagination = LatentImagination(head, horizon=max_blind_steps)
        self.imagination = imagination
        self.guard = BlindRolloutGuard(
            imagination, confidence_floor=confidence_floor, max_blind_steps=max_blind_steps
        )

        # -- the slow half of the dual-speed engine (woken only on ignition) ---
        if slow_layer is None:
            slow_layer = SlowThreatAssessor(
                zone_center=self.zone_center, danger_radius=self.danger_radius,
                dt=self.dt, horizon=int(slow_horizon), engage_eta=int(engage_eta),
            )
        self.slow_layer = slow_layer

        self.reset()

    @property
    def n_parameters(self) -> int:
        """Trainable parameters the *orchestrator* adds: 0. The only parameters in
        play belong to the optional world-model head it coasts on."""
        return 0

    def reset(self) -> None:
        """Clear all running state (caches, FSMs, counters)."""
        self.detector.reset()
        self.breaker.reset()
        self.guard.reset()
        self._tick = 0
        self._last_pos: Optional[torch.Tensor] = None
        self._last_vel: Optional[torch.Tensor] = None
        self._last_az = 0.0
        self._last_dist = 0.0
        self._last_doppler = self.freq
        self._last_inside = False
        self._prev_pos: Optional[torch.Tensor] = None
        self._prev_vel: Optional[torch.Tensor] = None

    # -- perception helpers -----------------------------------------------------

    def _perceive(self, pos: torch.Tensor, vel: torch.Tensor):
        """Acoustic cues for a target at ``pos`` moving at ``vel``."""
        itd = interaural_time_difference(pos, self.left_ear, self.right_ear, speed=self.speed)
        az = float(localize_azimuth(itd, head_width=self.head_width, speed=self.speed).item())
        dist = float((pos - self.head_center).norm().item())
        dop = float(doppler_shift(pos, vel, self.head_center, torch.zeros_like(vel),
                                  freq=self.freq, speed=self.speed).item())
        return az, dist, dop

    def _encode(self, az: float, dist: float, dop: float, vel: torch.Tensor) -> torch.Tensor:
        """Bounded perceptual feature vector for the world-model head (1, feature_dim)."""
        raw = [
            math.sin(az), math.cos(az),
            math.tanh(dist / max(self.danger_radius, 1e-6)),
            math.tanh((dop - self.freq) / max(self.freq, 1e-6) * 50.0),
            math.tanh(float(vel[0].item())), math.tanh(float(vel[-1].item())),
        ]
        feat = (raw + [0.0] * self.feature_dim)[: self.feature_dim]
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

    # -- the loop ---------------------------------------------------------------

    def step(self, target_pos, target_vel, *, sensor_ok: bool = True) -> SentryTick:
        """Advance one tick of the dual-speed loop.

        Parameters
        ----------
        target_pos, target_vel : sequence of float
            The target's true position / velocity this tick (the sensor's input).
            Ignored when ``sensor_ok`` is False (a dropped frame).
        sensor_ok : bool
            Whether the sensor delivered a frame this tick.
        """
        self._tick += 1
        pos = torch.as_tensor(target_pos, dtype=torch.float32)
        vel = torch.as_tensor(target_vel, dtype=torch.float32)

        surprise: Optional[float] = None

        if sensor_ok:
            # -- PERCEIVE (live) ----------------------------------------------
            az, dist, dop = self._perceive(pos, vel)
            self.guard.step(self._encode(az, dist, dop, vel))   # cache a good latent
            source, confidence, blind = "live", 1.0, 0

            # -- PREDICT -> surprise (predictive coding) ----------------------
            if self._prev_pos is not None:
                pred_pos, _ = integrate(self._prev_pos, self._prev_vel,
                                        torch.zeros_like(vel), self.dt)
                surprise = float((pos - pred_pos).norm().item())

            est_pos = pos
            self._last_pos, self._last_vel = pos, vel
            self._last_az, self._last_dist, self._last_doppler = az, dist, dop
            self._prev_pos, self._prev_vel = pos, vel
        else:
            # -- COAST / go dark on a dropped frame ---------------------------
            guard_out = self.guard.step(None)
            source, confidence, blind = guard_out.source, guard_out.confidence, guard_out.blind_steps
            if guard_out.source == "dark" or self._last_pos is None:
                # blind and untrusted: hold the last estimate, do not invent data.
                source = "dark"
                az, dist, dop = self._last_az, self._last_dist, self._last_doppler
                est_pos = self._last_pos if self._last_pos is not None else self.head_center
            else:
                # coast: dead-reckon the target forward with physics, re-perceive.
                self._last_pos, self._last_vel = integrate(
                    self._last_pos, self._last_vel, torch.zeros_like(self._last_vel), self.dt
                )
                az, dist, dop = self._perceive(self._last_pos, self._last_vel)
                self._last_az, self._last_dist, self._last_doppler = az, dist, dop
                est_pos = self._last_pos

        # -- REASON: protected-zone containment (recomputed, not stored) -------
        inside = bool(in_ball(est_pos.unsqueeze(0), self.zone_center, self.danger_radius)[0].item())
        self._last_inside = inside

        # -- TRIGGER: wake the slow layer only on a salient surprise -----------
        # Ignition can only fire on a live tick (``surprise`` is computed solely
        # from a real observation), so ``pos``/``vel`` below are genuine.
        event: Optional[PerceptionEvent] = None
        if surprise is not None:
            ev = self.detector.update(surprise)
            if ev is not None and ev.kind == "ignition":
                assessment = self.slow_layer.assess(pos, vel, tick=self._tick)
                event = PerceptionEvent(tick=self._tick, azimuth=az, distance=dist,
                                        inside_zone=inside, doppler=dop, salience=ev.z,
                                        assessment=assessment)

        # -- ACT SAFELY: aim command through the output circuit breaker --------
        raw_cmd = az if source != "dark" else self.breaker.last_safe
        res = self.breaker.step(raw_cmd)

        return SentryTick(
            tick=self._tick, azimuth=az, distance=dist, doppler=dop, inside_zone=inside,
            surprise=surprise, source=source, confidence=confidence, blind_steps=blind,
            command=res.value, command_tripped=res.tripped, event=event,
        )
