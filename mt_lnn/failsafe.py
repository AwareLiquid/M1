"""
mt_lnn/failsafe.py -- input-dropout blind rollout + model-external output breaker.

Why this module exists
----------------------
A fast liquid core can be deployed where two things go wrong that the *training*
objective never sees:

1. **The input stream stutters.** A sensor drops a frame, a packet is late, the
   upstream feed gaps. Feeding the core a zero / stale frame silently corrupts
   its recurrent state. A robust system should instead *coast on its own
   prediction* for a short, **confidence-budgeted** window -- and, crucially,
   stop coasting (declare "I can no longer see") rather than hallucinate forever.

2. **The output goes out of bounds.** A controller's command must respect hard
   physical red-lines (actuator limits, slew rate) no matter what the learned
   model emits -- including NaN/Inf. That guarantee cannot live *inside* the
   learned graph; it must be an external, model-decoupled last-metre gate.

This module is those two safety actuators, mirroring real biology:

* **BlindRolloutGuard** -- when the input is present it caches the live latent
  and passes it through; when the input drops it *imagines* forward from the
  last good state (using the world model's already-trained one-step map via
  :class:`mt_lnn.imagination.LatentImagination`) and serves the imagined latent,
  but only while the imagination's own **confidence** stays above a floor and
  only for a bounded number of blind steps. This is the brain coasting on a
  forward model through a brief occlusion -- and knowing when to give up.

* **CircuitBreaker** -- a model-*external* output limiter. It *always* hard-clamps
  (NaN/Inf scrub, absolute bounds, slew-rate limit) as an unconditional
  guarantee, and additionally trips to a **fallback** (hold-last-safe, or a
  caller-supplied controller such as a PID) with debounce when the raw command
  repeatedly violates the red-lines -- a Schmitt-style protective reflex with
  bumpless transfer back to the model when it recovers.

* **TopologyBreaker** -- a model-*external* *representation*-health tripwire. The
  two guards above watch a scalar output and a single latent; this one watches
  the *shape* of the whole live state population. It latches a healthy reference
  point cloud, then each tick measures the Symmetric Relative-Topology Divergence
  (:func:`mt_lnn.topology_ops.srtd`) of the live cloud's H0 barcode from that
  reference (and, optionally, a change in the connected-component count
  :func:`mt_lnn.topology_ops.betti0`). Benign jitter leaves the topology intact
  (SRTD ~ 0); a mode collapse / regime change / steering edit gone wrong fuses or
  splits clusters and spikes SRTD -- and after a debounce the breaker trips. This
  is the zero-parameter topological failsafe that ``demo_topology_ops`` anticipated.

Design / coupling
-----------------
* All three are small **stateful actuators** with **zero trainable parameters**;
  none is an ``nn.Module`` and none imports ``model.py``. ``BlindRolloutGuard`` /
  ``CircuitBreaker`` import only the standard library + ``torch``;
  ``TopologyBreaker`` additionally composes the sibling *operator* module
  :mod:`mt_lnn.topology_ops` (itself pure, model-free) -- composition of
  zero-param primitives, not backbone coupling. ``BlindRolloutGuard`` is
  duck-typed on a ``LatentImagination``-like object; ``CircuitBreaker`` is pure
  scalar/tensor arithmetic; ``TopologyBreaker`` is pure point-cloud topology.
* Online: one ``step`` call per tick, deterministic, fully resettable.

Pinned against deterministic / analytic streams and a real ``LatentImagination``
bridge in ``tests/test_failsafe.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .topology_ops import betti0, srtd

__all__ = [
    "GuardOutput",
    "BlindRolloutGuard",
    "BreakerResult",
    "CircuitBreaker",
    "TopologyResult",
    "TopologyBreaker",
]


# --------------------------------------------------------------------------- #
# 1. input-dropout blind rollout                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GuardOutput:
    """One tick of :class:`BlindRolloutGuard`.

    Attributes
    ----------
    latent : torch.Tensor or None
        The latent to act on this tick -- (B, P), L2-normalised. ``None`` when
        the guard has gone *dark*: the input is absent and the imagination is no
        longer trustworthy (confidence below floor, or the blind budget is
        spent). A ``None`` latent is the guard's explicit "I can no longer see"
        signal; the caller should hold / safe-stop, not act on a stale frame.
    source : str
        "live"     -- the input was present; this is the real latent.
        "imagined" -- the input dropped; this is a trusted blind-rollout latent.
        "dark"     -- the input dropped and the guard has stopped serving.
    confidence : float
        Trust in ``latent`` this tick: 1.0 while live, the imagination's per-step
        confidence while imagining, 0.0 when dark.
    blind_steps : int
        Number of *consecutive* blind (input-absent) ticks so far, including this
        one. 0 whenever the input is present.
    """

    latent: Optional[torch.Tensor]
    source: str
    confidence: float
    blind_steps: int


class BlindRolloutGuard:
    """Coast on the world model through brief input dropouts, with a budget.

    The guard sits between the input stream and the consumer of latents. On every
    tick the caller hands it the current hidden state (or ``None`` if the input
    dropped this tick):

    * **input present** -- the guard encodes and caches the live present latent,
      resets the blind counter, and returns it (``source="live"``,
      ``confidence=1.0``).
    * **input absent** -- the guard imagines a trajectory once from the last good
      hidden state and serves the imagined latent for the current blind step,
      gated two ways:
        - the imagined step's **confidence** must stay ``>= confidence_floor``;
        - the number of consecutive blind steps must stay ``<= max_blind_steps``.
      If either gate fails the guard goes **dark** (``latent=None``,
      ``source="dark"``, ``confidence=0.0``) instead of hallucinating onward.

    Parameters
    ----------
    imagination : LatentImagination-like
        Anything exposing ``imagine(hidden, horizon) -> ImaginedTrajectory`` with
        ``.latents`` (B, H, P) / ``.confidence`` (B, H), a ``horizon`` int and a
        ``head`` with ``online_proj``. The real
        :class:`mt_lnn.imagination.LatentImagination` satisfies this.
    confidence_floor : float
        Minimum per-step imagination confidence to keep serving while blind
        (∈ [0, 1)). Default 0.3.
    max_blind_steps : int, optional
        Hard cap on consecutive blind ticks before going dark. Defaults to the
        imagination's ``horizon`` (you cannot coast further than you imagined).

    Notes
    -----
    Zero trainable parameters (``n_parameters == 0``). Stateful and resettable.
    """

    def __init__(
        self,
        imagination,
        *,
        confidence_floor: float = 0.3,
        max_blind_steps: Optional[int] = None,
    ) -> None:
        if not hasattr(imagination, "imagine"):
            raise TypeError(
                "imagination must expose `imagine(hidden, horizon)`; expected a "
                "LatentImagination-like object."
            )
        if not (0.0 <= confidence_floor < 1.0):
            raise ValueError(
                f"confidence_floor must be in [0, 1), got {confidence_floor}"
            )
        horizon = int(getattr(imagination, "horizon", 0)) or 1
        if max_blind_steps is None:
            max_blind_steps = horizon
        if max_blind_steps < 1:
            raise ValueError(f"max_blind_steps must be >= 1, got {max_blind_steps}")
        self.imagination = imagination
        self.confidence_floor = float(confidence_floor)
        self.max_blind_steps = int(max_blind_steps)
        # the imagination can only supply `horizon` blind frames; never index past it
        self._horizon = horizon
        self.reset()

    @property
    def n_parameters(self) -> int:
        """Zero trainable parameters -- a pure inference-time actuator."""
        return 0

    @property
    def blind_steps(self) -> int:
        """Consecutive input-absent ticks served so far (0 while live)."""
        return self._blind

    def reset(self) -> None:
        """Clear cached state, the blind counter and any standing imagination."""
        self._last_hidden: Optional[torch.Tensor] = None
        self._traj = None
        self._blind = 0

    @staticmethod
    def _present_latent(imagination, hidden: torch.Tensor) -> torch.Tensor:
        """L2-normalised present latent for a live hidden state -- (B, P)."""
        if hidden.dim() == 3:
            hidden = hidden[:, -1, :]
        elif hidden.dim() != 2:
            raise ValueError(
                f"hidden must be (B, d_model) or (B, T, d_model), got {tuple(hidden.shape)}"
            )
        proj = imagination.head.online_proj(hidden)
        return F.normalize(proj, dim=-1, eps=1e-8)

    @torch.no_grad()
    def step(self, hidden: Optional[torch.Tensor]) -> GuardOutput:
        """Advance one tick.

        Parameters
        ----------
        hidden : torch.Tensor or None
            The live hidden state this tick -- (B, d_model) or (B, T, d_model) --
            or ``None`` if the input dropped this tick.

        Returns
        -------
        GuardOutput
        """
        if hidden is not None:
            # live: cache the good state, reset the blind window, serve the truth.
            self._last_hidden = hidden
            self._traj = None
            self._blind = 0
            latent = self._present_latent(self.imagination, hidden)
            return GuardOutput(latent=latent, source="live", confidence=1.0, blind_steps=0)

        # input dropped this tick.
        if self._last_hidden is None:
            # never saw a good frame -> nothing to imagine from.
            self._blind += 1
            return GuardOutput(latent=None, source="dark", confidence=0.0,
                               blind_steps=self._blind)

        self._blind += 1
        # budget gate: cannot coast past the cap or past what we imagined.
        if self._blind > self.max_blind_steps or self._blind > self._horizon:
            return GuardOutput(latent=None, source="dark", confidence=0.0,
                               blind_steps=self._blind)

        # imagine once from the last good state, then index per blind step.
        if self._traj is None:
            self._traj = self.imagination.imagine(self._last_hidden, self._horizon)

        idx = self._blind - 1                            # 0-based into (B, H, P)
        conf = float(self._traj.confidence[:, idx].min().item())
        if conf < self.confidence_floor:
            # the dream is no longer trustworthy -> go dark instead of hallucinate.
            return GuardOutput(latent=None, source="dark", confidence=conf,
                               blind_steps=self._blind)

        latent = self._traj.latents[:, idx, :]           # (B, P), already normalised
        return GuardOutput(latent=latent, source="imagined", confidence=conf,
                           blind_steps=self._blind)


# --------------------------------------------------------------------------- #
# 2. model-external output circuit breaker                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BreakerResult:
    """One tick of :class:`CircuitBreaker`.

    Attributes
    ----------
    value : float
        The safe command actually emitted this tick -- always finite and always
        within ``[lo, hi]`` and within the slew limit, whether or not the breaker
        is tripped.
    tripped : bool
        Whether the breaker is in its protective (fallback) state this tick.
    reason : str
        Why the *raw* command was unsafe this tick: "" (clean), "nonfinite"
        (NaN/Inf), "bounds" (outside ``[lo, hi]``), or "rate" (slew too fast).
        Reflects the raw input even on ticks where clamping already fixed it, so
        a caller can log the underlying fault.
    overridden : bool
        Whether ``value`` came from the fallback (``True``) rather than the
        clamped model command (``False``).
    """

    value: float
    tripped: bool
    reason: str
    overridden: bool


class CircuitBreaker:
    """Model-external last-metre output safety: hard-clamp + tripping fallback.

    Two guarantees, in order:

    1. **Unconditional hard clamp.** Every raw command is first scrubbed of
       NaN/Inf, clamped into ``[lo, hi]``, and rate-limited to ``max_rate`` per
       tick relative to the last emitted value. This happens *every* tick, so the
       output is *always* finite and inside the red-lines -- even before any trip.

    2. **Debounced trip to fallback.** Each tick is judged unsafe if the raw
       command is non-finite, outside ``[lo, hi]``, or exceeds the slew limit.
       After ``trip_after`` consecutive unsafe ticks the breaker *trips*: it stops
       trusting the model and emits the **fallback** instead (hold-last-safe if
       ``fallback`` is ``None``, else ``fallback(last_safe) -> float`` such as a
       PID). After ``reset_after`` consecutive clean ticks it *closes* again,
       resuming the model with **bumpless transfer** (the slew limit caps the
       reconnection jump).

    Parameters
    ----------
    lo, hi : float
        Hard absolute output bounds. ``lo < hi`` required.
    max_rate : float, optional
        Maximum change in the emitted value per tick (slew limit, > 0). ``None``
        disables rate limiting / the "rate" fault.
    fallback : callable, optional
        ``fallback(last_safe: float) -> float`` invoked while tripped. ``None``
        holds the last safe value.
    trip_after : int
        Consecutive unsafe ticks before tripping (>= 1). Default 3.
    reset_after : int
        Consecutive clean ticks before closing again (>= 1). Default 3.
    initial : float
        The seed "last safe value" before the first tick. Default 0.0.

    Notes
    -----
    Zero trainable parameters (``n_parameters == 0``). Operates on plain Python
    floats (scalar control signal); deterministic and resettable.
    """

    def __init__(
        self,
        *,
        lo: float,
        hi: float,
        max_rate: Optional[float] = None,
        fallback: Optional[Callable[[float], float]] = None,
        trip_after: int = 3,
        reset_after: int = 3,
        initial: float = 0.0,
    ) -> None:
        if not (lo < hi):
            raise ValueError(f"need lo < hi, got lo={lo}, hi={hi}")
        if max_rate is not None and not (max_rate > 0.0):
            raise ValueError(f"max_rate must be > 0 or None, got {max_rate}")
        if trip_after < 1:
            raise ValueError(f"trip_after must be >= 1, got {trip_after}")
        if reset_after < 1:
            raise ValueError(f"reset_after must be >= 1, got {reset_after}")
        if not (lo <= initial <= hi):
            raise ValueError(f"initial must be in [lo, hi], got {initial}")
        self.lo = float(lo)
        self.hi = float(hi)
        self.max_rate = None if max_rate is None else float(max_rate)
        self.fallback = fallback
        self.trip_after = int(trip_after)
        self.reset_after = int(reset_after)
        self.initial = float(initial)
        self.reset()

    @property
    def n_parameters(self) -> int:
        """Zero trainable parameters -- a pure external safety actuator."""
        return 0

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def last_safe(self) -> float:
        return self._last_safe

    def reset(self) -> None:
        """Clear the FSM, debounce counters and last-safe value."""
        self._last_safe = self.initial
        self._tripped = False
        self._bad_run = 0
        self._good_run = 0

    @staticmethod
    def _as_float(x) -> float:
        try:
            return float(x.item())                       # torch / numpy scalar
        except AttributeError:
            return float(x)

    def step(self, raw) -> BreakerResult:
        """Judge and gate one raw model command.

        Parameters
        ----------
        raw : float or scalar tensor
            The model's proposed output this tick (may be NaN/Inf/out-of-range).

        Returns
        -------
        BreakerResult
        """
        x = self._as_float(raw)
        prev = self._last_safe

        # -- classify the *raw* command (for the fault reason + debounce) -------
        finite = (x == x) and (x not in (float("inf"), float("-inf")))
        reason = ""
        if not finite:
            reason = "nonfinite"
        elif x < self.lo or x > self.hi:
            reason = "bounds"
        elif self.max_rate is not None and abs(x - prev) > self.max_rate:
            reason = "rate"
        unsafe = reason != ""

        # -- debounce FSM -------------------------------------------------------
        if unsafe:
            self._bad_run += 1
            self._good_run = 0
        else:
            self._good_run += 1
            self._bad_run = 0
        if not self._tripped and self._bad_run >= self.trip_after:
            self._tripped = True
        elif self._tripped and self._good_run >= self.reset_after:
            self._tripped = False

        # -- choose the source --------------------------------------------------
        overridden = False
        if self._tripped:
            overridden = True
            if self.fallback is None:
                candidate = prev                         # hold last safe
            else:
                candidate = self._as_float(self.fallback(prev))
        else:
            candidate = x

        # -- unconditional hard clamp (NaN scrub -> bounds -> slew) -------------
        safe = self._hard_clamp(candidate, prev)
        self._last_safe = safe
        return BreakerResult(value=safe, tripped=self._tripped, reason=reason,
                             overridden=overridden)

    __call__ = step

    def _hard_clamp(self, candidate: float, prev: float) -> float:
        """Scrub NaN/Inf, clamp to bounds, then rate-limit relative to ``prev``."""
        c = candidate
        if not (c == c) or c in (float("inf"), float("-inf")):
            c = prev                                     # NaN/Inf -> hold
        if c < self.lo:
            c = self.lo
        elif c > self.hi:
            c = self.hi
        if self.max_rate is not None:
            delta = c - prev
            if delta > self.max_rate:
                c = prev + self.max_rate
            elif delta < -self.max_rate:
                c = prev - self.max_rate
        return c


# --------------------------------------------------------------------------- #
# 3. model-external topological (representation-shape) tripwire               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TopologyResult:
    """One tick of :class:`TopologyBreaker`.

    Attributes
    ----------
    srtd : float
        Symmetric Relative-Topology Divergence of the live cloud's H0 barcode
        from the healthy reference this tick. ``0.0`` for a topologically
        identical cloud; grows as clusters fuse / split. ``inf`` is never
        produced -- it is a bounded, sorted-barcode distance.
    betti0 : int or None
        Connected-component count of the live cloud at the probe scale ``eps``
        (``None`` when ``eps`` was not configured, i.e. the Betti check is off).
    tripped : bool
        Whether the breaker is in its protective state this tick (the live
        representation has drifted in shape and stayed drifted past the debounce).
    reason : str
        Why the live cloud was judged unhealthy this tick: "" (clean), "noref"
        (no reference latched yet), "srtd" (divergence over threshold), or
        "betti" (component count changed from the reference). Reflects the raw
        per-tick judgement even before/after a trip, for logging.
    """

    srtd: float
    betti0: Optional[int]
    tripped: bool
    reason: str


class TopologyBreaker:
    """Model-external tripwire on the *shape* of the live state population.

    Latch a healthy reference point cloud (at construction, via
    :meth:`set_reference`, or by auto-latching the first clean tick), then on
    every tick judge the live cloud:

    1. **SRTD drift.** ``srtd(reference, live)`` (the sorted-barcode H0 distance)
       must stay ``<= srtd_threshold``. Benign jitter keeps it ~ 0; a mode
       collapse / regime change spikes it.
    2. **Betti-0 change (optional).** If a probe scale ``eps`` is given, the
       connected-component count ``betti0(live, eps)`` must equal the reference's
       count; a changed cluster count is also a fault.

    A **debounced FSM** (mirroring :class:`CircuitBreaker`): after ``trip_after``
    consecutive unhealthy ticks the breaker *trips*; after ``reset_after``
    consecutive clean ticks it *closes* again. The breaker does not modify the
    representation -- it is a health *signal* (``tripped`` / ``reason``) a caller
    acts on (freeze, roll back a steering edit, fall back to a safe policy).

    Parameters
    ----------
    srtd_threshold : float
        Maximum tolerated topology divergence from the reference (> 0).
    eps : float, optional
        Probe scale for the optional Betti-0 component-count check. ``None``
        disables it (SRTD-only).
    trip_after : int
        Consecutive unhealthy ticks before tripping (>= 1). Default 2.
    reset_after : int
        Consecutive clean ticks before closing again (>= 1). Default 3.
    reference : torch.Tensor, optional
        A healthy reference cloud ``(N, D)`` to latch immediately. If ``None`` the
        first :meth:`step` latches its cloud as the reference (always clean).

    Notes
    -----
    Zero trainable parameters (``n_parameters == 0``). Stateful and resettable.
    Composes :func:`mt_lnn.topology_ops.srtd` / :func:`~mt_lnn.topology_ops.betti0`;
    no backbone coupling.
    """

    def __init__(
        self,
        *,
        srtd_threshold: float,
        eps: Optional[float] = None,
        trip_after: int = 2,
        reset_after: int = 3,
        reference: Optional[torch.Tensor] = None,
    ) -> None:
        if not (srtd_threshold > 0.0):
            raise ValueError(f"srtd_threshold must be > 0, got {srtd_threshold}")
        if eps is not None and not (eps > 0.0):
            raise ValueError(f"eps must be > 0 or None, got {eps}")
        if trip_after < 1:
            raise ValueError(f"trip_after must be >= 1, got {trip_after}")
        if reset_after < 1:
            raise ValueError(f"reset_after must be >= 1, got {reset_after}")
        self.srtd_threshold = float(srtd_threshold)
        self.eps = None if eps is None else float(eps)
        self.trip_after = int(trip_after)
        self.reset_after = int(reset_after)
        self.reset()
        if reference is not None:
            self.set_reference(reference)

    @property
    def n_parameters(self) -> int:
        """Zero trainable parameters -- a pure external health actuator."""
        return 0

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def has_reference(self) -> bool:
        return self._reference is not None

    def reset(self) -> None:
        """Clear the reference, the FSM and the debounce counters."""
        self._reference: Optional[torch.Tensor] = None
        self._ref_betti: Optional[int] = None
        self._tripped = False
        self._bad_run = 0
        self._good_run = 0

    def set_reference(self, cloud: torch.Tensor) -> None:
        """Latch ``cloud`` (``(N, D)``) as the healthy topological reference."""
        ref = torch.as_tensor(cloud)
        self._reference = ref
        self._ref_betti = None if self.eps is None else betti0(ref, self.eps)

    @torch.no_grad()
    def step(self, cloud: torch.Tensor) -> TopologyResult:
        """Judge one live state population ``cloud`` -- ``(N, D)``.

        The first call with no latched reference adopts ``cloud`` as the reference
        (clean by definition). Subsequent calls measure drift from it.
        """
        live = torch.as_tensor(cloud)

        if self._reference is None:
            # auto-latch the first cloud as the healthy reference.
            self.set_reference(live)
            return TopologyResult(srtd=0.0, betti0=self._ref_betti, tripped=False,
                                  reason="noref")

        div = float(srtd(self._reference, live))
        b0 = None if self.eps is None else betti0(live, self.eps)

        reason = ""
        if div > self.srtd_threshold:
            reason = "srtd"
        elif self._ref_betti is not None and b0 != self._ref_betti:
            reason = "betti"
        unhealthy = reason != ""

        # -- debounce FSM (mirrors CircuitBreaker) -----------------------------
        if unhealthy:
            self._bad_run += 1
            self._good_run = 0
        else:
            self._good_run += 1
            self._bad_run = 0
        if not self._tripped and self._bad_run >= self.trip_after:
            self._tripped = True
        elif self._tripped and self._good_run >= self.reset_after:
            self._tripped = False

        return TopologyResult(srtd=div, betti0=b0, tripped=self._tripped, reason=reason)

    __call__ = step
