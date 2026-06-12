"""
mt_lnn/salience_events.py -- global-workspace ignition / state-change events.

Why this module exists
----------------------
The fast liquid core runs hot at every tick. A slow / symbolic / cloud layer
(business logic, safety policy, an escalation LLM) must NOT poll that hot loop --
it should be *woken* only when something salient changes. This module is that
tripwire: a zero-parameter, model-decoupled **observer** that turns a stream of
per-step salience signals into discrete state-change events, so the dual-speed
engine has a real interface instead of a diagram.

The brain mapping is literal, not decorative:

* **surprise (predictive coding)** -- the natural salience signal is the world
  model's normalised prediction error (``last_pred_error``): how badly "what
  happens next" was mispredicted. High surprise == something changed.
* **adaptation / homeostasis** -- an EMA baseline tracks the *recent normal*, and
  salience is measured as a z-score against it. A slow drift the system has
  adapted to does not fire; a sharp departure does. The baseline is updated only
  during quiet periods, so a salient event does not contaminate "normal".
* **global-workspace ignition (GWT)** -- an event fires only when the z-score
  crosses an *ignition* threshold (it "wins the workspace"); the event carries
  the salience snapshot as its broadcast payload.
* **Schmitt hysteresis** -- a lower *release* threshold ends the ignited state,
  debouncing the on/off transition the way membrane dynamics do.
* **refractory period** -- after any event the detector is briefly silent, so a
  noisy signal cannot trigger an event storm.

Design / coupling
-----------------
* A small **stateful observer**, but **zero trainable parameters**; not an
  ``nn.Module``. Imports only the standard library; never imports ``model.py``.
* Duck-typed on the model: :func:`world_model_surprise` reads ``last_pred_error``
  off any predictive head, but the detector itself just consumes a scalar -- feed
  it GWTB bid margin, a sensor residual, anything.
* Online: :meth:`SalienceEventDetector.update` takes one sample and returns an
  optional event; :meth:`observe` runs a whole sequence. Deterministic.

Pinned against deterministic synthetic streams in
``tests/test_salience_events.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = [
    "StateChangeEvent",
    "SalienceEventDetector",
    "world_model_surprise",
]


@dataclass(frozen=True)
class StateChangeEvent:
    """A salient state change, broadcast to the slow layer.

    Attributes
    ----------
    step : int          1-based index of the sample that triggered it
    kind : str          "ignition" (salience crossed up) or "quiescence" (settled)
    salience : float    the z-score that triggered it (surprise in sigma)
    signal : float      raw signal value at the trigger
    baseline : float    running baseline (recent normal) at the trigger
    z : float           alias of ``salience`` (signed std above baseline)
    """

    step: int
    kind: str
    salience: float
    signal: float
    baseline: float
    z: float


class SalienceEventDetector:
    """Turn a salience stream into ignition / quiescence events.

    Parameters
    ----------
    ignite_z : float
        z-score (std above the adaptive baseline) needed to fire an ``ignition``.
    release_z : float
        z-score at/below which an ignited state ends with a ``quiescence`` event.
        Must be ``< ignite_z`` (the hysteresis gap).
    refractory : int
        Number of steps the detector stays silent after any event (>= 0).
    ema_decay : float
        Baseline/variance EMA retention in ``(0, 1)``; higher = slower adaptation.
    warmup : int
        Number of initial samples used only to seed the baseline (no events).
    eps : float
        Variance floor, keeps the z-score finite on a flat baseline.
    """

    def __init__(self, *, ignite_z: float = 3.0, release_z: float = 1.0,
                 refractory: int = 5, ema_decay: float = 0.9,
                 warmup: int = 8, eps: float = 1e-6) -> None:
        if not (release_z < ignite_z):
            raise ValueError(f"need release_z < ignite_z, got {release_z} !< {ignite_z}")
        if refractory < 0:
            raise ValueError(f"refractory must be >= 0, got {refractory}")
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")
        if warmup < 1:
            raise ValueError(f"warmup must be >= 1, got {warmup}")
        self.ignite_z = float(ignite_z)
        self.release_z = float(release_z)
        self.refractory = int(refractory)
        self.ema_decay = float(ema_decay)
        self.warmup = int(warmup)
        self.eps = float(eps)
        self.reset()

    # -- introspection -----------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """Zero trainable parameters -- a pure observer."""
        return 0

    @property
    def state(self) -> str:
        return self._state                              # "quiescent" | "ignited"

    @property
    def baseline(self) -> float:
        return self._mean

    @property
    def sigma(self) -> float:
        return math.sqrt(max(self._sq - self._mean * self._mean, self.eps))

    @property
    def n_events(self) -> int:
        return self._n_events

    def reset(self) -> None:
        """Clear all running state (baseline, FSM, counters)."""
        self._step = 0
        self._mean = 0.0
        self._sq = 0.0
        self._init = False
        self._state = "quiescent"
        self._refractory_left = 0
        self._n_events = 0

    # -- core --------------------------------------------------------------

    def _accumulate(self, x: float) -> None:
        if not self._init:
            self._mean, self._sq, self._init = x, x * x, True
            return
        d = self.ema_decay
        self._mean = d * self._mean + (1.0 - d) * x
        self._sq = d * self._sq + (1.0 - d) * (x * x)

    def update(self, signal: float) -> Optional[StateChangeEvent]:
        """Feed one salience sample; return an event if one fired, else ``None``.

        Salience is treated as one-sided: events fire on *upward* departures from
        the baseline (surprise / error magnitude grows). Feed a non-negative
        signal such as a prediction error or a residual magnitude.
        """
        self._step += 1
        x = float(signal)

        # warmup: only learn the baseline, never fire
        if self._step <= self.warmup:
            self._accumulate(x)
            return None

        sigma = math.sqrt(max(self._sq - self._mean * self._mean, self.eps))
        z = (x - self._mean) / sigma

        event: Optional[StateChangeEvent] = None
        if self._refractory_left > 0:
            self._refractory_left -= 1
        elif self._state == "quiescent":
            if z >= self.ignite_z:
                event = StateChangeEvent(self._step, "ignition", z, x, self._mean, z)
                self._state = "ignited"
                self._refractory_left = self.refractory
                self._n_events += 1
        else:  # ignited
            if z <= self.release_z:
                event = StateChangeEvent(self._step, "quiescence", z, x, self._mean, z)
                self._state = "quiescent"
                self._refractory_left = self.refractory
                self._n_events += 1

        # adapt the baseline on any event-free, non-refractory sample (in either
        # state). The trigger sample and the refractory window are skipped so a
        # salient event does not pull "normal" toward itself, but a *sustained*
        # regime change is slowly re-baselined -- so the detector quiesces and
        # re-arms to catch the next change instead of latching forever.
        if event is None and self._refractory_left == 0:
            self._accumulate(x)
        return event

    __call__ = update

    def observe(self, signals: Sequence[float]) -> List[StateChangeEvent]:
        """Reset, run an entire sequence, and return every event it produced."""
        self.reset()
        out: List[StateChangeEvent] = []
        for s in signals:
            ev = self.update(s)
            if ev is not None:
                out.append(ev)
        return out


def world_model_surprise(head) -> float:
    """Read the normalised prediction error (surprise) off a predictive head.

    Duck-typed: any object exposing ``last_pred_error`` works (e.g.
    :class:`mt_lnn.world_model.PredictiveStateHead`). This is the decoupled bridge
    that lets the detector watch the world model without importing it. Returns a
    plain ``float`` in ``[0, 1]``.
    """
    if not hasattr(head, "last_pred_error"):
        raise TypeError("head must expose `last_pred_error` (a predictive head)")
    val = head.last_pred_error
    try:
        return float(val.item())                        # torch scalar
    except AttributeError:
        return float(val)
