"""
mt_lnn/ingest_ops.py -- composable sensor-ingestion / stream-alignment operators
(L4 step 5).

Why this module exists
----------------------
The liquid core discretizes its continuous-time dynamics with a **fixed** step:
``ProtofilamentLTC`` decays as ``exp(-dt / tau)`` where ``dt = config.dt`` is a
compile-time constant. That is correct *iff* the inputs actually arrive on a
uniform ``dt`` grid. Real sensors do not oblige -- samples come with jittered
inter-arrival times, occasional dropouts, and clock drift. Feeding those raw,
unevenly-spaced samples into a fixed-``dt`` recurrence silently violates the
discretization (a 2x-late sample is integrated as if it were on time).

``streaming.py`` solves a *different* problem -- carrying recurrent state across
single-token steps in O(1) -- and assumes its caller already hands it one step
per ``dt``. Nothing upstream turns a *timestamped*, irregular sensor stream into
that uniform grid. This module is that missing front-end: a small set of
**composable, zero-parameter, analytic** operators that resample a timestamped
stream onto the model's uniform ``dt`` lattice, and -- the honest part -- mark
which lattice steps fall inside a dropout too long to interpolate across, so the
downstream :class:`~mt_lnn.failsafe.BlindRolloutGuard` can *coast* the dark
middle instead of fabricating data over a hole.

The brain mapping is literal: a uniform clock (the ``dt`` lattice) plus
short-gap interpolation (the senses fill small holes), but when the signal is
gone long enough the agent stops trusting the input and *predicts* forward -- it
does not hallucinate a smooth interpolation across a second-long blackout.

What each operator does
-----------------------
* **resample_uniform** -- the substrate: piecewise-linear (or zero-order-hold)
  resampling of ``(T, F)`` samples at monotonic ``timestamps`` onto a uniform
  ``dt`` grid. Linear interpolation is *exact* on a linear ramp (the pinned
  ground truth); ZOH reproduces sample-and-hold.
* **nearest_sample_gap** -- for each lattice step, the time to the nearest real
  sample. The raw trust signal.
* **coverage_mask** -- thresholds that gap: ``True`` where a real sample is
  within ``max_gap`` (interpolation is trustworthy), ``False`` deep inside a
  dropout (hand off to blind rollout).
* **interval_jitter** -- per-interval deviation of arrival times from nominal
  ``dt`` (a diagnostic / clock-health probe).
* **align_stream** -- the flagship: compose resample + coverage into an
  :class:`AlignedStream` (uniform values, their times, the coverage mask, the
  gap-step count), ready to drive the fixed-``dt`` core and its failsafe.

Honest scope
------------
A deterministic *operator* layer (the clock alignment), not a learned filter:
no Kalman state, no learned interpolation. Linear / ZOH resampling is exact only
to its model order (linear signals for linear mode); across a long gap it does
not invent a high-order fit -- it flags the gap and defers to the predictor.

Design / coupling
-----------------
* Pure functions, **stateless, zero trainable parameters**. Nothing here is an
  ``nn.Module``; no backbone coupling. Imports only ``torch`` (+ ``dataclasses``,
  ``math``); never imports ``model.py``.
* Time-major: ``values`` are ``(T, F)`` with time on axis 0 (a 1-D ``(T,)``
  stream is accepted and squeezed back), mirroring ``acoustic_ops`` trajectories.
  ``timestamps`` are a strictly-increasing ``(T,)`` vector of seconds.
* The resampling is differentiable in the sample *values*; the coverage
  thresholds and are piecewise-constant (documented, not hidden).

Every operator is pinned against an analytic ground truth in
``tests/test_ingest_ops.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

__all__ = [
    "AlignedStream",
    "resample_uniform",
    "nearest_sample_gap",
    "coverage_mask",
    "interval_jitter",
    "uniform_grid",
    "align_stream",
]


def _as_2d(values: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Return ``(values_2d, had_feature_axis)``; add a feature axis if 1-D."""
    if values.dim() == 1:
        return values.unsqueeze(-1), False
    if values.dim() == 2:
        return values, True
    raise ValueError(
        f"values must be (T,) or (T, F), got shape {tuple(values.shape)}"
    )


def _validate(values: torch.Tensor, timestamps: torch.Tensor) -> None:
    if timestamps.dim() != 1:
        raise ValueError(
            f"timestamps must be 1-D (T,), got {tuple(timestamps.shape)}"
        )
    if values.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"values and timestamps disagree on T: "
            f"{values.shape[0]} vs {timestamps.shape[0]}"
        )
    if timestamps.numel() < 2:
        raise ValueError("need at least 2 timestamped samples to resample")
    diffs = timestamps[1:] - timestamps[:-1]
    if not bool(torch.all(diffs > 0)):
        raise ValueError("timestamps must be strictly increasing")


# --------------------------------------------------------------------------- #
# 1. uniform query lattice                                                    #
# --------------------------------------------------------------------------- #


def uniform_grid(timestamps: torch.Tensor, *, dt: float,
                 t_start: Optional[float] = None,
                 n_steps: Optional[int] = None) -> torch.Tensor:
    """The uniform ``dt`` query lattice ``t_start + dt * [0, 1, ..., n_steps-1]``.

    ``t_start`` defaults to the first timestamp; ``n_steps`` defaults to the
    smallest count that reaches the last timestamp. Returns ``(M,)`` seconds.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    t0 = float(timestamps[0].item()) if t_start is None else float(t_start)
    if n_steps is None:
        span = float(timestamps[-1].item()) - t0
        n_steps = int(math.floor(span / dt + 1e-9)) + 1
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    return t0 + dt * torch.arange(n_steps, dtype=timestamps.dtype,
                                  device=timestamps.device)


# --------------------------------------------------------------------------- #
# 2. resampling (the substrate)                                               #
# --------------------------------------------------------------------------- #


def resample_uniform(values: torch.Tensor, timestamps: torch.Tensor, *,
                     dt: float, t_start: Optional[float] = None,
                     n_steps: Optional[int] = None,
                     mode: str = "linear",
                     eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample a timestamped stream onto a uniform ``dt`` grid.

    Parameters
    ----------
    values : (T, F) or (T,)
        ``T`` samples (feature axis optional).
    timestamps : (T,)
        Strictly-increasing sample times (seconds).
    dt : float
        Target grid spacing; the fixed step the recurrent core assumes.
    t_start, n_steps : optional
        Override the query lattice (see :func:`uniform_grid`).
    mode : {"linear", "previous"}
        ``"linear"`` -- piecewise-linear interpolation (exact on a linear ramp).
        ``"previous"`` -- zero-order hold (sample-and-hold: the most recent
        sample at or before each query time).

    Returns
    -------
    (values_out, query_times)
        ``values_out`` is ``(M, F)`` (or ``(M,)`` if ``values`` was 1-D);
        ``query_times`` is ``(M,)``. Query times outside the sample span hold the
        nearest endpoint (clamped, never extrapolated). Differentiable in
        ``values``.
    """
    if mode not in ("linear", "previous"):
        raise ValueError(f"mode must be 'linear' or 'previous', got {mode!r}")
    v2d, had_feat = _as_2d(values)
    _validate(v2d, timestamps)
    ts = timestamps
    q = uniform_grid(ts, dt=dt, t_start=t_start, n_steps=n_steps)
    qc = q.clamp(ts[0], ts[-1])                          # never extrapolate
    T = ts.shape[0]

    if mode == "previous":
        # most recent sample with timestamp <= qc (sample-and-hold)
        j = (torch.searchsorted(ts, qc, right=True) - 1).clamp(0, T - 1)
        out = v2d.index_select(0, j)
    else:
        hi = torch.searchsorted(ts, qc, right=True).clamp(1, T - 1)
        lo = hi - 1
        t0 = ts.index_select(0, lo)
        t1 = ts.index_select(0, hi)
        v0 = v2d.index_select(0, lo)
        v1 = v2d.index_select(0, hi)
        w = ((qc - t0) / (t1 - t0).clamp_min(eps)).clamp(0.0, 1.0).unsqueeze(-1)
        out = v0 + w * (v1 - v0)

    return (out if had_feat else out.squeeze(-1)), q


# --------------------------------------------------------------------------- #
# 3. trust signals (gap / coverage / jitter)                                  #
# --------------------------------------------------------------------------- #


def nearest_sample_gap(timestamps: torch.Tensor,
                       query_times: torch.Tensor) -> torch.Tensor:
    """Time from each query step to the nearest real sample (seconds, ``(M,)``).

    Small where a true sample sits near the query (interpolation is well
    supported); large in the middle of a dropout. The raw trust signal that
    :func:`coverage_mask` thresholds. Endpoints are handled (a query before the
    first / after the last sample measures to that endpoint).
    """
    if timestamps.dim() != 1:
        raise ValueError("timestamps must be 1-D (T,)")
    ts = timestamps
    T = ts.shape[0]
    pos = torch.searchsorted(ts, query_times)            # left insertion
    hi = pos.clamp(0, T - 1)
    lo = (pos - 1).clamp(0, T - 1)
    d_hi = (query_times - ts.index_select(0, hi)).abs()
    d_lo = (query_times - ts.index_select(0, lo)).abs()
    return torch.minimum(d_lo, d_hi)


def coverage_mask(timestamps: torch.Tensor, query_times: torch.Tensor, *,
                  max_gap: float) -> torch.Tensor:
    """Boolean ``(M,)``: is each query step backed by a real sample within ``max_gap``?

    ``True``  -- a true sample lies within ``max_gap`` seconds: the resampled
    value is trustworthy.
    ``False`` -- the nearest sample is farther than ``max_gap`` (deep inside a
    dropout, or outside the sample span): defer to the predictor / blind rollout
    rather than trusting a value interpolated across the hole.
    """
    if max_gap <= 0.0:
        raise ValueError(f"max_gap must be > 0, got {max_gap}")
    return nearest_sample_gap(timestamps, query_times) <= float(max_gap)


def interval_jitter(timestamps: torch.Tensor, *,
                    dt: Optional[float] = None) -> torch.Tensor:
    """Per-interval deviation of arrival spacing from nominal ``dt`` (``(T-1,)``).

    ``timestamps[i+1] - timestamps[i] - dt``: zero for a perfectly uniform clock,
    positive for a late sample, negative for an early one. ``dt`` defaults to the
    mean inter-arrival time (self-referential jitter). A clock-health diagnostic;
    the RMS of this is the usual scalar jitter figure.
    """
    if timestamps.dim() != 1 or timestamps.numel() < 2:
        raise ValueError("timestamps must be 1-D (T,) with T >= 2")
    iv = timestamps[1:] - timestamps[:-1]
    nominal = float(iv.mean().item()) if dt is None else float(dt)
    return iv - nominal


# --------------------------------------------------------------------------- #
# 4. flagship: compose resample + coverage                                    #
# --------------------------------------------------------------------------- #


@dataclass
class AlignedStream:
    """A timestamped sensor stream aligned onto the core's fixed-``dt`` lattice.

    Attributes
    ----------
    values : (M, F) or (M,)
        Resampled values on the uniform grid -- feed these to the recurrent core.
    times : (M,)
        The uniform query times (``t_start + dt * k``).
    covered : (M,) bool
        ``True`` where a real sample backs the step within ``max_gap`` (trust the
        value), ``False`` inside a dropout (coast: hand the step to
        :class:`~mt_lnn.failsafe.BlindRolloutGuard`).
    dt : float
        The grid spacing the stream was aligned to.
    """

    values: torch.Tensor
    times: torch.Tensor
    covered: torch.Tensor
    dt: float

    @property
    def n_steps(self) -> int:
        """Number of uniform steps produced."""
        return int(self.times.shape[0])

    @property
    def n_gap_steps(self) -> int:
        """Number of steps inside a dropout (handed off to blind rollout)."""
        return int((~self.covered).sum().item())

    @property
    def fully_covered(self) -> bool:
        """True when no step fell inside an untrusted gap."""
        return bool(self.covered.all().item())


def align_stream(values: torch.Tensor, timestamps: torch.Tensor, *,
                 dt: float, max_gap: Optional[float] = None,
                 t_start: Optional[float] = None,
                 n_steps: Optional[int] = None,
                 mode: str = "linear") -> AlignedStream:
    """Resample a jittered stream to uniform ``dt`` and mark its trustworthy steps.

    The flagship composition: turn a timestamped, irregular sensor stream into
    the uniform-``dt`` grid the fixed-step liquid core assumes, *and* report which
    steps are real-sample-backed (trust the value) versus deep inside a dropout
    (coast). ``max_gap`` defaults to ``1.5 * dt`` -- one slightly-late sample is
    tolerated, but a hole two or more steps wide is flagged.

    Parameters
    ----------
    values : (T, F) or (T,)
    timestamps : (T,)  strictly increasing seconds
    dt : float  target grid spacing (the core's fixed step)
    max_gap : float, optional  coverage threshold (default ``1.5 * dt``)
    mode : {"linear", "previous"}  resampling kernel

    Returns
    -------
    AlignedStream
    """
    out, q = resample_uniform(values, timestamps, dt=dt, t_start=t_start,
                              n_steps=n_steps, mode=mode)
    gap = 1.5 * dt if max_gap is None else max_gap
    covered = coverage_mask(timestamps, q, max_gap=gap)
    return AlignedStream(values=out, times=q, covered=covered, dt=float(dt))
