"""
mt_lnn/acoustic_ops.py -- composable acoustic / binaural-hearing operators (L4 step 4).

Why this module exists
----------------------
``spatial_ops.py`` computes over *geometry* and ``physics_ops.py`` evolves
*dynamics*. A third way an embodied agent grounds space is **hearing**: the brain
does not store "the sound is on my left" -- it *computes* direction from the tiny
physical differences between what the two ears receive. This module adds that
missing layer: a small set of **composable, zero-parameter, analytic** operators
that turn the geometry of a sound scene into the brain's auditory spatial cues,
and -- the flagship -- *invert* those cues back into a heading.

The brain mapping is literal, not decorative:

* **time of flight** -- sound travels at a finite speed ``c``; the delay from a
  source to a receiver is ``distance / c``. Everything else is built on this.
* **spherical spreading** -- a point source's pressure falls as ``1 / r``
  (energy ``1 / r^2`` over an expanding wavefront); this is the inverse-distance
  loudness cue.
* **ITD (interaural time difference)** -- the *primary* low-frequency
  localization cue: the wavefront reaches the near ear before the far ear. The
  medial superior olive reads this delay out as azimuth (the **Jeffress**
  coincidence-detector model).
* **ILD (interaural level difference)** -- the high-frequency cue: the near ear
  is louder than the far ear (head shadow / spreading), measured in dB.
* **Doppler shift** -- relative radial motion compresses or stretches the
  wavefront, raising the pitch of an approaching source and lowering a receding
  one -- the "coming or going?" cue.
* **wavefront superposition** -- multiple coherent sources sum *as phasors* at a
  receiver, so path differences produce constructive / destructive interference.
* **localization (inverse readout)** -- :func:`localize_azimuth` turns an ITD
  back into a bearing. This is the "compute, don't memorise" move: feed it cues
  produced by geometry and it recovers the direction.

Honest scope
------------
A deterministic *operator* layer (the acoustics), not a learned auditory model.
ITD->azimuth uses the standard far-field plane-wave model
``ITD = (head_width / c) * sin(theta)``; it is exact in the far field and
documented as approximate near the head -- tests pin the round-trip where the
model holds, and the cue operators against closed-form distances/phasors, not
against an unrealistic exact head-related transfer function.

Design / coupling
-----------------
* Pure functions, **stateless, zero trainable parameters**. Nothing here is an
  ``nn.Module``; no backbone coupling. Imports only ``torch`` (+ ``dataclasses``,
  ``math``); never imports ``model.py``.
* Batched / broadcasting: positions are ``(..., D)`` with ``D`` the spatial
  dimension (2 or 3); the cue operators broadcast over any leading shape, so a
  whole source trajectory ``(T, D)`` flows through at once.
* Everything is differentiable except the explicit ``clamp`` in the localization
  arcsin (documented per-function).

Every operator is pinned against an analytic ground truth in
``tests/test_acoustic_ops.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

__all__ = [
    "SPEED_OF_SOUND",
    "BinauralScene",
    "propagation_delay",
    "spherical_spreading_gain",
    "interaural_time_difference",
    "interaural_level_difference",
    "doppler_shift",
    "superpose_arrivals",
    "localize_azimuth",
    "binaural_scene",
]

# Speed of sound in dry air at 20 degrees C (m/s). The default propagation speed.
SPEED_OF_SOUND = 343.0


def _distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Euclidean distance over the last axis, broadcasting ``a`` and ``b``."""
    diff = a - b
    return torch.sqrt((diff * diff).sum(dim=-1).clamp_min(eps * eps))


# --------------------------------------------------------------------------- #
# 1. propagation primitives                                                   #
# --------------------------------------------------------------------------- #


def propagation_delay(source: torch.Tensor, receiver: torch.Tensor, *,
                      speed: float = SPEED_OF_SOUND) -> torch.Tensor:
    """Time of flight ``|source - receiver| / speed`` (seconds).

    ``source`` and ``receiver`` are ``(..., D)`` and broadcast against each other;
    the result is their broadcasted leading shape. This is the substrate every
    other cue is built on.
    """
    if speed <= 0.0:
        raise ValueError(f"speed must be > 0, got {speed}")
    return _distance(source, receiver) / speed


def spherical_spreading_gain(source: torch.Tensor, receiver: torch.Tensor, *,
                             ref_dist: float = 1.0, eps: float = 1e-9) -> torch.Tensor:
    """Inverse-distance pressure gain of a point source: ``ref_dist / r``.

    A point source radiates onto an expanding spherical wavefront, so the
    *pressure* amplitude falls as ``1 / r`` (intensity as ``1 / r^2``). The gain
    is normalised to ``1.0`` at ``ref_dist`` and clamped near the source so it
    stays finite. ``source`` / ``receiver`` are ``(..., D)``; returns ``(...)``.
    """
    if ref_dist <= 0.0:
        raise ValueError(f"ref_dist must be > 0, got {ref_dist}")
    r = _distance(source, receiver).clamp_min(eps)
    return ref_dist / r


# --------------------------------------------------------------------------- #
# 2. binaural cues                                                            #
# --------------------------------------------------------------------------- #


def interaural_time_difference(source: torch.Tensor, left_ear: torch.Tensor,
                               right_ear: torch.Tensor, *,
                               speed: float = SPEED_OF_SOUND) -> torch.Tensor:
    """ITD = ``delay_left - delay_right`` (seconds).

    Positive when the source is on the **right** (it reaches the right ear first,
    so the right delay is smaller). This is the primary low-frequency
    localization cue; :func:`localize_azimuth` reads it back out as a bearing.
    """
    dl = propagation_delay(source, left_ear, speed=speed)
    dr = propagation_delay(source, right_ear, speed=speed)
    return dl - dr


def interaural_level_difference(source: torch.Tensor, left_ear: torch.Tensor,
                                right_ear: torch.Tensor, *, ref_dist: float = 1.0,
                                eps: float = 1e-9) -> torch.Tensor:
    """ILD = ``level_right - level_left`` in decibels.

    Positive when the source is on the **right** (the near, right ear is louder).
    Computed from the spherical-spreading gains:
    ``20 * log10(gain_right / gain_left) = 20 * log10(r_left / r_right)``. This is
    the high-frequency localization cue, sign-aligned with the ITD.
    """
    gl = spherical_spreading_gain(source, left_ear, ref_dist=ref_dist, eps=eps)
    gr = spherical_spreading_gain(source, right_ear, ref_dist=ref_dist, eps=eps)
    return 20.0 * torch.log10((gr / gl.clamp_min(eps)).clamp_min(eps))


def doppler_shift(source: torch.Tensor, source_vel: torch.Tensor,
                  receiver: torch.Tensor, receiver_vel: torch.Tensor, *,
                  freq: float, speed: float = SPEED_OF_SOUND,
                  eps: float = 1e-9) -> torch.Tensor:
    """Observed frequency of a moving source heard by a moving receiver (Hz).

    Uses the standard non-relativistic formula along the source->receiver line of
    sight ``rhat``::

        f_obs = freq * (speed - v_recv . rhat) / (speed - v_src . rhat)

    where ``v . rhat`` is the velocity component *along* the source->receiver
    direction. An approaching source (``v_src . rhat > 0``) shrinks the
    denominator and raises the pitch; a receiver moving toward the source
    (``v_recv . rhat < 0``) raises it too. All tensors are ``(..., D)``; returns
    ``(...)``. The denominator is clamped away from zero (no shock-wave regime).
    """
    if freq <= 0.0:
        raise ValueError(f"freq must be > 0, got {freq}")
    if speed <= 0.0:
        raise ValueError(f"speed must be > 0, got {speed}")
    diff = receiver - source
    dist = torch.sqrt((diff * diff).sum(dim=-1, keepdim=True).clamp_min(eps * eps))
    rhat = diff / dist                                   # unit source -> receiver
    v_src_r = (source_vel * rhat).sum(dim=-1)            # (...,)
    v_recv_r = (receiver_vel * rhat).sum(dim=-1)         # (...,)
    denom = (speed - v_src_r)
    denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
    return freq * (speed - v_recv_r) / denom


# --------------------------------------------------------------------------- #
# 3. wavefront superposition (interference)                                   #
# --------------------------------------------------------------------------- #


def superpose_arrivals(sources: torch.Tensor, receiver: torch.Tensor, *,
                       freq: float, speed: float = SPEED_OF_SOUND,
                       amplitudes: Optional[torch.Tensor] = None,
                       ref_dist: float = 1.0, eps: float = 1e-9) -> torch.Tensor:
    """Complex pressure phasor of several coherent sources summed at a receiver.

    Each source contributes a delayed, spread, phase-rotated phasor; coherent
    sources interfere::

        p = sum_i  a_i * (ref_dist / r_i) * exp(-j * 2*pi * freq * (r_i / speed))

    Parameters
    ----------
    sources : (..., N, D)
        ``N`` point-source positions (the source axis is the second-to-last).
    receiver : (..., D)
        Receiver position; broadcast against the per-source axis.
    freq : float
        Common (coherent) source frequency in Hz.
    amplitudes : (..., N), optional
        Per-source source strength ``a_i`` (default all ones).
    ref_dist : float
        Reference distance for the spreading gain (gain ``1`` at ``ref_dist``).

    Returns
    -------
    torch.Tensor (complex), shape ``(...)``
        The summed phasor. ``.abs()`` is the received amplitude (interference
        magnitude); ``.angle()`` the received phase. Two equal in-phase arrivals
        double the amplitude; a half-wavelength path difference cancels them.
    """
    if freq <= 0.0:
        raise ValueError(f"freq must be > 0, got {freq}")
    recv = receiver.unsqueeze(-2)                        # (..., 1, D)
    r = _distance(sources, recv).clamp_min(eps)          # (..., N)
    gain = ref_dist / r                                  # (..., N)
    phase = -2.0 * math.pi * freq * (r / speed)          # (..., N)
    if amplitudes is None:
        amp = torch.ones_like(gain)
    else:
        amp = amplitudes.to(gain.dtype)
    phasor = (amp * gain) * torch.exp(1j * phase.to(torch.complex64))
    return phasor.sum(dim=-1)                            # (...,)


# --------------------------------------------------------------------------- #
# 4. localization (inverse readout) + scene composition                       #
# --------------------------------------------------------------------------- #


def localize_azimuth(itd: torch.Tensor, *, head_width: float,
                     speed: float = SPEED_OF_SOUND) -> torch.Tensor:
    """Recover source azimuth (radians) from an interaural time difference.

    Inverts the far-field plane-wave model ``ITD = (head_width / speed) sin(theta)``::

        theta = arcsin( clamp( ITD * speed / head_width, -1, 1 ) )

    ``theta`` is measured from straight ahead, **positive toward the right ear**
    (sign-aligned with :func:`interaural_time_difference`); ``+pi/2`` is hard
    right, ``-pi/2`` hard left, ``0`` dead ahead. The ``clamp`` keeps the arcsin
    valid when an out-of-range / noisy ITD is fed in (it saturates at the poles),
    which is the only non-differentiable point. Exact in the far field; an
    approximation close to the head.
    """
    if head_width <= 0.0:
        raise ValueError(f"head_width must be > 0, got {head_width}")
    sin_theta = (itd * speed / head_width).clamp(-1.0, 1.0)
    return torch.asin(sin_theta)


@dataclass
class BinauralScene:
    """Per-step auditory cues of a source moving relative to a binaural head.

    All fields are shaped ``(T,)`` over the trajectory length ``T``.

    Attributes
    ----------
    distance_left, distance_right : (T,)
        Source distance to each ear.
    itd : (T,)
        Interaural time difference (s); ``> 0`` when the source is to the right.
    ild : (T,)
        Interaural level difference (dB); ``> 0`` when the source is to the right.
    azimuth : (T,)
        Localized bearing (radians) recovered from ``itd`` (right-positive).
    doppler : (T,) or None
        Observed frequency (Hz) if ``freq`` was supplied and velocities given,
        else ``None`` -- ``> freq`` while approaching, ``< freq`` while receding.
    head_width : float
        Interaural distance used for the localization readout.
    """

    distance_left: torch.Tensor
    distance_right: torch.Tensor
    itd: torch.Tensor
    ild: torch.Tensor
    azimuth: torch.Tensor
    doppler: Optional[torch.Tensor]
    head_width: float


def binaural_scene(source: torch.Tensor, left_ear: torch.Tensor,
                   right_ear: torch.Tensor, *, source_vel: Optional[torch.Tensor] = None,
                   receiver_vel: Optional[torch.Tensor] = None,
                   freq: Optional[float] = None,
                   speed: float = SPEED_OF_SOUND, ref_dist: float = 1.0) -> BinauralScene:
    """Compose the cue operators over a whole source trajectory -- the flagship.

    Given a source path ``source`` ``(T, D)`` and a fixed binaural head
    (``left_ear`` / ``right_ear`` ``(D,)``), this rolls every per-instant cue at
    once and *localizes* the source back from its own ITD, so a caller gets, for
    each tick, "where is it (azimuth) and is it coming or going (Doppler)?" --
    computed from raw geometry, not memorised.

    Parameters
    ----------
    source : (T, D)
        Source position at each tick.
    left_ear, right_ear : (D,)
        The two ear positions (a stationary head).
    source_vel : (T, D), optional
        Source velocity per tick; required (with ``freq``) for the Doppler field.
    receiver_vel : (T, D), optional
        Head velocity per tick (default stationary).
    freq : float, optional
        Rest frequency (Hz) for the Doppler readout; ``None`` skips it.

    Returns
    -------
    BinauralScene
    """
    if source.dim() != 2:
        raise ValueError(f"source must be (T, D), got {tuple(source.shape)}")
    head_width = float(_distance(left_ear, right_ear).item())
    if head_width <= 0.0:
        raise ValueError("left_ear and right_ear must be distinct (head_width > 0)")

    dl = _distance(source, left_ear)
    dr = _distance(source, right_ear)
    itd = interaural_time_difference(source, left_ear, right_ear, speed=speed)
    ild = interaural_level_difference(source, left_ear, right_ear, ref_dist=ref_dist)
    azimuth = localize_azimuth(itd, head_width=head_width, speed=speed)

    doppler: Optional[torch.Tensor] = None
    if freq is not None and source_vel is not None:
        head_center = 0.5 * (left_ear + right_ear)
        rv = receiver_vel if receiver_vel is not None else torch.zeros_like(source_vel)
        center = head_center.expand_as(source)
        doppler = doppler_shift(source, source_vel, center, rv, freq=freq, speed=speed)

    return BinauralScene(distance_left=dl, distance_right=dr, itd=itd, ild=ild,
                         azimuth=azimuth, doppler=doppler, head_width=head_width)
