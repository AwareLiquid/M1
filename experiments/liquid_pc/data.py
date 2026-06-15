"""
experiments/liquid_pc/data.py — multi-timescale sequence tasks for the
predictive-coding liquid-core validation experiment.

Why this task
-------------
The claimed advantage of a *multi-timescale liquid core* is that it can separate
slow and fast structure in a signal across its tau channels, and the claimed
advantage of *predictive coding* is that a hierarchy predicting itself captures
temporal regularity better. The cleanest signal to expose (or falsify) that is a
**superposition of oscillations at well-separated timescales** plus noise: a
model that only captures one timescale will mispredict, a model that factorises
timescales will predict the far future well.

Everything here is seeded and CPU-cheap so the whole experiment is reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class SignalSpec:
    """Parameters of a multi-timescale signal regime."""

    n_components: int = 3          # oscillations summed per output dim
    period_min: float = 4.0        # fastest period (steps)
    period_max: float = 60.0       # slowest period (steps)
    noise_std: float = 0.02        # observation noise
    amp_decay: float = 0.7         # slower components get larger amplitude


def _make_components(d_in: int, spec: SignalSpec, gen: torch.Generator):
    """Sample per-dim oscillation banks: (d_in, K) freqs / phases / amps."""
    K = spec.n_components
    # Geometrically spaced periods slow->fast, shared structure but per-dim phase.
    periods = torch.logspace(
        math.log10(spec.period_max), math.log10(spec.period_min), K
    )                                                        # (K,)
    omegas = (2.0 * math.pi / periods).view(1, K).expand(d_in, K).contiguous()
    phases = 2.0 * math.pi * torch.rand(d_in, K, generator=gen)
    # Slower (earlier) components carry more amplitude -> slow trend dominates.
    amps = torch.tensor([spec.amp_decay ** k for k in range(K)]).view(1, K)
    amps = amps.expand(d_in, K).contiguous()
    return omegas, phases, amps


def make_signal(
    n_seq: int,
    seq_len: int,
    d_in: int = 1,
    *,
    spec: SignalSpec | None = None,
    seed: int = 0,
    phase_jitter: bool = True,
) -> torch.Tensor:
    """Generate ``(n_seq, seq_len, d_in)`` multi-timescale signals.

    Each sequence is a sum of ``spec.n_components`` sinusoids at geometrically
    spaced periods (slow -> fast), with a per-sequence random phase offset so the
    model must infer phase online rather than memorise one trajectory.
    """
    spec = spec or SignalSpec()
    gen = torch.Generator().manual_seed(seed)
    omegas, phases, amps = _make_components(d_in, spec, gen)   # (d_in, K) each

    t = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1, 1)   # (1,T,1,1)
    om = omegas.view(1, 1, d_in, -1)                          # (1,1,d_in,K)
    ph = phases.view(1, 1, d_in, -1)
    am = amps.view(1, 1, d_in, -1)

    if phase_jitter:
        # per-sequence phase offset (N,1,d_in,K) broadcast over time
        seq_phase = 2.0 * math.pi * torch.rand(n_seq, 1, d_in, om.shape[-1], generator=gen)
    else:
        seq_phase = torch.zeros(n_seq, 1, d_in, om.shape[-1])

    waves = am * torch.sin(om * t + ph + seq_phase)          # (N,T,d_in,K)
    x = waves.sum(dim=-1)                                     # (N,T,d_in)
    # Normalise to unit-ish scale, then add observation noise.
    x = x / (x.std() + 1e-6)
    noise = spec.noise_std * torch.randn(n_seq, seq_len, d_in, generator=gen)
    return x + noise


def train_test_split(
    x: torch.Tensor, frac_train: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_train = int(round(frac_train * x.shape[0]))
    return x[:n_train], x[n_train:]


def two_regimes(
    n_seq: int, seq_len: int, d_in: int = 1, *, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two *different* timescale regimes for a continual-learning / forgetting probe.

    Regime A is slow-dominated, regime B is fast-dominated; a model that overfits
    B and forgets A will show degraded A-error after training on B.
    """
    spec_a = SignalSpec(period_min=20.0, period_max=80.0, amp_decay=0.6)
    spec_b = SignalSpec(period_min=3.0, period_max=12.0, amp_decay=0.6)
    a = make_signal(n_seq, seq_len, d_in, spec=spec_a, seed=seed)
    b = make_signal(n_seq, seq_len, d_in, spec=spec_b, seed=seed + 1000)
    return a, b


def three_regimes(
    n_seq: int, seq_len: int, d_in: int = 1, *, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three well-separated timescale regimes for A->B->C continual learning.

    Regime A is slow-dominated (periods 20-80), B is fast-dominated (3-12), and C
    is a mid-band regime (8-30) that overlaps neither extreme. The three bands are
    distinct enough that naively training A->B->C catastrophically overwrites the
    earlier regimes unless old knowledge is rehearsed. Each regime uses a
    well-separated seed offset so the per-sequence phases differ across tasks.
    """
    spec_a = SignalSpec(period_min=20.0, period_max=80.0, amp_decay=0.6)
    spec_b = SignalSpec(period_min=3.0, period_max=12.0, amp_decay=0.6)
    spec_c = SignalSpec(period_min=8.0, period_max=30.0, amp_decay=0.6)
    a = make_signal(n_seq, seq_len, d_in, spec=spec_a, seed=seed)
    b = make_signal(n_seq, seq_len, d_in, spec=spec_b, seed=seed + 1000)
    c = make_signal(n_seq, seq_len, d_in, spec=spec_c, seed=seed + 2000)
    return a, b, c
