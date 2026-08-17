"""
tests/test_ingest_ops_properties.py -- property-based contracts for the sensor
ingestion / stream-alignment operators.

``tests/test_ingest_ops.py`` pins these operators against hand-picked analytic
examples. This file pins the *invariants themselves* with Hypothesis: instead of
trusting a ramp at three chosen points, we assert that linear resampling is exact
on **every** affine signal over **every** strictly-increasing clock, that ZOH
only ever emits real sample values, that coverage is monotone in ``max_gap``, and
so on. Hypothesis searches the input space and shrinks any violation to a minimal
counterexample -- exactly the regime hand-written cases miss.

Everything runs in float64 so the "exact" invariants can be asserted tightly.

Run:  python -m pytest tests/test_ingest_ops_properties.py -v
"""
import sys

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, ".")

from mt_lnn.ingest_ops import (                                   # noqa: E402
    align_stream,
    coverage_mask,
    interval_jitter,
    nearest_sample_gap,
    resample_uniform,
)

# Every tensor below is built with an explicit float64 dtype -- we deliberately
# do NOT touch torch's global default dtype, which would leak into the float32
# model tests sharing this pytest process.

# The autouse RNG fixture in conftest.py is function-scoped; Hypothesis re-runs
# the body many times per test, so silence that (expected) health check. None of
# these properties depend on torch's RNG -- all data comes from Hypothesis.
PROP = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_finite = dict(allow_nan=False, allow_infinity=False, width=64)


# --------------------------------------------------------------------------- #
# strategies                                                                  #
# --------------------------------------------------------------------------- #
@st.composite
def monotonic_clock(draw, min_len=2, max_len=10):
    """A strictly-increasing (T,) timestamp vector, gaps in [0.05, 3.0]."""
    n = draw(st.integers(min_len, max_len))
    t0 = draw(st.floats(-5.0, 5.0, **_finite))
    gaps = draw(st.lists(st.floats(0.05, 3.0, **_finite),
                         min_size=n - 1, max_size=n - 1))
    ts = [t0]
    for g in gaps:
        ts.append(ts[-1] + g)
    return torch.tensor(ts, dtype=torch.float64)


@st.composite
def clock_and_values(draw, n_feat=1, min_len=2, max_len=10):
    """A clock plus arbitrary (T, n_feat) sample values."""
    ts = draw(monotonic_clock(min_len, max_len))
    T = int(ts.shape[0])
    rows = draw(st.lists(
        st.lists(st.floats(-50.0, 50.0, **_finite), min_size=n_feat, max_size=n_feat),
        min_size=T, max_size=T))
    v = torch.tensor(rows, dtype=torch.float64)
    return ts, (v.squeeze(-1) if n_feat == 1 else v)


_dt = st.floats(0.05, 2.0, **_finite)


# --------------------------------------------------------------------------- #
# resample_uniform -- linear mode                                             #
# --------------------------------------------------------------------------- #
@PROP
@given(ts=monotonic_clock(), dt=_dt,
       a=st.floats(-5.0, 5.0, **_finite), b=st.floats(-10.0, 10.0, **_finite))
def test_linear_resample_is_exact_on_any_affine_signal(ts, dt, a, b):
    """The defining property of linear interpolation: it reproduces *any* affine
    signal exactly at every query (queries are clamped into the sample span)."""
    values = a * ts + b
    out, q = resample_uniform(values, ts, dt=dt, mode="linear")
    expected = a * q.clamp(ts[0], ts[-1]) + b
    assert torch.allclose(out, expected, atol=1e-7, rtol=1e-7)


@PROP
@given(data=clock_and_values(n_feat=2), dt=_dt)
def test_linear_resample_never_leaves_the_convex_hull(data, dt):
    """Clamped linear interpolation cannot extrapolate: every output value stays
    within the per-feature [min, max] of the inputs."""
    ts, v = data
    out, _ = resample_uniform(v, ts, dt=dt, mode="linear")
    lo = v.amin(dim=0)
    hi = v.amax(dim=0)
    assert torch.all(out >= lo - 1e-9)
    assert torch.all(out <= hi + 1e-9)


# --------------------------------------------------------------------------- #
# resample_uniform -- zero-order hold                                         #
# --------------------------------------------------------------------------- #
@PROP
@given(data=clock_and_values(n_feat=1), dt=_dt)
def test_zoh_only_ever_emits_a_real_sample_value(data, dt):
    """Sample-and-hold may only return values that actually occurred."""
    ts, v = data
    out, _ = resample_uniform(v, ts, dt=dt, mode="previous")
    sample_set = set(v.tolist())
    for x in out.tolist():
        assert any(abs(x - s) <= 1e-12 for s in sample_set)


@PROP
@given(data=clock_and_values(n_feat=1), dt=_dt)
def test_zoh_holds_the_most_recent_past_sample(data, dt):
    """Each ZOH output equals the sample whose timestamp is the latest one at or
    before the (clamped) query time."""
    ts, v = data
    out, q = resample_uniform(v, ts, dt=dt, mode="previous")
    qc = q.clamp(ts[0], ts[-1])
    T = ts.shape[0]
    j = (torch.searchsorted(ts, qc, right=True) - 1).clamp(0, T - 1)
    assert torch.equal(out, v.index_select(0, j))


# --------------------------------------------------------------------------- #
# nearest_sample_gap / coverage_mask                                          #
# --------------------------------------------------------------------------- #
@PROP
@given(ts=monotonic_clock(), dt=_dt)
def test_gap_is_nonnegative_and_vanishes_at_the_samples(ts, dt):
    _, q = resample_uniform(torch.zeros_like(ts), ts, dt=dt)
    gap = nearest_sample_gap(ts, q)
    assert torch.all(gap >= 0.0)
    # measured at the sample times themselves, the gap is exactly zero
    at_samples = nearest_sample_gap(ts, ts)
    assert torch.allclose(at_samples, torch.zeros_like(ts), atol=1e-9)


@PROP
@given(ts=monotonic_clock(), dt=_dt,
       g1=st.floats(0.05, 5.0, **_finite), g2=st.floats(0.05, 5.0, **_finite))
def test_coverage_is_monotone_in_max_gap(ts, dt, g1, g2):
    """A looser tolerance can only ever *add* covered steps, never remove one."""
    assume(g1 <= g2)
    _, q = resample_uniform(torch.zeros_like(ts), ts, dt=dt)
    c1 = coverage_mask(ts, q, max_gap=g1)
    c2 = coverage_mask(ts, q, max_gap=g2)
    # everything covered at the tighter threshold is covered at the looser one
    assert torch.all(c2 | ~c1)


@PROP
@given(ts=monotonic_clock(), dt=_dt, g=st.floats(0.05, 5.0, **_finite))
def test_coverage_is_exactly_the_gap_threshold(ts, dt, g):
    _, q = resample_uniform(torch.zeros_like(ts), ts, dt=dt)
    gap = nearest_sample_gap(ts, q)
    assert torch.equal(coverage_mask(ts, q, max_gap=g), gap <= g)


# --------------------------------------------------------------------------- #
# interval_jitter                                                             #
# --------------------------------------------------------------------------- #
@PROP
@given(t0=st.floats(-5.0, 5.0, **_finite), dt=_dt, n=st.integers(2, 20))
def test_jitter_is_zero_on_a_perfectly_uniform_clock(t0, dt, n):
    ts = t0 + dt * torch.arange(n, dtype=torch.float64)
    assert torch.allclose(interval_jitter(ts, dt=dt),
                          torch.zeros(n - 1, dtype=torch.float64), atol=1e-9)


@PROP
@given(ts=monotonic_clock())
def test_self_referential_jitter_sums_to_zero(ts):
    """Deviations from the *mean* interval must cancel out."""
    j = interval_jitter(ts)               # dt=None -> nominal = mean interval
    assert abs(float(j.sum())) <= 1e-6 * (1.0 + abs(float(ts[-1] - ts[0])))


# --------------------------------------------------------------------------- #
# align_stream -- the flagship composition                                    #
# --------------------------------------------------------------------------- #
@PROP
@given(data=clock_and_values(n_feat=2), dt=_dt)
def test_align_stream_is_deterministic_and_consistent_with_its_parts(data, dt):
    ts, v = data
    a = align_stream(v, ts, dt=dt)
    b = align_stream(v, ts, dt=dt)
    # deterministic
    assert torch.equal(a.values, b.values)
    assert torch.equal(a.covered, b.covered)
    # the composition equals resample + coverage applied separately
    out, q = resample_uniform(v, ts, dt=dt)
    assert torch.equal(a.values, out)
    assert torch.equal(a.times, q)
    assert torch.equal(a.covered, coverage_mask(ts, q, max_gap=1.5 * dt))
    # the gap-step accounting matches the mask
    assert a.n_gap_steps == int((~a.covered).sum())
    assert a.fully_covered == bool(a.covered.all())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
