"""
tests/test_ingest_ops.py -- composable sensor-ingestion / stream-alignment ops.

Every operator is pinned against an analytic ground truth:

  * linear resampling is *exact* on a linear ramp (and on a constant), and at
    the original timestamps returns the original samples;
  * zero-order-hold returns the most recent sample at-or-before each query;
  * query times outside the sample span hold the endpoints (no extrapolation);
  * nearest_sample_gap / coverage_mask flag the middle of a dropout and trust
    the well-sampled edges;
  * interval_jitter is zero for a perfectly uniform clock and signed for
    late / early samples;
  * the align_stream flagship composes the two: a jittered stream with a hole
    is resampled to a uniform grid whose gap steps are exactly the dropout.

The flagship handoff is the point -- covered steps feed the fixed-dt core, the
uncovered run is what BlindRolloutGuard coasts.
"""
import math
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.ingest_ops import (  # noqa: E402
    AlignedStream,
    resample_uniform,
    nearest_sample_gap,
    coverage_mask,
    interval_jitter,
    uniform_grid,
    align_stream,
)


# --- uniform_grid ----------------------------------------------------------


def test_uniform_grid_spans_the_samples():
    ts = torch.tensor([0.0, 0.3, 0.55, 1.0])
    g = uniform_grid(ts, dt=0.25)
    # smallest count of 0.25-steps from 0.0 that reaches 1.0 -> {0,.25,.5,.75,1.0}
    assert torch.allclose(g, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_uniform_grid_honours_overrides():
    ts = torch.tensor([0.0, 1.0])
    g = uniform_grid(ts, dt=0.5, t_start=2.0, n_steps=3)
    assert torch.allclose(g, torch.tensor([2.0, 2.5, 3.0]))


def test_uniform_grid_rejects_bad_dt():
    with pytest.raises(ValueError):
        uniform_grid(torch.tensor([0.0, 1.0]), dt=0.0)


# --- resample_uniform: linear is exact on a linear ramp --------------------


def test_linear_resample_is_exact_on_a_ramp():
    # samples of f(t) = 3 + 2t at jittered times; linear interp must recover it
    ts = torch.tensor([0.0, 0.17, 0.41, 0.73, 1.0])
    vals = 3.0 + 2.0 * ts
    out, q = resample_uniform(vals, ts, dt=0.25)
    assert torch.allclose(out, 3.0 + 2.0 * q, atol=1e-6)


def test_linear_resample_is_exact_on_a_constant():
    ts = torch.tensor([0.0, 0.31, 0.9, 1.4])
    vals = torch.full((4,), 7.5)
    out, _ = resample_uniform(vals, ts, dt=0.2)
    assert torch.allclose(out, torch.full_like(out, 7.5), atol=1e-6)


def test_resample_at_original_uniform_times_is_identity():
    ts = torch.tensor([0.0, 0.5, 1.0, 1.5])
    vals = torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]])
    out, q = resample_uniform(vals, ts, dt=0.5)
    assert torch.allclose(q, ts)
    assert torch.allclose(out, vals, atol=1e-6)


def test_resample_preserves_feature_axis_shape():
    ts = torch.tensor([0.0, 1.0, 2.0])
    vals2d = torch.randn(3, 4)
    out2d, _ = resample_uniform(vals2d, ts, dt=1.0)
    assert out2d.dim() == 2 and out2d.shape[1] == 4
    vals1d = torch.randn(3)
    out1d, _ = resample_uniform(vals1d, ts, dt=1.0)
    assert out1d.dim() == 1


# --- resample_uniform: zero-order hold -------------------------------------


def test_zoh_holds_the_most_recent_sample():
    ts = torch.tensor([0.0, 1.0, 2.0])
    vals = torch.tensor([10.0, 20.0, 30.0])
    out, q = resample_uniform(vals, ts, dt=0.5, mode="previous")
    # q = 0,.5,1,1.5,2 -> hold(0)=10, hold(.5)=10, hold(1)=20, hold(1.5)=20, hold(2)=30
    assert torch.allclose(out, torch.tensor([10.0, 10.0, 20.0, 20.0, 30.0]))


def test_zoh_differs_from_linear_midway():
    ts = torch.tensor([0.0, 1.0])
    vals = torch.tensor([0.0, 10.0])
    lin, _ = resample_uniform(vals, ts, dt=0.5, n_steps=2)   # q = 0, .5
    zoh, _ = resample_uniform(vals, ts, dt=0.5, n_steps=2, mode="previous")
    assert math.isclose(lin[1].item(), 5.0, abs_tol=1e-6)    # interpolated
    assert math.isclose(zoh[1].item(), 0.0, abs_tol=1e-6)    # held


# --- resample_uniform: endpoint clamping (no extrapolation) ----------------


def test_resample_holds_endpoints_outside_the_span():
    ts = torch.tensor([1.0, 2.0])
    vals = torch.tensor([5.0, 9.0])
    out, q = resample_uniform(vals, ts, dt=1.0, t_start=-1.0, n_steps=5)
    # q = -1,0,1,2,3 -> clamp to [1,2]: 5,5,5,9,9 (never extrapolated past 9)
    assert torch.allclose(out, torch.tensor([5.0, 5.0, 5.0, 9.0, 9.0]))


# --- resample_uniform: validation ------------------------------------------


def test_resample_rejects_non_monotonic_timestamps():
    with pytest.raises(ValueError):
        resample_uniform(torch.tensor([1.0, 2.0, 3.0]),
                         torch.tensor([0.0, 0.0, 1.0]), dt=0.5)


def test_resample_rejects_bad_mode():
    with pytest.raises(ValueError):
        resample_uniform(torch.tensor([1.0, 2.0]),
                         torch.tensor([0.0, 1.0]), dt=0.5, mode="cubic")


def test_resample_is_differentiable_in_values():
    ts = torch.tensor([0.0, 1.0, 2.0])
    vals = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out, _ = resample_uniform(vals, ts, dt=0.5)
    out.sum().backward()
    assert vals.grad is not None and torch.isfinite(vals.grad).all()


# --- nearest_sample_gap / coverage_mask ------------------------------------


def test_gap_is_zero_at_a_real_sample_and_peaks_mid_hole():
    ts = torch.tensor([0.0, 0.1, 2.0, 2.1])               # a hole between .1 and 2.0
    q = torch.tensor([0.0, 1.0, 1.05, 2.0])
    gap = nearest_sample_gap(ts, q)
    assert math.isclose(gap[0].item(), 0.0, abs_tol=1e-6)  # on a sample
    assert math.isclose(gap[3].item(), 0.0, abs_tol=1e-6)  # on a sample
    # 1.05 is the midpoint of the [0.1, 2.0] hole -> ~0.95 from either edge
    assert gap[2].item() > 0.9


def test_coverage_flags_the_dropout_middle_only():
    ts = torch.tensor([0.0, 0.1, 0.2, 1.5, 1.6])          # ~1.3s dropout
    q = torch.tensor([0.0, 0.2, 0.8, 1.5])
    cov = coverage_mask(ts, q, max_gap=0.2)
    assert cov.tolist() == [True, True, False, True]      # only mid-hole untrusted


def test_coverage_rejects_bad_max_gap():
    with pytest.raises(ValueError):
        coverage_mask(torch.tensor([0.0, 1.0]), torch.tensor([0.0]), max_gap=0.0)


# --- interval_jitter -------------------------------------------------------


def test_jitter_is_zero_for_a_uniform_clock():
    ts = torch.arange(5, dtype=torch.float32) * 0.1
    j = interval_jitter(ts, dt=0.1)
    assert torch.allclose(j, torch.zeros_like(j), atol=1e-6)


def test_jitter_signs_late_and_early_samples():
    ts = torch.tensor([0.0, 0.1, 0.25, 0.30])             # nominal dt=0.1
    j = interval_jitter(ts, dt=0.1)
    # intervals .1, .15, .05 -> deviations 0, +.05 (late), -.05 (early)
    assert torch.allclose(j, torch.tensor([0.0, 0.05, -0.05]), atol=1e-6)


# --- align_stream flagship -------------------------------------------------


def test_align_stream_resamples_and_marks_the_gap():
    # a clean run, a ~1s dropout, then recovery; dt=0.25, default max_gap=1.5*dt
    ts = torch.tensor([0.0, 0.25, 0.5, 1.6, 1.85, 2.1])
    vals = 1.0 + ts                                       # linear so values are exact
    s = align_stream(vals, ts, dt=0.25)
    assert isinstance(s, AlignedStream)
    assert torch.allclose(s.values, 1.0 + s.times, atol=1e-6)   # exact on the ramp
    assert not s.fully_covered                            # the hole was flagged
    assert s.n_gap_steps > 0
    # every uncovered step lies inside the [0.5, 1.6] dropout
    holes = s.times[~s.covered]
    assert bool(((holes > 0.5) & (holes < 1.6)).all())


def test_align_stream_clean_input_is_fully_covered():
    ts = torch.arange(6, dtype=torch.float32) * 0.2
    vals = torch.sin(ts)
    s = align_stream(vals, ts, dt=0.2)
    assert s.fully_covered and s.n_gap_steps == 0
    assert s.n_steps == s.times.shape[0]


def test_align_stream_is_deterministic():
    ts = torch.tensor([0.0, 0.3, 0.9, 1.0, 1.7])
    vals = torch.tensor([0.0, 1.0, -1.0, 2.0, 0.5])
    a = align_stream(vals, ts, dt=0.25)
    b = align_stream(vals, ts, dt=0.25)
    assert torch.equal(a.values, b.values)
    assert torch.equal(a.covered, b.covered)
