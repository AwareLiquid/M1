import pytest
pytest.importorskip("hypothesis")  # optional dev dep — see requirements-dev.txt

"""
tests/test_stdp_ops_properties.py -- property-based invariants for the
composable STDP operators.

``tests/test_stdp_ops.py`` pins these against chosen spike patterns; here we pin
the *plasticity laws themselves* with Hypothesis, over the whole space of spike
trains and window parameters:

- the window's sign is causal (>=0 for dt>0, <=0 for dt<0), it is bounded by the
  amplitudes, and decays monotonically away from the origin within each lobe;
- a balanced window (equal amplitudes and taus) is antisymmetric;
- the online eligibility-trace update equals the all-to-all pairwise sum exactly,
  for arbitrary multi-neuron trains;
- delta_w = potentiation - depression with both lobes non-negative;
- a purely causal train (all pre before all post) only potentiates; the reverse
  only depresses;
- the update is linear in the amplitudes and invariant to a global time shift.

All tensors are explicit float64; the global default dtype is left untouched so
the float32 model tests sharing this process are unaffected.

Run:  python -m pytest tests/test_stdp_ops_properties.py -v
"""
import sys

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, ".")

from mt_lnn.stdp_ops import (                                    # noqa: E402
    STDPParams,
    pairwise_stdp,
    stdp_kernel,
    stdp_trace_update,
)

PROP = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_F = dict(allow_nan=False, allow_infinity=False, width=64)
_f64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=_f64)


# --------------------------------------------------------------------------- #
# strategies                                                                  #
# --------------------------------------------------------------------------- #
@st.composite
def params(draw):
    return STDPParams(
        a_plus=draw(st.floats(0.0, 2.0, **_F)),
        a_minus=draw(st.floats(0.0, 2.0, **_F)),
        tau_plus=draw(st.floats(1.0, 40.0, **_F)),
        tau_minus=draw(st.floats(1.0, 40.0, **_F)),
        dt=1.0,
    )


@st.composite
def balanced_params(draw):
    a = draw(st.floats(0.1, 2.0, **_F))
    tau = draw(st.floats(1.0, 40.0, **_F))
    return STDPParams(a_plus=a, a_minus=a, tau_plus=tau, tau_minus=tau, dt=1.0)


@st.composite
def _train(draw, T, n):
    bits = draw(st.lists(st.integers(0, 1), min_size=T * n, max_size=T * n))
    return _t(bits).reshape(T, n)


@st.composite
def pre_post(draw, max_t=20, max_n=3):
    T = draw(st.integers(2, max_t))
    n_pre = draw(st.integers(1, max_n))
    n_post = draw(st.integers(1, max_n))
    return draw(_train(T, n_pre)), draw(_train(T, n_post)), T


# --------------------------------------------------------------------------- #
# window                                                                      #
# --------------------------------------------------------------------------- #
@PROP
@given(dt=st.floats(-60.0, 60.0, **_F), p=params())
def test_kernel_sign_is_causal_and_bounded(dt, p):
    w = float(stdp_kernel(_t(dt), p))
    if dt > 0:
        assert w >= -1e-12
    elif dt < 0:
        assert w <= 1e-12
    else:
        assert abs(w) < 1e-12
    assert abs(w) <= max(p.a_plus, p.a_minus) + 1e-9


@PROP
@given(d1=st.floats(0.01, 30.0, **_F), d2=st.floats(0.01, 30.0, **_F), p=params())
def test_kernel_decays_monotonically_within_each_lobe(d1, d2, p):
    lo, hi = min(d1, d2), max(d1, d2)
    # potentiation lobe (dt>0): larger gap -> smaller magnitude
    assert float(stdp_kernel(_t(hi), p)) <= float(stdp_kernel(_t(lo), p)) + 1e-12
    # depression lobe (dt<0): larger gap -> smaller magnitude (less negative)
    assert float(stdp_kernel(_t(-hi), p)) >= float(stdp_kernel(_t(-lo), p)) - 1e-12


@PROP
@given(dt=st.floats(0.01, 50.0, **_F), p=balanced_params())
def test_balanced_window_is_antisymmetric(dt, p):
    assert torch.isclose(stdp_kernel(_t(dt), p), -stdp_kernel(_t(-dt), p), atol=1e-9)


# --------------------------------------------------------------------------- #
# trace == pairwise                                                           #
# --------------------------------------------------------------------------- #
@PROP
@given(data=pre_post(), p=params())
def test_trace_update_equals_pairwise_for_every_synapse(data, p):
    pre, post, T = data
    res = stdp_trace_update(pre, post, p)
    grid = torch.arange(T, dtype=_f64)
    for i in range(pre.shape[1]):
        for j in range(post.shape[1]):
            expect = pairwise_stdp(grid[pre[:, i] > 0], grid[post[:, j] > 0], p)
            assert torch.isclose(res.delta_w[i, j], expect, atol=1e-9)


@PROP
@given(data=pre_post(), p=params())
def test_delta_w_splits_into_nonneg_lobes(data, p):
    pre, post, _ = data
    res = stdp_trace_update(pre, post, p)
    assert torch.allclose(res.delta_w, res.potentiation - res.depression, atol=1e-12)
    assert bool((res.potentiation >= -1e-12).all())
    assert bool((res.depression >= -1e-12).all())


# --------------------------------------------------------------------------- #
# causal structure                                                            #
# --------------------------------------------------------------------------- #
@PROP
@given(p=params())
def test_purely_causal_train_only_potentiates(p):
    assume(p.a_plus > 0 and p.a_minus > 0)
    T = 12
    pre = torch.zeros(T, dtype=_f64); pre[[1, 2, 3]] = 1.0      # all early
    post = torch.zeros(T, dtype=_f64); post[[8, 9, 10]] = 1.0   # all late
    res = stdp_trace_update(pre, post, p)
    assert torch.isclose(res.depression.squeeze(), _t(0.0), atol=1e-12)
    assert float(res.delta_w) >= -1e-12
    # role reversal: pure depression, no potentiation
    rev = stdp_trace_update(post, pre, p)
    assert torch.isclose(rev.potentiation.squeeze(), _t(0.0), atol=1e-12)
    assert float(rev.delta_w) <= 1e-12


# --------------------------------------------------------------------------- #
# linearity / invariance                                                      #
# --------------------------------------------------------------------------- #
@PROP
@given(data=pre_post(), p=params(), c=st.floats(0.1, 5.0, **_F))
def test_update_is_linear_in_the_amplitudes(data, p, c):
    pre, post, _ = data
    scaled = STDPParams(a_plus=c * p.a_plus, a_minus=c * p.a_minus,
                        tau_plus=p.tau_plus, tau_minus=p.tau_minus, dt=p.dt)
    base = stdp_trace_update(pre, post, p).delta_w
    got = stdp_trace_update(pre, post, scaled).delta_w
    assert torch.allclose(got, c * base, atol=1e-8)


@PROP
@given(p=params(), shift=st.floats(-50.0, 50.0, **_F))
def test_pairwise_is_time_shift_invariant(p, shift):
    pre = _t([1.0, 4.0, 9.0])
    post = _t([2.0, 5.0, 11.0])
    base = pairwise_stdp(pre, post, p)
    shifted = pairwise_stdp(pre + shift, post + shift, p)
    assert torch.isclose(base, shifted, atol=1e-8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
