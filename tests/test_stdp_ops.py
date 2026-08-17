"""
tests/test_stdp_ops.py -- composable STDP operators pinned against the analytic
spike-timing window.

These fix the spike-timing-dependent-plasticity rule on hand-computable spike
patterns (single pairs, pre-leading vs post-leading trains, simultaneous
spikes), where the canonical exponential window value, the all-to-all pair sum,
and the online eligibility-trace update are all known by hand.
``tests/test_stdp_ops_properties.py`` pins the laws over the whole input space.

All tensors are explicit float64; the global default dtype is left untouched.

Run:  python -m pytest tests/test_stdp_ops.py -v
"""
import math
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn.stdp_ops import (                                    # noqa: E402
    STDPParams,
    STDPResult,
    pairwise_stdp,
    stdp_kernel,
    stdp_trace_update,
    window_integral,
)

_f64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=_f64)


_P = STDPParams(a_plus=1.2, a_minus=0.8, tau_plus=15.0, tau_minus=25.0, dt=1.0)
_BAL = STDPParams(a_plus=1.0, a_minus=1.0, tau_plus=20.0, tau_minus=20.0, dt=1.0)


# --------------------------------------------------------------------------- #
# the window                                                                  #
# --------------------------------------------------------------------------- #
def test_kernel_matches_the_exponential_window_at_the_taus():
    assert torch.isclose(stdp_kernel(_t(15.0), _P), _t(1.2 * math.exp(-1.0)), atol=1e-12)
    assert torch.isclose(stdp_kernel(_t(-25.0), _P), _t(-0.8 * math.exp(-1.0)), atol=1e-12)


def test_kernel_is_zero_at_simultaneity():
    assert torch.isclose(stdp_kernel(_t(0.0), _P), _t(0.0), atol=1e-12)


def test_kernel_sign_is_causal():
    # pre before post (dt>0) potentiates; post before pre (dt<0) depresses
    assert float(stdp_kernel(_t(5.0), _P)) > 0.0
    assert float(stdp_kernel(_t(-5.0), _P)) < 0.0


def test_kernel_is_antisymmetric_when_balanced():
    dt = _t([1.0, 3.0, 8.0, 20.0])
    assert torch.allclose(stdp_kernel(dt, _BAL), -stdp_kernel(-dt, _BAL), atol=1e-12)


def test_window_integral_is_amplitude_times_tau():
    ltp, ltd = window_integral(_P)
    assert math.isclose(ltp, 1.2 * 15.0, rel_tol=1e-12)
    assert math.isclose(ltd, 0.8 * 25.0, rel_tol=1e-12)
    # the balanced window has equal lobes
    bltp, bltd = window_integral(_BAL)
    assert math.isclose(bltp, bltd, rel_tol=1e-12)


# --------------------------------------------------------------------------- #
# pairwise (definition)                                                       #
# --------------------------------------------------------------------------- #
def test_pairwise_single_pair_is_the_window_value():
    # one pre at t=0, one post at t=4 -> dt=+4 -> LTP
    dw = pairwise_stdp(_t([0.0]), _t([4.0]), _P)
    assert torch.isclose(dw, _t(1.2 * math.exp(-4.0 / 15.0)), atol=1e-12)
    # reverse the order -> LTD (negative)
    dw_rev = pairwise_stdp(_t([4.0]), _t([0.0]), _P)
    assert torch.isclose(dw_rev, _t(-0.8 * math.exp(-4.0 / 25.0)), atol=1e-12)


def test_pairwise_empty_train_is_zero():
    assert torch.isclose(pairwise_stdp(_t([]), _t([1.0, 2.0]), _P), _t(0.0))
    assert torch.isclose(pairwise_stdp(_t([1.0]), _t([]), _P), _t(0.0))


def test_pairwise_depends_only_on_time_differences():
    a = pairwise_stdp(_t([1.0, 4.0]), _t([2.0, 7.0]), _P)
    b = pairwise_stdp(_t([101.0, 104.0]), _t([102.0, 107.0]), _P)   # shifted +100
    assert torch.isclose(a, b, atol=1e-10)


# --------------------------------------------------------------------------- #
# trace form == pairwise                                                      #
# --------------------------------------------------------------------------- #
def test_trace_update_equals_pairwise_on_a_single_synapse():
    T = 10
    pre = torch.zeros(T, dtype=_f64); pre[[1, 4]] = 1.0
    post = torch.zeros(T, dtype=_f64); post[[2, 7]] = 1.0
    res = stdp_trace_update(pre, post, _P)
    assert isinstance(res, STDPResult)
    pw = pairwise_stdp(_t([1.0, 4.0]), _t([2.0, 7.0]), _P)
    assert torch.isclose(res.delta_w.squeeze(), pw, atol=1e-10)


def test_trace_update_matches_pairwise_per_synapse_multineuron():
    g = torch.Generator().manual_seed(0)
    T, n_pre, n_post = 18, 3, 2
    pre = (torch.rand(T, n_pre, generator=g) > 0.6).to(_f64)
    post = (torch.rand(T, n_post, generator=g) > 0.6).to(_f64)
    res = stdp_trace_update(pre, post, _P)
    grid = torch.arange(T, dtype=_f64)
    for i in range(n_pre):
        for j in range(n_post):
            pre_times = grid[pre[:, i] > 0]
            post_times = grid[post[:, j] > 0]
            expect = pairwise_stdp(pre_times, post_times, _P)
            assert torch.isclose(res.delta_w[i, j], expect, atol=1e-9)


def test_delta_w_is_potentiation_minus_depression():
    g = torch.Generator().manual_seed(1)
    pre = (torch.rand(20, 2, generator=g) > 0.5).to(_f64)
    post = (torch.rand(20, 2, generator=g) > 0.5).to(_f64)
    res = stdp_trace_update(pre, post, _P)
    assert torch.allclose(res.delta_w, res.potentiation - res.depression, atol=1e-12)
    assert bool((res.potentiation >= -1e-12).all())
    assert bool((res.depression >= -1e-12).all())


# --------------------------------------------------------------------------- #
# causal structure                                                            #
# --------------------------------------------------------------------------- #
def test_pre_leading_train_potentiates_post_leading_depresses():
    T = 12
    # pre spikes early, post spikes late -> pure causal -> net LTP, no depression
    pre = torch.zeros(T, dtype=_f64); pre[[1, 2, 3]] = 1.0
    post = torch.zeros(T, dtype=_f64); post[[8, 9, 10]] = 1.0
    causal = stdp_trace_update(pre, post, _P)
    assert float(causal.delta_w) > 0.0
    assert torch.isclose(causal.depression.squeeze(), _t(0.0), atol=1e-12)
    # swap the roles -> pure acausal -> net LTD, no potentiation
    acausal = stdp_trace_update(post, pre, _P)
    assert float(acausal.delta_w) < 0.0
    assert torch.isclose(acausal.potentiation.squeeze(), _t(0.0), atol=1e-12)


def test_simultaneous_spikes_contribute_nothing():
    # a SINGLE coincident pre/post spike: the only pair has dt=0 -> no change.
    # (Two coincident spikes at {2,5} would still form cross-pairs 2->5 and 5->2,
    # which are NOT simultaneous and do contribute -- so we use one spike each.)
    T = 8
    pre = torch.zeros(T, dtype=_f64); pre[3] = 1.0
    post = torch.zeros(T, dtype=_f64); post[3] = 1.0
    res = stdp_trace_update(pre, post, _P)
    assert torch.isclose(res.delta_w.squeeze(), _t(0.0), atol=1e-12)


# --------------------------------------------------------------------------- #
# guards                                                                       #
# --------------------------------------------------------------------------- #
def test_params_validation():
    with pytest.raises(ValueError):
        STDPParams(a_plus=-1.0)
    with pytest.raises(ValueError):
        STDPParams(tau_plus=0.0)
    with pytest.raises(ValueError):
        STDPParams(dt=-1.0)


def test_trace_update_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        stdp_trace_update(torch.zeros(5, dtype=_f64), torch.zeros(6, dtype=_f64), _P)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
