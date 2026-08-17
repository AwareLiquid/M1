import pytest
pytest.importorskip("hypothesis")  # optional dev dep — see requirements-dev.txt

"""
tests/test_geometry_ops_properties.py -- property-based information-geometry
invariants for the composable Fisher-Rao operators.

``tests/test_geometry_ops.py`` pins these against chosen scenarios; here we pin
the *geometric laws themselves* with Hypothesis, over the whole space of
distributions:

- the Fisher-Rao distance is symmetric, zero on the diagonal, non-negative,
  bounded by pi, obeys the triangle inequality, and is invariant under a common
  permutation of the categories (the symmetry the Euclidean readout breaks);
- the Bhattacharyya coefficient is in [0, 1] and 1 only at coincidence;
- geodesics hit their endpoints, stay on the simplex, run at constant speed
  (d(p, gamma(t)) = t d(p, q)), and reverse consistently;
- the exp / log maps are mutual inverses, log is a zero-sum tangent whose Fisher
  length equals the distance, and log(p, p) = 0;
- parallel transport preserves the Fisher-Rao norm and tangency (an isometry);
- the geodesic barycenter is permutation invariant, reproduces a repeated code,
  and equals the geodesic midpoint for two equal weights.

All tensors are explicit float64; the global default dtype is left untouched so
the float32 model tests sharing this process are unaffected.

Run:  python -m pytest tests/test_geometry_ops_properties.py -v
"""
import math
import sys

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, ".")

from mt_lnn.geometry_ops import (                                  # noqa: E402
    bhattacharyya_coefficient,
    exp_map,
    fisher_norm,
    fisher_rao_distance,
    geodesic_barycenter,
    geodesic_interpolate,
    log_map,
    parallel_transport,
    is_on_simplex,
)

PROP = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_F = dict(allow_nan=False, allow_infinity=False, width=64)
_f64 = torch.float64


def _t(rows):
    return torch.tensor(rows, dtype=_f64)


# --------------------------------------------------------------------------- #
# strategies                                                                  #
# --------------------------------------------------------------------------- #
@st.composite
def simplex(draw, min_n=2, max_n=6):
    """A strictly-interior simplex point (softmax of bounded logits)."""
    n = draw(st.integers(min_n, max_n))
    logits = draw(st.lists(st.floats(-4.0, 4.0, **_F), min_size=n, max_size=n))
    return torch.softmax(_t(logits), dim=-1), n


@st.composite
def two_simplex(draw, **kw):
    p, n = draw(simplex(**kw))
    logits = draw(st.lists(st.floats(-4.0, 4.0, **_F), min_size=n, max_size=n))
    q = torch.softmax(_t(logits), dim=-1)
    return p, q, n


@st.composite
def distinct_two(draw, **kw):
    p, q, n = draw(two_simplex(**kw))
    assume(float(fisher_rao_distance(p, q)) > 1e-3)
    return p, q, n


@st.composite
def three_simplex(draw, min_n=2, max_n=6):
    """Three points sharing one dimension n (fix n first, then draw three).

    Drawing three independent ``simplex()`` and ``assume``-ing equal n filters
    out ~96% of inputs (Hypothesis raises ``filter_too_much``); fixing n up
    front keeps every draw usable.
    """
    n = draw(st.integers(min_n, max_n))

    def _pt():
        lg = draw(st.lists(st.floats(-4.0, 4.0, **_F), min_size=n, max_size=n))
        return torch.softmax(_t(lg), dim=-1)

    return _pt(), _pt(), _pt(), n


# --------------------------------------------------------------------------- #
# distance / BC                                                               #
# --------------------------------------------------------------------------- #
@PROP
@given(data=two_simplex())
def test_distance_is_a_symmetric_nonnegative_bounded_metric(data):
    p, q, _ = data
    d = fisher_rao_distance(p, q)
    assert torch.isclose(d, fisher_rao_distance(q, p), atol=1e-9)   # symmetric
    assert float(d) >= -1e-9                                        # non-negative
    assert float(d) <= math.pi + 1e-6                               # bounded
    # d = 2 arccos(BC); near coincidence BC = 1 - eps so the self-distance
    # floor is 2 sqrt(2 eps) ~ 3e-8 (arccos has infinite slope at 1). Tolerate
    # that amplified rounding -- the hand-picked scenario test pins 1e-9.
    assert torch.isclose(fisher_rao_distance(p, p), _t(0.0), atol=1e-6)


@PROP
@given(data=two_simplex())
def test_bhattacharyya_is_in_unit_interval(data):
    p, q, _ = data
    bc = bhattacharyya_coefficient(p, q)
    assert -1e-9 <= float(bc) <= 1.0 + 1e-9
    assert torch.isclose(bhattacharyya_coefficient(p, p), _t(1.0), atol=1e-9)


@PROP
@given(data=three_simplex())
def test_distance_obeys_the_triangle_inequality(data):
    pp, qq, rr, _ = data
    d_pr = fisher_rao_distance(pp, rr)
    d_pq = fisher_rao_distance(pp, qq)
    d_qr = fisher_rao_distance(qq, rr)
    assert float(d_pr) <= float(d_pq) + float(d_qr) + 1e-6


@PROP
@given(data=two_simplex())
def test_distance_is_invariant_under_a_common_permutation(data):
    p, q, n = data
    perm = torch.randperm(n)
    # BC is a sum over categories; permuting both arguments only reorders the
    # summands, so the distance is invariant up to float summation order.
    assert torch.isclose(
        fisher_rao_distance(p, q),
        fisher_rao_distance(p[perm], q[perm]), atol=1e-6)


# --------------------------------------------------------------------------- #
# geodesics                                                                   #
# --------------------------------------------------------------------------- #
@PROP
@given(data=two_simplex(), t=st.floats(0.0, 1.0, **_F))
def test_geodesic_stays_on_the_simplex_and_hits_endpoints(data, t):
    p, q, _ = data
    g = geodesic_interpolate(p, q, t)
    assert bool(is_on_simplex(g))
    assert torch.allclose(geodesic_interpolate(p, q, 0.0), p, atol=1e-7)
    assert torch.allclose(geodesic_interpolate(p, q, 1.0), q, atol=1e-7)


@PROP
@given(data=distinct_two(), t=st.floats(0.0, 1.0, **_F))
def test_geodesic_runs_at_constant_speed(data, t):
    p, q, _ = data
    d = fisher_rao_distance(p, q)
    g = geodesic_interpolate(p, q, t)
    assert torch.isclose(fisher_rao_distance(p, g), t * d, atol=1e-6)


@PROP
@given(data=two_simplex(), t=st.floats(0.0, 1.0, **_F))
def test_geodesic_reverses_consistently(data, t):
    p, q, _ = data
    fwd = geodesic_interpolate(p, q, t)
    bwd = geodesic_interpolate(q, p, 1.0 - t)
    assert torch.allclose(fwd, bwd, atol=1e-6)


@PROP
@given(data=two_simplex(), t=st.floats(0.0, 1.0, **_F))
def test_geodesic_is_permutation_equivariant(data, t):
    p, q, n = data
    perm = torch.randperm(n)
    g = geodesic_interpolate(p, q, t)
    g_perm = geodesic_interpolate(p[perm], q[perm], t)
    assert torch.allclose(g[perm], g_perm, atol=1e-6)


# --------------------------------------------------------------------------- #
# exp / log / transport                                                       #
# --------------------------------------------------------------------------- #
@PROP
@given(data=two_simplex())
def test_exp_and_log_are_mutual_inverses(data):
    p, q, _ = data
    v = log_map(p, q)
    assert torch.isclose(v.sum(), _t(0.0), atol=1e-7)               # zero-sum tangent
    assert torch.isclose(fisher_norm(p, v), fisher_rao_distance(p, q), atol=1e-6)
    assert torch.allclose(exp_map(p, v), q, atol=1e-6)              # round-trip


@PROP
@given(data=simplex())
def test_log_of_self_is_zero(data):
    p, n = data
    assert torch.allclose(log_map(p, p), torch.zeros(n, dtype=_f64), atol=1e-7)


@PROP
@given(data=three_simplex())
def test_parallel_transport_is_an_isometry_into_the_tangent_space(data):
    pp, qq, rr, _ = data
    v = log_map(pp, rr)                                # a tangent at p
    vt = parallel_transport(pp, qq, v)
    assert torch.isclose(vt.sum(), _t(0.0), atol=1e-7)              # tangent at q
    assert torch.isclose(fisher_norm(pp, v), fisher_norm(qq, vt), atol=1e-6)


# --------------------------------------------------------------------------- #
# barycenter                                                                  #
# --------------------------------------------------------------------------- #
@PROP
@given(data=two_simplex())
def test_barycenter_of_two_equal_weights_is_the_midpoint(data):
    p, q, _ = data
    b = geodesic_barycenter(torch.stack([p, q]), _t([0.5, 0.5]))
    assert torch.allclose(b, geodesic_interpolate(p, q, 0.5), atol=1e-5)


@PROP
@given(data=simplex())
def test_barycenter_reproduces_a_repeated_code(data):
    p, _ = data
    b = geodesic_barycenter(torch.stack([p, p, p]))
    assert torch.allclose(b, p, atol=1e-5)


@PROP
@given(data=two_simplex())
def test_barycenter_is_permutation_equivariant(data):
    p, q, n = data
    perm = torch.randperm(n)
    w = _t([0.7, 0.3])
    b = geodesic_barycenter(torch.stack([p, q]), w)
    b_perm = geodesic_barycenter(torch.stack([p[perm], q[perm]]), w)
    assert torch.allclose(b[perm], b_perm, atol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
