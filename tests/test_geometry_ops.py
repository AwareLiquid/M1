"""
tests/test_geometry_ops.py -- composable information-geometry operators pinned
against analytic ground truth.

These tests fix the Fisher-Rao operators on the probability simplex at chosen,
hand-computable scenarios (vertices, two-point distributions, the uniform
centre); ``tests/test_geometry_ops_properties.py`` pins the geometric laws over
the whole input space with Hypothesis.

All tensors are explicit float64; the global default dtype is left untouched.

Run:  python -m pytest tests/test_geometry_ops.py -v
"""
import math
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn.geometry_ops import (                                  # noqa: E402
    Geodesic,
    bhattacharyya_coefficient,
    exp_map,
    fisher_information_matrix,
    fisher_inner,
    fisher_norm,
    fisher_rao_distance,
    fisher_rao_geodesic,
    geodesic_barycenter,
    geodesic_interpolate,
    is_on_simplex,
    log_map,
    parallel_transport,
    simplex_project,
    sphere_embed,
    sphere_unembed,
)

_f64 = torch.float64


def _t(rows):
    return torch.tensor(rows, dtype=_f64)


# --------------------------------------------------------------------------- #
# simplex housekeeping                                                        #
# --------------------------------------------------------------------------- #
def test_simplex_project_clamps_and_renormalises():
    x = _t([-0.5, 1.0, 1.0])                       # a negative + unnormalised
    p = simplex_project(x)
    assert torch.allclose(p, _t([0.0, 0.5, 0.5]), atol=1e-12)
    assert torch.isclose(p.sum(), _t(1.0))


def test_is_on_simplex_accepts_distributions_and_rejects_others():
    assert bool(is_on_simplex(_t([0.2, 0.3, 0.5])))
    assert not bool(is_on_simplex(_t([0.2, 0.3, 0.6])))     # sums to 1.1
    assert not bool(is_on_simplex(_t([-0.1, 0.6, 0.5])))    # negative entry


def test_sphere_embed_lands_on_the_unit_sphere_and_inverts():
    p = _t([0.1, 0.2, 0.7])
    u = sphere_embed(p)
    assert torch.isclose((u * u).sum(), _t(1.0), atol=1e-12)   # unit sphere
    assert torch.allclose(sphere_unembed(u), p, atol=1e-12)    # phi^{-1} o phi = id


# --------------------------------------------------------------------------- #
# distance / inner product                                                    #
# --------------------------------------------------------------------------- #
def test_bhattacharyya_is_one_for_equal_and_zero_for_disjoint():
    p = _t([0.3, 0.3, 0.4])
    assert torch.isclose(bhattacharyya_coefficient(p, p), _t(1.0), atol=1e-12)
    e0, e1 = _t([1.0, 0.0, 0.0]), _t([0.0, 1.0, 0.0])
    assert torch.isclose(bhattacharyya_coefficient(e0, e1), _t(0.0), atol=1e-12)


def test_distance_between_vertices_is_pi():
    e0, e1 = _t([1.0, 0.0, 0.0]), _t([0.0, 1.0, 0.0])
    assert torch.isclose(fisher_rao_distance(e0, e1), _t(math.pi), atol=1e-9)


def test_distance_two_point_matches_closed_form():
    # On the 1-simplex d = 2 arccos(sqrt(p0 q0) + sqrt(p1 q1)).
    p, q = _t([0.25, 0.75]), _t([0.81, 0.19])
    bc = math.sqrt(0.25 * 0.81) + math.sqrt(0.75 * 0.19)
    assert torch.isclose(fisher_rao_distance(p, q), _t(2.0 * math.acos(bc)), atol=1e-12)


def test_distance_to_self_is_zero():
    p = _t([0.2, 0.2, 0.2, 0.4])
    assert torch.isclose(fisher_rao_distance(p, p), _t(0.0), atol=1e-9)


def test_fisher_information_matrix_is_diag_inverse_and_reproduces_inner():
    p = _t([0.1, 0.4, 0.5])
    G = fisher_information_matrix(p)
    assert torch.allclose(G, torch.diag(1.0 / p), atol=1e-12)
    u = _t([0.3, -0.1, -0.2])                      # tangent: sums to 0
    v = _t([-0.2, 0.5, -0.3])
    quad = u @ G @ v
    assert torch.isclose(quad, fisher_inner(p, u, v), atol=1e-12)


def test_fisher_norm_is_sqrt_of_self_inner():
    p = _t([0.3, 0.3, 0.4])
    v = _t([0.2, -0.1, -0.1])
    assert torch.isclose(fisher_norm(p, v), torch.sqrt(fisher_inner(p, v, v)), atol=1e-12)


# --------------------------------------------------------------------------- #
# geodesics                                                                   #
# --------------------------------------------------------------------------- #
def test_geodesic_hits_its_endpoints():
    p, q = _t([0.2, 0.3, 0.5]), _t([0.6, 0.1, 0.3])
    assert torch.allclose(geodesic_interpolate(p, q, 0.0), p, atol=1e-9)
    assert torch.allclose(geodesic_interpolate(p, q, 1.0), q, atol=1e-9)


def test_geodesic_midpoint_is_equidistant_and_half_length():
    p, q = _t([0.2, 0.3, 0.5]), _t([0.6, 0.1, 0.3])
    m = geodesic_interpolate(p, q, 0.5)
    dp, dq = fisher_rao_distance(p, m), fisher_rao_distance(m, q)
    assert torch.isclose(dp, dq, atol=1e-9)                       # equidistant
    assert torch.isclose(dp + dq, fisher_rao_distance(p, q), atol=1e-9)  # additive


def test_geodesic_stays_on_the_simplex():
    p, q = _t([0.05, 0.9, 0.05]), _t([0.5, 0.25, 0.25])
    for t in (0.1, 0.4, 0.6, 0.9):
        g = geodesic_interpolate(p, q, t)
        assert bool(is_on_simplex(g))


def test_exp_and_log_are_inverse():
    p, q = _t([0.2, 0.3, 0.5]), _t([0.6, 0.1, 0.3])
    v = log_map(p, q)
    assert torch.isclose(v.sum(), _t(0.0), atol=1e-9)             # tangent
    assert torch.isclose(fisher_norm(p, v), fisher_rao_distance(p, q), atol=1e-9)
    assert torch.allclose(exp_map(p, v), q, atol=1e-7)            # round-trip


def test_log_of_self_is_zero_tangent():
    p = _t([0.3, 0.3, 0.4])
    assert torch.allclose(log_map(p, p), torch.zeros(3, dtype=_f64), atol=1e-9)


def test_parallel_transport_preserves_fisher_norm_and_tangency():
    p, q = _t([0.2, 0.3, 0.5]), _t([0.6, 0.1, 0.3])
    v = log_map(p, _t([0.1, 0.8, 0.1]))            # some tangent at p
    vt = parallel_transport(p, q, v)
    assert torch.isclose(vt.sum(), _t(0.0), atol=1e-9)            # tangent at q
    assert torch.isclose(fisher_norm(p, v), fisher_norm(q, vt), atol=1e-9)  # isometry


def test_parallel_transport_is_identity_when_endpoints_coincide():
    p = _t([0.3, 0.3, 0.4])
    v = _t([0.2, -0.1, -0.1])
    assert torch.allclose(parallel_transport(p, p, v), v, atol=1e-9)


# --------------------------------------------------------------------------- #
# barycenter                                                                  #
# --------------------------------------------------------------------------- #
def test_barycenter_of_identical_codes_is_that_code():
    p = _t([0.2, 0.3, 0.5])
    b = geodesic_barycenter(torch.stack([p, p, p]))
    assert torch.allclose(b, p, atol=1e-6)


def test_barycenter_of_two_equal_weights_is_the_geodesic_midpoint():
    p, q = _t([0.2, 0.3, 0.5]), _t([0.6, 0.1, 0.3])
    b = geodesic_barycenter(torch.stack([p, q]), _t([0.5, 0.5]))
    assert torch.allclose(b, geodesic_interpolate(p, q, 0.5), atol=1e-6)


def test_barycenter_respects_weights_and_stays_on_simplex():
    p, q = _t([0.2, 0.3, 0.5]), _t([0.6, 0.1, 0.3])
    b = geodesic_barycenter(torch.stack([p, q]), _t([0.9, 0.1]))
    assert bool(is_on_simplex(b))
    # heavily weighting p pulls the mean nearer p than q
    assert float(fisher_rao_distance(b, p)) < float(fisher_rao_distance(b, q))


# --------------------------------------------------------------------------- #
# flagship                                                                     #
# --------------------------------------------------------------------------- #
def test_fisher_rao_geodesic_flagship_summary():
    p, q = _t([0.2, 0.3, 0.5]), _t([0.6, 0.1, 0.3])
    geo = fisher_rao_geodesic(p, q, n_steps=8)
    assert isinstance(geo, Geodesic)
    assert geo.points.shape == (9, 3)
    assert torch.allclose(geo.points[0], p, atol=1e-9)
    assert torch.allclose(geo.points[-1], q, atol=1e-9)
    assert torch.isclose(geo.length, fisher_rao_distance(p, q), atol=1e-9)
    assert torch.allclose(geo.midpoint, geodesic_interpolate(p, q, 0.5), atol=1e-9)


def test_geodesic_differs_from_the_euclidean_chord():
    # the metric-consistency argument: the Euclidean midpoint is generally NOT
    # the Fisher-Rao midpoint (the chord cuts through the simplex's interior).
    p, q = _t([0.05, 0.9, 0.05]), _t([0.9, 0.05, 0.05])
    geo_mid = geodesic_interpolate(p, q, 0.5)
    euc_mid = 0.5 * (p + q)
    assert float((geo_mid - euc_mid).abs().max()) > 1e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
