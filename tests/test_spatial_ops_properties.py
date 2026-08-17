import pytest
pytest.importorskip("hypothesis")  # optional dev dep — see requirements-dev.txt

"""
tests/test_spatial_ops_properties.py -- property-based geometric invariants for
the composable spatial operators.

``tests/test_spatial_ops.py`` pins these against chosen scenarios; here we pin
the *geometric laws themselves* with Hypothesis, over the whole input space:

- the distance matrix is symmetric, zero on the diagonal, non-negative, obeys
  the triangle inequality, and is invariant under rigid motion (translation +
  rotation);
- relative directions are unit and antisymmetric; bearings agree with them;
- the radius graph is symmetric, edges exactly threshold the distance, and the
  edge set grows monotonically with the radius;
- the kNN graph is symmetric with degree >= k;
- reachability contains its seeds, grows monotonically with the hop budget, and
  agrees with finite hop-distance; on a complete graph everything is reachable;
- connected-component labels are constant along edges and identify exactly the
  mutually reachable nodes.

All tensors are explicit float64; the global default dtype is left untouched so
the float32 model tests sharing this process are unaffected.

Run:  python -m pytest tests/test_spatial_ops_properties.py -v
"""
import math
import sys

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, ".")

from mt_lnn.spatial_ops import (                                  # noqa: E402
    bearing,
    connected_components,
    hop_distance,
    in_ball,
    in_bounding_box,
    knn_graph,
    pairwise_distance,
    radius_graph,
    reachable_from,
    relative_direction,
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
def vecs(draw, n, d, lo, hi):
    rows = draw(st.lists(
        st.lists(st.floats(lo, hi, **_F), min_size=d, max_size=d),
        min_size=n, max_size=n))
    return _t(rows)


@st.composite
def cloud(draw, min_n=2, max_n=6, dims=(2, 3)):
    d = draw(st.sampled_from(list(dims)))
    n = draw(st.integers(min_n, max_n))
    return draw(vecs(n, d, -10.0, 10.0)), n, d


@st.composite
def distinct_cloud(draw, **kw):
    """A cloud whose points are pairwise separated (so directions are well defined)."""
    coords, n, d = draw(cloud(**kw))
    dist = pairwise_distance(coords)
    off = dist + torch.eye(n, dtype=_f64) * 1e9
    assume(float(off.min()) > 1e-2)
    return coords, n, d


# --------------------------------------------------------------------------- #
# metric structure                                                            #
# --------------------------------------------------------------------------- #
@PROP
@given(data=cloud())
def test_distance_is_a_symmetric_zero_diagonal_metric(data):
    coords, n, _ = data
    d = pairwise_distance(coords)
    assert torch.allclose(d, d.transpose(-1, -2), atol=1e-9)        # symmetric
    assert torch.allclose(d.diagonal(), torch.zeros(n, dtype=_f64), atol=1e-9)
    assert torch.all(d >= -1e-9)                                    # non-negative


@PROP
@given(data=cloud())
def test_distance_obeys_the_triangle_inequality(data):
    coords, n, _ = data
    d = pairwise_distance(coords)
    # d[i,k] <= d[i,j] + d[j,k] for every triple
    # build explicitly to avoid axis confusion: dij[i,j,1] + djk[1,j,k]
    dij = d.unsqueeze(-1)                          # (N, N, 1) over k
    djk = d.unsqueeze(0)                           # (1, N, N) over i
    bound = dij + djk                              # (N, N, N): [i,j,k] = d[i,j]+d[j,k]
    dik = d.unsqueeze(1)                           # (N, 1, N): [i,*,k] = d[i,k]
    assert torch.all(dik <= bound + 1e-6)


@PROP
@given(data=cloud(), shift=st.lists(st.floats(-20.0, 20.0, **_F), min_size=3, max_size=3),
       theta=st.floats(-math.pi, math.pi, **_F))
def test_distance_is_invariant_under_rigid_motion(data, shift, theta):
    coords, n, d = data
    c = _t(shift)[:d]
    moved = coords + c
    if d == 2:                                     # add a rotation in the plane
        R = _t([[math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)]])
        moved = moved @ R.transpose(0, 1)
    assert torch.allclose(pairwise_distance(coords), pairwise_distance(moved),
                          atol=1e-6, rtol=1e-6)


@PROP
@given(data=distinct_cloud())
def test_relative_direction_is_unit_and_antisymmetric(data):
    coords, n, _ = data
    u = relative_direction(coords)
    off = ~torch.eye(n, dtype=torch.bool)
    norms = u.norm(dim=-1)
    assert torch.allclose(norms[off], torch.ones(int(off.sum()), dtype=_f64), atol=1e-6)
    assert torch.allclose(u, -u.transpose(0, 1), atol=1e-6)         # i->j == -(j->i)


@PROP
@given(data=distinct_cloud(dims=(2,)))
def test_bearing_agrees_with_relative_direction(data):
    coords, n, _ = data
    u = relative_direction(coords)
    ang = bearing(coords)
    off = ~torch.eye(n, dtype=torch.bool)
    assert torch.allclose(torch.atan2(u[..., 1], u[..., 0])[off], ang[off], atol=1e-6)


# --------------------------------------------------------------------------- #
# containment                                                                 #
# --------------------------------------------------------------------------- #
@PROP
@given(data=cloud(), r=st.floats(0.5, 20.0, **_F))
def test_in_ball_matches_the_distance_to_centre(data, r):
    coords, n, d = data
    centre = torch.zeros(d, dtype=_f64)
    inside = in_ball(coords, centre, r)
    assert torch.equal(inside, coords.norm(dim=-1) <= r)


@PROP
@given(data=cloud())
def test_in_bounding_box_is_an_and_over_axes(data):
    coords, n, d = data
    lo = torch.full((d,), -3.0, dtype=_f64)
    hi = torch.full((d,), 3.0, dtype=_f64)
    inside = in_bounding_box(coords, lo, hi)
    assert torch.equal(inside, ((coords >= lo) & (coords <= hi)).all(dim=-1))


# --------------------------------------------------------------------------- #
# proximity graphs                                                            #
# --------------------------------------------------------------------------- #
@PROP
@given(data=cloud(), r=st.floats(0.5, 25.0, **_F))
def test_radius_graph_is_symmetric_and_thresholds_distance(data, r):
    coords, n, _ = data
    adj = radius_graph(coords, r)
    assert torch.equal(adj, adj.transpose(-1, -2))                  # symmetric
    assert torch.all(adj.diagonal() == 0)                          # no self-loops
    d = pairwise_distance(coords)
    off = ~torch.eye(n, dtype=torch.bool)
    assert torch.equal(adj[off] > 0, d[off] <= r)


@PROP
@given(data=cloud(), r1=st.floats(0.5, 25.0, **_F), r2=st.floats(0.5, 25.0, **_F))
def test_radius_graph_edges_grow_monotonically_with_radius(data, r1, r2):
    assume(r1 <= r2)
    coords, _, _ = data
    a1 = radius_graph(coords, r1)
    a2 = radius_graph(coords, r2)
    assert torch.all((a2 > 0) | (a1 == 0))                         # a1 edges subset of a2


@PROP
@given(data=cloud(min_n=3, max_n=6))
def test_knn_graph_is_symmetric_with_degree_at_least_k(data):
    coords, n, _ = data
    k = 1 if n <= 2 else 2
    adj = knn_graph(coords, k, symmetric=True)
    assert torch.equal(adj, adj.transpose(-1, -2))
    assert torch.all(adj.diagonal() == 0)
    assert torch.all(adj.sum(dim=-1) >= k)                         # symmetrize only adds


# --------------------------------------------------------------------------- #
# reachability / topology                                                     #
# --------------------------------------------------------------------------- #
@PROP
@given(data=cloud(min_n=3, max_n=6), r=st.floats(0.5, 25.0, **_F))
def test_reachability_contains_seed_and_agrees_with_hop_distance(data, r):
    coords, n, _ = data
    adj = radius_graph(coords, r)
    src = torch.zeros(n, dtype=_f64)
    src[0] = 1.0
    reach = reachable_from(adj, src)
    assert reach[0] == 1.0                                          # seed reachable
    # reachable iff its BFS hop-distance is finite
    hops = hop_distance(adj, src)
    assert torch.equal(reach > 0, torch.isfinite(hops))


@PROP
@given(data=cloud(min_n=3, max_n=6), r=st.floats(0.5, 25.0, **_F),
       h1=st.integers(0, 6), h2=st.integers(0, 6))
def test_reachability_grows_monotonically_with_hop_budget(data, r, h1, h2):
    assume(h1 <= h2)
    coords, n, _ = data
    adj = radius_graph(coords, r)
    src = torch.zeros(n, dtype=_f64)
    src[0] = 1.0
    r1 = reachable_from(adj, src, max_hops=h1)
    r2 = reachable_from(adj, src, max_hops=h2)
    assert torch.all((r2 > 0) | (r1 == 0))                         # more hops -> superset


@PROP
@given(n=st.integers(2, 7))
def test_complete_graph_reaches_everything_from_any_seed(n):
    adj = torch.ones(n, n, dtype=_f64) - torch.eye(n, dtype=_f64)
    src = torch.zeros(n, dtype=_f64)
    src[0] = 1.0
    assert torch.all(reachable_from(adj, src) > 0)


@PROP
@given(data=cloud(min_n=3, max_n=7), r=st.floats(0.5, 25.0, **_F))
def test_components_are_constant_along_edges_and_match_reachability(data, r):
    coords, n, _ = data
    adj = radius_graph(coords, r)
    labels = connected_components(adj)
    # every edge connects equal labels
    ii, jj = (adj > 0).nonzero(as_tuple=True)
    assert torch.all(labels[ii] == labels[jj])
    # two nodes share a label iff node-0's component is exactly its reachable set
    src = torch.zeros(n, dtype=_f64)
    src[0] = 1.0
    reach = reachable_from(adj, src) > 0
    assert torch.equal(labels == labels[0], reach)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
