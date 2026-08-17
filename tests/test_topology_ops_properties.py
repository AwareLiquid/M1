"""
tests/test_topology_ops_properties.py -- property-based TDA invariants for the
composable topology operators.

``tests/test_topology_ops.py`` pins these against chosen clouds; here we pin the
*topological laws themselves* with Hypothesis, over the whole space of point
clouds:

- the MST has exactly N-1 edges and non-negative total weight equal to the total
  persistence;
- Betti-0 is N at scale 0 (distinct points), 1 at a huge scale, and monotonically
  non-increasing in between;
- the barcode has N bars, its finite deaths are sorted and non-negative, and its
  ``betti0_at`` agrees with the union-find Betti-0;
- total persistence is permutation- and translation-invariant and scales linearly
  with the cloud;
- SRTD is symmetric, non-negative, zero on coincidence, permutation invariant
  (it depends only on the barcode multiset), and obeys the triangle inequality
  for clouds of equal size.

All tensors are explicit float64; the global default dtype is left untouched so
the float32 model tests sharing this process are unaffected.

Run:  python -m pytest tests/test_topology_ops_properties.py -v
"""
import sys

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, ".")

from mt_lnn.spatial_ops import pairwise_distance                  # noqa: E402
from mt_lnn.topology_ops import (                                 # noqa: E402
    betti0,
    minimum_spanning_tree,
    persistent_homology_h0,
    srtd,
    total_persistence,
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
def cloud(draw, min_n=2, max_n=8, min_d=2, max_d=4):
    """A point cloud whose points are pairwise distinct (no degenerate edges)."""
    n = draw(st.integers(min_n, max_n))
    d = draw(st.integers(min_d, max_d))
    rows = [draw(st.lists(st.floats(-5.0, 5.0, **_F), min_size=d, max_size=d))
            for _ in range(n)]
    pts = _t(rows)
    dist = pairwise_distance(pts) + torch.eye(n, dtype=_f64) * 1e9
    assume(float(dist.min()) > 1e-2)                  # well-separated points
    return pts, n


@st.composite
def _one_of_dim(draw, n, d):
    rows = [draw(st.lists(st.floats(-5.0, 5.0, **_F), min_size=d, max_size=d))
            for _ in range(n)]
    pts = _t(rows)
    dist = pairwise_distance(pts) + torch.eye(n, dtype=_f64) * 1e9
    assume(float(dist.min()) > 1e-2)
    return pts


@st.composite
def two_clouds_same_n(draw, **kw):
    px, n = draw(cloud(**kw))
    py = draw(_one_of_dim(n, px.shape[1]))
    return px, py, n


@st.composite
def k_clouds_same_n(draw, k=3, min_n=2, max_n=8, min_d=2, max_d=4):
    """``k`` clouds that all share one (N, D): fix N and D first, then draw each.

    Drawing ``k`` independent ``cloud()`` and ``assume``-ing equal N filters out
    nearly everything (Hypothesis ``filter_too_much``); fixing N/D up front keeps
    every draw usable.
    """
    n = draw(st.integers(min_n, max_n))
    d = draw(st.integers(min_d, max_d))
    clouds = [draw(_one_of_dim(n, d)) for _ in range(k)]
    return clouds, n


# --------------------------------------------------------------------------- #
# MST                                                                         #
# --------------------------------------------------------------------------- #
@PROP
@given(data=cloud())
def test_mst_has_n_minus_one_edges_and_nonneg_weight(data):
    pts, n = data
    mst = minimum_spanning_tree(pts)
    assert mst.edges.shape == (n - 1, 2)
    assert mst.weights.numel() == n - 1
    assert float(mst.total_weight) >= -1e-12
    assert torch.isclose(mst.total_weight, total_persistence(pts), atol=1e-9)


# --------------------------------------------------------------------------- #
# Betti-0                                                                     #
# --------------------------------------------------------------------------- #
@PROP
@given(data=cloud())
def test_betti0_runs_from_n_down_to_one(data):
    pts, n = data
    assert betti0(pts, 0.0) == n                       # distinct points, scale 0
    assert betti0(pts, 1e6) == 1                        # everything merged


@PROP
@given(data=cloud(), e1=st.floats(0.0, 20.0, **_F), e2=st.floats(0.0, 20.0, **_F))
def test_betti0_is_monotone_non_increasing_in_scale(data, e1, e2):
    pts, _ = data
    lo, hi = min(e1, e2), max(e1, e2)
    assert betti0(pts, lo) >= betti0(pts, hi)          # only merges as scale grows


@PROP
@given(data=cloud())
def test_betti0_is_permutation_invariant(data):
    pts, n = data
    perm = torch.randperm(n)
    assert betti0(pts, 0.5) == betti0(pts[perm], 0.5)


# --------------------------------------------------------------------------- #
# barcode                                                                     #
# --------------------------------------------------------------------------- #
@PROP
@given(data=cloud())
def test_barcode_is_sorted_nonneg_and_has_n_bars(data):
    pts, n = data
    bc = persistent_homology_h0(pts)
    assert bc.n_bars == n
    assert bc.finite_deaths.numel() == n - 1
    fd = bc.finite_deaths
    assert bool((fd >= -1e-12).all())                  # non-negative lifetimes
    assert torch.allclose(fd, torch.sort(fd).values, atol=0)   # sorted ascending


@PROP
@given(data=cloud(), eps=st.floats(0.0, 20.0, **_F))
def test_barcode_betti0_matches_union_find(data, eps):
    pts, _ = data
    bc = persistent_homology_h0(pts)
    assert bc.betti0_at(eps) == betti0(pts, eps)       # MST view == union-find view


# --------------------------------------------------------------------------- #
# total persistence invariances                                              #
# --------------------------------------------------------------------------- #
@PROP
@given(data=cloud())
def test_total_persistence_is_permutation_and_translation_invariant(data):
    pts, n = data
    perm = torch.randperm(n)
    shift = _t([3.0, -2.0, 1.5, 0.7][: pts.shape[1]])
    base = total_persistence(pts)
    assert torch.isclose(total_persistence(pts[perm]), base, atol=1e-9)
    assert torch.isclose(total_persistence(pts + shift), base, atol=1e-9)


@PROP
@given(data=cloud(), c=st.floats(0.1, 5.0, **_F))
def test_total_persistence_scales_linearly_with_the_cloud(data, c):
    pts, _ = data
    assert torch.isclose(total_persistence(c * pts),
                         c * total_persistence(pts), atol=1e-8)


# --------------------------------------------------------------------------- #
# SRTD                                                                        #
# --------------------------------------------------------------------------- #
@PROP
@given(data=two_clouds_same_n())
def test_srtd_is_symmetric_nonneg_and_zero_on_self(data):
    px, py, _ = data
    d = srtd(px, py)
    assert float(d) >= -1e-12
    assert torch.isclose(d, srtd(py, px), atol=1e-9)
    assert torch.isclose(srtd(px, px), _t(0.0), atol=1e-12)


@PROP
@given(data=two_clouds_same_n())
def test_srtd_is_permutation_invariant_in_both_clouds(data):
    px, py, n = data
    pp, pq = torch.randperm(n), torch.randperm(n)
    # SRTD depends only on the barcode multiset, invariant to point relabeling
    assert torch.isclose(srtd(px, py), srtd(px[pp], py[pq]), atol=1e-9)


@PROP
@given(data=k_clouds_same_n(k=3))
def test_srtd_obeys_triangle_inequality_for_equal_n(data):
    (x, y, z), _ = data               # the sorted-barcode metric needs equal cardinality
    d_xz = float(srtd(x, z))
    d_xy = float(srtd(x, y))
    d_yz = float(srtd(y, z))
    assert d_xz <= d_xy + d_yz + 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
