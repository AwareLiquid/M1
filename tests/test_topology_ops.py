"""
tests/test_topology_ops.py -- composable TDA operators pinned against analytic
ground truth.

These fix the H0 persistent-homology operators on hand-computable clouds (a line
of well-separated clusters, a single blob, a square), where the minimum spanning
tree, the Betti-0 curve, the barcode, and the symmetric topology divergence are
all known by hand. ``tests/test_topology_ops_properties.py`` pins the laws over
the whole input space with Hypothesis.

All tensors are explicit float64; the global default dtype is left untouched.

Run:  python -m pytest tests/test_topology_ops.py -v
"""
import math
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn.topology_ops import (                                 # noqa: E402
    Barcode,
    MinimumSpanningTree,
    betti0,
    betti0_curve,
    minimum_spanning_tree,
    persistent_homology_h0,
    srtd,
    total_persistence,
)

_f64 = torch.float64


def _t(rows):
    return torch.tensor(rows, dtype=_f64)


# three pairs of points, clusters spaced 5 apart, intra-cluster gap 0.1
_THREE_CLUSTERS = _t([[0.0, 0.0], [0.1, 0.0],
                      [5.0, 0.0], [5.1, 0.0],
                      [10.0, 0.0], [10.1, 0.0]])


# --------------------------------------------------------------------------- #
# minimum spanning tree                                                       #
# --------------------------------------------------------------------------- #
def test_mst_of_a_line_is_the_chain_of_nearest_neighbours():
    # 4 collinear points at 0,1,2,3: the MST is the 3 unit edges (total 3).
    pts = _t([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    mst = minimum_spanning_tree(pts)
    assert isinstance(mst, MinimumSpanningTree)
    assert mst.edges.shape == (3, 2)
    assert torch.allclose(torch.sort(mst.weights).values, _t([1.0, 1.0, 1.0]), atol=1e-12)
    assert torch.isclose(mst.total_weight, _t(3.0), atol=1e-12)


def test_mst_total_weight_matches_total_persistence():
    mst = minimum_spanning_tree(_THREE_CLUSTERS)
    assert torch.isclose(mst.total_weight,
                         total_persistence(_THREE_CLUSTERS), atol=1e-12)


def test_mst_of_single_point_is_empty():
    mst = minimum_spanning_tree(_t([[1.0, 2.0]]))
    assert mst.edges.shape == (0, 2)
    assert mst.weights.numel() == 0
    assert torch.isclose(mst.total_weight, _t(0.0))


# --------------------------------------------------------------------------- #
# Betti-0                                                                      #
# --------------------------------------------------------------------------- #
def test_betti0_counts_clusters_at_each_scale():
    assert betti0(_THREE_CLUSTERS, 0.05) == 6      # nothing merged: 6 points
    assert betti0(_THREE_CLUSTERS, 0.2) == 3       # each pair merged: 3 clusters
    assert betti0(_THREE_CLUSTERS, 6.0) == 1       # all merged: 1 component


def test_betti0_curve_is_monotone_non_increasing():
    th = torch.linspace(0.0, 11.0, 23, dtype=_f64)
    curve = betti0_curve(_THREE_CLUSTERS, th)
    assert curve[0].item() == 6
    assert curve[-1].item() == 1
    diffs = curve[1:] - curve[:-1]
    assert bool((diffs <= 0).all())                 # components only merge


def test_betti0_of_single_point_is_one():
    assert betti0(_t([[0.3, 0.7]]), 0.5) == 1


# --------------------------------------------------------------------------- #
# H0 barcode                                                                  #
# --------------------------------------------------------------------------- #
def test_barcode_finite_deaths_are_the_sorted_mst_weights():
    bc = persistent_homology_h0(_THREE_CLUSTERS)
    assert isinstance(bc, Barcode)
    assert bc.n_points == 6
    assert bc.n_bars == 6                            # 5 finite + 1 infinite
    # 3 intra-cluster merges at 0.1, then 2 inter-cluster merges at 4.9
    assert torch.allclose(bc.finite_deaths, _t([0.1, 0.1, 0.1, 4.9, 4.9]), atol=1e-12)


def test_barcode_betti0_at_agrees_with_union_find_betti0():
    bc = persistent_homology_h0(_THREE_CLUSTERS)
    for eps in (0.05, 0.2, 1.0, 4.95, 6.0):
        assert bc.betti0_at(eps) == betti0(_THREE_CLUSTERS, eps)


def test_total_persistence_is_sum_of_finite_bars():
    bc = persistent_homology_h0(_THREE_CLUSTERS)
    assert torch.isclose(bc.total_persistence, _t(0.1 * 3 + 4.9 * 2), atol=1e-12)


def test_total_persistence_is_differentiable():
    pts = _THREE_CLUSTERS.clone().requires_grad_(True)
    total_persistence(pts).backward()
    assert pts.grad is not None
    assert bool(torch.isfinite(pts.grad).all())


# --------------------------------------------------------------------------- #
# SRTD                                                                         #
# --------------------------------------------------------------------------- #
def test_srtd_is_zero_for_identical_clouds():
    assert torch.isclose(srtd(_THREE_CLUSTERS, _THREE_CLUSTERS), _t(0.0), atol=1e-12)


def test_srtd_is_symmetric_and_nonnegative():
    a = _THREE_CLUSTERS
    b = _t([[0.0, 0.0], [0.05, 0.0], [3.0, 0.0], [3.05, 0.0], [9.0, 0.0], [9.1, 0.0]])
    d_ab = srtd(a, b)
    d_ba = srtd(b, a)
    assert torch.isclose(d_ab, d_ba, atol=1e-12)
    assert float(d_ab) >= 0.0


def test_srtd_grows_when_topology_differs():
    tight = _t([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0],
                [0.15, 0.0], [0.2, 0.0], [0.25, 0.0]])     # one blob
    d_same = srtd(_THREE_CLUSTERS, _THREE_CLUSTERS + 1e-3)  # near-identical topology
    d_diff = srtd(_THREE_CLUSTERS, tight)                   # very different topology
    assert float(d_diff) > float(d_same)
    assert float(d_diff) > 1.0                              # the 4.9-scale bars dominate


def test_srtd_known_value_on_a_one_simplex_pair():
    # two 2-point clouds: a single MST edge each (lengths 2 and 5).
    x = _t([[0.0, 0.0], [2.0, 0.0]])
    y = _t([[0.0, 0.0], [5.0, 0.0]])
    # one finite bar each: |2 - 5| / 1 = 3
    assert torch.isclose(srtd(x, y), _t(3.0), atol=1e-12)


def test_srtd_is_differentiable():
    a = _THREE_CLUSTERS.clone().requires_grad_(True)
    b = (_THREE_CLUSTERS + 0.3)
    srtd(a, b).backward()
    assert bool(torch.isfinite(a.grad).all())


# --------------------------------------------------------------------------- #
# guards                                                                      #
# --------------------------------------------------------------------------- #
def test_persistence_rejects_batched_input():
    with pytest.raises(ValueError):
        persistent_homology_h0(torch.zeros(2, 3, 2, dtype=_f64))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
