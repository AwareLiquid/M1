"""
tests/test_spatial_ops.py — composable geometric operators (L4 step 2).

Each operator is pinned against an ANALYTIC ground truth (not a snapshot), and a
final composition test chains distance -> radius graph -> reachability to answer
a real "can the agent cross this gap?" query — geometric reasoning that is
*computed*, not memorised.
"""
import math
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.spatial_ops import (  # noqa: E402
    pairwise_distance,
    relative_direction,
    bearing,
    in_bounding_box,
    in_ball,
    radius_graph,
    knn_graph,
    reachable_from,
    hop_distance,
    connected_components,
)


# --- metric structure -----------------------------------------------------

def test_pairwise_distance_matches_cdist():
    g = torch.Generator().manual_seed(0)
    coords = torch.randn(2, 7, 3, generator=g)
    d = pairwise_distance(coords)
    ref = torch.cdist(coords, coords)
    assert d.shape == (2, 7, 7)
    assert torch.allclose(d, ref, atol=1e-5)
    # diagonal is zero, symmetric
    assert torch.allclose(d.diagonal(dim1=-2, dim2=-1), torch.zeros(2, 7), atol=1e-6)
    assert torch.allclose(d, d.transpose(-1, -2), atol=1e-6)


def test_pairwise_distance_2d_input_squeezes_batch():
    coords = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    d = pairwise_distance(coords)
    assert d.shape == (2, 2)
    assert d[0, 1].item() == pytest.approx(5.0)


def test_pairwise_distance_is_differentiable():
    coords = torch.randn(1, 4, 2, requires_grad=True)
    pairwise_distance(coords, eps=1e-12).sum().backward()
    assert coords.grad is not None and torch.isfinite(coords.grad).all()


def test_relative_direction_points_from_i_to_j_and_is_unit():
    coords = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 5.0]])
    u = relative_direction(coords)                       # (3, 3, 2)
    # i=0 -> j=1 is +x
    assert torch.allclose(u[0, 1], torch.tensor([1.0, 0.0]), atol=1e-6)
    # i=0 -> j=2 is +y
    assert torch.allclose(u[0, 2], torch.tensor([0.0, 1.0]), atol=1e-6)
    # i=1 -> j=0 is -x
    assert torch.allclose(u[1, 0], torch.tensor([-1.0, 0.0]), atol=1e-6)
    # self entries are zero; off-diagonals are unit length
    assert torch.allclose(u[0, 0], torch.zeros(2), atol=1e-6)
    off = u[0, 1:].norm(dim=-1)
    assert torch.allclose(off, torch.ones_like(off), atol=1e-6)


def test_bearing_matches_atan2():
    coords = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
    b = bearing(coords)
    assert b[0, 1].item() == pytest.approx(math.pi / 4, abs=1e-6)   # NE
    assert abs(b[0, 2].item()) == pytest.approx(math.pi, abs=1e-6)  # due west
    with pytest.raises(ValueError):
        bearing(torch.randn(5, 3))                       # 3-D not allowed


# --- containment ----------------------------------------------------------

def test_in_bounding_box():
    coords = torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.2, 0.9]])
    inside = in_bounding_box(coords, lo=[0.0, 0.0], hi=[1.0, 1.0])
    assert inside.tolist() == [True, False, True]


def test_in_ball():
    coords = torch.tensor([[0.0, 0.0], [0.9, 0.0], [2.0, 0.0]])
    inside = in_ball(coords, center=[0.0, 0.0], radius=1.0)
    assert inside.tolist() == [True, True, False]


# --- proximity graphs -----------------------------------------------------

def test_radius_graph_is_symmetric_and_thresholds_correctly():
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.5, 0.0]])
    adj = radius_graph(coords, radius=1.2)
    # 0-1 distance 1.0 <= 1.2 (edge); 1-2 distance 1.5 > 1.2 (no edge); 0-2 no
    assert adj[0, 1] == 1.0 and adj[1, 0] == 1.0
    assert adj[1, 2] == 0.0 and adj[0, 2] == 0.0
    assert adj.diagonal().sum() == 0.0                  # no self-loops by default
    assert torch.equal(adj, adj.T)


def test_radius_graph_include_self():
    coords = torch.tensor([[0.0, 0.0], [5.0, 0.0]])
    adj = radius_graph(coords, radius=0.1, include_self=True)
    assert adj[0, 0] == 1.0 and adj[1, 1] == 1.0


def test_knn_graph_picks_nearest():
    # a line of 4 points; each point's nearest neighbour is its immediate one
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    adj = knn_graph(coords, k=1, symmetric=True)
    # 0<->1 nearest, 1<->2 nearest; 3's nearest is 2 -> symmetric edge 2-3
    assert adj[0, 1] == 1.0
    assert adj[1, 2] == 1.0
    assert adj[2, 3] == 1.0 and adj[3, 2] == 1.0
    assert torch.equal(adj, adj.T)
    with pytest.raises(ValueError):
        knn_graph(coords, k=4)                          # k must be < N


# --- reachability / topology ----------------------------------------------

def _chain_adj(n):
    # path graph 0-1-2-...-(n-1)
    a = torch.zeros(n, n)
    for i in range(n - 1):
        a[i, i + 1] = 1.0
        a[i + 1, i] = 1.0
    return a


def test_reachable_from_full_closure_on_a_path():
    a = _chain_adj(5)
    src = torch.zeros(5); src[0] = 1.0
    reach = reachable_from(a, src)
    assert reach.tolist() == [1, 1, 1, 1, 1]            # whole path reachable


def test_reachable_from_respects_a_disconnection():
    # two separate edges: 0-1 and 2-3 (no link between the pairs)
    a = torch.zeros(4, 4)
    a[0, 1] = a[1, 0] = 1.0
    a[2, 3] = a[3, 2] = 1.0
    reach = reachable_from(a, torch.tensor([1.0, 0, 0, 0]))
    assert reach.tolist() == [1, 1, 0, 0]               # cannot cross the gap


def test_reachable_from_max_hops_limits_range():
    a = _chain_adj(6)
    src = torch.zeros(6); src[0] = 1.0
    reach = reachable_from(a, src, max_hops=2)
    assert reach.tolist() == [1, 1, 1, 0, 0, 0]         # only within 2 hops


def test_hop_distance_matches_bfs_on_a_path():
    a = _chain_adj(5)
    src = torch.zeros(5); src[0] = 1.0
    dist = hop_distance(a, src)
    assert dist.tolist() == [0, 1, 2, 3, 4]


def test_hop_distance_marks_unreachable_as_inf():
    a = torch.zeros(3, 3)
    a[0, 1] = a[1, 0] = 1.0                              # node 2 isolated
    dist = hop_distance(a, torch.tensor([1.0, 0, 0]))
    assert dist[0] == 0 and dist[1] == 1
    assert math.isinf(dist[2].item())


def test_connected_components_labels_partition():
    # components {0,1,2} and {3,4}
    a = torch.zeros(5, 5)
    for i, j in [(0, 1), (1, 2), (3, 4)]:
        a[i, j] = a[j, i] = 1.0
    labels = connected_components(a)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]
    # canonical = min index in component
    assert labels[0].item() == 0 and labels[3].item() == 3


def test_batched_reachability():
    a = torch.stack([_chain_adj(4), torch.zeros(4, 4)])  # batch: a path and an empty graph
    src = torch.zeros(2, 4); src[:, 0] = 1.0
    reach = reachable_from(a, src)
    assert reach[0].tolist() == [1, 1, 1, 1]             # path: all reachable
    assert reach[1].tolist() == [1, 0, 0, 0]             # empty graph: only source


# --- composition: the flagship spatial-reasoning query --------------------

def test_composition_reachability_across_a_gap_depends_on_step_length():
    """Two clusters of stepping-stones separated by a gap. The agent can only
    step <= `reach` at a time. Compose: distances -> radius graph -> reachability.
    Whether B is reachable from A is *computed* from the geometry and the step
    length — not stored. A larger stride bridges the gap; a smaller one cannot."""
    # left cluster near x=0, right cluster near x=3, gap ~1.5 between nearest stones
    coords = torch.tensor([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],     # A side (indices 0,1,2)
        [2.5, 0.0], [3.0, 0.0], [3.5, 0.0],     # B side (indices 3,4,5)
    ])
    src = torch.zeros(6); src[0] = 1.0                  # start on the far-left stone
    goal = 5

    # short stride: can hop within a cluster (<=0.6) but not the 1.5 gap
    short = radius_graph(coords, radius=0.6)
    reach_short = reachable_from(short, src)
    assert reach_short[goal] == 0.0                     # gap not crossable
    assert reach_short[:3].sum() == 3.0                 # whole A side reachable

    # long stride: 1.5 now bridges stone 2 -> stone 3, unlocking the B side
    long = radius_graph(coords, radius=1.6)
    reach_long = reachable_from(long, src)
    assert reach_long[goal] == 1.0                      # goal now reachable
    assert reach_long.sum() == 6.0                      # everything reachable

    # and the hop distance to the goal is finite only with the long stride
    assert math.isinf(hop_distance(short, src)[goal].item())
    assert torch.isfinite(hop_distance(long, src)[goal]).item()


def test_spatial_ops_module_does_not_import_model():
    import mt_lnn.spatial_ops as so
    src = open(so.__file__, encoding="utf-8").read()
    assert "import model" not in src and "from .model" not in src


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except Exception:
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
            raise
