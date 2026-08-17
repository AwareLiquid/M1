"""
mt_lnn/spatial_ops.py — composable geometric operators (L4 step 2).

Why this module exists
----------------------
``spatial.py`` gives the backbone a way to *perceive* geometry (turn coordinates
/ point clouds / voxels into tokens). That is necessary but not sufficient for
"真正的空间推理": perception describes a scene, it does not *compute* over its
geometry. This module adds the missing layer — a small set of **composable,
zero-parameter, differentiable-where-it-makes-sense** spatial operators that
answer geometric questions directly from coordinates:

    distance / direction   — metric structure (Euclidean distance, bearings,
                             unit relative-direction vectors);
    containment            — point ∈ axis-aligned box / ball;
    proximity graphs       — radius graph, kNN graph (the adjacency that turns a
                             point set into a navigable structure);
    topology               — connected components (label propagation);
    REACHABILITY           — which points are reachable from a source over a
                             proximity graph, and in how many hops (transitive
                             closure / BFS as batched tensor ops).

The point is **composition**: the operators chain. Build a radius graph from
pairwise distances, then ask reachability over it with an obstacle gap, and you
get a genuine spatial-reasoning query ("can the agent get from A to B given this
step length?") that is *computed*, not memorised — exactly the
memory-vs-reasoning distinction the spatial-reasoning roadmap calls for.

Design / coupling
-----------------
* Pure functions, **stateless, zero trainable parameters**. Nothing here is an
  ``nn.Module``; there is no backbone coupling. Imports only ``torch``; never
  imports ``model.py``.
* Batched: every operator accepts ``coords`` shaped ``(B, N, D)`` or ``(N, D)``
  (a missing batch dim is added and squeezed back). Graphs are ``(B, N, N)``.
* Metric operators (distance, direction) are differentiable. Graph/topology
  operators threshold and are piecewise-constant (discrete spatial reasoning);
  this is documented per-function rather than hidden.

Every operator is pinned against an analytic ground truth in
``tests/test_spatial_ops.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "pairwise_distance",
    "relative_direction",
    "bearing",
    "in_bounding_box",
    "in_ball",
    "radius_graph",
    "knn_graph",
    "reachable_from",
    "hop_distance",
    "connected_components",
]


def _batched(coords: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Return ``(coords_3d, had_batch)``; add a batch dim if ``coords`` is 2-D."""
    if coords.dim() == 2:
        return coords.unsqueeze(0), False
    if coords.dim() == 3:
        return coords, True
    raise ValueError(
        f"coords must be (N, D) or (B, N, D), got shape {tuple(coords.shape)}"
    )


# ---------------------------------------------------------------------------
# Metric structure (differentiable)
# ---------------------------------------------------------------------------

def pairwise_distance(coords: torch.Tensor, *, eps: float = 0.0) -> torch.Tensor:
    """Euclidean distance matrix.

    Parameters
    ----------
    coords : (B, N, D) or (N, D)
    eps : float
        Optional value added under the sqrt for a smooth gradient at distance 0.

    Returns
    -------
    (B, N, N) or (N, N) distances; ``d[..., i, j] = ||p_i - p_j||``.
    Differentiable.
    """
    c, had_batch = _batched(coords)
    diff = c.unsqueeze(-2) - c.unsqueeze(-3)              # (B, N, N, D)
    d = (diff.pow(2).sum(-1) + eps).clamp_min(0).sqrt()  # (B, N, N)
    return d if had_batch else d.squeeze(0)


def relative_direction(coords: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Unit vectors pointing from point ``i`` to point ``j``.

    Returns ``(B, N, N, D)``; the ``i == j`` entries are zero. Differentiable.
    """
    c, had_batch = _batched(coords)
    diff = c.unsqueeze(-2) - c.unsqueeze(-3)              # (B, N, N, D): p_i - p_j... careful
    # We want i -> j, i.e. p_j - p_i:
    diff = -diff
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(eps)
    unit = diff / norm
    # zero out the self entries (direction to oneself is undefined)
    N = c.shape[-2]
    eye = torch.eye(N, dtype=torch.bool, device=c.device).view(1, N, N, 1)
    unit = unit.masked_fill(eye, 0.0)
    return unit if had_batch else unit.squeeze(0)


def bearing(coords: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Planar bearing angle (radians, ``atan2``) from ``i`` to ``j`` for 2-D points.

    Returns ``(B, N, N)`` in ``(-pi, pi]``; ``i == j`` entries are 0. 2-D only.
    """
    c, had_batch = _batched(coords)
    if c.shape[-1] != 2:
        raise ValueError(f"bearing is defined for 2-D coords, got D={c.shape[-1]}")
    dvec = c.unsqueeze(-2) - c.unsqueeze(-3)              # p_i - p_j
    dvec = -dvec                                          # i -> j = p_j - p_i
    ang = torch.atan2(dvec[..., 1], dvec[..., 0])        # (B, N, N)
    N = c.shape[-2]
    eye = torch.eye(N, dtype=torch.bool, device=c.device).unsqueeze(0)
    ang = ang.masked_fill(eye, 0.0)
    return ang if had_batch else ang.squeeze(0)


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------

def in_bounding_box(coords: torch.Tensor, lo, hi) -> torch.Tensor:
    """Axis-aligned box containment: ``lo <= p <= hi`` on every axis.

    ``lo`` / ``hi`` are broadcastable to ``(D,)``. Returns a boolean mask
    ``(B, N)`` or ``(N,)``.
    """
    c, had_batch = _batched(coords)
    lo = torch.as_tensor(lo, dtype=c.dtype, device=c.device)
    hi = torch.as_tensor(hi, dtype=c.dtype, device=c.device)
    inside = ((c >= lo) & (c <= hi)).all(dim=-1)         # (B, N)
    return inside if had_batch else inside.squeeze(0)


def in_ball(coords: torch.Tensor, center, radius: float) -> torch.Tensor:
    """Ball containment: ``||p - center|| <= radius``.

    ``center`` is broadcastable to ``(D,)``. Returns boolean ``(B, N)`` / ``(N,)``.
    """
    c, had_batch = _batched(coords)
    center = torch.as_tensor(center, dtype=c.dtype, device=c.device)
    inside = (c - center).norm(dim=-1) <= float(radius)  # (B, N)
    return inside if had_batch else inside.squeeze(0)


# ---------------------------------------------------------------------------
# Proximity graphs
# ---------------------------------------------------------------------------

def radius_graph(coords: torch.Tensor, radius: float, *,
                 include_self: bool = False) -> torch.Tensor:
    """Symmetric radius graph: edge ``i—j`` iff ``||p_i - p_j|| <= radius``.

    Returns a float adjacency ``(B, N, N)`` / ``(N, N)`` in {0, 1}. Symmetric by
    construction. Thresholded (piecewise-constant, not differentiable in
    ``radius``).
    """
    d = pairwise_distance(coords)
    adj = (d <= float(radius)).to(coords.dtype)
    if not include_self:
        N = adj.shape[-1]
        eye = torch.eye(N, dtype=torch.bool, device=adj.device)
        adj = adj.masked_fill(eye, 0.0)
    return adj


def knn_graph(coords: torch.Tensor, k: int, *,
              symmetric: bool = True) -> torch.Tensor:
    """k-nearest-neighbour graph (excluding self).

    Returns a float adjacency in {0, 1}. With ``symmetric=True`` (default) an
    edge exists if *either* endpoint lists the other among its k nearest — the
    usual undirected kNN graph.
    """
    c, had_batch = _batched(coords)
    N = c.shape[-2]
    if not (1 <= k < N):
        raise ValueError(f"k must be in [1, N-1]={N - 1}, got {k}")
    d = pairwise_distance(c)                              # (B, N, N)
    eye = torch.eye(N, dtype=torch.bool, device=c.device)
    d = d.masked_fill(eye.unsqueeze(0), float("inf"))    # exclude self
    idx = d.topk(k, dim=-1, largest=False).indices       # (B, N, k)
    adj = torch.zeros(c.shape[0], N, N, dtype=c.dtype, device=c.device)
    adj.scatter_(-1, idx, 1.0)
    if symmetric:
        adj = torch.maximum(adj, adj.transpose(-1, -2))
    return adj if had_batch else adj.squeeze(0)


# ---------------------------------------------------------------------------
# Reachability / topology (batched transitive closure)
# ---------------------------------------------------------------------------

def _prep_adj_source(adjacency: torch.Tensor, source: torch.Tensor):
    a = adjacency.unsqueeze(0) if adjacency.dim() == 2 else adjacency
    s = source
    if s.dim() == 1:
        s = s.unsqueeze(0)
    if a.shape[0] != s.shape[0]:
        if a.shape[0] == 1:
            a = a.expand(s.shape[0], -1, -1)
        elif s.shape[0] == 1:
            s = s.expand(a.shape[0], -1)
    return a, s


def reachable_from(adjacency: torch.Tensor, source: torch.Tensor, *,
                   max_hops: Optional[int] = None) -> torch.Tensor:
    """Transitive closure: which nodes are reachable from ``source`` over the graph.

    Parameters
    ----------
    adjacency : (B, N, N) or (N, N)
        Edge weights > 0 are treated as edges (e.g. from :func:`radius_graph`).
    source : (B, N) or (N,)
        Boolean / {0,1} mask of seed nodes.
    max_hops : int, optional
        Cap on path length. ``None`` runs to the fixed point (full closure).

    Returns
    -------
    Float reachability mask ``(B, N)`` / ``(N,)`` in {0, 1}. The source itself is
    always reachable. Discrete (thresholded), not differentiable.
    """
    had_batch = adjacency.dim() == 3 and source.dim() == 2
    a, reach = _prep_adj_source(adjacency, source)
    B, N, _ = a.shape
    a_bool = (a > 0).to(torch.float32)
    # self-loops so that already-reached nodes persist
    eye = torch.eye(N, device=a.device).unsqueeze(0)
    a_hat = torch.maximum(a_bool, eye)
    reach = (reach > 0).to(torch.float32)                # (B, N)
    steps = N if max_hops is None else int(max_hops)
    for _ in range(steps):
        # next[b, j] = OR_i reach[b, i] AND a_hat[b, i, j]
        nxt = (reach.unsqueeze(1) @ a_hat).squeeze(1)    # (B, N)
        nxt = (nxt > 0).to(torch.float32)
        if max_hops is None and torch.equal(nxt, reach):
            break
        reach = nxt
    return reach if had_batch else reach.squeeze(0)


def hop_distance(adjacency: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Shortest hop count from any seed in ``source`` to every node (BFS).

    Returns ``(B, N)`` / ``(N,)`` of non-negative hop counts; unreachable nodes
    get ``inf``. Seeds are distance 0. Discrete, not differentiable.
    """
    had_batch = adjacency.dim() == 3 and source.dim() == 2
    a, seed = _prep_adj_source(adjacency, source)
    B, N, _ = a.shape
    a_bool = (a > 0).to(torch.float32)
    eye = torch.eye(N, device=a.device).unsqueeze(0)
    a_hat = torch.maximum(a_bool, eye)

    reach = (seed > 0).to(torch.float32)                 # (B, N)
    dist = torch.where(reach > 0,
                       torch.zeros(B, N, device=a.device),
                       torch.full((B, N), float("inf"), device=a.device))
    for h in range(1, N):
        nxt = (reach.unsqueeze(1) @ a_hat).squeeze(1)
        nxt = (nxt > 0).to(torch.float32)
        newly = (nxt > 0) & (reach == 0)                 # nodes first reached at hop h
        dist = torch.where(newly, torch.full_like(dist, float(h)), dist)
        if torch.equal(nxt, reach):
            break
        reach = nxt
    return dist if had_batch else dist.squeeze(0)


def connected_components(adjacency: torch.Tensor) -> torch.Tensor:
    """Connected-component labels via min-index label propagation.

    Each node is labelled with the smallest node index in its component (a
    canonical, deterministic labelling). Expects a symmetric adjacency. Returns
    integer labels ``(B, N)`` / ``(N,)``. Discrete, not differentiable.
    """
    a = adjacency.unsqueeze(0) if adjacency.dim() == 2 else adjacency
    had_batch = adjacency.dim() == 3
    B, N, _ = a.shape
    a_bool = (a > 0)
    eye = torch.eye(N, dtype=torch.bool, device=a.device).unsqueeze(0)
    a_hat = a_bool | eye                                 # include self
    labels = torch.arange(N, device=a.device).unsqueeze(0).expand(B, N).clone().float()
    big = float(N + 1)
    for _ in range(N):
        # neighbour label broadcast: cand[b, i, j] = labels[b, j] if edge else big
        cand = torch.where(a_hat, labels.unsqueeze(1).expand(B, N, N),
                           torch.full((B, N, N), big, device=a.device))
        new_labels = cand.min(dim=-1).values             # (B, N)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
    labels = labels.long()
    return labels if had_batch else labels.squeeze(0)
