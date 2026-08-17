"""
mt_lnn/topology_ops.py -- composable topological-data-analysis (TDA) operators
for the hidden-state point cloud (P0#2).

Why this module exists
----------------------
``geometry_ops.py`` measures *distance and curvature* on the simplex; this layer
measures *shape* -- the connectivity invariants of a whole population of states.
A set of place-cell codes, L2 memory keys, or hidden states is a point cloud,
and the question "how many clusters / attractor basins does it have, and how
strongly separated are they?" is a topological one. The answer is its
**0-dimensional persistent homology** (Betti-0 over a growing distance scale):
the number of connected components as a function of the merge radius, and the
*persistence* (lifetime) of each component before it merges into another.

The operators here compute that exactly and cheaply:

    minimum_spanning_tree  -- single-linkage skeleton of the cloud (Prim);
    persistent_homology_h0 -- the H0 barcode: every point is born at scale 0 and
                              dies (merges) at the weight of the MST edge that
                              first connects its component to a smaller-indexed
                              one, except one component that never dies (the
                              infinite bar). This is the standard, exact
                              single-linkage = MST characterisation of H0.
    betti0 / betti0_curve  -- the number of connected components at a scale eps,
                              computed by REUSING spatial_ops' union-find-style
                              connected_components on the radius graph (and
                              equivalently derivable from the MST, cross-checked
                              in the tests).
    total_persistence      -- a single scalar "how clustered" summary (sum of
                              finite bar lengths): a differentiable topology size.
    srtd                   -- Symmetric Relative-Topology Divergence: a symmetric,
                              zero-at-coincidence divergence between the H0
                              barcodes of two clouds; usable as a regulariser
                              signal that *penalises topology change* (e.g. across
                              a steering edit or two augmented views).

Why MST gives H0 persistence (the one fact this module rests on)
----------------------------------------------------------------
Grow a ball of radius eps around every point and connect points within 2*eps;
as eps increases, components only ever *merge*, never split (H0 is monotone).
Two components merge exactly when eps reaches half the shortest edge bridging
them -- and the set of those bridging edges, over the whole filtration, is
precisely the minimum spanning tree. So the multiset of H0 death scales equals
the multiset of MST edge weights, and Betti-0 at scale eps is
``N - #{MST edges with weight <= eps}``. This is exact, not an approximation.

Design / coupling
-----------------
* Pure functions, **stateless, zero trainable parameters**. Nothing here is an
  ``nn.Module``. Imports only ``torch`` / ``dataclasses`` / ``math`` and the
  sibling operator module ``spatial_ops`` (for distances + the union-find
  connected components the roadmap asked to reuse). **Never imports model.py.**
* Metric quantities (MST edge weights, total persistence, SRTD) are
  differentiable in the point positions. The integer Betti numbers threshold and
  are piecewise-constant (genuine discrete topology) -- documented, not hidden.
* Honest scope: this is H0 (connected components) only -- the cheap, exact part
  of persistent homology. Higher homology (loops H1, voids H2) and the full
  multi-scale Representation-Topology-Divergence are deliberately out of scope;
  ``srtd`` is the H0 sorted-barcode surrogate, stated as such.

Pinned against analytic ground truth in ``tests/test_topology_ops.py`` and over
the whole input space in ``tests/test_topology_ops_properties.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .spatial_ops import connected_components, pairwise_distance, radius_graph

__all__ = [
    "minimum_spanning_tree",
    "MinimumSpanningTree",
    "betti0",
    "betti0_curve",
    "persistent_homology_h0",
    "Barcode",
    "total_persistence",
    "srtd",
]


def _as_cloud(points: torch.Tensor) -> torch.Tensor:
    """Validate and return a 2-D ``(N, D)`` point cloud (no batch dim)."""
    if points.dim() != 2:
        raise ValueError(
            f"points must be a single cloud of shape (N, D), got "
            f"{tuple(points.shape)}; the persistence operators are per-cloud."
        )
    if points.shape[0] < 1:
        raise ValueError("need at least one point")
    return points


# --------------------------------------------------------------------------- #
# minimum spanning tree (single-linkage skeleton)                             #
# --------------------------------------------------------------------------- #
@dataclass
class MinimumSpanningTree:
    """The MST of a point cloud.

    Attributes
    ----------
    edges : (N-1, 2) long
        Each row ``(i, j)`` is an undirected tree edge, in the order Prim added
        them (so weights are NOT sorted -- sort them for the barcode).
    weights : (N-1,)
        ``weights[k] = ||points[edges[k,0]] - points[edges[k,1]]||``. Differentiable.
    n_points : int
    """

    edges: torch.Tensor
    weights: torch.Tensor
    n_points: int

    @property
    def total_weight(self) -> torch.Tensor:
        """Sum of edge lengths (the classic MST cost)."""
        return self.weights.sum()


def minimum_spanning_tree(points: torch.Tensor) -> MinimumSpanningTree:
    """Euclidean minimum spanning tree via Prim's algorithm on the distance matrix.

    Prim grows one tree from node 0, repeatedly attaching the nearest node not
    yet in the tree. ``O(N^2)`` with dense tensor updates -- ample for the small
    populations (codes / states) this layer is meant for. The edge *weights* are
    differentiable in ``points`` (each is a Euclidean norm); the *combinatorial*
    tree (which edges) is piecewise-constant.
    """
    p = _as_cloud(points)
    n = p.shape[0]
    dev, dt = p.device, p.dtype
    if n == 1:
        return MinimumSpanningTree(
            edges=torch.empty(0, 2, dtype=torch.long, device=dev),
            weights=torch.empty(0, dtype=dt, device=dev),
            n_points=1,
        )

    dist = pairwise_distance(p)                       # (N, N), differentiable
    visited = torch.zeros(n, dtype=torch.bool, device=dev)
    visited[0] = True
    # best[j] = shortest known distance from the tree to unvisited node j,
    # achieved via the tree node best_from[j].
    best = dist[0].clone()                            # distances from node 0
    best_from = torch.zeros(n, dtype=torch.long, device=dev)

    edge_rows = []
    weight_idx = []                                   # (from, to) to re-read diff'able dist
    for _ in range(n - 1):
        masked = best.masked_fill(visited, math.inf)
        j = int(torch.argmin(masked).item())          # nearest unvisited node
        i = int(best_from[j].item())
        edge_rows.append((i, j))
        weight_idx.append((i, j))
        visited[j] = True
        # relax: can node j offer a shorter connection to any unvisited node?
        dj = dist[j]
        better = (dj < best) & (~visited)
        best = torch.where(better, dj, best)
        best_from = torch.where(better, torch.full_like(best_from, j), best_from)

    edges = torch.tensor(edge_rows, dtype=torch.long, device=dev)
    # Recompute the chosen edge lengths straight from the point differences
    # (NOT by indexing the dense ``dist`` matrix): the matrix's zero diagonal
    # makes ``sqrt`` non-differentiable there, and the unused-entry gradient is
    # ``0 * inf = NaN``. Tree edges join distinct points, so these norms are
    # differentiable. ``dist`` stays for the (non-differentiable) Prim choices.
    ii = edges[:, 0]
    jj = edges[:, 1]
    weights = (p[ii] - p[jj]).norm(dim=-1)
    return MinimumSpanningTree(edges=edges, weights=weights, n_points=n)


# --------------------------------------------------------------------------- #
# Betti-0 via the reused union-find connected components                      #
# --------------------------------------------------------------------------- #
def betti0(points: torch.Tensor, eps: float, *, include_self: bool = True) -> int:
    """Number of connected components of the cloud at merge scale ``eps``.

    Builds the radius graph (edge iff ``||p_i - p_j|| <= eps``) and counts the
    components with the sibling ``spatial_ops.connected_components`` (the
    union-find-style label propagation the roadmap asked to reuse). Returns a
    Python ``int`` (a genuine integer topological invariant). Discrete: not
    differentiable in ``points`` or ``eps``.
    """
    p = _as_cloud(points)
    n = p.shape[0]
    if n == 1:
        return 1
    adj = radius_graph(p, radius=float(eps), include_self=False)
    labels = connected_components(adj)                # (N,) canonical labels
    return int(torch.unique(labels).numel())


def betti0_curve(points: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Betti-0 as a function of scale: the component count at each ``thresholds``.

    Returns an integer tensor of the same length as ``thresholds``. Monotonically
    non-increasing (components only merge as the scale grows) -- the discrete
    signature of the H0 persistence barcode. Discrete, not differentiable.
    """
    p = _as_cloud(points)
    th = torch.as_tensor(thresholds, dtype=p.dtype, device=p.device).flatten()
    return torch.tensor(
        [betti0(p, float(e)) for e in th], dtype=torch.long, device=p.device
    )


# --------------------------------------------------------------------------- #
# H0 persistent homology (the barcode)                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Barcode:
    """An H0 persistence barcode.

    Every point is born at scale 0; ``finite_deaths`` are the scales at which
    components merge (exactly the sorted MST edge weights), and exactly one
    component never dies (the infinite bar). Stored split so the infinite bar
    never poisons sums.

    Attributes
    ----------
    finite_deaths : (N-1,)
        Sorted ascending merge scales. Differentiable in the cloud.
    n_points : int
        Total components at scale 0 (= number of points).
    """

    finite_deaths: torch.Tensor
    n_points: int

    @property
    def n_bars(self) -> int:
        """Total H0 bars = N (one per point); ``N-1`` finite + 1 infinite."""
        return self.n_points

    @property
    def total_persistence(self) -> torch.Tensor:
        """Sum of finite bar lifetimes (births are 0, so just the death sum).

        A single differentiable scalar measuring "how spread out / clustered"
        the cloud is: large when components persist (well-separated clusters),
        small when everything merges immediately (one tight blob).
        """
        if self.finite_deaths.numel() == 0:
            return torch.zeros((), dtype=torch.float32)
        return self.finite_deaths.sum()

    def betti0_at(self, eps: float) -> int:
        """Components alive at scale ``eps`` = ``N - #{finite deaths <= eps}``.

        Derived purely from the barcode (the MST characterisation); the tests
        cross-check this against the union-find :func:`betti0`.
        """
        merged = int((self.finite_deaths <= float(eps)).sum().item())
        return self.n_points - merged


def persistent_homology_h0(points: torch.Tensor) -> Barcode:
    """The exact H0 persistence barcode of a Euclidean point cloud.

    Uses the single-linkage = MST fact: the H0 death scales are exactly the MST
    edge weights. Births are all 0; one component is immortal (the infinite bar),
    represented by storing only the ``N-1`` finite deaths. Differentiable in the
    point positions through the MST edge lengths.
    """
    p = _as_cloud(points)
    mst = minimum_spanning_tree(p)
    finite_deaths, _ = torch.sort(mst.weights)
    return Barcode(finite_deaths=finite_deaths, n_points=p.shape[0])


def total_persistence(points: torch.Tensor) -> torch.Tensor:
    """Convenience: ``persistent_homology_h0(points).total_persistence``.

    The differentiable scalar topology-size of the cloud (sum of H0 bar
    lifetimes / equivalently the total MST weight).
    """
    return persistent_homology_h0(points).total_persistence


# --------------------------------------------------------------------------- #
# Symmetric Relative-Topology Divergence (a regulariser signal)               #
# --------------------------------------------------------------------------- #
def srtd(points_x: torch.Tensor, points_y: torch.Tensor, *, p: float = 1.0) -> torch.Tensor:
    """Symmetric topology divergence between two clouds' H0 barcodes.

    Both clouds' finite-death multisets are sorted and compared termwise (the
    1-D / sorted optimal-transport matching), padding the shorter barcode with
    zero-length bars (a component that is born and dies immediately, i.e. sits on
    the diagonal). The result is::

        SRTD_p(X, Y) = ( mean_k | sorted_deaths_X[k] - sorted_deaths_Y[k] |^p )^(1/p)

    Properties (pinned in the tests): **symmetric** in its arguments,
    **non-negative**, and **zero iff the two H0 barcodes coincide** -- so driving
    it to zero forces two representations to share the same coarse cluster
    structure. Differentiable in both clouds (through the MST edge lengths).

    Honest scope: this is the H0 (connected-components) sorted-barcode surrogate
    for Representation-Topology-Divergence, not the full multi-dimensional RTD.
    It is a regulariser *signal*, not a certified bottleneck distance.
    """
    bx = persistent_homology_h0(points_x).finite_deaths
    by = persistent_homology_h0(points_y).finite_deaths
    nx, ny = bx.numel(), by.numel()
    dt = bx.dtype if nx >= ny else by.dtype
    dev = bx.device

    # sorted termwise comparison; pad the shorter with zero-length (diagonal) bars
    n = max(nx, ny)
    if nx < n:
        bx = torch.cat([bx, torch.zeros(n - nx, dtype=dt, device=dev)])
    if ny < n:
        by = torch.cat([by, torch.zeros(n - ny, dtype=dt, device=dev)])
    if n == 0:
        return torch.zeros((), dtype=dt, device=dev)

    diff = (bx - by).abs()
    if p == 1.0:
        return diff.mean()
    return diff.pow(p).mean().pow(1.0 / p)
