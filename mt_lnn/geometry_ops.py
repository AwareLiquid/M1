"""
mt_lnn/geometry_ops.py -- composable information-geometry operators on the
probability simplex (L4 / P0 step: the Riemannian metric for place-cell codes).

Why this module exists
----------------------
``spatial_ops.py`` computes over *Euclidean* geometry, ``physics_ops.py`` evolves
*dynamics*. But the spatial **memory** layer (``spatial_memory.py``) keys its
associative map with *soft place-cell codes* -- and those codes are produced by
:class:`mt_lnn.spatial.PlaceCellCode` as ``softmax(...)``: non-negative rows that
sum to one. **They are categorical probability distributions, i.e. points on the
probability simplex** ``Δ^{n-1}``, not free vectors in ``R^n``.

A precise scoping note (honesty first). The associative read-out
``out = key @ M`` is **not** a metric operation and is *not* in scope for this
module: it is the classical linear-associator readout of Bicanski & Burgess
(2021) -- a convex combination of the *content* rows ``M[p]`` (free vectors in
``R^{d_model}``, not simplex points) weighted by the soft place key. Re-deriving
it through a Fisher-Rao metric would be a category error, so ``spatial_memory``
deliberately leaves the readout as-is.

Where the Euclidean metric is genuinely *wrong* is whenever two place-cell
**codes** are compared, interpolated, or averaged -- ``d(p, q)``, the midpoint
of two codes, the mean of several recalls. Those operands are softmax rows:
non-negative, sum-to-one, i.e. points on the probability simplex ``Δ^{n-1}``. On
the simplex the natural, reparameterisation-invariant geometry is the
**Fisher-Rao information metric**

    g_p(u, v) = sum_i  u_i v_i / p_i          (u, v tangent: sum u_i = sum v_i = 0)

Mixing a Euclidean metric on a curved statistical manifold is exactly the kind of
*metric inconsistency* that lets small errors accumulate and breaks the symmetry
between "before" and "after" a state transition. This module supplies the
**correct** geometry as a set of composable, zero-parameter, analytic operators:
distance, geodesics, the exponential / logarithm maps, parallel transport, the
Fisher inner product, and a geodesic (Karcher) barycenter -- ready for the
code-comparison / consolidation use cases above (today its in-tree consumer is
``topology_ops``).

The closed form -- the sphere isometry
---------------------------------------
The map ``phi(p) = sqrt(p)`` is an isometry from the simplex with the Fisher-Rao
metric onto the **positive orthant of the unit sphere** ``{u : sum u_i^2 = 1}``
carrying *four times* the round metric. So everything reduces to spherical
trigonometry on ``u = sqrt(p)``:

* Bhattacharyya coefficient   ``BC(p, q) = sum_i sqrt(p_i q_i) = <sqrt(p), sqrt(q)>``
* Fisher-Rao distance         ``d(p, q) = 2 * arccos(BC(p, q))``   in ``[0, pi]``
* geodesic                    slerp of ``sqrt(p), sqrt(q)`` on the sphere, squared
* exp / log / transport        the sphere exp / log / great-circle transport,
                               pushed through the ``sqrt`` / ``( )^2`` change of
                               variables (a tangent ``v`` at ``p`` corresponds to
                               the sphere tangent ``xi = v / (2 sqrt(p))``).

Because ``sqrt(p) >= 0`` the embedding stays in the positive orthant, so the
great-circle angle is in ``[0, pi/2]`` and the distance in ``[0, pi]``; the only
degenerate point (``sin(angle) = 0``) is ``p == q``, handled explicitly.

Brain / architecture mapping
----------------------------
* The L2 place-cell key is a distribution over fields -> a simplex point. The
  Fisher-Rao distance is the metric-consistent "how different are these two
  remembered locations?" -- invariant to how we *label/permute* the fields, which
  is the precise statement of the symmetry a Euclidean *code comparison* would
  silently break (the linear readout itself is unaffected -- see the scoping note
  above).
* :func:`geodesic_interpolate` is the metric-consistent way to blend two codes
  (path integration / morphing between remembered places); the Euclidean
  midpoint of two codes is generally *not* even the right "halfway" distribution.
* :func:`geodesic_barycenter` is the proper averaging of place-cell codes
  (consolidating several recalls into one), replacing the naive arithmetic mean.

Design / coupling
-----------------
* Pure functions, **stateless, zero trainable parameters**. Nothing here is an
  ``nn.Module``; no backbone coupling. Imports only ``torch`` (+ ``dataclasses``,
  ``math``); **never imports ``model.py``**.
* Batched / broadcasting: a distribution is ``(..., n)`` with ``n`` the number of
  categories (place fields); the pairwise operators broadcast over any leading
  shape, so a whole trajectory of codes ``(T, n)`` flows through at once.
* Everything is differentiable except explicit ``clamp`` guards (documented per
  function) that keep ``arccos`` / divisions finite on the boundary.

Every operator is pinned against an analytic ground truth in
``tests/test_geometry_ops.py`` and against the geometric laws themselves
(symmetry, the triangle inequality, permutation invariance, exp/log inverse,
isometric parallel transport, constant-speed geodesics) in
``tests/test_geometry_ops_properties.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

__all__ = [
    "simplex_project",
    "is_on_simplex",
    "sphere_embed",
    "sphere_unembed",
    "bhattacharyya_coefficient",
    "fisher_rao_distance",
    "fisher_inner",
    "fisher_norm",
    "fisher_information_matrix",
    "geodesic_interpolate",
    "exp_map",
    "log_map",
    "parallel_transport",
    "geodesic_barycenter",
    "Geodesic",
    "fisher_rao_geodesic",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# 0. simplex housekeeping                                                     #
# --------------------------------------------------------------------------- #


def simplex_project(x: torch.Tensor, *, eps: float = _EPS) -> torch.Tensor:
    """Clamp to non-negative and renormalise the last axis to sum to one.

    A defensive projection back onto the (interior of the) simplex: it floors
    tiny negatives from numerical error and divides by the row sum. ``x`` is
    ``(..., n)``; returns ``(..., n)`` rows that sum to one.
    """
    x = x.clamp_min(0.0)
    s = x.sum(dim=-1, keepdim=True).clamp_min(eps)
    return x / s


def is_on_simplex(p: torch.Tensor, *, atol: float = 1e-5) -> torch.Tensor:
    """Boolean ``(...)``: each row is non-negative and sums to one within ``atol``."""
    nonneg = (p >= -atol).all(dim=-1)
    sums_one = (p.sum(dim=-1) - 1.0).abs() <= atol
    return nonneg & sums_one


def sphere_embed(p: torch.Tensor) -> torch.Tensor:
    """The isometry ``phi(p) = sqrt(p)`` onto the unit-sphere positive orthant."""
    return torch.sqrt(p.clamp_min(0.0))


def sphere_unembed(u: torch.Tensor) -> torch.Tensor:
    """The inverse ``phi^{-1}(u) = u^2`` (then renormalised for safety)."""
    return simplex_project(u * u)


# --------------------------------------------------------------------------- #
# 1. distance / inner product                                                 #
# --------------------------------------------------------------------------- #


def bhattacharyya_coefficient(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``BC(p, q) = sum_i sqrt(p_i q_i)`` in ``[0, 1]`` -- the sphere cosine.

    ``p`` and ``q`` are ``(..., n)`` and broadcast against each other; returns
    their broadcast leading shape ``(...)``. ``BC == 1`` iff ``p == q`` and
    ``BC == 0`` iff the supports are disjoint.
    """
    return (torch.sqrt(p.clamp_min(0.0)) * torch.sqrt(q.clamp_min(0.0))).sum(dim=-1)


def fisher_rao_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Fisher-Rao geodesic distance ``2 * arccos(BC(p, q))`` in ``[0, pi]``.

    The intrinsic distance on the simplex under the Fisher information metric:
    symmetric, zero iff ``p == q``, obeys the triangle inequality, and is
    invariant under a common permutation of the categories (relabelling the
    place fields). The Bhattacharyya cosine is clamped to ``[-1, 1]`` before
    ``arccos`` (the only non-differentiable guard). ``p, q`` are ``(..., n)``;
    returns ``(...)``.
    """
    bc = bhattacharyya_coefficient(p, q).clamp(-1.0, 1.0)
    return 2.0 * torch.arccos(bc)


def fisher_inner(p: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *,
                 eps: float = _EPS) -> torch.Tensor:
    """Fisher-Rao inner product ``<u, v>_p = sum_i u_i v_i / p_i`` at ``p``.

    ``u`` and ``v`` are tangent vectors at ``p`` (each summing to zero). ``p`` is
    floored by ``eps`` so the metric stays finite on the boundary. All are
    ``(..., n)``; returns ``(...)``.
    """
    return (u * v / p.clamp_min(eps)).sum(dim=-1)


def fisher_norm(p: torch.Tensor, v: torch.Tensor, *, eps: float = _EPS) -> torch.Tensor:
    """Fisher-Rao length ``sqrt(<v, v>_p)`` of a tangent vector ``v`` at ``p``."""
    return torch.sqrt(fisher_inner(p, v, v, eps=eps).clamp_min(0.0))


def fisher_information_matrix(p: torch.Tensor, *, eps: float = _EPS) -> torch.Tensor:
    """The Fisher metric tensor at ``p`` in ambient coordinates: ``diag(1 / p)``.

    For a categorical distribution the Fisher information of the mean parameters
    is diagonal with entries ``1 / p_i``; restricted to the tangent space
    (``sum v_i = 0``) this reproduces :func:`fisher_inner`. ``p`` is ``(..., n)``;
    returns ``(..., n, n)``.
    """
    inv = 1.0 / p.clamp_min(eps)
    return torch.diag_embed(inv)


# --------------------------------------------------------------------------- #
# 2. geodesics                                                                #
# --------------------------------------------------------------------------- #


def geodesic_interpolate(p: torch.Tensor, q: torch.Tensor, t) -> torch.Tensor:
    """Constant-speed Fisher-Rao geodesic from ``p`` (t=0) to ``q`` (t=1).

    Spherical-linear interpolation (slerp) of ``sqrt(p), sqrt(q)`` on the unit
    sphere, squared back onto the simplex::

        w(t) = [sin((1-t) * Omega) * sqrt(p) + sin(t * Omega) * sqrt(q)] / sin(Omega)
        gamma(t) = w(t)^2            (Omega = arccos(BC(p, q)))

    The result stays exactly on the simplex (``w(t)`` is a unit vector) and moves
    at constant Fisher-Rao speed, so ``d(p, gamma(t)) = t * d(p, q)``. ``t`` may
    be a Python float or a tensor broadcasting against the leading shape. ``p, q``
    are ``(..., n)``; returns ``(..., n)``. The ``Omega -> 0`` limit (``p == q``)
    falls back to a plain blend.
    """
    u = torch.sqrt(p.clamp_min(0.0))
    v = torch.sqrt(q.clamp_min(0.0))
    cos_omega = (u * v).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.arccos(cos_omega)                       # (..., 1)
    sin_omega = torch.sin(omega)
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=p.dtype, device=p.device)
    t = t.unsqueeze(-1) if t.dim() == p.dim() - 1 else t
    # slerp with a safe small-angle fallback (linear blend) where sin(Omega) ~ 0
    small = sin_omega < 1e-7
    so = sin_omega.clamp_min(1e-7)
    w_slerp = (torch.sin((1.0 - t) * omega) * u + torch.sin(t * omega) * v) / so
    w_lin = (1.0 - t) * u + t * v
    w = torch.where(small, w_lin, w_slerp)
    return simplex_project(w * w)


def exp_map(p: torch.Tensor, v: torch.Tensor, *, eps: float = _EPS) -> torch.Tensor:
    """Riemannian exponential map ``exp_p(v)`` on the simplex.

    Follows the geodesic leaving ``p`` with initial velocity the tangent vector
    ``v`` (which must satisfy ``sum_i v_i = 0``) for unit time. Implemented via
    the sphere: the simplex tangent ``v`` corresponds to the sphere tangent
    ``xi = v / (2 sqrt(p))`` (so ``||v||_p = 2 ||xi||``); apply the sphere
    exponential and square back. ``p, v`` are ``(..., n)``; returns ``(..., n)``.
    """
    u = torch.sqrt(p.clamp_min(eps))                      # on the sphere
    xi = v / (2.0 * u)                                    # sphere tangent at u
    norm = torch.linalg.vector_norm(xi, dim=-1, keepdim=True)
    safe = norm.clamp_min(eps)
    w = torch.cos(norm) * u + torch.sin(norm) * (xi / safe)
    w = torch.where(norm < eps, u, w)                     # exp_p(0) = p
    return simplex_project(w * w)


def log_map(p: torch.Tensor, q: torch.Tensor, *, eps: float = _EPS) -> torch.Tensor:
    """Riemannian logarithm ``log_p(q)``: the tangent at ``p`` pointing to ``q``.

    Inverse of :func:`exp_map`: returns the tangent vector ``v`` at ``p`` (with
    ``sum_i v_i = 0``) such that ``exp_p(v) == q`` and ``||v||_p == d(p, q)``.
    Computed as the sphere logarithm of ``sqrt(q)`` at ``sqrt(p)``, pushed back
    through the ``2 sqrt(p)`` change of variables. ``p, q`` are ``(..., n)``;
    returns ``(..., n)``.
    """
    u = torch.sqrt(p.clamp_min(eps))
    z = torch.sqrt(q.clamp_min(0.0))
    cos_theta = (u * z).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.arccos(cos_theta)
    diff = z - cos_theta * u                              # sphere geodesic dir (unnormed)
    ndiff = torch.linalg.vector_norm(diff, dim=-1, keepdim=True)
    safe = ndiff.clamp_min(eps)
    xi = theta * (diff / safe)                            # sphere tangent, ||xi|| = theta
    xi = torch.where(ndiff < eps, torch.zeros_like(xi), xi)
    return 2.0 * u * xi                                   # back to simplex tangent


def parallel_transport(p: torch.Tensor, q: torch.Tensor, v: torch.Tensor, *,
                       eps: float = _EPS) -> torch.Tensor:
    """Parallel-transport tangent ``v`` at ``p`` to the tangent space at ``q``.

    Transports ``v`` (``sum_i v_i = 0``) along the Fisher-Rao geodesic from ``p``
    to ``q``, preserving the Fisher-Rao inner product (an isometry between the two
    tangent spaces). Implemented as great-circle parallel transport of the
    corresponding sphere tangent and pushed back to ``q``. This is the rule the
    spatial *steering* operation must obey so that "directions" stay comparable
    before and after a transition (the symmetry the Euclidean update breaks).
    ``p, q, v`` are ``(..., n)``; returns the transported tangent ``(..., n)`` at
    ``q``.
    """
    u = torch.sqrt(p.clamp_min(eps))
    w = torch.sqrt(q.clamp_min(eps))
    xi = v / (2.0 * u)                                    # sphere tangent at u
    cos_theta = (u * w).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.arccos(cos_theta)
    sin_theta = torch.sin(theta)
    n = w - cos_theta * u                                 # geodesic direction at u (unnormed)
    nn = torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    safe = nn.clamp_min(eps)
    nhat = n / safe
    coef = (xi * nhat).sum(dim=-1, keepdim=True)          # component of xi along the geodesic
    xi_perp = xi - coef * nhat
    xi_rot = xi_perp + coef * (-sin_theta * u + cos_theta * nhat)
    xi_trans = torch.where(nn < eps, xi, xi_rot)          # p == q: nothing to rotate
    return 2.0 * w * xi_trans                             # sphere tangent at w -> simplex tangent at q


# --------------------------------------------------------------------------- #
# 3. geodesic (Karcher) barycenter                                            #
# --------------------------------------------------------------------------- #


def geodesic_barycenter(codes: torch.Tensor, weights: Optional[torch.Tensor] = None,
                        *, iters: int = 32, tol: float = 1e-9,
                        eps: float = _EPS) -> torch.Tensor:
    """Weighted Fisher-Rao (Karcher / Frechet) mean of several simplex points.

    The point minimising ``sum_k w_k * d(b, code_k)^2`` on the simplex -- the
    metric-consistent way to *average* place-cell codes (e.g. consolidating
    several recalls), unlike the arithmetic mean which ignores the curvature.

    Solved by Riemannian gradient descent: start from the closed-form sphere mean
    ``(normalise sum_k w_k sqrt(code_k))^2`` and iterate ``b <- exp_b(sum_k w_k
    log_b(code_k))`` until the update is below ``tol`` (or ``iters`` is reached).

    Parameters
    ----------
    codes : (K, n)
        ``K`` distributions on the ``n``-simplex.
    weights : (K,), optional
        Non-negative weights (default uniform); internally normalised to sum 1.

    Returns
    -------
    torch.Tensor (n,)
        The barycentric distribution. For a single code, or ``K`` identical
        codes, it returns that code; for two equally weighted codes it is the
        geodesic midpoint ``geodesic_interpolate(p, q, 0.5)``.
    """
    if codes.dim() != 2:
        raise ValueError(f"codes must be (K, n), got {tuple(codes.shape)}")
    K, n = codes.shape
    if weights is None:
        weights = codes.new_ones(K)
    w = (weights / weights.sum().clamp_min(eps)).reshape(K, 1)

    # closed-form sphere-mean initialisation
    roots = torch.sqrt(codes.clamp_min(0.0))              # (K, n)
    m = (w * roots).sum(dim=0)                            # (n,)
    m = m / torch.linalg.vector_norm(m).clamp_min(eps)
    b = simplex_project(m * m)                            # (n,)

    for _ in range(iters):
        # tangent-space mean of the logs, then step along it
        logs = log_map(b.expand(K, n), codes, eps=eps)    # (K, n) tangents at b
        g = (w * logs).sum(dim=0)                         # (n,) tangent at b
        step = torch.linalg.vector_norm(g)
        b = exp_map(b, g, eps=eps)
        if float(step) < tol:
            break
    return b


# --------------------------------------------------------------------------- #
# 4. flagship: a sampled geodesic path                                        #
# --------------------------------------------------------------------------- #


@dataclass
class Geodesic:
    """A sampled Fisher-Rao geodesic between two simplex points -- the flagship.

    Attributes
    ----------
    points : (S + 1, n)
        ``S + 1`` distributions sampled at ``t = 0, 1/S, ..., 1`` along the
        geodesic (``points[0] == p``, ``points[-1] == q``).
    length : (,)
        The Fisher-Rao distance ``d(p, q)`` (the total geodesic length).
    midpoint : (n,)
        ``points`` at ``t = 0.5`` -- the metric-consistent halfway distribution.
    """

    points: torch.Tensor
    length: torch.Tensor
    midpoint: torch.Tensor


def fisher_rao_geodesic(p: torch.Tensor, q: torch.Tensor, *, n_steps: int = 16) -> Geodesic:
    """Compose the geodesic operators into a sampled path -- the flagship.

    Samples the constant-speed Fisher-Rao geodesic between two place-cell codes
    ``p`` and ``q`` (both ``(n,)``) at ``n_steps + 1`` points, and reports its
    length and midpoint. This is the metric-consistent "morph" between two
    remembered locations -- contrast with the Euclidean segment ``(1-t) p + t q``,
    which is a straight chord that does not respect the information metric.
    """
    if p.dim() != 1 or q.dim() != 1:
        raise ValueError(f"p, q must be 1-D distributions, got {tuple(p.shape)}, {tuple(q.shape)}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    ts = torch.linspace(0.0, 1.0, n_steps + 1, dtype=p.dtype, device=p.device)
    pts = torch.stack([geodesic_interpolate(p, q, float(t)) for t in ts], dim=0)
    length = fisher_rao_distance(p, q)
    mid = geodesic_interpolate(p, q, 0.5)
    return Geodesic(points=pts, length=length, midpoint=mid)
