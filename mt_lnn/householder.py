"""Householder NDIT primitives — non-diagonal input-dependent transitions.

Design doc: docs/NONDIAGONAL_TRANSITION.md. A5 (NC1-complete) needs a
NON-DIAGONAL input-dependent state transition (Merrill ICML 2024 Cor 4.7);
Householder reflections are the only parameterisation with spectral radius
exactly 1 (unitary), so the state cannot explode.

This module provides the pure-linear-algebra primitives (M1 milestone):
  - householder_matrix(v):     Q = I - 2 v v^T   (v unit-norm)
  - householder_apply(v, x):   Q x in O(D^2), no materialised matrix
  - wy_representation(vs):     WY form of a Householder product Q_1...Q_k
  - wy_apply(U, Sigma, V, x):  apply the WY form in O(k D^2)

All functions are pure and parameter-free; the trainable NDIT layer wraps
these with an input-dependent v_t = normalize(W_v x_t + b_v).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def householder_matrix(v: torch.Tensor) -> torch.Tensor:
    """Householder reflection matrix Q = I - 2 v v^T.

    Args:
        v: (..., D) — does NOT need to be unit-norm (normalised internally).
    Returns:
        Q: (..., D, D)
    """
    v = F.normalize(v, dim=-1)                       # (..., D)
    return torch.eye(v.shape[-1], device=v.device, dtype=v.dtype) \
        - 2.0 * torch.einsum("...i,...j->...ij", v, v)


def householder_apply(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply Q = I - 2vv^T to x without materialising the matrix: O(D^2).

    Args:
        v: (..., D)
        x: (..., D)
    Returns:
        Q x: (..., D)
    """
    v = F.normalize(v, dim=-1)
    return x - 2.0 * torch.einsum("...i,...i->...", v, x).unsqueeze(-1) * v


def wy_representation(vs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """WY representation of a Householder product P = Q_1 Q_2 ... Q_k.

    Golub & Van Loan: P = I + U S V^T where
      V[:, j]  = v_j (the raw reflector directions),
      U[:, j]  = -2 (Q_1...Q_{j-1}) v_j,
      S        = upper-triangular coupling (identity here for Householder).
    For Householder reflectors the coupling is simply: S = I (upper-triangular
    unit), and U is built by applying the PREVIOUS reflectors to -2 v_j.
    The returned form applies as  x -> x + U (V^T x)  (S=I folded into U).

    Args:
        vs: (k, D) — k reflector direction vectors.
    Returns:
        U: (D, k), V: (D, k) such that P x = x + U (V^T x).
    """
    k, D = vs.shape
    vs = F.normalize(vs, dim=-1)
    V = vs.T                                          # (D, k)
    # U[:, j] = -2 (Q_1 ... Q_{j-1}) v_j — build by sequential application
    U = torch.empty(D, k, device=vs.device, dtype=vs.dtype)
    for j in range(k):
        u_j = -2.0 * vs[j]
        for i in range(j - 1, -1, -1):
            u_j = householder_apply(vs[i], u_j)
        U[:, j] = u_j
    return U, V


def wy_apply(U: torch.Tensor, V: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply the WY form P x = x + U (V^T x). O(k D^2).

    Args:
        U, V: (D, k) from wy_representation
        x:    (..., D)
    Returns:
        P x:  (..., D)
    """
    proj = torch.einsum("dj,...d->...j", V, x)         # (..., k)
    return x + torch.einsum("dj,...j->...d", U, proj)


def unitary_check(Q: torch.Tensor) -> float:
    """Max |Q Q^T - I| — a unitary-ness diagnostic (0 = perfect)."""
    with torch.no_grad():
        Id = torch.eye(Q.shape[-1], device=Q.device, dtype=Q.dtype)
        return float((Q @ Q.transpose(-1, -2) - Id).abs().max())
