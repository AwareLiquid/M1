"""
phi_spectral.py — Spectral / Gaussian integration metrics for MT-LNN.

Provides a drop-in replacement for the KSG kNN estimator in ``phi_hat.py``
based on the **Gaussian Total Correlation (Φ_G)**.

Theory
------
For a multivariate Gaussian X ~ 𝒩(0, Σ), the **total correlation** (also
called multi-information or redundancy) between K non-overlapping parts
s_1, …, s_K is:

    TC(X) = Σ_k H(s_k) − H(X)
           = (1/2) [Σ_k log det(Σ_k) − log det(Σ)]

where Σ_k is the diagonal block of Σ corresponding to partition k, and
det is evaluated via the sum of log-eigenvalues for numerical stability.

This is the *same conceptual quantity* as Φ̂ in ``phi_hat.py``, but:

1. **Exact under the Gaussian assumption** (Φ̂ is a biased kNN estimate).
2. **O(d³)** via SVD — no O(N²·d) pairwise-distance matrix.
3. **Sample-efficient**: works well even when N < d (using regularisation).
4. **Scale-invariant** when using the *correlation matrix* variant (default).

Secondary metrics
-----------------
``effective_rank``
    Participation ratio PR = (Σ λ_i)² / (Σ λ_i²) — how many dimensions
    are meaningfully used by the hidden state.

``integration_ratio``
    Φ_G / H_whole — fraction of total entropy attributable to cross-part
    correlations; interpretable as a "degree of integration" in [0, 1].

Anesthesia compatibility
------------------------
``phi_spectral_anesthesia_sweep`` and ``anesthesia_test_result_spectral``
mirror the interface of ``phi_hat.py`` so either metric can be swapped in.

References
----------
- Studeny, M. & Vejnarova, J. (1998). The multiinformation function as a
  tool for measuring stochastic dependence. Learning in Graphical Models.
- Tononi, G. (2004). An information integration theory of consciousness.
  BMC Neuroscience, 5:42.
- Roy, O. & Vetterli, M. (2007). The effective rank: A measure of effective
  dimensionality. EUSIPCO.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log_det(C: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Numerically stable log-determinant of a symmetric PSD matrix.

    Uses eigendecomposition so that near-singular matrices get a clean
    log-det from clamped positive eigenvalues instead of a NaN from a
    Cholesky failure.
    """
    # torch.linalg.eigvalsh returns eigenvalues in ascending order for
    # real symmetric matrices — guaranteed non-negative for PSD.
    eigs = torch.linalg.eigvalsh(C)            # (d,)
    return eigs.clamp(min=eps).log().sum().item()


def _covariance(X: torch.Tensor, regularise: bool = True) -> torch.Tensor:
    """
    Unbiased sample covariance matrix of X ∈ (N, d).

    Optionally adds a Tikhonov regulariser scaled to the mean eigenvalue
    (Ledoit-Wolf shrinkage, simplified):  Σ_reg = Σ + ρ·I  where
    ρ = trace(Σ) / d × 0.01.  Ensures positive-definiteness even when
    N < d.
    """
    X = X.detach().float()
    N, d = X.shape
    X = X - X.mean(dim=0, keepdim=True)
    C = (X.T @ X) / max(N - 1, 1)                 # (d, d)
    if regularise:
        rho = C.trace() / d * 0.01
        C = C + rho * torch.eye(d, device=C.device, dtype=C.dtype)
    return C


def _correlation(X: torch.Tensor) -> torch.Tensor:
    """Normalise covariance to a correlation matrix (diagonal = 1)."""
    C = _covariance(X, regularise=True)
    std = C.diag().sqrt().clamp(min=1e-8)
    C_corr = C / (std.unsqueeze(0) * std.unsqueeze(1))
    return C_corr


# ---------------------------------------------------------------------------
# Core: Gaussian Total Correlation  Φ_G
# ---------------------------------------------------------------------------

@torch.no_grad()
def gaussian_total_correlation(
    hidden: torch.Tensor,
    K: int = 4,
    use_correlation: bool = True,
) -> float:
    """
    Φ_G(hidden) = (1/2) [ Σ_k log det(Σ_k) − log det(Σ) ]

    Parameters
    ----------
    hidden          : (N, d) — N samples of d-dimensional activations.
    K               : number of contiguous equal-size partitions.
    use_correlation : if True, use the correlation matrix (normalises out
                      per-dimension variance scales). Recommended.

    Returns
    -------
    float — Gaussian total correlation in nats (≥ 0 for PSD matrices).
    """
    assert hidden.dim() == 2, f"hidden must be (N, d), got {hidden.shape}"
    N, d = hidden.shape
    assert d % K == 0, f"d={d} is not divisible by K={K}"
    part_size = d // K

    cov_fn = _correlation if use_correlation else _covariance

    C_whole = cov_fn(hidden)
    log_det_whole = _log_det(C_whole)

    log_det_parts = 0.0
    for j in range(K):
        s_j = hidden[:, j * part_size: (j + 1) * part_size]
        C_j = cov_fn(s_j)
        log_det_parts += _log_det(C_j)

    # Φ_G = (1/2)(Σ log|Σ_k| − log|Σ|)
    phi_g = 0.5 * (log_det_parts - log_det_whole)
    return phi_g


# ---------------------------------------------------------------------------
# Secondary metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def effective_rank(hidden: torch.Tensor) -> float:
    """
    Participation Ratio  PR = (Σ λ_i)² / (Σ λ_i²).

    PR ≈ 1  → representation collapsed to a line.
    PR ≈ d  → representation uniform across all dimensions.

    A rising PR under normal operation and falling PR under anesthesia
    mirrors the information-integration story: integrated representations
    require many correlated dimensions.
    """
    X = hidden.detach().float()
    X = X - X.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(X, full_matrices=False)
    eigs = (s ** 2).clamp(min=0.0)
    return (eigs.sum() ** 2 / (eigs ** 2).sum()).item()


@torch.no_grad()
def integration_ratio(
    hidden: torch.Tensor,
    K: int = 4,
    use_correlation: bool = True,
    eps: float = 1e-6,
) -> float:
    """
    Φ_G / H_whole — fraction of total differential entropy attributable
    to cross-part correlations.

    Bounded in [0, 1] for well-behaved covariance matrices; values close
    to 1 indicate near-total integration (all entropy is shared), values
    close to 0 indicate independent parts.
    """
    assert hidden.dim() == 2
    N, d = hidden.shape
    assert d % K == 0

    cov_fn = _correlation if use_correlation else _covariance
    C_whole = cov_fn(hidden)
    H_whole = 0.5 * _log_det(C_whole)

    part_size = d // K
    H_parts = 0.0
    for j in range(K):
        s_j = hidden[:, j * part_size: (j + 1) * part_size]
        H_parts += 0.5 * _log_det(cov_fn(s_j))

    phi_g = 0.5 * (H_parts - H_whole)
    return phi_g / (abs(H_whole) + eps)


# ---------------------------------------------------------------------------
# Model-level helpers (mirror phi_hat.py interface)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _hidden_states_from_model(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Run forward and return final hidden states (B, T, d)."""
    x = model.embedding(input_ids)
    for block in model.blocks:
        x, _ = block(x)
    if model.gwtb is not None:
        x, _ = model.gwtb(x)
    x, _ = model.coherence(x)
    return model.final_norm(x)


@torch.no_grad()
def compute_phi_spectral_from_model(
    model,
    input_ids: torch.Tensor,
    K: int = 4,
    n_batches: int = 10,
    use_correlation: bool = True,
    include_secondary: bool = False,
) -> Union[float, Dict[str, float]]:
    """
    Compute Φ_G over the model's final hidden states.

    Accumulates hidden states across ``n_batches`` independent random
    forward passes so that the covariance estimate is based on B·T·n_batches
    samples — dramatically more stable than the kNN estimator in the
    high-d / low-N regime.

    Parameters
    ----------
    model            : MTLNNModel
    input_ids        : (B, T) seed batch (shape used for all passes).
    K                : number of contiguous partitions.
    n_batches        : random forward passes to pool (default 10).
    use_correlation  : normalise covariance → correlation matrix.
    include_secondary: if True, also return effective_rank and
                       integration_ratio in a dict.

    Returns
    -------
    float if include_secondary=False, else dict with keys:
        "phi_g", "effective_rank", "integration_ratio"
    """
    model.eval()
    B, T = input_ids.shape
    device = input_ids.device
    d = model.config.d_model

    # Ensure d is divisible by K; shrink K if needed
    while K > 1 and d % K != 0:
        K -= 1

    all_hidden: List[torch.Tensor] = []
    for i in range(n_batches):
        ids = input_ids if i == 0 else torch.randint(
            0, model.config.vocab_size, (B, T), device=device
        )
        x = _hidden_states_from_model(model, ids)   # (B, T, d)
        all_hidden.append(x.reshape(-1, d).cpu())

    flat = torch.cat(all_hidden, dim=0)             # (N_total, d)

    phi_g = gaussian_total_correlation(flat, K=K, use_correlation=use_correlation)

    if not include_secondary:
        return phi_g

    return {
        "phi_g": phi_g,
        "effective_rank": effective_rank(flat),
        "integration_ratio": integration_ratio(flat, K=K, use_correlation=use_correlation),
    }


# ---------------------------------------------------------------------------
# Anesthesia sweep
# ---------------------------------------------------------------------------

@torch.no_grad()
def phi_spectral_anesthesia_sweep(
    model,
    input_ids: torch.Tensor,
    kappas: List[float] = (1.0, 2.0, 5.0, 10.0),
    K: int = 4,
    n_batches: int = 10,
    use_correlation: bool = True,
) -> Dict[float, float]:
    """
    Compute Φ_G at each anesthesia level κ ∈ kappas.

    Returns ``{κ: Φ_G}``. Semantics identical to ``phi_hat_anesthesia_sweep``
    so ``anesthesia_test_result`` from ``phi_hat.py`` can be reused directly.
    """
    from .anesthesia import anesthetize

    results: Dict[float, float] = {}
    for kappa in kappas:
        level = max(0.0, min(1.0, (kappa - 1.0) / 9.0))
        with anesthetize(model, level):
            phi_g = compute_phi_spectral_from_model(
                model, input_ids, K=K, n_batches=n_batches,
                use_correlation=use_correlation,
            )
        results[kappa] = phi_g
    return results


def anesthesia_test_result_spectral(
    sweep: Dict[float, float],
    delta: float = 0.7,
) -> dict:
    """
    Same logic as ``phi_hat.anesthesia_test_result`` — collapse ≥ δ·100%.

    Accepts the Φ_G sweep dict produced by ``phi_spectral_anesthesia_sweep``.
    Since Φ_G ≥ 0 (Gaussian TC is non-negative), ``collapse_pct`` is more
    interpretable here than with the potentially-negative KSG estimator.
    """
    kappa_min = min(sweep)
    kappa_max = max(sweep)
    phi_clean = sweep[kappa_min]
    phi_full  = sweep[kappa_max]

    abs_change   = phi_full - phi_clean
    signed_rel   = abs_change / max(abs(phi_clean), 1e-6)
    collapse_pct = max(0.0, -signed_rel) * 100.0

    sorted_kappas = sorted(sweep.keys())
    monotone_decrease = all(
        sweep[sorted_kappas[i]] >= sweep[sorted_kappas[i + 1]]
        for i in range(len(sorted_kappas) - 1)
    )
    passed = (collapse_pct >= delta * 100.0) and monotone_decrease

    return {
        "phi_g_clean": phi_clean,
        "phi_g_full":  phi_full,
        "abs_change":  abs_change,
        "signed_relative_change": signed_rel,
        "collapse_pct":  collapse_pct,
        "monotone_decrease": monotone_decrease,
        "delta_threshold": delta,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def compare_phi_metrics(
    model,
    input_ids: torch.Tensor,
    K: int = 4,
    k_nn: int = 3,
    n_batches: int = 10,
) -> Dict[str, float]:
    """
    Compute both Φ̂ (KSG) and Φ_G (Gaussian TC) for direct comparison.

    Returns a dict with keys: "phi_hat", "phi_g", "effective_rank",
    "integration_ratio".
    """
    from .phi_hat import compute_phi_hat_from_model

    phi_hat = compute_phi_hat_from_model(model, input_ids, K=K, k_nn=k_nn,
                                         n_batches=n_batches)
    spectral = compute_phi_spectral_from_model(model, input_ids, K=K,
                                               n_batches=n_batches,
                                               include_secondary=True)
    return {"phi_hat": phi_hat, **spectral}
