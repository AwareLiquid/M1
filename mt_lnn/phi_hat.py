"""
phi_hat.py — Information-integration proxy Φ̂ for MT-LNN.

Implements the kNN differential-entropy estimator of Kraskov, Stögbauer, &
Grassberger (2004) with the Chebyshev (L∞) metric, applied to the partition

    h ∈ ℝ^d  ↦  (s_1, …, s_K) with s_k ∈ ℝ^{d/K}

The integration proxy is

    Φ̂(h) = Σ_k Ĥ(s_k) - Ĥ(h)

— the difference between summed part-wise entropies and the whole-vector
entropy. Φ̂ > 0 indicates strong inter-part correlation (= integrated
information). Φ̂ ≈ 0 indicates the parts are independent. Φ̂ is the
quantity our anesthesia test targets: it should *collapse* as anesthesia
level rises.

References
----------
- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual
  information. Physical Review E, 69(6):066138.
- The exact estimator used here is Eq. (8) of that paper, single-variable
  variant (differential entropy), with the L∞ neighbour-distance rule for
  algorithmic simplicity.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F

# Avoid SciPy dependency for digamma — torch has it.
from torch.special import digamma


# ---------------------------------------------------------------------------
# Core kNN entropy estimator
# ---------------------------------------------------------------------------

@torch.no_grad()
def knn_entropy_chebyshev(X: torch.Tensor, k: int = 3) -> float:
    """
    Kraskov–Stögbauer–Grassberger entropy estimator with L∞ neighbour metric.

    Ĥ(X) = -ψ(k) + ψ(N) + d · ⟨log(2 · ε_i)⟩

    where ε_i is the distance from sample i to its k-th nearest neighbour
    under the L∞ metric.

    Parameters
    ----------
    X : (N, d) tensor of activations
    k : neighbour index (paper default 3)

    Returns
    -------
    float — estimated differential entropy in nats
    """
    X = X.detach().float()
    N, d = X.shape
    if N <= k + 1:
        return float("nan")

    # Pairwise L∞ distances. O(N²·d) memory & time; fine for N≤2048.
    diffs = (X.unsqueeze(0) - X.unsqueeze(1)).abs()                # (N, N, d)
    dist = diffs.max(dim=-1).values                                # (N, N)
    # Mask self-distance so kthvalue doesn't pick it.
    dist.fill_diagonal_(float("inf"))

    # k-th nearest neighbour distance per row
    eps, _ = torch.kthvalue(dist, k, dim=1)                        # (N,)

    eps = eps.clamp(min=1e-12)
    mean_log_2eps = (eps * 2.0).log().mean()

    psi_k = digamma(torch.tensor(float(k)))
    psi_N = digamma(torch.tensor(float(N)))
    H_hat = -psi_k + psi_N + d * mean_log_2eps
    return H_hat.item()


# ---------------------------------------------------------------------------
# Φ̂ for a single mini-batch of hidden states
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_phi_hat(hidden: torch.Tensor, K: int = 4, k_nn: int = 3) -> float:
    """
    Φ̂(hidden) = Σ_k Ĥ(s_k) - Ĥ(hidden)

    Parameters
    ----------
    hidden : (N, d) — N samples of d-dimensional activations
    K      : number of contiguous partitions (paper default 4)
    k_nn   : kNN index for entropy estimator (paper default 3)
    """
    assert hidden.dim() == 2, f"hidden must be (N, d), got {hidden.shape}"
    N, d = hidden.shape
    assert d % K == 0, f"d={d} not divisible by K={K}"
    part_size = d // K

    H_whole = knn_entropy_chebyshev(hidden, k=k_nn)
    H_parts = 0.0
    for j in range(K):
        s_j = hidden[:, j * part_size: (j + 1) * part_size]
        H_parts += knn_entropy_chebyshev(s_j, k=k_nn)

    return H_parts - H_whole


# ---------------------------------------------------------------------------
# Φ̂ from a model + token batch
# ---------------------------------------------------------------------------

@torch.no_grad()
def _hidden_states_from_model(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Run a forward pass and return final hidden states (B, T, d) — just before
    lm_head. Handles both gwtb_per_block=True (GWTB inside each block) and
    gwtb_per_block=False (single top-level GWTB).
    """
    x = model.embedding(input_ids)
    for block in model.blocks:
        x, _ = block(x)
    # Top-level GWTB is None when gwtb_per_block=True (each block ran its own).
    if model.gwtb is not None:
        x, _ = model.gwtb(x)
    x, _ = model.coherence(x)
    return model.final_norm(x)


@torch.no_grad()
def compute_phi_hat_from_model(
    model,
    input_ids: torch.Tensor,
    K: int = 4,
    k_nn: int = 3,
    n_batches: int = 10,
) -> float:
    """
    Compute Φ̂ over the model's final hidden states.

    Runs `n_batches` independent forward passes with fresh random token batches
    of the same shape as `input_ids`, collects all (B·T) activation samples,
    and returns a single averaged Φ̂.  Averaging reduces kNN-estimator variance
    by ~√n_batches — important for the pass/fail stability of the anesthesia test.

    Parameters
    ----------
    model      : MTLNNModel
    input_ids  : (B, T) seed batch — shape used for all n_batches passes
    K          : partition count for Φ̂ (default 4)
    k_nn       : kNN index for entropy estimator (default 3)
    n_batches  : number of independent random batches to average over (default 10)
    """
    model.eval()
    B, T = input_ids.shape
    device = input_ids.device

    phi_vals = []
    for i in range(n_batches):
        # First pass uses the provided ids; subsequent passes use random tokens
        # of the same shape so the estimate is not tied to one specific prompt.
        ids = input_ids if i == 0 else torch.randint(
            0, model.config.vocab_size, (B, T), device=device
        )
        x = _hidden_states_from_model(model, ids)       # (B, T, d)
        flat = x.reshape(-1, x.shape[-1])               # (B·T, d)
        phi = compute_phi_hat(flat, K=K, k_nn=k_nn)
        if not (phi != phi):                            # skip NaN (too few samples)
            phi_vals.append(phi)

    return float(sum(phi_vals) / len(phi_vals)) if phi_vals else float("nan")


# ---------------------------------------------------------------------------
# Φ̂ sweep over anesthesia levels — the AVP curve
# ---------------------------------------------------------------------------

@torch.no_grad()
def phi_hat_anesthesia_sweep(
    model,
    input_ids: torch.Tensor,
    kappas: List[float] = (1.0, 2.0, 5.0, 10.0),
    K: int = 4,
    k_nn: int = 3,
) -> Dict[float, float]:
    """
    Compute Φ̂ at each anesthesia level κ ∈ kappas. Returns dict
    {κ: Φ̂}. The 'anesthesia test' passes if Φ̂(κ_max) / Φ̂(κ=1) ≤ 0.3
    (i.e. at least 70% collapse), matching Casali et al. 2013.
    """
    from .anesthesia import anesthetize

    results: Dict[float, float] = {}
    for kappa in kappas:
        # Map κ ∈ [1, 10] → anesthesia_level ∈ [0, 1] using the inverse of
        # Casali et al.'s sigmoid dose-response curve. We use a simple linear
        # mapping for clarity: level = (κ - 1) / 9 — κ=1 → 0, κ=10 → 1.
        level = max(0.0, min(1.0, (kappa - 1.0) / 9.0))
        with anesthetize(model, level):
            phi = compute_phi_hat_from_model(model, input_ids, K=K, k_nn=k_nn)
        results[kappa] = phi
    return results


def anesthesia_test_result(sweep: Dict[float, float], delta: float = 0.7) -> dict:
    """
    Anesthesia test from the paper: Φ̂(κ_max) / Φ̂(κ=1) ≤ 1 - δ (default δ=0.7).

    Robust to negative Φ̂ baselines (the Kraskov estimator can be negatively
    biased for high d / small N, so absolute values are noisy but the
    *direction and magnitude of change with κ* remain meaningful).

    Returns:
      phi_clean, phi_full           — Φ̂ at min and max κ
      abs_change                    — phi_full - phi_clean (negative = collapse)
      signed_relative_change        — abs_change / max(|phi_clean|, 1e-6)
      collapse_pct                  — max(0, -signed_relative_change) × 100;
                                       ≥0% even when Φ̂ rises (only counts collapse)
      monotone_decrease             — True iff Φ̂ is non-increasing across κ
      passed                        — collapse_pct ≥ δ·100 AND monotone_decrease
    """
    kappa_min = min(sweep)
    kappa_max = max(sweep)
    phi_clean = sweep[kappa_min]
    phi_full = sweep[kappa_max]
    abs_change = phi_full - phi_clean
    signed_rel = abs_change / max(abs(phi_clean), 1e-6)
    # Collapse percentage: only counts decrease in Φ̂, not increase.
    # If Φ̂ rises under anesthesia (unexpected — model integration *grows*
    # when damped) collapse_pct stays 0%.
    collapse_pct = max(0.0, -signed_rel) * 100.0

    sorted_kappas = sorted(sweep.keys())
    monotone_decrease = all(
        sweep[sorted_kappas[i]] >= sweep[sorted_kappas[i + 1]]
        for i in range(len(sorted_kappas) - 1)
    )

    passed = (collapse_pct >= delta * 100.0) and monotone_decrease
    return {
        "phi_clean": phi_clean,
        "phi_full": phi_full,
        "abs_change": abs_change,
        "signed_relative_change": signed_rel,
        "collapse_pct": collapse_pct,
        "monotone_decrease": monotone_decrease,
        "delta_threshold": delta,
        "passed": passed,
    }
