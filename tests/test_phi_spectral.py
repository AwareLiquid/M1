"""
test_phi_spectral.py — Tests for Gaussian Total Correlation (Φ_G) and
spectral integration metrics.

Run:  python -m pytest tests/test_phi_spectral.py -v
"""

import sys
import math
import warnings

import torch
import numpy as np

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import (
    MTLNNConfig, MTLNNModel,
    gaussian_total_correlation,
    effective_rank,
    integration_ratio,
    compute_phi_spectral_from_model,
    phi_spectral_anesthesia_sweep,
    anesthesia_test_result_spectral,
    compare_phi_metrics,
)
from mt_lnn.phi_spectral import _covariance, _correlation, _log_det


def small_cfg():
    return MTLNNConfig(
        vocab_size=200, max_seq_len=32, d_model=128,
        n_layers=2, n_heads=4, n_kv_heads=2, d_head=32,
        dropout=0.0, attention_dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Unit: matrix utilities
# ---------------------------------------------------------------------------

def test_log_det_identity():
    """log|I_d| = 0."""
    I = torch.eye(8)
    assert abs(_log_det(I)) < 1e-5


def test_log_det_diagonal():
    """log|diag(σ²)| = Σ log(σ²)."""
    diag_vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
    C = torch.diag(diag_vals)
    expected = diag_vals.log().sum().item()
    assert abs(_log_det(C) - expected) < 1e-4


def test_covariance_shape():
    X = torch.randn(50, 16)
    C = _covariance(X)
    assert C.shape == (16, 16)


def test_correlation_diagonal_ones():
    X = torch.randn(100, 8)
    R = _correlation(X)
    diag = R.diag()
    assert torch.allclose(diag, torch.ones(8), atol=1e-5)


# ---------------------------------------------------------------------------
# Unit: Φ_G mathematical properties
# ---------------------------------------------------------------------------

def test_phi_g_independent_parts_near_zero():
    """When parts are independent, Φ_G ≈ 0."""
    N, d, K = 200, 16, 4
    # Generate truly independent parts
    parts = [torch.randn(N, d // K) for _ in range(K)]
    X = torch.cat(parts, dim=1)
    phi = gaussian_total_correlation(X, K=K)
    # Should be close to 0 (within estimation noise); allow generous tolerance
    assert phi < 1.0, f"Independent Φ_G should be near 0, got {phi:.4f}"


def test_phi_g_correlated_higher_than_independent():
    """Highly correlated data should have larger Φ_G than independent data."""
    N, d = 300, 16
    K = 4

    # Independent
    X_ind = torch.randn(N, d)
    phi_ind = gaussian_total_correlation(X_ind, K=K)

    # Correlated: all dims driven by a shared factor
    shared = torch.randn(N, 1).expand(N, d)
    noise = torch.randn(N, d) * 0.1
    X_corr = shared + noise
    phi_corr = gaussian_total_correlation(X_corr, K=K)

    assert phi_corr > phi_ind, (
        f"Correlated Φ_G ({phi_corr:.4f}) should exceed independent ({phi_ind:.4f})"
    )


def test_phi_g_non_negative():
    """Φ_G = (1/2)(Σ log|Σ_k| - log|Σ|) ≥ 0 by the chain rule inequality."""
    for _ in range(5):
        X = torch.randn(100, 16)
        phi = gaussian_total_correlation(X, K=4)
        assert phi >= -1e-6, f"Φ_G should be non-negative, got {phi:.6f}"


def test_phi_g_invariant_to_shift():
    """Φ_G should be translation-invariant (correlation matrix mode)."""
    X = torch.randn(100, 16)
    phi_base = gaussian_total_correlation(X, K=4, use_correlation=True)
    phi_shifted = gaussian_total_correlation(X + 100.0, K=4, use_correlation=True)
    assert abs(phi_base - phi_shifted) < 1e-4


# ---------------------------------------------------------------------------
# Unit: effective_rank
# ---------------------------------------------------------------------------

def test_effective_rank_rank1():
    """Rank-1 matrix → PR ≈ 1."""
    v = torch.randn(100, 1).expand(100, 16)
    pr = effective_rank(v + torch.randn(100, 16) * 0.01)
    assert pr < 3.0, f"Near-rank-1 PR should be low, got {pr:.2f}"


def test_effective_rank_full():
    """Isotropic Gaussian → PR ≈ d."""
    d = 16
    X = torch.randn(500, d)
    pr = effective_rank(X)
    # PR of isotropic should be near d (within 30%)
    assert pr > d * 0.5, f"Isotropic PR should be near {d}, got {pr:.2f}"


# ---------------------------------------------------------------------------
# Integration: model-level Φ_G
# ---------------------------------------------------------------------------

def test_compute_phi_spectral_from_model_returns_float():
    cfg = small_cfg()
    model = MTLNNModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    phi = compute_phi_spectral_from_model(model, ids, K=4, n_batches=2)
    assert isinstance(phi, float)
    assert not math.isnan(phi)


def test_compute_phi_spectral_include_secondary():
    cfg = small_cfg()
    model = MTLNNModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    result = compute_phi_spectral_from_model(model, ids, K=4, n_batches=2,
                                              include_secondary=True)
    assert set(result.keys()) == {"phi_g", "effective_rank", "integration_ratio"}
    assert result["effective_rank"] > 0


def test_compare_phi_metrics():
    cfg = small_cfg()
    model = MTLNNModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    result = compare_phi_metrics(model, ids, K=4, n_batches=2)
    assert "phi_hat" in result
    assert "phi_g" in result
    assert "effective_rank" in result


# ---------------------------------------------------------------------------
# Integration: anesthesia sweep
# ---------------------------------------------------------------------------

def test_anesthesia_sweep_returns_dict():
    cfg = small_cfg()
    model = MTLNNModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    sweep = phi_spectral_anesthesia_sweep(
        model, ids, kappas=[1.0, 5.0], K=4, n_batches=2
    )
    assert set(sweep.keys()) == {1.0, 5.0}
    for v in sweep.values():
        assert isinstance(v, float)
        assert not math.isnan(v)


def test_anesthesia_test_result_spectral_structure():
    sweep = {1.0: 2.0, 5.0: 0.4}
    result = anesthesia_test_result_spectral(sweep)
    required = {
        "phi_g_clean", "phi_g_full", "abs_change",
        "signed_relative_change", "collapse_pct",
        "monotone_decrease", "delta_threshold", "passed",
    }
    assert required.issubset(result.keys())


def test_anesthesia_test_result_spectral_passing():
    """A 90% Φ_G collapse (1.0 → 0.1) should pass at δ=0.7."""
    sweep = {1.0: 1.0, 5.0: 0.1}
    result = anesthesia_test_result_spectral(sweep, delta=0.7)
    assert result["collapse_pct"] > 70.0
    assert result["passed"] is True


def test_anesthesia_test_result_spectral_failing():
    """A 10% Φ_G collapse should fail at δ=0.7."""
    sweep = {1.0: 1.0, 5.0: 0.9}
    result = anesthesia_test_result_spectral(sweep, delta=0.7)
    assert result["passed"] is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
