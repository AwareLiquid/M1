"""
tests/test_pscan_chunkwise.py — chunkwise scan equivalence tests.

The chunkwise path is ONLY allowed to differ from pscan()/pscan_sequential()
by float reassociation (different summation order, same math). These tests
pin that equivalence in fp32 and are the MERGE GATE for the
use_chunkwise_scan switch: if any of them fail in Phase B, the switch must
not ship enabled.

Tolerances are hardcoded constants (see CHUNKWISE_RTOL / CHUNKWISE_ATOL):
they are loose enough to absorb C-step reassociation + log-space segsum
rounding (~1e-5 rel per matrix entry) and tight enough that any structural
bug (off-by-one chunk boundary, wrong carry, bad mask) — which produces
O(1) errors — fails loudly.

Designed for the CPU <2min budget: all shapes are tiny, total runtime is
well under 10s.
"""

import sys

import torch
from torch.testing import assert_close

sys.path.insert(0, ".")

from mt_lnn.parallel_scan import (
    pscan,
    pscan_chunkwise,
    pscan_chunkwise_constant_A,
    pscan_constant_A,
    pscan_sequential,
)

# Hardcoded fp32 equivalence tolerances (merge-gate constants — do not tune
# per-test; a failure here means the chunkwise math diverged, not noise).
CHUNKWISE_RTOL = 1e-3
CHUNKWISE_ATOL = 1e-3
# Tighter pair for the single-short-chunk regime (C <= 8), where
# reassociation error is ~1e-5 and the gate can afford 10x more headroom.
SHORT_RTOL = 1e-4
SHORT_ATOL = 1e-4


def _rand_case(*batch_T, d=6, lo=0.05, hi=0.95, seed=0, signed=False):
    """Deterministic (A, X) with multipliers in (+-lo, +-hi), |A| < 1."""
    torch.manual_seed(seed)
    T = batch_T[-1]
    A = torch.rand(*batch_T) * (hi - lo) + lo
    if signed:
        A = A * (torch.randint(0, 2, batch_T) * 2.0 - 1.0)   # random sign
    X = torch.randn(*batch_T, d)
    return A, X


# ---------------------------------------------------------------------------
# Test 1: numerical equivalence vs pscan_sequential — multi-shape, non-pow2
# ---------------------------------------------------------------------------

def test_chunkwise_matches_sequential_multishape():
    shapes = [
        (2, 64, 8),            # exact chunk multiple (C=64)
        (2, 100, 5),           # partial trailing chunk (100 = 64 + 36)
        (2, 7, 3),             # T < chunk_size (single short chunk)
        (1, 129, 4),           # C+1 boundary
        (2, 3, 4, 37, 5),      # multi-dim batch, non-pow2 T
    ]
    for i, shape in enumerate(shapes):
        A, X = _rand_case(*shape, seed=100 + i)
        H_seq = pscan_sequential(A, X)
        H_chk = pscan_chunkwise(A, X, chunk_size=64)
        assert_close(H_chk, H_seq, rtol=CHUNKWISE_RTOL, atol=CHUNKWISE_ATOL)
    print("[ok] test_chunkwise_matches_sequential_multishape")


# ---------------------------------------------------------------------------
# Test 2: tight single-chunk parity + T sweep across chunk boundaries vs pscan
# ---------------------------------------------------------------------------

def test_chunkwise_chunk_boundary_sweep():
    # Single short chunk, zero carry: rounding-dominated regime -> tight gate
    A, X = _rand_case(2, 8, 4, seed=7)
    assert_close(
        pscan_chunkwise(A, X, chunk_size=8), pscan_sequential(A, X),
        rtol=SHORT_RTOL, atol=SHORT_ATOL,
    )
    # Boundary-prone lengths vs the production pscan: T = C, C-1, C+1, ...
    for T in [1, 63, 64, 65, 127, 128, 129, 257]:
        A, X = _rand_case(2, T, 5, seed=200 + T)
        assert_close(
            pscan_chunkwise(A, X, chunk_size=64), pscan(A, X),
            rtol=CHUNKWISE_RTOL, atol=CHUNKWISE_ATOL,
        )
    print("[ok] test_chunkwise_chunk_boundary_sweep")


# ---------------------------------------------------------------------------
# Test 3: constant-A specialisation — chunkwise vs the production constant-A
# path (the repo's default decay regime, constant along T)
# ---------------------------------------------------------------------------

def test_chunkwise_constant_A_specialisation():
    torch.manual_seed(300)
    B, P, S, T, D = 2, 3, 2, 100, 6
    decay = torch.rand(B, P, S) * 0.9 + 0.05
    X = torch.randn(B, P, S, T, D)
    H_ref = pscan_constant_A(decay, X)                       # production path
    H_chk = pscan_chunkwise_constant_A(decay, X, chunk_size=64)
    assert_close(H_chk, H_ref, rtol=CHUNKWISE_RTOL, atol=CHUNKWISE_ATOL)
    # Also against the general chunkwise entry (broadcast must agree)
    A_full = decay.unsqueeze(-1).expand(B, P, S, T)
    assert_close(
        H_chk, pscan_chunkwise(A_full, X, chunk_size=64),
        rtol=CHUNKWISE_RTOL, atol=CHUNKWISE_ATOL,
    )
    print("[ok] test_chunkwise_constant_A_specialisation")


# ---------------------------------------------------------------------------
# Test 4: carry correctness — non-zero h_init, and chunk-boundary carry
# handoff (two-step continuation == one-shot full-sequence run)
# ---------------------------------------------------------------------------

def test_chunkwise_carry_correctness():
    B, T, D = 2, 100, 5
    A, X = _rand_case(B, T, D, seed=400)
    h_init = torch.randn(B, D)
    assert_close(
        pscan_chunkwise(A, X, h_init=h_init, chunk_size=32),
        pscan_sequential(A, X, h_init=h_init),
        rtol=CHUNKWISE_RTOL, atol=CHUNKWISE_ATOL,
    )
    # Carry handoff identity: running [0:50] then feeding its last state as
    # h_init into [50:100] must equal the single fused run over all 100.
    A1, X1, A2, X2 = A[..., :50], X[..., :50, :], A[..., 50:], X[..., 50:, :]
    H_full = pscan_chunkwise(A, X, chunk_size=32)
    H1 = pscan_chunkwise(A1, X1, chunk_size=32)
    H2 = pscan_chunkwise(A2, X2, h_init=H1[..., -1, :], chunk_size=32)
    assert_close(H1, H_full[..., :50, :], rtol=SHORT_RTOL, atol=SHORT_ATOL)
    assert_close(H2, H_full[..., 50:, :], rtol=SHORT_RTOL, atol=SHORT_ATOL)
    print("[ok] test_chunkwise_carry_correctness")


# ---------------------------------------------------------------------------
# Test 5: signed multipliers (negative A) + gradient flow parity — the
# signed_decay / selective_decay regimes depend on negative eigenvalues
# ---------------------------------------------------------------------------

def test_chunkwise_signed_and_gradients():
    A, X = _rand_case(2, 96, 5, seed=500, signed=True)       # |A| in (.05,.95), sign mixed
    assert_close(
        pscan_chunkwise(A, X, chunk_size=32), pscan_sequential(A, X),
        rtol=CHUNKWISE_RTOL, atol=CHUNKWISE_ATOL,
    )
    # Gradients flow and match the sequential reference (loose reassociation
    # bounds — backward sums reassociate just like the forward pass)
    A_g = A.clone().requires_grad_()
    X_g = X.clone().requires_grad_()
    pscan_chunkwise(A_g, X_g, chunk_size=32).pow(2).mean().backward()
    A_s = A.clone().requires_grad_()
    X_s = X.clone().requires_grad_()
    pscan_sequential(A_s, X_s).pow(2).mean().backward()
    assert torch.isfinite(A_g.grad).all() and A_g.grad.abs().sum() > 0
    assert torch.isfinite(X_g.grad).all() and X_g.grad.abs().sum() > 0
    assert_close(A_g.grad, A_s.grad, rtol=5e-2, atol=5e-3)
    assert_close(X_g.grad, X_s.grad, rtol=5e-2, atol=5e-3)
    print("[ok] test_chunkwise_signed_and_gradients")


# ---------------------------------------------------------------------------
# Test 6: near-zero decay inside a chunk — regression for the segsum
# overflow (exp of a POSITIVE upper-triangle segsum used to reach fp32 inf,
# then inf * 0 -> NaN). Two |A| ~ 1e-30 steps span ~138 nats > ln(fp32 max).
# ---------------------------------------------------------------------------

def test_chunkwise_near_zero_decay_no_nan():
    T, D = 5, 3
    A = torch.tensor([[0.5, 1e-30, 1e-30, 0.5, 0.9]])
    torch.manual_seed(600)
    X = torch.randn(1, T, D)
    H = pscan_chunkwise(A, X, chunk_size=4)
    assert torch.isfinite(H).all(), "near-zero decay produced NaN/Inf"
    assert_close(H, pscan_sequential(A, X),
                 rtol=CHUNKWISE_RTOL, atol=CHUNKWISE_ATOL)
    print("[ok] test_chunkwise_near_zero_decay_no_nan")


def run_all():
    print("=" * 60)
    print("Chunkwise scan equivalence suite")
    print("=" * 60)
    test_chunkwise_matches_sequential_multishape()
    test_chunkwise_chunk_boundary_sweep()
    test_chunkwise_constant_A_specialisation()
    test_chunkwise_carry_correctness()
    test_chunkwise_signed_and_gradients()
    test_chunkwise_near_zero_decay_no_nan()
    print("=" * 60)
    print("ALL CHUNKWISE SCAN TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
