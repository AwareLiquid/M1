"""
tests/test_parallel_scan.py — pscan correctness tests.

The parallel scan must match the sequential reference implementation to
machine precision (1e-6 fp32). This is the load-bearing piece for the
"liquid recurrence is real" claim — if the scan is buggy, the whole
temporal-state advantage collapses to noise.
"""

import sys
import torch

sys.path.insert(0, ".")

from mt_lnn.parallel_scan import pscan, pscan_sequential, pscan_constant_A


def _max_abs_diff(a, b):
    return (a - b).abs().max().item()


# ---------------------------------------------------------------------------
# Test 1: power-of-2 T, single batch
# ---------------------------------------------------------------------------

def test_pscan_pow2():
    torch.manual_seed(0)
    B, T, D = 2, 16, 8
    A = torch.rand(B, T) * 0.9 + 0.05         # in (0.05, 0.95)
    X = torch.randn(B, T, D)
    H_seq = pscan_sequential(A, X)
    H_par = pscan(A, X)
    diff = _max_abs_diff(H_seq, H_par)
    print(f"  T=16 (pow2)  diff = {diff:.2e}")
    assert diff < 1e-5, f"pscan != sequential: {diff}"
    print("[ok] test_pscan_pow2")


# ---------------------------------------------------------------------------
# Test 2: arbitrary T (not power of 2)
# ---------------------------------------------------------------------------

def test_pscan_arbitrary_T():
    torch.manual_seed(1)
    for T in [1, 7, 13, 37, 100, 256, 257]:
        B, D = 3, 5
        A = torch.rand(B, T) * 0.9 + 0.05
        X = torch.randn(B, T, D)
        H_seq = pscan_sequential(A, X)
        H_par = pscan(A, X)
        diff = _max_abs_diff(H_seq, H_par)
        assert diff < 1e-5, f"T={T}: pscan diff = {diff}"
    print(f"  tested T ∈ {{1, 7, 13, 37, 100, 256, 257}}, all match")
    print("[ok] test_pscan_arbitrary_T")


# ---------------------------------------------------------------------------
# Test 3: multi-dim batching (e.g. our (B, P, S, T, D) layout)
# ---------------------------------------------------------------------------

def test_pscan_multidim():
    torch.manual_seed(2)
    B, P, S, T, D = 2, 3, 4, 13, 8
    A = torch.rand(B, P, S, T) * 0.9 + 0.05
    X = torch.randn(B, P, S, T, D)
    H_seq = pscan_sequential(A, X)
    H_par = pscan(A, X)
    diff = _max_abs_diff(H_seq, H_par)
    print(f"  shape (B={B}, P={P}, S={S}, T={T}, D={D})  diff = {diff:.2e}")
    assert diff < 1e-5
    print("[ok] test_pscan_multidim")


# ---------------------------------------------------------------------------
# Test 4: non-zero initial state h_init
# ---------------------------------------------------------------------------

def test_pscan_h_init():
    torch.manual_seed(3)
    B, T, D = 2, 23, 5
    A = torch.rand(B, T) * 0.9 + 0.05
    X = torch.randn(B, T, D)
    h_init = torch.randn(B, D)
    H_seq = pscan_sequential(A, X, h_init=h_init)
    H_par = pscan(A, X, h_init=h_init)
    diff = _max_abs_diff(H_seq, H_par)
    print(f"  h_init provided  diff = {diff:.2e}")
    assert diff < 1e-5
    print("[ok] test_pscan_h_init")


# ---------------------------------------------------------------------------
# Test 5: constant-A specialisation matches general pscan
# ---------------------------------------------------------------------------

def test_pscan_constant_A():
    torch.manual_seed(4)
    B, P, S, T, D = 2, 3, 4, 31, 6
    decay = torch.rand(B, P, S) * 0.9 + 0.05    # constant in T
    X = torch.randn(B, P, S, T, D)

    # Reference: explicit broadcast then general pscan
    A_full = decay.unsqueeze(-1).expand(B, P, S, T)
    H_ref = pscan(A_full, X)
    H_spec = pscan_constant_A(decay, X)
    diff = _max_abs_diff(H_ref, H_spec)
    print(f"  constant-A specialisation  diff = {diff:.2e}")
    assert diff < 1e-5
    print("[ok] test_pscan_constant_A")


# ---------------------------------------------------------------------------
# Test 6: gradient flow
# ---------------------------------------------------------------------------

def test_pscan_gradient():
    torch.manual_seed(5)
    B, T, D = 2, 11, 4
    A = (torch.rand(B, T) * 0.9 + 0.05).requires_grad_()
    X = torch.randn(B, T, D, requires_grad=True)

    H = pscan(A, X)
    loss = H.pow(2).mean()
    loss.backward()
    assert A.grad is not None and torch.isfinite(A.grad).all()
    assert X.grad is not None and torch.isfinite(X.grad).all()
    assert A.grad.abs().sum() > 0
    assert X.grad.abs().sum() > 0

    # Also check gradients match the sequential reference
    A2 = A.detach().clone().requires_grad_()
    X2 = X.detach().clone().requires_grad_()
    H_seq = pscan_sequential(A2, X2)
    loss2 = H_seq.pow(2).mean()
    loss2.backward()
    grad_A_diff = _max_abs_diff(A.grad, A2.grad)
    grad_X_diff = _max_abs_diff(X.grad, X2.grad)
    print(f"  grad diff:  A: {grad_A_diff:.2e}    X: {grad_X_diff:.2e}")
    assert grad_A_diff < 1e-5
    assert grad_X_diff < 1e-5
    print("[ok] test_pscan_gradient")


# ---------------------------------------------------------------------------
# Test 7: long sequence (T=1024) — checks numerical stability
# ---------------------------------------------------------------------------

def test_pscan_long_sequence():
    torch.manual_seed(6)
    B, T, D = 1, 1024, 4
    # decay close to 1.0 — long-memory regime where naïve geometric-kernel
    # tricks would explode; pscan should remain stable
    A = torch.full((B, T), 0.99)
    X = torch.randn(B, T, D)
    H_seq = pscan_sequential(A, X)
    H_par = pscan(A, X)
    diff = _max_abs_diff(H_seq, H_par)
    print(f"  T=1024, decay=0.99  diff = {diff:.2e}")
    assert diff < 1e-3, f"long-T stability broke: {diff}"
    print("[ok] test_pscan_long_sequence")


def run_all():
    print("=" * 60)
    print("Parallel scan correctness suite")
    print("=" * 60)
    test_pscan_pow2()
    test_pscan_arbitrary_T()
    test_pscan_multidim()
    test_pscan_h_init()
    test_pscan_constant_A()
    test_pscan_gradient()
    test_pscan_long_sequence()
    print("=" * 60)
    print("ALL PSCAN TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
