"""
parallel_scan.py — Blelloch parallel prefix scan for linear recurrences.

Solves h_t = A_t * h_{t-1} + X_t,  h_0 = 0,  for t = 0 ... T-1
in O(log T) sequential depth (O(T log T) total work) instead of O(T)
sequential time.

This is the same scan algorithm used by Mamba / S4 / S5 for selective SSM
training. It is what makes MT-LNN's "liquid" recurrence actually recurrent
during training — without this, training reduces to a gated FFN.

Implementation
--------------
Recursive form of François Fleuret's pscan, with multi-dim batching:
    https://fleuret.org/dlc/materials/pscan.py

A trivially-batched PyTorch port also appears in:
    https://github.com/alxndrTL/mamba.py/blob/main/mambapy/pscan.py
    (MIT licence, attribution preserved)
    https://github.com/sustcsonglin/mamba-triton  (Triton variant)

Both A and X are batched over any number of leading dims; the *last* dim of A
and the *second-to-last* dim of X are the scan dimension T. This matches the
shape we already pass through the model: (..., T) for the recurrence
multiplier and (..., T, D) for the inputs.

Correctness is checked in tests/test_parallel_scan.py against a sequential
reference implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sequential reference (slow, used for testing and small T)
# ---------------------------------------------------------------------------

def pscan_sequential(A: torch.Tensor, X: torch.Tensor,
                      h_init: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Sequential O(T) reference: h_t = A_t * h_{t-1} + X_t.

    A:      (..., T)        — per-step multipliers
    X:      (..., T, D)     — per-step inputs
    h_init: (..., D) or None — initial state h_{-1}; defaults to zeros

    Returns H: (..., T, D)
    """
    T = X.shape[-2]
    H = torch.empty_like(X)
    h = (h_init if h_init is not None else torch.zeros_like(X[..., 0, :]))
    for t in range(T):
        h = A[..., t : t + 1] * h + X[..., t, :]
        H[..., t, :] = h
    return H


# ---------------------------------------------------------------------------
# Parallel scan (Blelloch, recursive form)
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    # Tracers (torch.jit / torch.onnx) hand this a 0-dim Tensor rather than a
    # Python int, and Tensor has no .bit_length(). The scan depth is a function
    # of the sequence length, which is static whenever we are tracing, so
    # coercing to int here is safe and is what makes ONNX export possible.
    n = int(n)
    return 1 << (max(n, 1) - 1).bit_length()


def _pscan_pow2(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Recursive parallel scan assuming T is a power of 2.

    A: (..., T),   X: (..., T, D).   Returns H: (..., T, D).

    Each level of recursion combines (even, odd) pairs:
        Xo_new = Ao * Xa + Xo,    Ao_new = Ao * Aa
    Recurses on the odd subsequence of length T/2, then reconstructs the
    even outputs from a one-step-shifted version of the recursive result.
    """
    T = A.shape[-1]
    if T == 1:
        return X

    # Split scan dim into even / odd indices
    Aa = A[..., 0::2]                    # (..., T/2)
    Ao = A[..., 1::2]                    # (..., T/2)
    Xa = X[..., 0::2, :]                 # (..., T/2, D)
    Xo = X[..., 1::2, :]                 # (..., T/2, D)

    # Combine pairs so the odd sequence absorbs its even predecessor
    Xo_new = Ao.unsqueeze(-1) * Xa + Xo  # (..., T/2, D)
    Ao_new = Ao * Aa                     # (..., T/2)

    # Recurse on the odd (combined) sub-scan
    Yo = _pscan_pow2(Ao_new, Xo_new)     # (..., T/2, D)

    # Reconstruct the even outputs: Ya[t] = Aa[t] * Yo[t-1] + Xa[t]
    # with Yo[-1] = 0.  We shift Yo right by 1 along the scan dim.
    zero = torch.zeros_like(Yo[..., :1, :])
    Yo_shifted = torch.cat([zero, Yo[..., :-1, :]], dim=-2)
    Ya = Aa.unsqueeze(-1) * Yo_shifted + Xa   # (..., T/2, D)

    # Interleave Ya and Yo back into a length-T output
    new_shape = list(Yo.shape)
    new_shape[-2] = T                          # restore full scan length
    Y = torch.empty(new_shape, dtype=X.dtype, device=X.device)
    Y[..., 0::2, :] = Ya
    Y[..., 1::2, :] = Yo
    return Y


def pscan(A: torch.Tensor, X: torch.Tensor,
           h_init: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Parallel scan: h_t = A_t * h_{t-1} + X_t,  h_{-1} = h_init (or 0).

    A:      (..., T)        — per-step multipliers. Any sign is valid — the
                               scan is plain multiplication; stability needs
                               |A| <= 1, enforced upstream. Signed (negative)
                               multipliers are used by the signed-decay
                               extension (config.signed_decay, λ = decay·tanh(s));
                               parity vs pscan_sequential is covered by
                               tests/test_signed_decay.py.
    X:      (..., T, D)     — per-step inputs
    h_init: (..., D) or None — initial state (absorbed into X[..., 0, :])

    Returns H: (..., T, D), same shape and dtype as X.

    O(log T) sequential depth via Blelloch scan; pads T to the next power of 2.
    """
    T_orig = A.shape[-1]
    assert X.shape[-2] == T_orig, \
        f"A and X scan dims disagree: A[..., T={T_orig}], X[..., T={X.shape[-2]}, D]"

    # Absorb non-zero initial state into the first input: h_0 = A_0 * h_init + X_0
    # so we can keep the pscan formula h_{-1} = 0.
    if h_init is not None:
        X = X.clone()
        X[..., 0, :] = X[..., 0, :] + A[..., 0:1] * h_init

    # Pad to next power of 2 along the scan dim
    Tpow2 = _next_pow2(T_orig)
    if Tpow2 != T_orig:
        pad = Tpow2 - T_orig
        # Pad A with 1 (so h propagates unchanged past the real input)
        A_padded = F.pad(A, (0, pad), value=1.0)
        # Pad X with 0
        X_padded = F.pad(X, (0, 0, 0, pad), value=0.0)
    else:
        A_padded, X_padded = A, X

    H = _pscan_pow2(A_padded, X_padded)

    if Tpow2 != T_orig:
        H = H[..., :T_orig, :]
    return H


# ---------------------------------------------------------------------------
# Specialised helper for the case A is constant along T
# (this is our default case — decay depends only on (proto, scale))
# ---------------------------------------------------------------------------

def pscan_constant_A(decay: torch.Tensor, X: torch.Tensor,
                      h_init: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Same as pscan() but A is broadcast from a smaller shape — typically the
    recurrence multiplier is constant along T (per protofilament / scale)
    and we don't want to materialise a (..., T) tensor just to scan.

    decay:  (...)            — per-channel decay, broadcasts over T
    X:      (..., T, D)
    h_init: (..., D) or None

    Returns H: (..., T, D).

    Implementation: expand decay along T and call pscan(). PyTorch handles
    the no-copy broadcast in the einsum / multiplications inside pscan.
    """
    T = X.shape[-2]
    # Broadcast decay to (..., T): add a scan dim and expand
    A = decay.unsqueeze(-1).expand(*decay.shape, T)
    return pscan(A, X, h_init=h_init)


# ---------------------------------------------------------------------------
# Chunkwise scan (SSD-style): intra-chunk matmul + inter-chunk carry
# ---------------------------------------------------------------------------
#
# Same recurrence h_t = A_t * h_{t-1} + X_t, decomposed into chunks of C
# steps (Mamba-2 SSD / GLA / DeltaNet training formulation):
#   * within a chunk every output is ONE masked matmul
#         H_local = L @ X_chunk,   L[i, j] = prod_{k=j+1..i} A_k  (j <= i)
#   * across chunks only a single state carry s propagates:
#         H[i] = L[i, :] @ X_chunk + g_i * s,   g_i = prod_{k=start..i} A_k
# Sequential depth drops from O(T log T) scan work to T/C small carry
# steps while the per-chunk work is dense matmul (Tensor-Core friendly).
#
# L is built without divisions or cumprod quotients, which underflow fp32
# for |A| < 1 and long chunks (0.05^64 ~ 1e-83). Instead the magnitude is
# factorised log-space (segsum: exp(cumlog_i - cumlog_j), the Mamba-2
# formulation) and the SIGN is factorised exactly (scum_i * scum_j for
# sigma in {-1,+1}), so signed multipliers (config.signed_decay /
# selective_decay, lambda in (-1, 1)) are first-class. exp underflow
# flushes truly-negligible weights to 0 — the same flush-to-zero the
# sequential products already exhibit.
#
# Bit-level result differs from pscan() only by float reassociation
# (different summation order); equivalence to pscan_sequential is pinned
# by tests/test_pscan_chunkwise.py and is the merge gate for the
# use_chunkwise_scan switch.

def _chunk_decay_terms(A_c: torch.Tensor):
    """Shared per-chunk decay factorisation.

    A_c: (..., C). Returns (cumlog, scum):
      cumlog: (..., C) inclusive cumsum of log|A| (clamped away from 0)
      scum:   (..., C) inclusive cumprod of sign(A) in {-1, +1}
    """
    sign = (A_c >= 0).to(A_c.dtype) * 2.0 - 1.0          # sign(0) -> +1
    tiny = torch.finfo(A_c.dtype).tiny                   # dtype-safe log domain
    logm = A_c.abs().clamp_min(tiny).log()
    return logm.cumsum(-1), sign.cumprod(-1)


def _chunk_decay_matrix(cumlog: torch.Tensor, scum: torch.Tensor) -> torch.Tensor:
    """L[i, j] = prod_{k=j+1..i} A_k for j <= i, else 0.  (..., C, C)."""
    C = cumlog.shape[-1]
    segsum = cumlog.unsqueeze(-1) - cumlog.unsqueeze(-2)   # (..., i, j)
    sign_outer = scum.unsqueeze(-1) * scum.unsqueeze(-2)
    tri = torch.ones(C, C, device=cumlog.device,
                     dtype=cumlog.dtype).tril()            # incl. diagonal
    return segsum.exp() * sign_outer * tri


def _scan_chunk(A_c: torch.Tensor, X_c: torch.Tensor, carry: torch.Tensor):
    """One chunk: H = L @ X + g * carry; returns (H, next_carry = H[..., -1, :])."""
    cumlog, scum = _chunk_decay_terms(A_c)
    L = _chunk_decay_matrix(cumlog, scum)                # (..., C, C)
    g = scum * cumlog.exp()                              # (..., C) decay from chunk start
    H = L.matmul(X_c) + g.unsqueeze(-1) * carry          # (..., C, D)
    return H, H[..., -1, :]


def pscan_chunkwise(A: torch.Tensor, X: torch.Tensor,
                    h_init: Optional[torch.Tensor] = None,
                    chunk_size: int = 64) -> torch.Tensor:
    """
    Chunkwise parallel-form scan: same semantics as pscan(), same shapes.

    A:      (..., T)        — per-step multipliers (any sign, as in pscan)
    X:      (..., T, D)     — per-step inputs
    h_init: (..., D) or None — initial state h_{-1} (chunk-0 carry, NOT
                               absorbed into X, so X is never copied)
    chunk_size: C           — intra-chunk width; trade matmul size (C^2)
                              against carry-loop depth (T/C)

    Returns H: (..., T, D), same shape and dtype as X.
    """
    T = A.shape[-1]
    assert X.shape[-2] == T, \
        f"A and X scan dims disagree: A[..., T={T}], X[..., T={X.shape[-2]}, D]"
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")
    if T == 0:
        return torch.empty_like(X)

    carry = h_init if h_init is not None else torch.zeros_like(X[..., 0, :])
    H_chunks = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        H_c, carry = _scan_chunk(A[..., start:end], X[..., start:end, :], carry)
        H_chunks.append(H_c)
    return torch.cat(H_chunks, dim=-2)


def pscan_chunkwise_constant_A(decay: torch.Tensor, X: torch.Tensor,
                               h_init: Optional[torch.Tensor] = None,
                               chunk_size: int = 64) -> torch.Tensor:
    """
    Chunkwise counterpart of pscan_constant_A(): decay (...)
    broadcasts over the scan dim, then pscan_chunkwise() runs as usual.
    """
    T = X.shape[-2]
    A = decay.unsqueeze(-1).expand(*decay.shape, T)
    return pscan_chunkwise(A, X, h_init=h_init, chunk_size=chunk_size)
