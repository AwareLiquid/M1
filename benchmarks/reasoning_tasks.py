"""Depth-sensitive synthetic reasoning tasks for the P0 "thinking steps" study.

Two task families whose difficulty is controlled by a single reasoning-depth
knob, so we can plot "thinking iterations vs accuracy" (docs/ROADMAP_M2.md P0):

1. k-hop pointer chasing
   A random functional graph over `n_nodes` (each node has exactly one
   out-edge). Query: "starting at node s, where are you after k hops?".
   A parallel/shallow model must compose k lookups; one latent thinking
   iteration can propagate ~one hop, so accuracy should climb with
   thinking steps until steps >= k. This is the cleanest depth probe
   (HRM/TRM-style reasoning without language confounds).

2. modular arithmetic chain
   Evaluate ((a1 op a2) op a3 ...) mod m for k operands, ops in {+, -, *}.
   Sequential carry/accumulate structure; depth requirement grows with k.

Both are emitted as fixed-length token sequences with a single answer
position, tiny vocabularies, and deterministic generation given a seed —
so runs are reproducible and model-agnostic (no imports from mt_lnn here).

Sequence layout (both tasks):
    [BOS] <problem tokens...> [THINK] [ANS]
The model reads everything up to [THINK], iterates its core N times
("thinking"), then the logits at the [ANS] position are scored against the
answer token. Loss/accuracy is computed ONLY at the answer position.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

# ── Shared vocabulary ────────────────────────────────────────────────────────
# Token ids are packed as:
#   0..N_SPECIAL-1                    special tokens
#   N_SPECIAL..N_SPECIAL+V-1          value tokens (node ids / digits mod m)
# Value vocab size V = max(n_nodes, modulus) so both tasks share one embedding.

PAD, BOS, THINK, ANS, SEP, ARROW, START = range(7)
N_SPECIAL = 7
# Operator tokens live right after the specials in a reserved slot of 3.
OP_ADD, OP_SUB, OP_MUL = N_SPECIAL, N_SPECIAL + 1, N_SPECIAL + 2
N_OPS = 3
VALUE_BASE = N_SPECIAL + N_OPS  # first value-token id


def vocab_size(n_values: int) -> int:
    """Total vocab for a run where value tokens span 0..n_values-1."""
    return VALUE_BASE + n_values


def val_tok(v: int) -> int:
    return VALUE_BASE + int(v)


@dataclass
class Batch:
    """One generated batch. tokens: (B, T) int64; answer: (B,) value ids
    (already offset into vocab space); ans_pos: index of [ANS] in every row
    (constant per task config since sequences are fixed-length)."""

    tokens: np.ndarray
    answer: np.ndarray
    ans_pos: int


# ── Task 1: k-hop pointer chasing ────────────────────────────────────────────
#
# Layout: [BOS] u0 [ARROW] f(u0) [SEP] u1 [ARROW] f(u1) ... [SEP]
#         [START] s  k(as value token) [THINK] [ANS]
# The full edge list of the functional graph is presented (shuffled), then the
# start node and hop count. Answer = f^k(s).

def gen_pointer_chase(
    batch: int,
    n_nodes: int,
    k_hops: int,
    rng: np.random.Generator,
) -> Batch:
    T = 1 + 3 * n_nodes + 4 + 1  # BOS + n_nodes*(u ARROW v [SEP]) + START s k THINK + ANS
    toks = np.full((batch, T), PAD, dtype=np.int64)
    ans = np.zeros(batch, dtype=np.int64)

    for b in range(batch):
        # Random functional graph: f is a SINGLE n-cycle permutation.
        # A general random permutation admits a "copy the start node" shortcut:
        # f^k(s)=s iff the cycle length of s divides k, giving a content-free
        # accuracy of |{L in 1..n : L | k}|/n (measured 2026-07-29: models sat
        # exactly on this plateau — 0.379 at k=4 vs the predicted 3/8=0.375).
        # A single n-cycle guarantees f^k(s) != s for all 0 < k < n, and the
        # answer stays uniform over the other n-1 nodes — shortcut sealed.
        order = rng.permutation(n_nodes)
        f = np.empty(n_nodes, dtype=np.int64)
        f[order] = np.roll(order, -1)  # order[i] -> order[i+1], closing a cycle
        edges = rng.permutation(n_nodes)  # presentation order of edge list

        row = [BOS]
        for u in edges:
            row += [val_tok(u), ARROW, val_tok(f[u])]
        # NOTE: SEP omitted between triples to keep T small; the fixed
        # (u ARROW v) width already delimits edges unambiguously.
        s = int(rng.integers(n_nodes))
        row += [START, val_tok(s), val_tok(k_hops % n_nodes), THINK, ANS]

        cur = s
        for _ in range(k_hops):
            cur = int(f[cur])

        toks[b, : len(row)] = row
        ans[b] = val_tok(cur)

    return Batch(tokens=toks, answer=ans, ans_pos=T - 1)


# ── Task 2: modular arithmetic chain ─────────────────────────────────────────
#
# Layout: [BOS] a1 op a2 op a3 ... op ak [THINK] [ANS]
# Answer = ((a1 op a2) op a3 ...) mod m, left-to-right (no precedence).

_OPS = (OP_ADD, OP_SUB, OP_MUL)


def gen_mod_chain(
    batch: int,
    modulus: int,
    k_terms: int,
    rng: np.random.Generator,
    ops: tuple[int, ...] = _OPS,
) -> Batch:
    T = 1 + (2 * k_terms - 1) + 2  # BOS + terms/ops + THINK + ANS
    toks = np.full((batch, T), PAD, dtype=np.int64)
    ans = np.zeros(batch, dtype=np.int64)

    a = rng.integers(0, modulus, size=(batch, k_terms))
    op_idx = rng.integers(0, len(ops), size=(batch, k_terms - 1))

    for b in range(batch):
        row = [BOS, val_tok(a[b, 0])]
        acc = int(a[b, 0])
        for i in range(k_terms - 1):
            op = ops[op_idx[b, i]]
            v = int(a[b, i + 1])
            row += [op, val_tok(v)]
            if op == OP_ADD:
                acc = (acc + v) % modulus
            elif op == OP_SUB:
                acc = (acc - v) % modulus
            else:
                acc = (acc * v) % modulus
        row += [THINK, ANS]
        toks[b, : len(row)] = row
        ans[b] = val_tok(acc)

    return Batch(tokens=toks, answer=ans, ans_pos=T - 1)


# ── Task 3: parity ───────────────────────────────────────────────────────────
#
# Layout: [BOS] b1 b2 ... bk [THINK] [ANS], bits ∈ {0,1}, answer = XOR of all.
# Purpose (memory/m1-m2-depth-study, 2026-08-03): parity ∈ TC⁰, but
# finite-precision SSMs with NON-NEGATIVE gating provably cannot compute it
# (Sarrof et al. NeurIPS 2024 Thm 2). This is the DEBUG GATE for the liquid
# core's eigenvalue parameterization defect — failure here indicts the
# (0,1)-decay parameterization, not circuit depth, and is fixable by extending
# to signed eigenvalues (Grazzi et al. ICLR 2025). Do not confuse a parity
# failure with an NC¹ separation.

def gen_parity(
    batch: int,
    k_bits: int,
    rng: np.random.Generator,
) -> Batch:
    T = 1 + k_bits + 2  # BOS + bits + THINK + ANS
    toks = np.full((batch, T), PAD, dtype=np.int64)
    bits = rng.integers(0, 2, size=(batch, k_bits))
    toks[:, 0] = BOS
    toks[:, 1:1 + k_bits] = VALUE_BASE + bits
    toks[:, 1 + k_bits] = THINK
    toks[:, 2 + k_bits] = ANS
    ans = VALUE_BASE + (bits.sum(axis=1) % 2)
    return Batch(tokens=toks, answer=ans.astype(np.int64), ans_pos=T - 1)


# ── Task 4: S5 word problem ──────────────────────────────────────────────────
#
# Layout: [BOS] g1 g2 ... gk [THINK] [ANS], each g a uniformly random element
# of the symmetric group S5 (|S5| = 120, one value token per element), answer
# = the left-to-right product g1·g2·...·gk.
# Purpose: S5 is NON-SOLVABLE, so its word problem is NC¹-complete (Barrington)
# — the canonical task a TC⁰ model (diagonal, input-independent transition:
# Merrill et al. ICML 2024 Thm 4.2) cannot solve at fixed depth, while
# Θ(log n) weight-tied depth (stack_iterations) suffices (Merrill & Sabharwal
# NeurIPS 2025). This is THE separation experiment. NOTE mod_chain cannot play
# this role: abelian groups are solvable and admit O(1)-depth shortcuts
# (Krohn–Rhodes), so depth-flat results there are theory-consistent and
# uninformative. Products of uniform elements are uniform on S5 → content-free
# baseline is exactly 1/120; no cycle/shortcut structure to seal.

from itertools import permutations as _perms

_S5 = tuple(_perms(range(5)))                    # fixed enumeration, 120 elems
_S5_INDEX = {p: i for i, p in enumerate(_S5)}
# composition table: (a∘b)[x] = a[b[x]]; COMPOSE[a, b] = index of a∘b
_S5_COMPOSE = np.empty((120, 120), dtype=np.int64)
for _i, _a in enumerate(_S5):
    for _j, _b in enumerate(_S5):
        _S5_COMPOSE[_i, _j] = _S5_INDEX[tuple(_a[_b[x]] for x in range(5))]
S5_ORDER = 120


def gen_s5_word(
    batch: int,
    k_terms: int,
    rng: np.random.Generator,
) -> Batch:
    T = 1 + k_terms + 2  # BOS + elements + THINK + ANS
    toks = np.full((batch, T), PAD, dtype=np.int64)
    g = rng.integers(0, S5_ORDER, size=(batch, k_terms))
    toks[:, 0] = BOS
    toks[:, 1:1 + k_terms] = VALUE_BASE + g
    toks[:, 1 + k_terms] = THINK
    toks[:, 2 + k_terms] = ANS
    acc = g[:, 0].copy()
    for i in range(1, k_terms):
        acc = _S5_COMPOSE[acc, g[:, i]]
    return Batch(tokens=toks, answer=(VALUE_BASE + acc).astype(np.int64),
                 ans_pos=T - 1)


# ── Convenience: unified generator ───────────────────────────────────────────

def make_generator(task: str, difficulty: int, n_values: int, seed: int):
    """Returns (gen_fn(batch, rng) -> Batch, vocab, n_values).

    difficulty = k_hops for pointer_chase, k_terms for mod_chain.
    n_values   = n_nodes  for pointer_chase, modulus for mod_chain.
    """
    if task == "pointer_chase":
        def gen(batch: int, rng: np.random.Generator) -> Batch:
            return gen_pointer_chase(batch, n_values, difficulty, rng)
    elif task == "mod_chain":
        def gen(batch: int, rng: np.random.Generator) -> Batch:
            return gen_mod_chain(batch, n_values, difficulty, rng)
    elif task == "parity":
        n_values = 2  # bits; difficulty = k_bits
        def gen(batch: int, rng: np.random.Generator) -> Batch:
            return gen_parity(batch, difficulty, rng)
    elif task == "s5_word":
        n_values = S5_ORDER  # difficulty = k_terms
        def gen(batch: int, rng: np.random.Generator) -> Batch:
            return gen_s5_word(batch, difficulty, rng)
    else:
        raise ValueError(f"unknown task: {task}")
    return gen, vocab_size(n_values), n_values


# ── Self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    rng = np.random.default_rng(0)

    b = gen_pointer_chase(4, n_nodes=8, k_hops=3, rng=rng)
    assert b.tokens.shape[1] == b.ans_pos + 1
    assert (b.tokens[:, b.ans_pos] == ANS).all()
    assert (b.answer >= VALUE_BASE).all()

    # single-cycle guarantee: f^k(s) != s for all 0 < k < n (shortcut sealed)
    big = gen_pointer_chase(64, n_nodes=8, k_hops=4, rng=rng)
    for row_i in range(64):
        row_t = big.tokens[row_i]
        j, f_map = 1, {}
        while row_t[j] != START:
            f_map[int(row_t[j]) - VALUE_BASE] = int(row_t[j + 2]) - VALUE_BASE
            j += 3
        s0 = int(row_t[j + 1]) - VALUE_BASE
        cur0 = s0
        for _ in range(4):
            cur0 = f_map[cur0]
        assert cur0 != s0, "single-cycle must forbid f^k(s)=s for k<n"

    # answers must be reachable by manually replaying hops from the token row
    row = b.tokens[0]
    edges = {}
    i = 1
    while row[i] != START:
        u, v = int(row[i]) - VALUE_BASE, int(row[i + 2]) - VALUE_BASE
        edges[u] = v
        i += 3
    s = int(row[i + 1]) - VALUE_BASE
    cur = s
    for _ in range(3):
        cur = edges[cur]
    assert val_tok(cur) == b.answer[0], "pointer_chase replay mismatch"

    b2 = gen_mod_chain(4, modulus=10, k_terms=5, rng=rng)
    assert (b2.tokens[:, b2.ans_pos] == ANS).all()

    # parity: replay XOR from tokens
    bp = gen_parity(16, k_bits=9, rng=rng)
    for r in range(16):
        bits = bp.tokens[r, 1:10] - VALUE_BASE
        assert VALUE_BASE + (bits.sum() % 2) == bp.answer[r]

    # S5: group-theory sanity — identity element, closure, and manual replay
    ident = _S5_INDEX[tuple(range(5))]
    assert (_S5_COMPOSE[ident, :] == np.arange(120)).all()
    assert (_S5_COMPOSE[:, ident] == np.arange(120)).all()
    # associativity spot-check on random triples
    tri = np.random.default_rng(1).integers(0, 120, size=(50, 3))
    for a, b_, c in tri:
        assert _S5_COMPOSE[_S5_COMPOSE[a, b_], c] == _S5_COMPOSE[a, _S5_COMPOSE[b_, c]]
    bs = gen_s5_word(8, k_terms=6, rng=rng)
    for r in range(8):
        g = bs.tokens[r, 1:7] - VALUE_BASE
        acc = int(g[0])
        for i in range(1, 6):
            acc = int(_S5_COMPOSE[acc, int(g[i])])
        assert VALUE_BASE + acc == bs.answer[r]
    # deterministic given seed
    r1 = np.random.default_rng(42)
    r2 = np.random.default_rng(42)
    x1 = gen_mod_chain(2, 10, 4, r1)
    x2 = gen_mod_chain(2, 10, 4, r2)
    assert (x1.tokens == x2.tokens).all() and (x1.answer == x2.answer).all()

    print("reasoning_tasks selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        _selftest()
