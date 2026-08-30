# Parametric memory engine (v0)

`mt_lnn/parametric_memory.py` — an agent-memory runtime with a Mem0/Letta-shaped
interface whose memory body is **a fast-weight associative matrix, not a
database**. This page states what it is, how it compares to the external-store
route, and — with the same discipline as RESULTS.md — what it does and does
not claim. Measured numbers live in RESULTS.md / BENCHMARKS.md only.

## Objective (the optimization problem being solved)

The external-store route (Mem0 / Letta / Zep) pays O(n) storage for history,
cannot bit-exactly port state across processes, and implements selective
deletion by editing rows/logs. The claim under test: the fast-weight (F, z)
matrix this repo has already proven at the LM level (cross-window recall
0.56 anchor; snapshot/restore bit-exact) can BE the memory — O(1) state,
algebraic single-binding forgetting, bit-exact snapshots — behind a
Mem0-shaped API, and can be scored on the MemoryAgentBench four-competency
axes against both structural-zero controls and an external-store (BM25)
control.

## Iteration log (key decisions and fixed defects)

Decisions, each with its why:

1. **forget = delta projection, no index** (`F ← F − k(kᵀF)`). The task
   allowed a per-binding index for exact removal, but an index is O(n) state
   which would undermine the O(1) claim — algebra removes exactly one key's
   association at zero index cost.
2. **Relative zero-threshold in recall** (`|qF| ≤ 1e-5·|F|` → "no memory").
   Test-caught defect: after forget, fp32 rounding residue (~1e-7) still
   pointed mostly at the erased value — absolute thresholds leaked stale
   answers; scaling by |F| separates genuine reads from rounding by ~4
   orders of magnitude.
3. **D3 conflict is zero-shot** (training protocol stays identical to the
   0.56 anchor) so D2 remains anchor-comparable and D3 measures pure
   mechanism (sum blends, delta corrects).
4. **D4 runs in a real subprocess** (not just a fresh model object) — the
   snapshot provably survives process death; a no-restore control in the
   same child validates the harness has power.
5. **BM25 control reported as measured**: indexed exact-key lookup is flat
   latency and 1.0 accuracy — the honest comparison axis is storage O(n) vs
   constant state bytes and the parametric capacity boundary (~sqrt(d/N)).
6. **Worktree isolation** (`M1-pm`): the main checkout is shared by
   concurrent lines; committing there twice landed commits on foreign
   branches. The branch now lives in its own worktree.
7. **Steps discipline**: 3000 steps (budget cap) vs the anchor's 8000 —
   every JSON records its steps; docs never mix the two calibers.

Fixed defects along the way: `load_session` dropped the new `fw_state`
field (snapshots never round-tripped); argparse prefix-matching swallowed
`--config`/`--seed` into `--configs`/`--seeds` in child mode; BM25 store had
an unbound local (`n`) and an inconsistent value-token prefix (0% accuracy);
runtime curve at d=2048 would have allocated an 8 GB embedding table
(vocab narrowed to 20k); logging was silent between phases (now: timestamps,
PHASE markers, NaN watchdog, per-trial child output, tee'd run log).


## Interface

```python
mem = ParametricMemory(update_rule="sum")        # or "delta"
mem.write("s1", key=5001, value=7234)            # token-id path (anchor space)
mem.write("s1", key="favorite color", value="blue")   # text path (hash or injected encoder)
mem.recall("s1", query=5001, top_k=3)            # k -> F read -> NN decode
mem.forget("s1", key=5001)                       # ONE binding, surgically
mem.forget("s1")                                 # whole session -> zeroed
mem.save("s1", "s1.json"); mem.load("s1", "s1.json")  # bit-exact, atomic
mem.state_bytes("s1")                            # constant in writes
```

The per-session state is the same (F, z) pair the trained adapters carry
(`mt_lnn/mt_lnn_v2.py`), with both write rules this repo has benchmarked:

| rule | write | read | LM counterpart |
|---|---|---|---|
| `sum` | `F ← decay·F + k vᵀ`, `z ← decay·z + k` | `r = qF / (q·z + ε)` | `mt_v2` (the 0.56-anchor mechanism) |
| `delta` | `F ← decay·F − η·k(kᵀF − vᵀ)` | `r = qF` | `mt_v2_delta` (gradient-as-memory, corrects on rewrite) |

`forget(key)` is a delta projection `F ← F − k(kᵀF)` — algebra, not row
deletion: no binding index exists, so state size stays O(1) in writes.

## Mechanism comparison

|  | Mem0 / Letta / Zep (external store) | Titans-style parametric (online meta-gradient) | **this repo (fast-weight state)** |
|---|---|---|---|
| Memory body | vector DB / files / plain-text logs | model weights updated by surprise-driven gradients | per-session (F, z) fast-weight matrix |
| Storage vs history | **O(n)** — grows with every fact | O(1) | **O(1)** — (d² + d) floats per session, constant in writes |
| Snapshot / restore | exporter-dependent, rarely bit-exact | not addressed | **bit-exact** (base64 through the session envelope; unit-tested) |
| Selective delete | delete/edit rows or logs | not addressed | **single-binding algebraic removal** (forget) |
| Rewrite (conflict) | upsert a record | gradient step, approximate | `delta` write corrects; `sum` blends (measured in the four-competency bench) |
| Decode cost | index lookup | full forward pass | O(V·d) codebook NN (or O(candidates·d)) |
| Capacity boundary | unbounded (grows) | task-dependent | **~sqrt(d/N) read SNR** — at fixed d, accuracy degrades past O(d) bindings (measured curve in the bench) |

The honest one-line trade: external stores buy unbounded capacity with O(n)
storage and no bit-exact state; the parametric runtime buys constant storage,
bit-exact snapshots and single-binding forgetting at a finite per-state
capacity that scales with d. Accuracy-vs-writes at fixed state width d is the
capacity curve the bench emits as
`benchmarks/results/parametric_memory_runtime.json`.

## Complexity notes (full disclosure)

- `state_bytes(session)` counts the (F, z) tensors — **constant in the number
  of writes**. The session **registry** grows with the number of sessions
  (one fixed-size state each): per-session memory is O(1) in writes, total
  O(#sessions).
- Text **values** need a decode lexicon (a read vector must map back to
  text). It is a **fixed-capacity LRU** (default 1024) — bounded, unlike an
  append-only log. Token-id values decode against the fixed embedding table
  and need no lexicon.
- Recall **compute** is O(d²) for the matrix read plus O(candidates·d) for
  the decode — flat in the number of writes, unlike an O(n)-scan store, but
  growing with the chosen state width d.
- The default text encoder is a deterministic feature hash (exact-match
  keys, zero downloads). Inject a real sentence encoder for semantic recall
  (same contract as `fast_weight_store.build_session_key`).

## What this is NOT

- It does **not** improve base-model perplexity and does **not** provide
  long-context LM gains (see RESULTS.md nulls — the state is an episodic
  key→value memory, not compressed distributed context).
- It is v0: single-head, not thread-safe, decay defaults to 1.0 (forgetting
  is explicit). The PMB harness (`benchmarks/persistent_memory/`) is the
  follow-on integration point for text-session tasks.
