# PMB — Persistent Memory Benchmark (v0)

A benchmark for **cross-session persistent memory**. Existing suites test
retrieval inside one context window (NIAH/RULER), one very long sequence
(BABILong), or retrieval-product pipelines (LongMemEval). PMB tests the three
axes none of them cover:

1. **Session boundaries** — between sessions, ALL memory state must be
   serialized to disk and restored into a fresh process/instance. Bit-exact
   persistence is the entry ticket, not a bonus.
2. **Update semantics** — later writes overwrite earlier ones; answering with
   a superseded value is scored as a distinct failure mode.
3. **Forgetting curve** — recall must be reported as a function of the number
   of intervening distractor sessions, as a full curve, never a single point.

All tasks are program-generated from a seed (English templates in v0);
gold answers are unique 6-digit codes so exact-substring scoring is
unambiguous.

## Contestant protocol (submission interface)

Implement a subclass of `MemorySystem` (`systems.py`) with four methods:

```python
class MySystem(MemorySystem):
    mode = "evidence"   # or "generative" — declares your scoring path (§ below)

    def ingest(self, session_id: str, text: str) -> None:
        """Consume one session's text."""

    def snapshot(self, session_id: str) -> bytes:
        """Serialize ALL persistent state to bytes (the harness writes them
        to a file on disk). This is your system's entire memory."""

    def restore(self, blob: bytes) -> None:
        """Rebuild state on a FRESH instance from a snapshot blob."""

    def answer_context(self, question: str) -> str:
        """Evidence-recall mode: return the context your system assembles
        for the question. Scored by exact-substring match of the gold value
        against a harness-truncated prefix (see scoring modes)."""
        # Generative systems instead override answer(question) -> str and are
        # scored on the (truncated) generated answer text.
```

### The harness enforces persistence mechanically

For every session in an episode the runner (`run_pmb.py::run_episode`):

1. builds a **fresh** instance via your factory,
2. `restore()`s it from the previous session's on-disk snapshot file,
3. `ingest()`s the session text,
4. `snapshot()`s and writes the bytes to disk,
5. **discards the instance**.

The answer phase runs on yet another fresh instance restored from disk. State
that lives only in process memory scores zero by construction.

### Scoring modes (declared via the `mode` class attribute)

* **Evidence recall** (`mode = "evidence"`, the default): a question is
  correct iff the gold 6-digit value appears as an exact substring of
  `answer_context(question)` **after the harness truncates it to
  `--context_char_budget` chars (default 4000)**. This isolates the *memory*
  system from answer-generation quality.
* **Generative** (`mode = "generative"`, for weight/state-memory systems that
  cannot emit retrieved text, e.g. fast-weight models): the harness calls
  `answer(question)` instead — `answer_context` is never invoked — and
  substring-matches the gold against the **first 200 chars** of the generated
  answer. `stale_answer_rate` and `answer_latency_ms` are computed on the
  same truncated answer / the `answer()` call.

**Anti-enumeration truncation (why the budgets exist).** A cheat system that
ingests nothing and answers every question with all 900,000 possible 6-digit
codes (~5.4 MB of text) once saturated all three report columns. The harness
now scores only a truncated prefix: 4000 chars hold at most ~570 codes, so
blind enumeration hits a gold code with probability ~0.06% per question (200
chars ≈ 28 codes ≈ 0.003% for generative answers). The **pre-truncation**
length is still reported as `context_chars_mean`, so flooding is exposed in
the report instead of being rewarded. Returning a focused, minimal context is
the only winning strategy.

## Task families

| Task | What it tests | Variables | Metrics |
|---|---|---|---|
| **T1** Cross-session retention | N facts in session 1, K distractor sessions (~2K tokens each), then questions | N ∈ {4, 16, 64} × K ∈ {0, 4, 16} — the **full grid** | exact-match recall per (N, K) cell |
| **T2** Streaming update recall | the same key rewritten across sessions; only the LATEST value counts | 8 keys × 3 write rounds (v0 default) | latest-value recall **and** `stale_answer_rate` (stale present, gold absent) + `stale_context_rate` (stale present at all) |
| **T3** Forgetting curve | fixed injection, recall vs. gap | gap ∈ {0, 2, 8, 32} distractor sessions — the **full curve** | recall per gap |

**Anti-saturation hardening (all tasks).** Every queried person carries 2
extra *confuser* facts about other attributes (each with its own code) — the
question asks about exactly one attribute, so entity-level retrieval is not
enough. Every distractor session embeds 8 *decoy* facts ("other person -
attribute - code") phrased with the same templates as the real facts — "find
a 6-digit number" is not a strategy. All codes in an episode (gold, stale,
confuser, decoy) are drawn in one global without-replacement pass, so the
gold can never collide with anything and exact-substring scoring stays
unambiguous; an oracle with an unbounded window still recalls 100%.

## Mandatory three-column report

Every system on every task must report **all three** columns — the benchmark's
thesis is their trade-off surface, and a recall-only submission is invalid:

| Column | Definition |
|---|---|
| `recall` | exact-match recall (per T1 cell / T2 / per T3 gap), scored on the harness-truncated text |
| `state_bytes` | size in bytes of the final on-disk snapshot |
| `answer_latency_ms` | mean wall-clock time of context assembly (or generation) per question |

Supplementary fields: `mode` (which scoring path was used) and
`context_chars_mean` (mean **pre-truncation** length of the returned
context/answer — flooding shows up here).

## Honesty clauses (v0 scope)

* Generators are synthetic English templates; Chinese and real-dialogue
  corpora (LongMemEval-style) are v1.
* Fixed-size-state systems **must** publish the full T3 curve, including the
  unfavorable region — bounded state under competitive association provably
  overwrites, and hiding that disqualifies the submission.
* The `oracle` baseline reports a `truncated` flag when history overflows its
  window (`--max_tokens`); overflow failure is a data point, not an error.
* The benchmark authors publish their own system's numbers on all three
  columns, including the unfavorable T3 region, alongside every release.
* `none` / `oracle` / `rag` run CPU-only. The `fastweight` contestant
  (MT-LNN adapter (F, z) snapshots + `FastWeightSessionStore`) requires a
  GPU-fine-tuned adapter checkpoint and ships as an interface skeleton in v0.

## Running

```bash
# offline, CPU-only, no downloads:
python benchmarks/persistent_memory/run_pmb.py \
    --task all --system rag --encoder hash --seed 0 --out_json rag_hash.json

# reference baselines:
python benchmarks/persistent_memory/run_pmb.py --task all --system none
python benchmarks/persistent_memory/run_pmb.py --task all --system oracle --max_tokens 8000

# rag with the real sentence encoder (downloads multilingual-e5-small):
python benchmarks/persistent_memory/run_pmb.py --task t1 --system rag --encoder e5

# tests:
python benchmarks/persistent_memory/test_pmb.py
```

Key flags: `--task {t1,t2,t3,all}` · `--system {none,oracle,rag,fastweight}` ·
`--seed INT` · `--encoder {hash,e5}` · `--topk INT` · `--dim INT` (hash) ·
`--max_tokens INT` (oracle) · `--context_char_budget INT` (evidence-mode
scoring truncation, default 4000; generative answers are always cut at 200) ·
`--out_json PATH`.

### Output JSON shape

```jsonc
{
  "benchmark": "pmb-v0",
  "system": "rag",
  "seed": 0,
  "config": {"encoder": "hash", "topk": 8, "dim": 256, "max_tokens": 8000,
             "context_char_budget": 4000},
  "results": {
    "t1": {"task": "t1", "grid": [
      {"n": 4, "k": 0, "mode": "evidence", "recall": 1.0,
       "state_bytes": 13113, "answer_latency_ms": 0.06,
       "context_chars_mean": 480.0, "...": "..."}
    ]},
    "t2": {"task": "t2", "recall": 1.0, "stale_answer_rate": 0.0,
           "stale_context_rate": 0.5, "...": "..."},
    "t3": {"task": "t3", "curve": [
      {"gap": 0, "recall": 1.0, "...": "..."},
      {"gap": 32, "recall": 0.9, "...": "..."}
    ]}
  }
}
```

## Submitting a system

1. Implement the four-method protocol above (or wrap your product's API in
   it). Your `snapshot`/`restore` must round-trip through bytes on disk.
2. Run `--task all` on seeds {0, 1, 2} and report the mean of each cell.
3. Include: the full T1 grid, T2 recall **and** stale rates, the full T3
   curve, and all three columns everywhere.
4. Declare your scoring mode via the `mode` class attribute
   (`"evidence"` or `"generative"`) and state it — plus your encoder/model
   versions — in the submission. Scoring is always applied to the
   harness-truncated text (4000 chars evidence / 200 chars generative);
   context or answer flooding is reported via `context_chars_mean` and does
   not score. Do not tune on the test seeds you report.
