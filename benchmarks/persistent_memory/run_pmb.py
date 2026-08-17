"""PMB v0 CLI runner.

Enforces the spec's session-boundary rule mechanically: after EVERY session
the system's state is snapshot()-ed to bytes, written to a real file on disk,
the instance is discarded, and a FRESH instance is built and restore()-d from
that file before the next session (and before answering). A system that
"remembers" only in process memory scores zero by construction.

Mandatory three-column report per system per task (spec §4):
    recall · state_bytes · answer_latency_ms
T2 additionally reports stale_answer_rate (and stale_context_rate).
T1 reports the full (N, K) grid; T3 reports the full forgetting curve.

Anti-enumeration scoring (spec §7): the harness truncates whatever the system
returns BEFORE matching — evidence contexts to --context_char_budget (default
4000 chars), generative answers to 200 chars — and reports the pre-truncation
mean length as ``context_chars_mean`` so flooding shows up in the report.

Usage::

    python benchmarks/persistent_memory/run_pmb.py \
        --task t1 --system rag --encoder hash --seed 0 --out_json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, Optional

try:                                    # package mode (python -m ...)
    from .tasks import generate_t2, t1_grid, t3_curve
    from .systems import MemorySystem, make_system
except ImportError:                     # script mode (python run_pmb.py)
    from tasks import generate_t2, t1_grid, t3_curve
    from systems import MemorySystem, make_system


# ---------------------------------------------------------------------------
# Episode runner — disk-serialized session loop
# ---------------------------------------------------------------------------

# Anti-enumeration defense (spec §7). Scoring looks ONLY at a harness-
# truncated prefix of what the system returns. Why: a cheat system that
# ingests nothing and answers every question with ALL 900,000 possible
# 6-digit codes (~5.4 MB of text) once saturated all three report columns
# (recall 1.0, state_bytes 0, low latency). Under a 4000-char budget the
# scored text holds at most ~570 codes, so blind enumeration hits a uniformly
# drawn gold code with probability ~570/900000 ≈ 0.06% per question.
# Generative answers get a much tighter 200-char budget — a real answer is
# one sentence, and 200 chars hold at most ~28 codes (hit rate ~0.003%).
# The PRE-truncation length is still reported (context_chars_mean) so
# flooding is exposed in the report instead of being rewarded.
DEFAULT_CONTEXT_CHAR_BUDGET = 4000
GENERATIVE_ANSWER_CHAR_BUDGET = 200


def run_episode(factory: Callable[[], MemorySystem], episode: Dict,
                snap_dir: Path,
                context_char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET) -> Dict:
    """Run one episode under the mandatory persistence discipline.

    Returns recall / stale rates / state_bytes (final snapshot size on disk) /
    answer_latency_ms (mean wall time of answer_context — or, for
    ``mode == "generative"`` systems, of answer — per question) /
    context_chars_mean (pre-truncation length of the scored text).
    """
    snap_dir = Path(snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    blob_path: Optional[Path] = None
    state_bytes = 0
    for session_id, text in episode["sessions"]:
        system = factory()                          # fresh instance per session
        if blob_path is not None:
            system.restore(blob_path.read_bytes())  # resume from DISK
        system.ingest(session_id, text)
        blob = system.snapshot(session_id)
        blob_path = snap_dir / f"{session_id}.snap"
        blob_path.write_bytes(blob)                 # state must survive here
        state_bytes = len(blob)
        del system                                  # nothing survives in RAM

    # Answer phase: yet another fresh instance, restored from disk.
    system = factory()
    if blob_path is not None:
        system.restore(blob_path.read_bytes())

    mode = getattr(system, "mode", "evidence")
    scoring_budget = (GENERATIVE_ANSWER_CHAR_BUDGET if mode == "generative"
                      else int(context_char_budget))
    hits = stale_answers = stale_contexts = 0
    latencies_ms = []
    raw_chars = []
    for q in episode["questions"]:
        t0 = time.perf_counter()
        raw = (system.answer(q["q"]) if mode == "generative"
               else system.answer_context(q["q"]))
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        raw_chars.append(len(raw))
        scored = raw[:scoring_budget]     # HARNESS truncation — see above
        gold_in = q["gold"] in scored
        stale_in = any(s in scored for s in q.get("stale", []))
        hits += int(gold_in)
        stale_contexts += int(stale_in)
        stale_answers += int(stale_in and not gold_in)

    n_q = len(episode["questions"])
    result = {
        "mode": mode,
        "recall": hits / n_q,
        "stale_answer_rate": stale_answers / n_q,
        "stale_context_rate": stale_contexts / n_q,
        "state_bytes": state_bytes,
        "answer_latency_ms": (sum(latencies_ms) / len(latencies_ms)
                              if latencies_ms else 0.0),
        "context_chars_mean": (sum(raw_chars) / len(raw_chars)
                               if raw_chars else 0.0),
        "n_questions": n_q,
        "n_sessions": len(episode["sessions"]),
    }
    if hasattr(system, "truncated"):
        result["truncated"] = bool(system.truncated)
    return result


def _run_in_tmp(factory: Callable[[], MemorySystem], episode: Dict,
                context_char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET) -> Dict:
    with tempfile.TemporaryDirectory(prefix="pmb_snap_") as td:
        return run_episode(factory, episode, Path(td),
                           context_char_budget=context_char_budget)


# ---------------------------------------------------------------------------
# Per-task drivers
# ---------------------------------------------------------------------------

def run_t1(factory: Callable[[], MemorySystem], seed: int,
           context_char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET) -> Dict:
    """Full (N, K) grid — single-cell reporting is invalid."""
    grid = []
    for params, episode in t1_grid(seed):
        res = _run_in_tmp(factory, episode, context_char_budget)
        grid.append({"n": params["n_facts"], "k": params["k_distractors"],
                     **res})
    return {"task": "t1", "grid": grid}


def run_t2(factory: Callable[[], MemorySystem], seed: int,
           context_char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET) -> Dict:
    episode = generate_t2(seed)
    res = _run_in_tmp(factory, episode, context_char_budget)
    return {"task": "t2", "params": episode["params"], **res}


def run_t3(factory: Callable[[], MemorySystem], seed: int,
           context_char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET) -> Dict:
    """Full forgetting curve over all gaps — single-point reporting is
    invalid (capacity-honesty clause, spec §2/T3)."""
    curve = []
    for params, episode in t3_curve(seed):
        res = _run_in_tmp(factory, episode, context_char_budget)
        curve.append({"gap": params["gap"], **res})
    return {"task": "t3", "n_facts": params["n_facts"], "curve": curve}


TASK_RUNNERS = {"t1": run_t1, "t2": run_t2, "t3": run_t3}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PMB v0 — Persistent Memory Benchmark runner")
    p.add_argument("--task", choices=["t1", "t2", "t3", "all"], default="t1")
    p.add_argument("--system", choices=["none", "oracle", "rag", "fastweight"],
                   default="rag")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder", choices=["hash", "e5"], default="hash",
                   help="rag encoder: hash = offline hashed BoW; "
                        "e5 = mt_lnn SentenceEncoder (multilingual-e5-small)")
    p.add_argument("--topk", type=int, default=8, help="rag retrieval top-k")
    p.add_argument("--dim", type=int, default=256,
                   help="hash encoder dimensionality")
    p.add_argument("--max_tokens", type=int, default=8000,
                   help="oracle context window (whitespace-word proxy)")
    p.add_argument("--context_char_budget", type=int,
                   default=DEFAULT_CONTEXT_CHAR_BUDGET,
                   help="anti-enumeration: the harness truncates "
                        "answer_context() to this many chars before scoring "
                        "(generative answers are always truncated to "
                        f"{GENERATIVE_ANSWER_CHAR_BUDGET} chars)")
    p.add_argument("--model", default=None,
                   help="fastweight: path to the E2 GPU-tuned MT-LNN adapter "
                        "checkpoint (v0: loader not implemented yet)")
    p.add_argument("--out_json", default=None,
                   help="write the report JSON here (stdout always)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.system == "fastweight" and args.model is not None:
        raise NotImplementedError(
            "v0: the fastweight checkpoint loader is pending the E2 GPU "
            "adapter deliverable — FastWeightSystem's protocol wiring exists "
            "in systems.py, but no --model loading path ships yet.")

    def factory() -> MemorySystem:
        return make_system(args.system, encoder=args.encoder, topk=args.topk,
                           dim=args.dim, max_tokens=args.max_tokens)

    tasks = ["t1", "t2", "t3"] if args.task == "all" else [args.task]
    report = {
        "benchmark": "pmb-v0",
        "system": args.system,
        "seed": args.seed,
        "config": {"encoder": args.encoder, "topk": args.topk,
                   "dim": args.dim, "max_tokens": args.max_tokens,
                   "context_char_budget": args.context_char_budget},
        "results": {t: TASK_RUNNERS[t](factory, args.seed,
                                       args.context_char_budget)
                    for t in tasks},
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
