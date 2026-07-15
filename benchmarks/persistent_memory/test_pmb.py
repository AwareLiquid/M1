"""PMB v0 tests — pytest-style, but also directly runnable:

    E:/M1/.venv/Scripts/python.exe benchmarks/persistent_memory/test_pmb.py

Covers: generator determinism (byte-equal across regenerations), T2 latest-
value semantics, oracle upper bound (recall = 1 with an unbounded window),
none lower bound (recall = 0), and rag/hash beating none on a small T1 cell —
all under the mandatory disk-serialized session loop.
"""
from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path

try:                                    # package mode (pytest from repo root)
    from .tasks import generate_t1, generate_t2, t1_grid, t3_curve
    from .systems import MemorySystem, make_system
    from .run_pmb import (DEFAULT_CONTEXT_CHAR_BUDGET, run_episode)
except ImportError:                     # script mode (python test_pmb.py)
    from tasks import generate_t1, generate_t2, t1_grid, t3_curve
    from systems import MemorySystem, make_system
    from run_pmb import DEFAULT_CONTEXT_CHAR_BUDGET, run_episode


def _all_episodes(seed: int):
    return ([ep for _, ep in t1_grid(seed)]
            + [generate_t2(seed)]
            + [ep for _, ep in t3_curve(seed)])


def _run_factory(factory, episode,
                 context_char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET):
    with tempfile.TemporaryDirectory(prefix="pmb_test_") as td:
        return run_episode(factory, episode, Path(td),
                           context_char_budget=context_char_budget)


def _run(system_name: str, episode,
         context_char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET, **kwargs):
    return _run_factory(lambda: make_system(system_name, **kwargs),
                        episode, context_char_budget)


# ---------------------------------------------------------------------------


def test_generators_are_deterministic():
    """Same seed twice -> byte-identical serialized episodes; a different
    seed -> different content (the seed is actually wired through)."""
    a = json.dumps(_all_episodes(7), sort_keys=True).encode("utf-8")
    b = json.dumps(_all_episodes(7), sort_keys=True).encode("utf-8")
    assert a == b, "same-seed regeneration is not byte-equal"
    c = json.dumps(_all_episodes(8), sort_keys=True).encode("utf-8")
    assert a != c, "different seeds produced identical data"


def test_t2_gold_is_latest_value():
    """gold = the LAST-written value; every stale value was written in an
    earlier session; gold never collides with any stale value."""
    ep = generate_t2(3)
    assert ep["questions"], "T2 generated no questions"
    for q in ep["questions"]:
        assert len(q["stale"]) >= 1, "T2 question has no update history"
        assert q["gold"] not in q["stale"]
        gold_sessions = [i for i, (_, text) in enumerate(ep["sessions"])
                         if q["gold"] in text]
        assert gold_sessions, "gold value not present in any session"
        for s in q["stale"]:
            stale_sessions = [i for i, (_, text) in enumerate(ep["sessions"])
                              if s in text]
            assert stale_sessions, "stale value not present in any session"
            assert max(stale_sessions) < min(gold_sessions), (
                "a stale value was written after the gold value")


def test_oracle_unbounded_recall_is_one():
    ep = generate_t1(0, 4, 2)
    res = _run("oracle", ep, max_tokens=10**9)
    assert res["recall"] == 1.0, f"oracle full-history recall {res['recall']}"
    assert res["truncated"] is False
    assert res["state_bytes"] > 0


def test_none_recall_is_zero():
    ep = generate_t1(0, 4, 2)
    res = _run("none", ep)
    assert res["recall"] == 0.0
    assert res["state_bytes"] == 0


def test_rag_hash_beats_none_on_small_t1():
    ep = generate_t1(0, 4, 2)
    rag = _run("rag", ep, encoder="hash", topk=8, dim=256)
    none = _run("none", ep)
    assert rag["recall"] > none["recall"], (
        f"rag/hash recall {rag['recall']} did not beat none {none['recall']}")
    assert rag["state_bytes"] > 0


def test_snapshot_restore_survives_disk_roundtrip():
    """Direct check that rag state is bit-faithful through bytes: restore on
    a fresh instance answers identically to the live one."""
    ep = generate_t1(1, 4, 0)
    live = make_system("rag", encoder="hash", topk=8, dim=256)
    for sid, text in ep["sessions"]:
        live.ingest(sid, text)
    blob = live.snapshot("final")

    revived = make_system("rag", encoder="hash", topk=8, dim=256)
    revived.restore(blob)
    for q in ep["questions"]:
        assert live.answer_context(q["q"]) == revived.answer_context(q["q"])
    assert revived.snapshot("again") == blob, "snapshot not bit-stable"


# ---------------------------------------------------------------------------
# Anti-gaming defenses (spec §7)
# ---------------------------------------------------------------------------


def _all_codes_text() -> str:
    """Every possible 6-digit code, space-joined (~6.3 MB) — the enumeration
    cheat payload. Space separators keep 6-digit substrings unambiguous."""
    return " ".join(str(c) for c in range(100_000, 1_000_000))


def test_enumeration_cheat_scores_zero():
    """A system that ingests NOTHING and answers with all 900k codes used to
    saturate all three columns. The harness char budget (4000) holds ~570
    codes -> per-question hit rate ~0.06%, so recall collapses; the flooding
    itself is exposed via context_chars_mean."""
    payload = _all_codes_text()

    class EnumerationCheat(MemorySystem):
        name = "cheat"

        def ingest(self, session_id, text):
            pass

        def snapshot(self, session_id):
            return b""

        def restore(self, blob):
            pass

        def answer_context(self, question):
            return payload

    ep = generate_t1(0, 16, 4)
    res = _run_factory(lambda: EnumerationCheat(), ep)
    assert res["recall"] < 0.05, (
        f"enumeration cheat still scores {res['recall']}")
    assert res["context_chars_mean"] > DEFAULT_CONTEXT_CHAR_BUDGET, (
        "flooding must be visible in context_chars_mean")


def test_decoys_create_discrimination():
    """Hardened T1 (decoy facts in distractors + confuser attributes on the
    queried persons) must break rag/hash topk=8 saturation on the hardest
    cell, while an oracle with an unbounded window AND unbounded scoring
    budget still recalls 100% — the scoring-correctness anchor: every gold is
    present and extractable from the full history."""
    ep = generate_t1(0, 64, 16)
    rag = _run("rag", ep, encoder="hash", topk=8, dim=256)
    assert rag["recall"] < 0.999, (
        f"rag/hash topk=8 still saturates hardened T1: {rag['recall']}")
    oracle = _run("oracle", ep, context_char_budget=10**9, max_tokens=10**9)
    assert oracle["recall"] == 1.0, (
        f"oracle anchor broke: {oracle['recall']} (generator invariant)")


def test_generative_mode_scored_via_answer():
    """mode='generative' systems are scored on answer() truncated to 200
    chars: a dummy that answers the gold value scores 1.0; a flooder that
    answers ~6 MB of codes scores ~0 — and answer_context is never called."""
    ep = generate_t1(0, 16, 0)
    gold_by_q = {q["q"]: q["gold"] for q in ep["questions"]}

    class GoldGenerative(MemorySystem):
        name = "gold-gen"
        mode = "generative"

        def ingest(self, session_id, text):
            pass

        def snapshot(self, session_id):
            return b""

        def restore(self, blob):
            pass

        def answer_context(self, question):
            raise AssertionError(
                "harness must not call answer_context in generative mode")

        def answer(self, question):
            return f"The requested value is {gold_by_q[question]}."

    res = _run_factory(lambda: GoldGenerative(), ep)
    assert res["mode"] == "generative"
    assert res["recall"] == 1.0, f"gold generative recall {res['recall']}"

    payload = _all_codes_text()

    class FloodGenerative(GoldGenerative):
        name = "flood-gen"

        def answer(self, question):
            return payload

    res = _run_factory(lambda: FloodGenerative(), ep)
    assert res["recall"] < 0.05, (
        f"generative flooding still scores {res['recall']}")


# ---------------------------------------------------------------------------

def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
