import json
from pathlib import Path

from scripts.bench_cloud_inject_uplift import EchoBackend, hit, run


def test_hit_normalizes_punctuation_and_case():
    assert hit("Witten", "It was witten, in 1995.")
    assert hit("Tokyo", "  TOKYO!!  ")
    assert not hit("Saturn", "Jupiter has many moons")


def test_echo_backend_returns_fact_only_when_injected():
    b = EchoBackend()
    assert b.generate("Question: who?") == "I do not know."
    out = b.generate("[Absorbed fact] The answer is X.\nContinuing: Question: who?")
    assert out == "The answer is X."


def test_run_uplift_with_echo_backend(tmp_path):
    qs = [
        {"id": "a", "question": "who?", "answer": "Alice", "fact": "Alice is the one.", "topic": "a"},
        {"id": "b", "question": "what?", "answer": "Bob", "fact": "Bob is the one.", "topic": "b"},
    ]
    r = run(qs, EchoBackend())
    assert r["n_questions"] == 2
    assert r["no_inject_accuracy"] == 0.0
    assert r["inject_accuracy"] == 1.0
    assert r["uplift_abs"] == 1.0
    assert r["uplift_rel"] is None


def test_bundled_questions_file_is_well_formed():
    p = Path("benchmarks/cloud_inject_questions.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data) >= 20
    for row in data:
        for k in ("id", "question", "answer", "fact", "topic"):
            assert k in row, f"missing {k} in {row.get('id')}"
        assert row["answer"]
        assert row["fact"]
