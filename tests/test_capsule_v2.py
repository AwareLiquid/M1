"""Test Capsule v2 schema + backward compatibility with v1 files."""

import os
import tempfile

import torch

from mt_lnn import ModelCacheStruct
from mt_lnn.capsule import (
    CAPSULE_VERSION,
    save_capsule,
    load_capsule,
    add_open_question,
    add_evidence,
)


def _make_cache(num_layers: int = 2, hidden: int = 8) -> ModelCacheStruct:
    cache = ModelCacheStruct(token_count=42)
    for _ in range(num_layers):
        h = torch.zeros(1, hidden)
        cache.layers.append((None, h, None))
    return cache


def test_capsule_v2_roundtrip(tmp_path):
    cache = _make_cache()
    cache.open_questions = []
    cache.evidence_log = []
    add_open_question(cache, "What is M-theory?")
    add_evidence(cache, source="mock:gemini", query="m-theory", fact="placeholder " * 5)

    path = tmp_path / "session.capsule"
    save_capsule(
        cache,
        str(path),
        open_questions=cache.open_questions,
        evidence_log=cache.evidence_log,
    )
    assert path.exists()

    restored = load_capsule(str(path))
    assert restored.token_count == 42
    assert len(restored.layers) == 2
    assert restored.open_questions == ["What is M-theory?"]
    assert len(restored.evidence_log) == 1
    assert restored.evidence_log[0]["source"] == "mock:gemini"
    assert restored.evidence_log[0]["query"] == "m-theory"
    # The full fact text is intentionally not stored — only its length.
    assert "fact_len" in restored.evidence_log[0]


def test_capsule_v1_backward_compat(tmp_path):
    """A v1 file (no open_questions / evidence_log) must load cleanly."""
    cache = _make_cache()
    path = tmp_path / "legacy.capsule"

    # Write a v1-shaped payload manually.
    h_states = [layer[1].cpu() for layer in cache.layers]
    legacy_payload = {
        "version": "1.0",
        "token_count": cache.token_count,
        "h_states": h_states,
    }
    torch.save(legacy_payload, str(path))

    restored = load_capsule(str(path))
    assert restored.token_count == 42
    assert restored.open_questions == []
    assert restored.evidence_log == []


def test_capsule_v2_helpers_handle_missing_attrs():
    """add_open_question / add_evidence must initialise lists lazily."""
    cache = _make_cache()
    # Deliberately do not set open_questions / evidence_log first.
    add_open_question(cache, "Q1")
    add_evidence(cache, source="s", query="q", fact="f")
    assert cache.open_questions == ["Q1"]
    assert cache.evidence_log[0]["source"] == "s"


def test_capsule_version_constant():
    assert CAPSULE_VERSION == "2.0"
