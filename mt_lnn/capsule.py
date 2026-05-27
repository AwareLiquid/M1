"""
capsule.py — State persistence module for MT-LNN (AwareLiquid).

Capsule v2 (2026-05-28) extends v1 with a small structured "thinking
context" alongside the raw recurrent state:

    belief_state    — the O(1) h_prev tensors (the v1 payload)
    open_questions  — sub-questions the model could not yet resolve
                      (pushed when the entropy router intercepts)
    evidence_log    — provenance trail for cloud-injected facts
                      (source, query, timestamp, byte length)

The intent is to make a capsule a "resumable thinking session" instead of
just a memory snapshot. open_questions / evidence_log are intentionally
small lists (target capsule footprint ≤ 5 KB).

Backward compatibility: load_capsule() transparently upgrades v1 files by
filling open_questions / evidence_log with empty lists.
"""

import os
import time
from typing import Dict, List, Optional

import torch

from .model import ModelCacheStruct

CAPSULE_VERSION = "2.0"


def _h_states_from_cache(cache: ModelCacheStruct) -> List[Optional[torch.Tensor]]:
    h_states: List[Optional[torch.Tensor]] = []
    for layer_cache in cache.layers:
        h_prev = layer_cache[1] if (layer_cache is not None and len(layer_cache) > 1) else None
        h_states.append(h_prev.cpu() if h_prev is not None else None)
    return h_states


def save_capsule(
    cache: ModelCacheStruct,
    filepath: str,
    *,
    open_questions: Optional[List[str]] = None,
    evidence_log: Optional[List[Dict]] = None,
) -> None:
    """Serialize recurrent state + thinking context to ``filepath``.

    Parameters
    ----------
    cache : ModelCacheStruct holding the current h_prev per layer.
    filepath : destination path (e.g. ``session_alice.capsule``).
    open_questions : sub-questions unresolved at save time. Defaults to [].
    evidence_log : list of cloud-fact provenance dicts. Defaults to [].
    """
    capsule_data = {
        "version": CAPSULE_VERSION,
        "token_count": cache.token_count,
        "h_states": _h_states_from_cache(cache),
        "open_questions": list(open_questions or []),
        "evidence_log": list(evidence_log or []),
    }
    torch.save(capsule_data, filepath)

    num_layers = len([h for h in capsule_data["h_states"] if h is not None])
    file_size_kb = os.path.getsize(filepath) / 1024
    print(
        f"[State Capsule v{CAPSULE_VERSION}] Crystallized {num_layers} layers, "
        f"{len(capsule_data['open_questions'])} open Q, "
        f"{len(capsule_data['evidence_log'])} evidence rows → "
        f"{filepath} ({file_size_kb:.1f} KB)."
    )


def load_capsule(filepath: str, device: str = "cpu") -> ModelCacheStruct:
    """Load a capsule (v1 or v2) and return a seeded ModelCacheStruct.

    For v2 capsules, ``open_questions`` and ``evidence_log`` are attached
    as attributes on the returned cache so callers can read them without a
    second file load. v1 capsules upgrade transparently with empty lists.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Capsule file not found: {filepath}")

    capsule_data = torch.load(filepath, map_location=device, weights_only=False)
    version = capsule_data.get("version", "1.0")
    h_states = capsule_data.get("h_states", [])
    token_count = capsule_data.get("token_count", 0)
    open_questions = list(capsule_data.get("open_questions", []))
    evidence_log = list(capsule_data.get("evidence_log", []))

    new_cache = ModelCacheStruct(token_count=token_count)
    for h in h_states:
        h_tensor = h.to(device) if h is not None else None
        new_cache.layers.append((None, h_tensor, None))

    # Attach thinking context for downstream consumers (router, demo, UI).
    new_cache.open_questions = open_questions
    new_cache.evidence_log = evidence_log

    print(
        f"[State Capsule] Inherited v{version} capsule: {token_count} tokens, "
        f"{len(open_questions)} open Q, {len(evidence_log)} evidence rows."
    )
    return new_cache


def add_open_question(cache: ModelCacheStruct, question: str) -> None:
    """Push an unresolved sub-question onto the capsule context."""
    questions: List[str] = getattr(cache, "open_questions", None) or []
    questions.append(question)
    cache.open_questions = questions


def add_evidence(
    cache: ModelCacheStruct,
    *,
    source: str,
    query: str,
    fact: str,
) -> None:
    """Append a cloud-fact provenance row onto the capsule context."""
    log: List[Dict] = getattr(cache, "evidence_log", None) or []
    log.append(
        {
            "source": source,
            "query": query,
            "fact_len": len(fact),
            "ts": time.time(),
        }
    )
    cache.evidence_log = log
