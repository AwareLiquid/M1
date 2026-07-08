"""FastWeightSessionStore — durable, content-addressed fast-weight memory.

The bridge's slow/durable half. :func:`snapshot_adapter_streams` turns the
volatile (F, z) fast-weight into a persistable dict; this store keeps those
snapshots on disk keyed by a SENTENCE-embedding of the session's text, so a
later session can recall the right one by content and
:func:`restore_adapter_streams` it back into a fresh model. That is the
fast->slow (hippocampus->durable) transfer the consolidation stack never had a
source for (see the scout map).

Design decisions, each tracing to a scout-flagged failure mode:

* KEY IS DECOUPLED FROM PAYLOAD. The retrieval key is an e5 sentence
  embedding of the session TEXT (384-d), NOT a pooled (F, z) — pooling the
  associative matrix to make it addressable is lossy and collides. The whole
  (F, z)-per-adapter snapshot rides as the KB *content* (torch.save'd), fully
  intact.
* DEDICATED DB / KEY_DIM. PersistentKnowledgeMemory assumes one key_dim per
  file; mixing 384-d session keys with the conversation/KB stores' keys raises
  on write. This store owns its own db_path (default key_dim=384 for e5-small).
* CENTERED COSINE. LM/sentence keys share a dominant direction that compresses
  cosine into a narrow high band; ``center=True`` (Mu & Viswanath all-but-the-
  top) is the default so recall discriminates sessions instead of returning one
  near-universal nearest neighbour. The wrong-session control in
  ``benchmarks/cross_session_recall.py`` exists to catch it if this is too weak.

Append-only with an LRU capacity cap: correct for DISTINCT sessions (different
conversations -> different keys), which is the cross-session product case and
the C3 collision test. Per-session UPSERT (dropping a session's stale snapshot
when it is re-written) needs a KB delete-by-id API the store deliberately does
not reach into; it is a documented follow-on. Until then, capacity eviction
bounds stale-snapshot growth and recall returns the best content match.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import torch

from .knowledge_memory import PersistentKnowledgeMemory

E5_SMALL_DIM = 384


class FastWeightSessionStore:
    """Durable store mapping a session-text key -> a fast-weight snapshot."""

    def __init__(
        self,
        db_path: str = ".mt_lnn_fastweight.db",
        key_dim: int = E5_SMALL_DIM,
        max_entries: Optional[int] = 2000,
    ):
        self.kb = PersistentKnowledgeMemory(
            key_dim=key_dim, db_path=db_path, max_entries=max_entries
        )

    def write_session(
        self,
        session_id: str,
        key_vec: torch.Tensor,
        fw_snapshot: dict,
        meta: Optional[dict] = None,
    ) -> int:
        """Persist *fw_snapshot* (from snapshot_adapter_streams) under a
        content key. Returns the row id. The snapshot is stored verbatim as
        the KB content; session_id rides in both content and meta."""
        content = {"session_id": session_id, "snapshot": fw_snapshot}
        full_meta = {"session_id": session_id, **(meta or {})}
        return self.kb.write(key_vec, content=content, meta=full_meta)

    def recall_session(
        self,
        query_vec: torch.Tensor,
        top_k: int = 1,
        center: bool = True,
        score_floor: Optional[float] = None,
        expected_session_id: Optional[str] = None,
    ) -> List[Tuple[dict, float, Optional[str], Any]]:
        """Return the best-matching sessions as
        ``(fw_snapshot, score, session_id, meta)`` tuples, best first.

        ``score_floor`` (cosine in [-1, 1]) suppresses weak matches so an
        unrelated new session does not spuriously restore stale state (None =
        return the top hit unconditionally; set a floor in production so "no
        relevant memory" cleanly yields nothing).

        ``expected_session_id`` filters to that session so an EVICTED or absent
        session yields ``[]`` instead of a foreign snapshot — without it, a
        caller resuming a specific session that has been LRU-evicted would
        silently receive some other session's state (review finding [3]).

        Ties are broken toward the NEWEST row (append-only + an exact-id key
        produces several cosine-1.0 rows for a re-written session; the latest
        write must win, not the oldest — review finding [2])."""
        if len(self.kb) == 0:
            return []
        # Query a generous window so all exact-id tie duplicates are visible,
        # then re-rank (score desc, row-id desc) so recency breaks ties.
        window = min(len(self.kb), max(top_k * 8, 32))
        hits = self.kb.query(query_vec, top_k=window, center=center,
                             return_ids=True)                # (rid, content, score, meta)
        hits.sort(key=lambda h: (h[2], h[0]), reverse=True)
        out: List[Tuple[dict, float, Optional[str], Any]] = []
        for rid, content, score, meta in hits:
            if score_floor is not None and score < score_floor:
                continue
            sid = content.get("session_id")
            if expected_session_id is not None and sid != expected_session_id:
                continue
            out.append((content["snapshot"], float(score), sid, meta))
            if len(out) >= top_k:
                break
        return out

    def __len__(self) -> int:
        return len(self.kb)

    def clear(self) -> None:
        self.kb.clear()

    def close(self) -> None:
        """Close the underlying SQLite connection (release the db file)."""
        conn = getattr(self.kb, "_conn", None)
        if conn is not None:
            conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def id_key(session_id: str, dim: int) -> torch.Tensor:
    """Deterministic unit vector for an EXACT-id lookup (no encoder needed).

    Maps a session_id string to a fixed, well-spread direction so recalling
    with the same id yields cosine 1.0 (an oracle-by-id retrieval that reuses
    the content-addressed store unchanged). Distinct ids map to near-orthogonal
    directions in high dim, so id and e5-text keys can share a db with a
    score_floor separating exact-id hits from semantic ones.

    NOTE: append-only + this exact-id key means N re-writes of the same session
    produce N cosine-1.0 rows. ``recall_session`` breaks the tie toward the
    newest row, so RESUME correctly returns the latest snapshot; the remaining
    cost is storage bloat from stale duplicates, bounded by the store's
    ``max_entries`` LRU. True per-session UPSERT (deleting the stale rows on
    rewrite) still needs a KB delete-by-id API and is a documented follow-on.
    """
    import hashlib

    v = torch.zeros(dim, dtype=torch.float32)
    # Hash the id into `dim` buckets with signed contributions -> a stable,
    # spread-out direction (a tiny feature-hash, deterministic across processes).
    for i in range(max(dim, 32)):
        h = hashlib.sha256(f"{session_id}:{i}".encode()).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        v[idx] += sign
    return v / v.norm().clamp_min(1e-6)


def build_session_key(text: str, encode_fn: Callable[[str], torch.Tensor],
                      max_chars: int = 4000) -> torch.Tensor:
    """Retrieval key from a DEDICATED sentence encoder over the session text.

    ``encode_fn`` is typically ``SentenceEncoder(...).as_fn(is_query=...)`` —
    kept as an injected callable so the store is testable with a deterministic
    fake encoder (no model download) and so serve/CPU/GPU all share one path.
    Text is truncated to bound encoder cost on long sessions."""
    return encode_fn((text or " ")[:max_chars])
