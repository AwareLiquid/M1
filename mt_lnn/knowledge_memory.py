"""
knowledge_memory.py — persistent long-term DECLARATIVE memory for MT-LNN.

Why this exists (and how it differs from memory.py)
----------------------------------------------------
``memory.py / SessionMemory`` persists the recurrent *hidden state* (``h_prev``)
of a session — that is **working memory**: small, fast, position-tied, not
content-addressable. You cannot ask it "what do I know about X".

This module is the complementary tier: a **long-term declarative knowledge
store**. The brain does not cram every fact into synaptic weights; it keeps a
large, slow, content-addressable store (hippocampal/neocortical declarative
memory) that is queried *by similarity* and pulled in on demand. That is exactly
the right division of labour for an O(1)-state edge model:

    working memory  (SessionMemory)  → live recurrent state, tiny, real-time
    procedural memory (model weights) → skills, fixed by training
    declarative memory (THIS module)  → knowledge, large, queried on demand

Design goals
------------
* **Zero coupling.** Imports only ``torch`` + stdlib ``sqlite3``. It never
  imports ``model.py`` and ``model.py`` never imports it. It attaches to an
  agent/pipeline through a tiny, explicit interface — no edit to the core model
  is required to use it.
* **Local-first / privacy.** Everything lives in a single SQLite file on the
  device. Nothing leaves the machine. The same file can be copied/synced across
  ends so a small edge model keeps a big shared knowledge base without growing
  its parameter count.
* **Bounded footprint.** An optional ``max_entries`` cap evicts the
  least-recently-used record so the store stays small and predictable on an
  edge device ("small body + large memory, pulled on demand").

Interface
---------
>>> from mt_lnn.knowledge_memory import PersistentKnowledgeMemory
>>> kb = PersistentKnowledgeMemory(key_dim=64, db_path=":memory:")
>>> kb.write(key_vec, "Paris is the capital of France")
>>> hits = kb.query(query_vec, top_k=3)      # [(content, score, meta), ...]
>>> best = kb.recall(query_vec)              # content of the top hit, or None

Keys are L2-normalised on write and query, so ``query`` ranks by cosine
similarity. Contents are arbitrary picklable Python objects (strings, dicts,
small tensors) stored via ``torch.save``.
"""

from __future__ import annotations

import io
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key_vec     BLOB    NOT NULL,    -- float32 L2-normalised key, length = key_dim
    content     BLOB    NOT NULL,    -- torch.save(content) bytes
    meta        BLOB,                -- torch.save(meta) bytes or NULL
    created_at  TEXT    NOT NULL,
    accessed_at TEXT    NOT NULL,    -- ISO time of last access (informational)
    access_seq  INTEGER NOT NULL     -- monotonic recency counter; the LRU sort key
);
"""

_PRAGMA = "PRAGMA journal_mode=WAL;"


# ---------------------------------------------------------------------------
# (de)serialisation helpers
# ---------------------------------------------------------------------------

def _obj_to_bytes(obj: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _bytes_to_obj(data: bytes) -> Any:
    buf = io.BytesIO(data)
    # weights_only=True: everything this module stores is plain data — str,
    # dict/list/tuple of primitives, None, and torch tensors — all on the
    # weights-only unpickler's safe allowlist. The module docstring encourages
    # copying/syncing the db file across devices, so its bytes must be treated
    # as untrusted input: weights_only=False would be an arbitrary-code-
    # execution surface (pickle) on every query. Custom classes are therefore
    # NOT supported as content; store primitives/tensors (or pre-serialise).
    return torch.load(buf, weights_only=True, map_location="cpu")


def _key_to_bytes(vec: torch.Tensor) -> bytes:
    return vec.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()


def _bytes_to_key(data: bytes, key_dim: int) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.float32).view(key_dim)


def _normalise(vec: torch.Tensor) -> torch.Tensor:
    """L2-normalise a 1-D key so dot products are cosine similarities."""
    return F.normalize(vec.detach().to(torch.float32).flatten(), dim=0, eps=1e-8)


# ---------------------------------------------------------------------------
# PersistentKnowledgeMemory
# ---------------------------------------------------------------------------

class PersistentKnowledgeMemory:
    """SQLite-backed, content-addressable long-term knowledge store.

    Parameters
    ----------
    key_dim:
        Dimensionality of the key (embedding) vectors. All writes/queries must
        match this — a mismatch raises ``ValueError`` rather than silently
        corrupting the index.
    db_path:
        Path to the SQLite file. ``":memory:"`` gives an ephemeral in-process
        store (handy for tests). A real path persists across processes/restarts
        and can be synced across devices.
    max_entries:
        Optional cap on the number of stored records. When exceeded, the
        least-recently-accessed record is evicted (LRU). ``None`` = unbounded.
    """

    def __init__(
        self,
        key_dim: int,
        db_path: Union[str, Path] = ".mt_lnn_knowledge.db",
        max_entries: Optional[int] = None,
    ):
        if key_dim <= 0:
            raise ValueError(f"key_dim must be positive, got {key_dim}")
        if max_entries is not None and max_entries <= 0:
            raise ValueError(f"max_entries must be positive or None, got {max_entries}")

        self.key_dim = int(key_dim)
        self.db_path = str(db_path)
        self.max_entries = max_entries

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(_PRAGMA)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        # key_dim guard: a db file created with one embedding dimension must
        # never be reopened with another (e.g. after swapping the base model
        # while reusing the same KB_PATH). Without this, every query crashes on
        # `view(key_dim)` deserialisation, and worse, new writes silently mix
        # dimensions and permanently poison the store. The dimension is pinned
        # in a tiny kb_meta table at creation time and checked on every open.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kb_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        row = self._conn.execute(
            "SELECT value FROM kb_meta WHERE key = 'key_dim'"
        ).fetchone()
        if row is not None:
            stored_dim = int(row[0])
        else:
            # Legacy db without kb_meta (or a brand-new db): infer the true
            # dimension from any existing row (migration), else pin the
            # requested key_dim for a fresh store.
            first = self._conn.execute(
                "SELECT key_vec FROM knowledge LIMIT 1"
            ).fetchone()
            stored_dim = len(first[0]) // 4 if first is not None else self.key_dim
            self._conn.execute(
                "INSERT OR REPLACE INTO kb_meta (key, value) VALUES ('key_dim', ?)",
                (str(stored_dim),),
            )
            self._conn.commit()
        if stored_dim != self.key_dim:
            self._conn.close()
            raise ValueError(
                f"key_dim mismatch: {self.db_path!r} stores {stored_dim}-d keys "
                f"but was opened with key_dim={self.key_dim}. This usually means "
                f"the base/embedding model changed while reusing the same db "
                f"path. Point this store at a fresh db file (or delete the old "
                f"one) — mixing key dimensions would corrupt the index."
            )

        # Monotonic recency counter used as the LRU sort key. Resolution-
        # independent (a wall-clock timestamp can collide when many ops land in
        # the same microsecond, which silently breaks LRU ordering). Seed it from
        # the persisted max so recency survives reopen.
        row = self._conn.execute("SELECT MAX(access_seq) FROM knowledge").fetchone()
        self._seq = int(row[0]) if row and row[0] is not None else 0
        # One shared sqlite3.Connection (check_same_thread=False) + the _seq
        # counter are touched by every request; FastAPI runs the sync serve
        # endpoints on an anyio threadpool (concurrent). Serialize all access
        # behind an RLock (reentrant: write()->_enforce_capacity(),
        # query(touch=True) both re-enter). Prevents interleaved sqlite ops and
        # the non-atomic `self._seq += 1` read-modify-write.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        key: torch.Tensor,
        content: Any,
        meta: Optional[Any] = None,
    ) -> int:
        """Store *content* indexed by the embedding *key*.

        Returns the row id of the new record. After insertion the LRU cap (if
        any) is enforced, so a write may evict the oldest-accessed record.
        """
        key = self._validate_key(key)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._seq += 1
            cur = self._conn.execute(
                """
                INSERT INTO knowledge (key_vec, content, meta, created_at, accessed_at, access_seq)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    _key_to_bytes(key),
                    _obj_to_bytes(content),
                    _obj_to_bytes(meta) if meta is not None else None,
                    now,
                    now,
                    self._seq,
                ),
            )
            self._conn.commit()
            row_id = int(cur.lastrowid)
            self._enforce_capacity()
            return row_id

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        key: torch.Tensor,
        top_k: int = 5,
        touch: bool = True,
        center: bool = False,
        return_ids: bool = False,
    ) -> List[Tuple]:
        """Return the *top_k* records most similar to *key* by cosine similarity.

        Each hit is ``(content, score, meta)`` with ``score`` in [-1, 1], sorted
        descending (or ``(id, content, score, meta)`` when ``return_ids=True``).
        Returns an empty list when the store is empty.

        Parameters
        ----------
        return_ids:
            When True, each hit is prefixed with the record's integer row id
            ``(id, content, score, meta)`` so a caller (e.g. a graph layer) can
            map a recalled record back to a stable node id. Default False
            preserves the 3-tuple ``(content, score, meta)`` contract.
        touch:
            When True (default), the matched records' ``accessed_at`` is bumped so
            the LRU eviction policy treats recalled knowledge as "fresh". Pass
            False for a read-only peek that does not influence eviction.
        center:
            Anisotropy-robust scoring. Keys produced by mean-pooling a language
            model's hidden states share a dominant direction, which inflates and
            compresses every cosine score into a narrow high band -- so plain
            cosine makes one record a near-universal nearest neighbour and barely
            discriminates queries. When True, subtract the corpus mean of the
            stored keys from both the stored keys and the query before scoring
            (the "all-but-the-top" post-processing of Mu & Viswanath 2018), then
            renormalise. This cancels the shared direction so scores spread out
            and rank by genuine semantic content. Default False preserves the
            plain-cosine behaviour. Recommended True for LM-encoded keys.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        q = self._validate_key(key)

        with self._lock:
            rows = self._conn.execute(
                "SELECT id, key_vec, content, meta FROM knowledge"
            ).fetchall()
        if not rows:
            return []

        # Stack all keys into one matrix and score in a single mat-vec. The store
        # is meant for edge-scale knowledge bases (thousands, not billions), so a
        # brute-force cosine scan is the right, dependency-free choice.
        keys = torch.stack(
            [_bytes_to_key(r[1], self.key_dim) for r in rows], dim=0
        )                                                   # (N, key_dim), already unit-norm
        # Stored keys live on CPU (SQLite round-trip); the caller's query may
        # arrive on any device. Align BEFORE the centered-cosine math below,
        # otherwise `keys @ q` raises a CPU/CUDA device mismatch (observed on
        # GPU serve hosts in the /v1/sleep build_graph path).
        q = q.to(keys.device)
        if center and keys.shape[0] >= 2:
            # Remove the shared dominant direction (corpus mean) from keys AND
            # query, then renormalise so the scores are cosines on the residual
            # (semantic) component. Computed from the stored keys only, so the
            # query is projected against the same offset.
            mu = keys.mean(dim=0, keepdim=True)             # (1, key_dim)
            keys = F.normalize(keys - mu, dim=-1)
            q = F.normalize(q - mu.squeeze(0), dim=0)
        scores = keys @ q                                   # (N,) cosine sims

        k = min(top_k, scores.shape[0])
        top_scores, top_idx = torch.topk(scores, k)

        hits: List[Tuple[Any, float, Optional[Any]]] = []
        touched_ids: List[int] = []
        for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
            row = rows[idx]
            content = _bytes_to_obj(row[2])
            meta = _bytes_to_obj(row[3]) if row[3] is not None else None
            rid = int(row[0])
            if return_ids:
                hits.append((rid, content, float(score), meta))
            else:
                hits.append((content, float(score), meta))
            touched_ids.append(rid)

        if touch and touched_ids:
            now = datetime.now(timezone.utc).isoformat()
            # Bump the recency counter for each hit so they become the freshest
            # records for LRU purposes (counter, not wall-clock → collision-free).
            with self._lock:
                updates = []
                for rid in touched_ids:
                    self._seq += 1
                    updates.append((now, self._seq, rid))
                self._conn.executemany(
                    "UPDATE knowledge SET accessed_at = ?, access_seq = ? WHERE id = ?",
                    updates,
                )
                self._conn.commit()

        return hits

    def recall(
        self, key: torch.Tensor, touch: bool = True, center: bool = False
    ) -> Optional[Any]:
        """Convenience: the content of the single best match, or ``None`` if the
        store is empty. See :meth:`query` for the ``center`` flag."""
        hits = self.query(key, top_k=1, touch=touch, center=center)
        return hits[0][0] if hits else None

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0])

    def clear(self) -> None:
        """Delete all stored knowledge."""
        with self._lock:
            self._conn.execute("DELETE FROM knowledge")
            self._conn.commit()

    def delete(self, ids) -> int:
        """Delete records by row id. Returns the count requested. Enables true
        per-record consolidation/UPSERT (previously a documented follow-on:
        only LRU eviction could remove records)."""
        ids = [int(i) for i in ids]
        if not ids:
            return 0
        with self._lock:
            self._conn.executemany("DELETE FROM knowledge WHERE id = ?",
                                   [(i,) for i in ids])
            self._conn.commit()
        return len(ids)

    def all_meta(self):
        """Return ``[(id, meta_dict_or_None), ...]`` for every record — for
        consolidation policies that RANK by a meta field (e.g. surprise)
        without needing a query key."""
        with self._lock:
            rows = self._conn.execute("SELECT id, meta FROM knowledge").fetchall()
        return [(int(r[0]), _bytes_to_obj(r[1]) if r[1] is not None else None)
                for r in rows]

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_key(self, key: torch.Tensor) -> torch.Tensor:
        vec = key.detach().to(torch.float32).flatten()
        if vec.numel() != self.key_dim:
            raise ValueError(
                f"key has {vec.numel()} elements, expected key_dim={self.key_dim}"
            )
        return _normalise(vec)

    def _enforce_capacity(self) -> None:
        """Evict least-recently-accessed records until within ``max_entries``."""
        if self.max_entries is None:
            return
        n = len(self)
        if n <= self.max_entries:
            return
        overflow = n - self.max_entries
        # Lowest recency counter first → least-recently-used eviction.
        self._conn.execute(
            """
            DELETE FROM knowledge WHERE id IN (
                SELECT id FROM knowledge ORDER BY access_seq ASC, id ASC LIMIT ?
            )
            """,
            (overflow,),
        )
        self._conn.commit()

    # Context-manager support
    def __enter__(self) -> "PersistentKnowledgeMemory":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"PersistentKnowledgeMemory(key_dim={self.key_dim}, "
            f"db={self.db_path!r}, entries={len(self)}, max_entries={self.max_entries})"
        )
