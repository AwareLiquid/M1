"""
memory.py — SQLite-backed persistent recurrent state for MT-LNN.

Persists only the LNN recurrent h_prev tensors (one per layer) across
processes.  KV caches are NOT persisted: they are position-tied to a
specific token sequence and must be rebuilt from the token history.

Storage layout (SQLite, default: .mt_lnn_state.db in the working dir):

  sessions(
    session_id   TEXT PRIMARY KEY,
    token_count  INTEGER,          -- cumulative tokens seen in this session
    updated_at   TEXT,             -- ISO-8601 timestamp
    h_states     BLOB              -- torch.save([h0, h1, ...]) binary
  )

Quick-start
-----------
>>> from mt_lnn.memory import SessionMemory
>>> mem = SessionMemory()                    # creates / opens .mt_lnn_state.db
>>> mem.save("chat_001", cache, token_count=256)
>>> h_states = mem.load("chat_001")          # list[Tensor | None], len = n_layers
>>> # Restore into a fresh cache before inference:
>>> cache = mem.restore_cache(h_states, device="cpu")
"""

from __future__ import annotations

import io
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import torch

# Lazy import to avoid circular dependency at module level.
# model.py imports memory.py; use TYPE_CHECKING guard for type hints only.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model import ModelCacheStruct

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT    PRIMARY KEY,
    token_count  INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT    NOT NULL,
    h_states     BLOB    NOT NULL
);
"""

_PRAGMA = "PRAGMA journal_mode=WAL;"


# ---------------------------------------------------------------------------
# Tensor serialisation helpers
# ---------------------------------------------------------------------------

def _tensors_to_bytes(tensors: List[Optional[torch.Tensor]]) -> bytes:
    """Serialize a list of optional tensors to bytes via torch.save."""
    buf = io.BytesIO()
    # Convert to CPU before storing (device is re-applied at load time).
    cpu_tensors = [t.cpu() if t is not None else None for t in tensors]
    torch.save(cpu_tensors, buf)
    return buf.getvalue()


def _bytes_to_tensors(data: bytes) -> List[Optional[torch.Tensor]]:
    """Deserialize bytes back to a list of optional tensors."""
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True, map_location="cpu")


# ---------------------------------------------------------------------------
# SessionMemory
# ---------------------------------------------------------------------------

class SessionMemory:
    """Local-first SQLite store for MT-LNN recurrent states.

    This class is intended for demos, notebooks, and single-node experiments.
    It is not a high-QPS distributed state service. Production deployments
    should put a queue, connection pool, or external store in front of the
    recurrent-state persistence path.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created if absent.
        Pass ``":memory:"`` for an in-process ephemeral store (testing).
    """

    def __init__(self, db_path: Union[str, Path] = ".mt_lnn_state.db"):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(_PRAGMA)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # One shared sqlite3.Connection (check_same_thread=False) may be hit
        # from multiple threads (e.g. a threaded server persisting several
        # sessions). Serialize all execute/commit behind an RLock — the same
        # pattern PersistentKnowledgeMemory uses — to prevent interleaved
        # statements on the shared connection.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        session_id: str,
        cache: "ModelCacheStruct",
        token_count: int = 0,
    ) -> None:
        """Persist the h_prev states from *cache* under *session_id*.

        The h_states list has length ``n_layers``; each element is a tensor
        of shape ``(B, P, d_proto)`` or ``None`` if the layer has no state.

        Parameters
        ----------
        session_id:
            Unique identifier for the conversation / session.
        cache:
            The ``ModelCacheStruct`` returned by a ``use_cache=True`` forward
            pass.  Only the ``[layer][1]`` (h_prev) slot is stored.
        token_count:
            Cumulative token count so far in this session.  Used for bookkeeping
            only; does not affect recall.
        """
        h_states: List[Optional[torch.Tensor]] = []
        for layer_cache in cache.layers:
            if layer_cache is not None and len(layer_cache) > 1:
                h_states.append(layer_cache[1])  # slot 1 = h_prev
            else:
                h_states.append(None)

        blob = _tensors_to_bytes(h_states)
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (session_id, token_count, updated_at, h_states)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    token_count = excluded.token_count,
                    updated_at  = excluded.updated_at,
                    h_states    = excluded.h_states
                """,
                (session_id, token_count, now, blob),
            )
            self._conn.commit()

    def load(
        self,
        session_id: str,
        device: Union[str, torch.device] = "cpu",
    ) -> Optional[List[Optional[torch.Tensor]]]:
        """Load persisted h_prev tensors for *session_id*.

        Returns
        -------
        list[Tensor | None] of length n_layers, or ``None`` if the session
        does not exist in the store.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT h_states FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        tensors = _bytes_to_tensors(row[0])
        dev = torch.device(device)
        return [t.to(dev) if t is not None else None for t in tensors]

    def restore_cache(
        self,
        h_states: List[Optional[torch.Tensor]],
    ) -> "ModelCacheStruct":
        """Build a minimal ``ModelCacheStruct`` that seeds h_prev for each layer.

        The KV and GWTB-KV slots are left ``None`` — the model rebuilds them
        on the first cached forward pass.

        Parameters
        ----------
        h_states:
            List returned by :meth:`load`.
        """
        # Import here to avoid circular imports at module load time.
        from .model import ModelCacheStruct

        cache = ModelCacheStruct()
        for h in h_states:
            # LayerCache = (attn_kv, h_prev, gwtb_kv)
            cache.layers.append((None, h, None))
        return cache

    def session_info(self, session_id: str) -> Optional[dict]:
        """Return metadata for *session_id* without loading tensors.

        Returns ``None`` if the session does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id, token_count, updated_at FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {"session_id": row[0], "token_count": row[1], "updated_at": row[2]}

    def list_sessions(self) -> List[dict]:
        """Return metadata for all stored sessions, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, token_count, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [{"session_id": r[0], "token_count": r[1], "updated_at": r[2]} for r in rows]

    def delete(self, session_id: str) -> bool:
        """Delete a stored session.  Returns True if the session existed."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    # Context-manager support
    def __enter__(self) -> "SessionMemory":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return f"SessionMemory(db={self.db_path!r}, sessions={n})"
