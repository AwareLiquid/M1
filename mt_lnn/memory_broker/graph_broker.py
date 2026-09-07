"""GraphBroker — MemoryBroker protocol adapter over GraphKnowledgeMemory.

Wraps :class:`mt_lnn.graph_memory.GraphKnowledgeMemory` (the weighted typed
graph with multi-hop spreading-activation recall and the SUPERSEDES /
CONTRADICTS node lifecycle — see that module's provenance header for what is
borrowed from the Awareness-SDK memory graph vs what is new here). PURE
PROTOCOL ADAPTATION: no edge type, decay, threshold, or lifecycle policy is
changed; the constructor kwargs forward untouched.

Session scoping: the graph substrate has no native sessions, so this broker
gives each ``session_id`` its OWN GraphKnowledgeMemory (one SQLite file under
``root_dir``). That is the only mapping that preserves the interface's
session-isolation contract without touching the substrate's query logic —
filtering a shared graph's results by a metadata tag after a spreading-activation
walk would silently under-fill ``top_k`` and leak cross-session activation
through the hops, which is exactly the kind of quiet semantic drift this
package exists to prevent.

Capability notes (full matrix in docs/MEMORY_BROKER.md):

* recall is MULTI-HOP: a query seeds its top cosine neighbours and propagates
  activation along weighted edges (``spread_activation``), so weakly-similar
  but strongly-connected nodes surface. ``score`` on a RecallHit is the
  accumulated activation, not a cosine.
* write AUTO-LINKS (the substrate's duck-typed write with the instance's
  auto_link policy): similar nodes gain semantic edges as they are written.
* lifecycle: ``supersede``/``auto_supersede``/``mark_contradiction`` are
  exposed pass-through via :meth:`graph` (the interface stays small; the
  lifecycle policy is the substrate's, unchanged).
* ``forget``: whole-session only (``key=None`` deletes the session's store
  file). The substrate has no node-removal API — single-binding removal is
  the parametric backend's algebraic differentiator, and faking it here by
  hiding nodes would corrupt the graph's activation topology.
* ``snapshot``/``restore`` are BIT-EXACT at the store level: the session's
  SQLite file, WAL-checkpointed, transported as raw bytes + sha256. Restore
  swaps the file back and reopens, verifying the hash.
* ``state_bytes`` is the session's SQLite file size — O(n) in content,
  stated honestly against the parametric backend's O(1).
"""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Any, Dict, List, Optional

import torch

from ..graph_memory import GraphKnowledgeMemory
from .base import BrokerError, MemoryBroker, MemoryKey, RecallHit, Snapshot

_SNAP_FORMAT = "graph-broker-v1"
_EPS = 1e-8


def _hash_vec(text: str, dim: int) -> torch.Tensor:
    """Deterministic unit vector for a string (feature hashing, no download).

    Same construction as parametric_memory._hash_vec: sha256 into ``dim``
    signed buckets, L2-normalised. Exact-match keying; semantic neighbours
    need a real encoder — inject one via ``text_encoder`` (same contract as
    :class:`mt_lnn.sentence_encoder.SentenceEncoder`)."""
    v = torch.zeros(dim, dtype=torch.float32)
    for i in range(max(dim, 32)):
        h = hashlib.sha256(f"{text}:{i}".encode()).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        v[idx] += 1.0 if h[4] & 1 else -1.0
    return v / v.norm().clamp_min(1e-6)


class GraphBroker(MemoryBroker):
    """MemoryBroker face over per-session knowledge graphs."""

    kind = "graph"

    def __init__(self, key_dim: int = 64, root_dir: str = ".mt_lnn_graph_broker",
                 text_encoder=None, **graph_kwargs) -> None:
        """``key_dim``/``text_encoder`` define the embedding space; every
        other kwarg (``auto_link``, ``link_threshold``, ``max_entries``, ...)
        forwards untouched to GraphKnowledgeMemory."""
        self.key_dim = int(key_dim)
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
        self._text_encoder = text_encoder
        self._graph_kwargs = dict(graph_kwargs)
        self._graphs: Dict[str, GraphKnowledgeMemory] = {}

    # ------------------------------------------------------------ sessions

    def _db_path(self, session_id: str) -> str:
        safe = hashlib.sha256(session_id.encode()).hexdigest()[:16]
        return os.path.join(self.root_dir, f"{safe}.db")

    def graph(self, session_id: str) -> GraphKnowledgeMemory:
        """The session's GraphKnowledgeMemory (opened lazily, kept open).

        Exposed so lifecycle calls (supersede / mark_contradiction /
        auto_supersede) reach the substrate unchanged."""
        g = self._graphs.get(session_id)
        if g is None:
            g = GraphKnowledgeMemory(key_dim=self.key_dim,
                                     db_path=self._db_path(session_id),
                                     **self._graph_kwargs)
            self._graphs[session_id] = g
        return g

    def has_session(self, session_id: str) -> bool:
        return (session_id in self._graphs
                or os.path.exists(self._db_path(session_id)))

    def _embed(self, key) -> torch.Tensor:
        if isinstance(key, bool):
            raise BrokerError("bool is not a memory key; use int token ids or str")
        if isinstance(key, int):
            # Same domain as the parametric backend: token ids ride the
            # deterministic hash space too (the graph has no vocab table).
            key = f"<tok:{key}>"
        if isinstance(key, str):
            v = (self._text_encoder(key, self.key_dim) if self._text_encoder
                 else _hash_vec(key, self.key_dim))
            v = v.to(torch.float32)
            return v / v.norm().clamp_min(_EPS)
        raise BrokerError(f"memory key must be int or str, got {type(key)!r}")

    # ------------------------------------------------------------- broker

    def write(self, session_id: str, key: MemoryKey, value, *,
              meta: Optional[Any] = None) -> str:
        try:
            node_id = self.graph(session_id).write(
                self._embed(key), value, meta)
        except BrokerError:
            raise
        except Exception as exc:   # sqlite/storage layer — wrap per contract
            raise BrokerError(f"graph write failed: {exc}") from exc
        return str(node_id)

    def recall(self, session_id: str, query: MemoryQuery,
               top_k: int = 5) -> List[RecallHit]:
        if not self.has_session(session_id):
            return []
        try:
            hits = self.graph(session_id).spread_activation(
                self._embed(query), top_k=top_k)
        except Exception as exc:
            raise BrokerError(f"graph recall failed: {exc}") from exc
        return [RecallHit(value=content, score=float(act), meta=meta)
                for content, act, meta in hits]

    def forget(self, session_id: str, key: Optional[MemoryKey] = None) -> bool:
        if not self.has_session(session_id):
            return False
        if key is not None:
            raise NotImplementedError(
                "the graph backend supports whole-session forget only: the "
                "substrate has no node-removal API, and hiding nodes would "
                "corrupt spreading-activation topology (docs/MEMORY_BROKER.md)")
        self.graph(session_id).close()
        self._graphs.pop(session_id, None)
        try:
            os.remove(self._db_path(session_id))
        except FileNotFoundError:
            pass
        return True

    def snapshot(self, session_id: str) -> Snapshot:
        """The session's SQLite file, checkpointed, as bytes + sha256.

        Bit-exact: the memory body IS the file; raw bytes carry it. The open
        graph is closed first (WAL checkpoint into the main db file) and
        reopened fresh from the same path afterwards."""
        if not self.has_session(session_id):
            raise KeyError(f"no session {session_id!r} to snapshot")
        g = self._graphs.pop(session_id, None)
        if g is not None:
            g.close()   # WAL checkpoint lands in the main db file on close
        try:
            with open(self._db_path(session_id), "rb") as f:
                raw = f.read()
        except OSError as exc:
            raise BrokerError(f"graph snapshot failed: {exc}") from exc
        self.graph(session_id)   # reopen from the same file
        return {
            "format": _SNAP_FORMAT,
            "key_dim": self.key_dim,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "db_bytes_b64": base64.b64encode(raw).decode("ascii"),
        }

    def restore(self, session_id: str, snap: Snapshot) -> None:
        if snap.get("format") != _SNAP_FORMAT:
            raise ValueError(f"unknown snapshot format {snap.get('format')!r}")
        raw = base64.b64decode(snap["db_bytes_b64"])
        if hashlib.sha256(raw).hexdigest() != snap["sha256"]:
            raise BrokerError("snapshot bytes fail sha256 verification")
        g = self._graphs.pop(session_id, None)
        if g is not None:
            g.close()
        path = self._db_path(session_id)
        try:
            with open(path, "wb") as f:
                f.write(raw)
        except OSError as exc:
            raise BrokerError(f"graph restore failed: {exc}") from exc
        self.graph(session_id)   # reopen from the restored file

    def state_bytes(self, session_id: str) -> int:
        """Size of the session's memory body on disk: the SQLite file plus
        its (un-checkpointed) WAL — the honest O(n) number for this backend."""
        total = 0
        for suffix in ("", "-wal"):
            try:
                total += os.path.getsize(self._db_path(session_id) + suffix)
            except OSError:
                pass
        return total

    def close(self) -> None:
        for g in self._graphs.values():
            g.close()
        self._graphs.clear()
