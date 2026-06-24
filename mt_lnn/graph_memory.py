"""
mt_lnn/graph_memory.py -- associative GRAPH memory for M1 (System-2 cognition).

Why this exists (and how it differs from knowledge_memory.py)
-------------------------------------------------------------
``PersistentKnowledgeMemory`` is a flat, content-addressable store: a query
returns its Top-K nearest neighbours by cosine and stops there. That is recall,
not *association* -- it cannot answer "what is reachable FROM what I just
remembered" along a chain of related facts.

This module adds the missing relational tier the M1 ("slow thinking" / System-2)
track is meant to own: a **weighted, typed knowledge graph** over the same
content-addressable nodes, plus **multi-hop spreading activation** recall. A query
seeds activation at its Top-K cosine neighbours, then propagates that activation
along edge weights for a few hops (with per-hop decay), returning nodes ranked by
*accumulated* activation. So a node that is only weakly similar to the query but
strongly connected to a strong match still surfaces -- genuine associative recall,
not a flat Top-K.

What is borrowed, and what is new (honest provenance)
-----------------------------------------------------
Borrowed from the Awareness-SDK memory graph (everest-an/Awareness-SDK):
  * typed nodes + **weighted typed edges** as the memory substrate;
  * **semantic-cosine linking** with a similarity threshold (their
    ``isSemanticallyRelated``: link two cards when cosine >= threshold, edge
    weight = the similarity).
New here (Awareness retrieves by hybrid lexical+vector+RRF, NOT by graph walk):
  * **spreading-activation / multi-hop recall** over the weighted edges -- the
    associative step a flat Top-K (and Awareness's RRF rerank) does not do.

Design discipline (same as session_consolidation.py / adapter_homeostasis.py)
-----------------------------------------------------------------------------
  * Reuses, does not duplicate -- the node vector store, cosine/center scoring and
    (de)serialisation all live in PersistentKnowledgeMemory; this file adds only
    an edges table and the graph walk. Node id == the knowledge row id (exposed
    via ``query(..., return_ids=True)``).
  * Zero model coupling -- imports only torch + stdlib + knowledge_memory.
  * Local-first -- nodes and edges live in one SQLite file, copyable/syncable as a
    unit (the "small body, large portable memory" property).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import torch

from .knowledge_memory import PersistentKnowledgeMemory, _bytes_to_obj

__all__ = ["GraphKnowledgeMemory"]


_EDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS edges (
    src_id INTEGER NOT NULL,
    dst_id INTEGER NOT NULL,
    weight REAL    NOT NULL DEFAULT 1.0,
    etype  TEXT    NOT NULL DEFAULT 'related',
    PRIMARY KEY (src_id, dst_id, etype)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
"""

_PRAGMA = "PRAGMA journal_mode=WAL;"


class GraphKnowledgeMemory:
    """Weighted knowledge graph with spreading-activation recall.

    Wraps a :class:`PersistentKnowledgeMemory` (the nodes / vector store) and adds
    an ``edges`` table in the SAME SQLite file, so a node and its relations travel
    together. Node ids are the knowledge store's row ids.

    Parameters
    ----------
    key_dim, db_path, max_entries:
        Forwarded to the underlying PersistentKnowledgeMemory. Note: when
        ``max_entries`` evicts a node, edges that point at it become dangling;
        graph queries skip dangling edges (they JOIN against existing nodes), so
        recall stays correct, but the edge rows are not garbage-collected.
    """

    def __init__(
        self,
        key_dim: int,
        db_path: str = ".mt_lnn_graph.db",
        max_entries: Optional[int] = None,
    ) -> None:
        self.nodes = PersistentKnowledgeMemory(
            key_dim=key_dim, db_path=db_path, max_entries=max_entries
        )
        self.key_dim = self.nodes.key_dim
        self.db_path = str(db_path)
        # A second connection to the same file for the edges table. SQLite
        # serialises access; both connections use WAL.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(_PRAGMA)
        self._conn.executescript(_EDGE_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def add_node(
        self,
        key: torch.Tensor,
        content: Any,
        meta: Optional[Any] = None,
        *,
        auto_link: bool = False,
        link_threshold: float = 0.55,
        link_top_k: int = 5,
        link_center: bool = False,
    ) -> int:
        """Add a content node; optionally auto-link it to similar existing nodes.

        Returns the new node id. With ``auto_link=True`` the node is connected to
        every existing node whose cosine similarity to ``key`` is >=
        ``link_threshold`` (edge weight = the similarity, ``etype='semantic'``) --
        the Awareness-style semantic linking, run BEFORE this node is itself in
        the store so it never links to itself.
        """
        # Link first (against the store WITHOUT this node), then insert, so the
        # new node cannot match itself.
        neighbours: List[Tuple[int, float]] = []
        if auto_link:
            hits = self.nodes.query(
                key, top_k=link_top_k, touch=False,
                center=link_center, return_ids=True,
            )
            neighbours = [
                (nid, float(score))
                for (nid, _content, score, _meta) in hits
                if score >= link_threshold
            ]
        node_id = self.nodes.write(key, content, meta)
        for nid, score in neighbours:
            self.link(node_id, nid, weight=score, etype="semantic")
        return node_id

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def link(
        self,
        src_id: int,
        dst_id: int,
        *,
        weight: float = 1.0,
        etype: str = "related",
        bidirectional: bool = True,
    ) -> None:
        """Create (or update) a weighted, typed edge between two nodes.

        Self-loops are ignored. ``bidirectional`` (default) adds the reverse edge
        too -- association is symmetric unless a caller models a directed relation.
        Re-linking the same (src, dst, etype) overwrites the weight.
        """
        if src_id == dst_id:
            return
        pairs = [(src_id, dst_id)]
        if bidirectional:
            pairs.append((dst_id, src_id))
        for a, b in pairs:
            self._conn.execute(
                """
                INSERT INTO edges (src_id, dst_id, weight, etype)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(src_id, dst_id, etype)
                DO UPDATE SET weight = excluded.weight
                """,
                (int(a), int(b), float(weight), str(etype)),
            )
        self._conn.commit()

    def auto_link_semantic(
        self,
        node_id: int,
        key: torch.Tensor,
        *,
        threshold: float = 0.55,
        top_k: int = 5,
        center: bool = False,
        etype: str = "semantic",
    ) -> int:
        """Link an EXISTING node to other nodes with cosine >= ``threshold``.

        Returns the number of edges created. Mirrors Awareness-SDK's
        ``isSemanticallyRelated`` linking; the edge weight is the cosine
        similarity, so stronger relations propagate more activation.
        """
        hits = self.nodes.query(
            key, top_k=top_k + 1, touch=False, center=center, return_ids=True
        )
        n_linked = 0
        for nid, _content, score, _meta in hits:
            if nid == node_id:
                continue
            if score >= threshold:
                self.link(node_id, nid, weight=float(score), etype=etype)
                n_linked += 1
        return n_linked

    # ------------------------------------------------------------------
    # Spreading-activation recall (the associative step)
    # ------------------------------------------------------------------

    def spread_activation(
        self,
        query_key: torch.Tensor,
        *,
        seeds: int = 5,
        hops: int = 2,
        decay: float = 0.5,
        top_k: int = 5,
        center: bool = False,
        touch: bool = False,
        min_activation: float = 1e-6,
    ) -> List[Tuple[Any, float, Optional[Any]]]:
        """Multi-hop associative recall by spreading activation over the graph.

        Step 1 (seed): activate the Top-``seeds`` cosine neighbours of
        ``query_key`` with activation = their (clamped) similarity. Step 2
        (spread): for ``hops`` rounds, push ``decay * activation(src) *
        edge_weight`` from each active node to its neighbours, accumulating.
        Step 3: return the Top-``top_k`` nodes by total accumulated activation as
        ``(content, activation, meta)``.

        A node only weakly similar to the query but strongly linked to a strong
        seed therefore surfaces -- the associative recall a flat Top-K cannot do.
        ``hops=0`` reduces exactly to a cosine Top-``seeds`` (no propagation), so
        the parameter directly trades flat recall against associative reach.
        """
        if seeds <= 0:
            raise ValueError(f"seeds must be positive, got {seeds}")
        if hops < 0:
            raise ValueError(f"hops must be >= 0, got {hops}")
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")

        seed_hits = self.nodes.query(
            query_key, top_k=seeds, touch=touch, center=center, return_ids=True
        )
        if not seed_hits:
            return []

        activation: Dict[int, float] = {}
        frontier: Dict[int, float] = {}
        for nid, _content, score, _meta in seed_hits:
            a = max(float(score), 0.0)        # negative cosine -> no activation
            if a <= 0.0:
                continue
            activation[nid] = activation.get(nid, 0.0) + a
            frontier[nid] = frontier.get(nid, 0.0) + a

        for _ in range(hops):
            if not frontier:
                break
            next_frontier: Dict[int, float] = {}
            for src, a_src in frontier.items():
                if a_src <= min_activation:
                    continue
                for dst, w in self._neighbours(src):
                    delta = decay * a_src * w
                    if delta <= min_activation:
                        continue
                    activation[dst] = activation.get(dst, 0.0) + delta
                    next_frontier[dst] = next_frontier.get(dst, 0.0) + delta
            frontier = next_frontier

        ranked = sorted(activation.items(), key=lambda kv: kv[1], reverse=True)
        out: List[Tuple[Any, float, Optional[Any]]] = []
        for nid, act in ranked[:top_k]:
            node = self._get_node(nid)
            if node is None:
                continue                       # evicted under max_entries
            content, meta = node
            out.append((content, float(act), meta))
        return out

    # ------------------------------------------------------------------
    # Internal graph access (reads the shared SQLite file)
    # ------------------------------------------------------------------

    def _neighbours(self, src_id: int) -> List[Tuple[int, float]]:
        """Out-edges of ``src_id`` whose destination node still exists.

        The JOIN against ``knowledge`` drops dangling edges (dst evicted under an
        LRU cap), so a graph walk never activates a deleted node.
        """
        rows = self._conn.execute(
            """
            SELECT e.dst_id, e.weight
            FROM edges e
            JOIN knowledge k ON k.id = e.dst_id
            WHERE e.src_id = ?
            """,
            (int(src_id),),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def _get_node(self, node_id: int) -> Optional[Tuple[Any, Optional[Any]]]:
        row = self._conn.execute(
            "SELECT content, meta FROM knowledge WHERE id = ?", (int(node_id),)
        ).fetchone()
        if row is None:
            return None
        content = _bytes_to_obj(row[0])
        meta = _bytes_to_obj(row[1]) if row[1] is not None else None
        return content, meta

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def n_nodes(self) -> int:
        return len(self.nodes)

    def n_edges(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
        self.nodes.close()

    def __enter__(self) -> "GraphKnowledgeMemory":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"GraphKnowledgeMemory(key_dim={self.key_dim}, db={self.db_path!r}, "
            f"nodes={self.n_nodes()}, edges={self.n_edges()})"
        )
