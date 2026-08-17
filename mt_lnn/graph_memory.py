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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .knowledge_memory import (
    PersistentKnowledgeMemory,
    _bytes_to_key,
    _bytes_to_obj,
)

__all__ = ["GraphKnowledgeMemory", "calibrate_threshold_from_keys"]


# ---------------------------------------------------------------------------
# Adaptive link threshold (anisotropy-robust, encoder-independent)
# ---------------------------------------------------------------------------

def calibrate_threshold_from_keys(
    keys: torch.Tensor,
    *,
    quantile: float = 0.9,
    max_pairs: int = 4000,
    center: bool = False,
    seed: int = 0,
) -> float:
    """Pick a semantic-link threshold from a corpus's OWN cosine distribution.

    The problem this solves: a fixed ``link_threshold`` is encoder- and
    corpus-specific. Sentence encoders (e5, bge, ...) are anisotropic -- their
    cosines pile into a high, narrow band, so the "right" absolute cut differs
    per model and per corpus and has to be hand-tuned. A *quantile of the actual
    pairwise-cosine distribution* is scale-free: ``quantile=0.9`` means "link only
    the top ~10% strongest relations", which transfers across encoders without
    retuning.

    Samples up to ``max_pairs`` distinct off-diagonal key pairs (all pairs when
    the corpus is small), computes their cosines, and returns the ``quantile``-th
    percentile. ``keys`` is ``(N, d)``; it is L2-normalised here so the dot
    products are cosines. ``center=True`` first removes the corpus mean direction
    (the Mu & Viswanath anisotropy fix) so the distribution reflects semantic
    residual similarity, matching a centered query. Fewer than 2 keys -> returns
    ``1.0`` (nothing can link), so the caller degrades safely.
    """
    if not (0.0 <= quantile <= 1.0):
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    if keys.ndim != 2:
        raise ValueError(f"keys must be 2-D (N, d), got shape {tuple(keys.shape)}")
    n = keys.shape[0]
    if n < 2:
        return 1.0

    k = F.normalize(keys.detach().to(torch.float32), dim=-1)
    if center:
        k = F.normalize(k - k.mean(dim=0, keepdim=True), dim=-1)

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        iu = torch.triu_indices(n, n, offset=1)         # all unique pairs
        cos = (k[iu[0]] * k[iu[1]]).sum(dim=-1)
    else:
        # Sample distinct unordered pairs (i < j) without materialising O(N^2).
        i = torch.randint(0, n, (max_pairs,), generator=gen)
        j = torch.randint(0, n, (max_pairs,), generator=gen)
        keep = i != j
        i, j = i[keep], j[keep]
        cos = (k[i] * k[j]).sum(dim=-1)

    if cos.numel() == 0:
        return 1.0
    return float(torch.quantile(cos, quantile))


_EDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS edges (
    src_id INTEGER NOT NULL,
    dst_id INTEGER NOT NULL,
    weight REAL    NOT NULL DEFAULT 1.0,
    etype  TEXT    NOT NULL DEFAULT 'related',
    PRIMARY KEY (src_id, dst_id, etype)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE TABLE IF NOT EXISTS node_status (
    node_id       INTEGER PRIMARY KEY,
    status        TEXT    NOT NULL DEFAULT 'active',  -- 'active' | 'superseded'
    superseded_by INTEGER,                            -- the node that replaced it
    updated_at    TEXT    NOT NULL
);
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
        *,
        auto_link: bool = False,
        link_threshold: float = 0.55,
        link_top_k: int = 5,
        link_center: bool = False,
    ) -> None:
        self.nodes = PersistentKnowledgeMemory(
            key_dim=key_dim, db_path=db_path, max_entries=max_entries
        )
        self.key_dim = self.nodes.key_dim
        self.db_path = str(db_path)
        # Default auto-link policy used by the duck-typed write() so a
        # GraphKnowledgeMemory can be a drop-in consolidation target that builds
        # the graph as it ingests.
        self.auto_link = bool(auto_link)
        self.link_threshold = float(link_threshold)
        self.link_top_k = int(link_top_k)
        self.link_center = bool(link_center)
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

    def write(self, key: torch.Tensor, content: Any, meta: Optional[Any] = None) -> int:
        """Duck-typed knowledge-store write (mirrors PersistentKnowledgeMemory).

        Lets a GraphKnowledgeMemory stand in as the ``knowledge_memory`` target of
        :meth:`mt_lnn.sleep_consolidation.SleepWakeConsolidator.nrem_replay`, so a
        sleep pass that consolidates sessions ALSO builds the relational graph.
        Honors the instance's auto-link policy (set at construction), so similar
        nodes are connected as they are written. Returns the new node id.
        """
        return self.add_node(
            key, content, meta,
            auto_link=self.auto_link,
            link_threshold=self.link_threshold,
            link_top_k=self.link_top_k,
            link_center=self.link_center,
        )

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

    def link_by_reference(
        self,
        src_id: int,
        text: str,
        *,
        title_of: str = "title",
        etype: str = "reference",
        bidirectional: bool = False,
    ) -> int:
        """Link ``src_id`` to every node it explicitly *mentions* in ``text``.

        Scans ``text`` for code-identifier references (backtick / file-path /
        PascalCase) and matches them against other nodes' titles, creating typed
        ``reference`` edges whose weight is the Awareness-SDK match confidence
        (exact path 1.0, backtick 0.8, PascalCase 0.7, basename 0.6). This is the
        content-driven complement to :meth:`auto_link_semantic`: cosine linking
        connects embedding-similar nodes, reference linking connects a note to the
        entities it names.

        A node's title is read from ``meta[title_of]`` (default key ``"title"``);
        nodes without that key are simply not matchable as targets. References are
        directional by default (the note points at what it mentions). Returns the
        number of edges created.
        """
        from .graph_linking import extract_references, match_references

        refs = extract_references(text)
        if not refs:
            return 0
        nodes = [
            {"id": nid, "title": title}
            for (nid, title) in self._titled_nodes(title_of)
            if nid != src_id
        ]
        links = match_references(refs, nodes)
        for link in links:
            self.link(
                src_id, int(link["target_id"]),
                weight=float(link["confidence"]),
                etype=etype, bidirectional=bidirectional,
            )
        return len(links)

    def calibrate_link_threshold(
        self,
        *,
        quantile: float = 0.9,
        max_pairs: int = 4000,
        center: Optional[bool] = None,
        seed: int = 0,
    ) -> float:
        """Adaptive link threshold from THIS store's own key distribution.

        Reads every stored node key and returns the ``quantile``-th percentile of
        their pairwise cosines (see :func:`calibrate_threshold_from_keys`), so the
        linker connects only the top ``1-quantile`` fraction of relations -- no
        per-encoder hand-tuning. ``center`` defaults to the instance's
        ``link_center``. Returns ``1.0`` when there are fewer than 2 nodes.
        """
        center = self.link_center if center is None else center
        rows = self._conn.execute("SELECT key_vec FROM knowledge").fetchall()
        if len(rows) < 2:
            return 1.0
        keys = torch.stack(
            [_bytes_to_key(r[0], self.key_dim) for r in rows], dim=0
        )
        return calibrate_threshold_from_keys(
            keys, quantile=quantile, max_pairs=max_pairs,
            center=center, seed=seed,
        )

    # ------------------------------------------------------------------
    # Living-knowledge lifecycle (supersede / contradict + self-correction)
    # ------------------------------------------------------------------
    #
    # Knowledge is not static: a fact gets updated ("the deadline moved"), and a
    # stale version should stop being recalled as current truth without being
    # destroyed (the history is still useful). This is the M1 analogue of the
    # Awareness-SDK lifecycle manager: typed SUPERSEDES / CONTRADICTS edges plus a
    # node status so recall returns the live view, while the superseded node and
    # its provenance edge are kept for audit.
    #
    # Honesty boundary -- detection vs bookkeeping:
    #   * SUPERSESSION is auto-detectable: a newer node that is a near-duplicate
    #     (cosine >= a high threshold) of an older one is, with high confidence,
    #     an update of it. Embeddings support that judgement, so auto_supersede
    #     does it. Recency is "newer = larger node id" (ids autoincrement).
    #   * CONTRADICTION (A and B assert incompatible things while NOT being near-
    #     duplicates) needs entailment/NLI, which this module does NOT have and
    #     does NOT fake. mark_contradiction records a caller-ASSERTED contradiction
    #     and runs the resolution policy; it never claims to have detected one.

    def status_of(self, node_id: int) -> str:
        """Lifecycle status of a node: 'active' (default) or 'superseded'."""
        row = self._conn.execute(
            "SELECT status FROM node_status WHERE node_id = ?", (int(node_id),)
        ).fetchone()
        return row[0] if row else "active"

    def _set_status(self, node_id: int, status: str, superseded_by: Optional[int]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO node_status (node_id, status, superseded_by, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                status = excluded.status,
                superseded_by = excluded.superseded_by,
                updated_at = excluded.updated_at
            """,
            (int(node_id), str(status),
             None if superseded_by is None else int(superseded_by), now),
        )
        self._conn.commit()

    def supersede(self, stale_id: int, current_id: int, *, weight: float = 1.0) -> None:
        """Mark ``stale_id`` as replaced by ``current_id`` (the live version).

        Records a directed ``supersedes`` edge ``current -> stale`` (provenance:
        "this node replaced that one") and flips the stale node's status to
        'superseded'. The stale node and its content stay in the store -- only its
        recall visibility changes (see ``exclude_superseded`` on the query/walk).
        A node cannot supersede itself.
        """
        if stale_id == current_id:
            raise ValueError("a node cannot supersede itself")
        self.link(current_id, stale_id, weight=weight, etype="supersedes",
                  bidirectional=False)
        self._set_status(stale_id, "superseded", current_id)

    def auto_supersede(
        self,
        new_id: int,
        key: torch.Tensor,
        *,
        threshold: float = 0.95,
        top_k: int = 5,
        center: bool = False,
        only_older: bool = True,
    ) -> int:
        """Self-correction: retire OLDER near-duplicates of a just-added node.

        Finds active nodes whose cosine to ``key`` is >= ``threshold`` (a HIGH cut
        -- these are near-duplicate restatements, i.e. updates of the same fact)
        and supersedes each by ``new_id``. With ``only_older`` (default) it only
        retires nodes added before ``new_id`` (lower id), so adding an updated fact
        replaces its stale versions but a fact is never retired by something that
        predates it. Returns the number of nodes superseded.

        ``threshold`` is intentionally high (0.95): supersession means "this IS the
        same fact, restated", not merely "related". Use a normal link for related.
        """
        hits = self.nodes.query(
            key, top_k=top_k + 1, touch=False, center=center, return_ids=True
        )
        n = 0
        for nid, _content, score, _meta in hits:
            if nid == new_id:
                continue
            if only_older and nid >= new_id:
                continue
            if score >= threshold and self.status_of(nid) == "active":
                self.supersede(nid, new_id, weight=float(score))
                n += 1
        return n

    def mark_contradiction(
        self,
        a_id: int,
        b_id: int,
        *,
        weight: float = 1.0,
        resolve: bool = True,
    ) -> Optional[int]:
        """Record a caller-ASSERTED contradiction between two nodes.

        Adds a symmetric ``contradicts`` edge (the relation is mutual). This method
        does NOT detect contradictions -- the caller asserts one (e.g. from an
        external checker or the user). With ``resolve=True`` (default) the
        lifecycle resolves it by recency: the OLDER node (smaller id) is superseded
        by the newer one, so recall returns the current claim while the edge keeps
        the conflict on record. Returns the id of the node that was superseded, or
        ``None`` when ``resolve`` is False. A node cannot contradict itself.
        """
        if a_id == b_id:
            raise ValueError("a node cannot contradict itself")
        self.link(a_id, b_id, weight=weight, etype="contradicts", bidirectional=True)
        if not resolve:
            return None
        older, newer = (a_id, b_id) if a_id < b_id else (b_id, a_id)
        self.supersede(older, newer, weight=weight)
        return older

    def n_superseded(self) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM node_status WHERE status = 'superseded'"
        ).fetchone()[0])

    def _superseded_ids(self) -> set:
        rows = self._conn.execute(
            "SELECT node_id FROM node_status WHERE status = 'superseded'"
        ).fetchall()
        return {int(r[0]) for r in rows}

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
        exclude_superseded: bool = True,
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

        ``exclude_superseded`` (default True): nodes retired by the lifecycle
        (see :meth:`supersede` / :meth:`auto_supersede`) are dropped from the
        RESULTS so recall returns the live view, but they still CONDUCT activation
        (a stale node can relay to its replacement's neighbours), keeping the graph
        connected. With no superseded nodes this is a no-op, so it is safe by
        default. Pass False to recall the full history including retired versions.
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

        # Superseded nodes still conducted activation above (kept the graph
        # connected); now drop them from the RESULTS so recall is the live view.
        retired = self._superseded_ids() if exclude_superseded else set()
        ranked = sorted(activation.items(), key=lambda kv: kv[1], reverse=True)
        out: List[Tuple[Any, float, Optional[Any]]] = []
        for nid, act in ranked:
            if nid in retired:
                continue
            node = self._get_node(nid)
            if node is None:
                continue                       # evicted under max_entries
            content, meta = node
            out.append((content, float(act), meta))
            if len(out) >= top_k:
                break
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

    def _titled_nodes(self, title_of: str) -> List[Tuple[int, str]]:
        """All (id, title) pairs whose meta carries a string title under
        ``title_of`` -- the matchable targets for reference linking."""
        rows = self._conn.execute("SELECT id, meta FROM knowledge").fetchall()
        out: List[Tuple[int, str]] = []
        for nid, meta_blob in rows:
            if meta_blob is None:
                continue
            meta = _bytes_to_obj(meta_blob)
            if isinstance(meta, dict):
                title = meta.get(title_of)
                if isinstance(title, str) and title:
                    out.append((int(nid), title))
        return out

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
