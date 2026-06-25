"""
mt_lnn/session_consolidation.py -- bridge persisted sessions into the sleep cycle.

What this is for
----------------
Step ③ made the inference server persist each conversation's recurrent LNN state
(``h_prev``) to SQLite via :class:`mt_lnn.memory.SessionMemory` -- fast, volatile
*working* memory, keyed by ``session_id`` and NOT content-addressable (you cannot
ask it "which past conversation looked like this").

This module is the missing "sleep" bridge (Gap 4, NREM stage): it replays those
persisted sessions and consolidates the salient ones, BY CONTENT, into the slow
long-term :class:`mt_lnn.knowledge_memory.PersistentKnowledgeMemory`. After a
consolidation cycle a session's recurrent signature becomes a content-addressable
key in the durable store, so the right past session can be *recalled by
similarity* -- the fast->slow (hippocampus->neocortex) transfer the sleep module
was built for.

It runs the EXISTING :class:`mt_lnn.sleep_consolidation.SleepWakeConsolidator`
(NREM ``nrem_replay``) unchanged; this file only supplies the duck-typed replay
buffer that presents sessions in the field-dict layout that consolidator expects.

Design discipline (same as memory.py / knowledge_memory.py / sleep_consolidation.py):
  * Zero model coupling -- imports only torch + the memory/sleep modules + stdlib.
    Never imports model.py.
  * Reuses, does not duplicate -- the consolidation policy (salience ranking,
    fraction, determinism) all lives in SleepWakeConsolidator; the long-term
    store is PersistentKnowledgeMemory; the source is SessionMemory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .knowledge_memory import PersistentKnowledgeMemory
from .graph_memory import GraphKnowledgeMemory, calibrate_threshold_from_keys
from .memory import SessionMemory
from .sleep_consolidation import ReplayConsolidationResult, SleepWakeConsolidator

__all__ = [
    "pool_session_key",
    "SessionReplayBuffer",
    "consolidate_sessions",
    "consolidate_sessions_to_graph",
]


# ---------------------------------------------------------------------------
# Recurrent state -> content key
# ---------------------------------------------------------------------------

def pool_session_key(
    h_states: List[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    """Reduce a session's per-layer recurrent state to a single content key.

    ``h_states`` is the list returned by :meth:`SessionMemory.load` -- one entry
    per layer, each ``(B, P, d_proto)`` or ``None``. Each present layer is
    mean-pooled over every axis but the last (``-> (d_proto,)``) and the layers
    are averaged, giving a fixed ``(d_proto,)`` key that is identical in
    dimensionality across all sessions of the same model. Returns ``None`` when
    the session carries no recurrent state at all (nothing to consolidate).

    The key is NOT normalised here -- PersistentKnowledgeMemory L2-normalises on
    write and query, and its ``center=True`` query option removes the shared
    dominant direction these pooled-state keys exhibit.
    """
    pooled: List[torch.Tensor] = []
    for h in h_states:
        if h is None:
            continue
        h = h.detach().to(torch.float32)
        pooled.append(h.reshape(-1, h.shape[-1]).mean(dim=0))   # (d_proto,)
    if not pooled:
        return None
    return torch.stack(pooled, dim=0).mean(dim=0)               # (d_proto,)


# ---------------------------------------------------------------------------
# Duck-typed replay buffer over persisted sessions
# ---------------------------------------------------------------------------

class SessionReplayBuffer:
    """Present persisted sessions in the field-dict layout NREM replay expects.

    Implements the minimal duck-typed contract used by
    :meth:`SleepWakeConsolidator.nrem_replay`: ``__len__`` and
    ``sample(k, *, replacement=False)`` returning a dict whose ``key``/``salience``
    fields are tensors and whose ``content``/``meta`` fields are parallel lists
    (the consolidator indexes them row-wise, so lists are valid -- it never
    requires content/meta to be tensors).

    Episodes are built eagerly from a :class:`SessionMemory` store: each session
    becomes (key = pooled recurrent signature, content = session_id, meta =
    bookkeeping, salience = cumulative token_count). Sessions with no recurrent
    state are skipped.
    """

    KEY_FIELD = "key"
    CONTENT_FIELD = "content"
    META_FIELD = "meta"
    SALIENCE_FIELD = "salience"

    def __init__(
        self,
        keys: torch.Tensor,                 # (n, key_dim)
        contents: List[Any],
        metas: List[Any],
        saliences: torch.Tensor,            # (n,)
        *,
        seed: int = 0,
    ) -> None:
        n = keys.shape[0]
        if not (len(contents) == len(metas) == saliences.shape[0] == n):
            raise ValueError("keys/contents/metas/saliences length mismatch")
        self.keys = keys
        self.contents = contents
        self.metas = metas
        self.saliences = saliences
        self.key_dim = int(keys.shape[1]) if n > 0 else 0
        self._gen = torch.Generator(device="cpu").manual_seed(int(seed))

    @classmethod
    def from_session_memory(
        cls, mem: SessionMemory, *, device: str = "cpu", seed: int = 0
    ) -> "SessionReplayBuffer":
        keys: List[torch.Tensor] = []
        contents: List[Any] = []
        metas: List[Any] = []
        saliences: List[float] = []
        for info in mem.list_sessions():
            sid = info["session_id"]
            h_states = mem.load(sid, device=device)
            if h_states is None:
                continue
            key = pool_session_key(h_states)
            if key is None:
                continue
            keys.append(key)
            contents.append(sid)
            metas.append(
                {"session_id": sid,
                 "token_count": info["token_count"],
                 "updated_at": info["updated_at"]}
            )
            saliences.append(float(info["token_count"]))
        if keys:
            key_mat = torch.stack(keys, dim=0)
            sal = torch.tensor(saliences, dtype=torch.float32)
        else:
            key_mat = torch.zeros(0, 0)
            sal = torch.zeros(0)
        return cls(key_mat, contents, metas, sal, seed=seed)

    def __len__(self) -> int:
        return self.keys.shape[0]

    def sample(self, k: int, *, replacement: bool = False) -> Dict[str, Any]:
        n = len(self)
        if n == 0:
            raise ValueError("cannot sample from an empty SessionReplayBuffer")
        if replacement:
            idx = torch.randint(0, n, (k,), generator=self._gen)
        else:
            k = min(k, n)
            idx = torch.randperm(n, generator=self._gen)[:k]
        idx_list = idx.tolist()
        return {
            self.KEY_FIELD: self.keys[idx],
            self.CONTENT_FIELD: [self.contents[i] for i in idx_list],
            self.META_FIELD: [self.metas[i] for i in idx_list],
            self.SALIENCE_FIELD: self.saliences[idx],
        }


# ---------------------------------------------------------------------------
# One-call orchestration: replay sessions -> long-term knowledge store
# ---------------------------------------------------------------------------

def consolidate_sessions(
    session_db: str,
    knowledge_db: str,
    *,
    consolidate_fraction: float = 1.0,
    seed: int = 0,
    max_entries: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run one NREM sleep pass: replay persisted sessions and consolidate the
    salient ones (by cumulative token_count) into the long-term knowledge store.

    Returns a summary dict: replayed / consolidated counts, the resulting number
    of knowledge entries, and the key dimensionality. A no-op (all-zero counts)
    when there are no consolidatable sessions, so it is always safe to call.
    """
    with SessionMemory(session_db) as mem:
        buffer = SessionReplayBuffer.from_session_memory(mem, device=device, seed=seed)

    if len(buffer) == 0:
        return {"replayed": 0, "consolidated": 0,
                "knowledge_entries": 0, "key_dim": 0}

    kb = PersistentKnowledgeMemory(
        key_dim=buffer.key_dim, db_path=knowledge_db, max_entries=max_entries
    )
    try:
        consolidator = SleepWakeConsolidator(seed=seed)
        result: ReplayConsolidationResult = consolidator.nrem_replay(
            buffer, kb,
            key_field=SessionReplayBuffer.KEY_FIELD,
            content_field=SessionReplayBuffer.CONTENT_FIELD,
            meta_field=SessionReplayBuffer.META_FIELD,
            salience_field=SessionReplayBuffer.SALIENCE_FIELD,
            consolidate_fraction=consolidate_fraction,
        )
        entries = len(kb)
    finally:
        kb.close()

    return {
        "replayed": result.replayed,
        "consolidated": result.consolidated,
        "knowledge_entries": entries,
        "key_dim": buffer.key_dim,
    }


def consolidate_sessions_to_graph(
    session_db: str,
    graph_db: str,
    *,
    consolidate_fraction: float = 1.0,
    seed: int = 0,
    link_threshold: float = 0.55,
    link_quantile: Optional[float] = None,
    link_top_k: int = 5,
    link_center: bool = True,
    max_entries: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run one NREM pass that consolidates sessions into a relational GRAPH.

    Identical replay/salience policy to :func:`consolidate_sessions`, but the
    long-term target is a :class:`mt_lnn.graph_memory.GraphKnowledgeMemory`: each
    consolidated session becomes a node AND is auto-linked (by recurrent-signature
    cosine >= the link threshold, edge weight = similarity) to the sessions
    already consolidated. So sleep does not just store sessions by content -- it
    sediments the RELATIONS between them, giving M1 a graph it can later traverse
    by spreading activation.

    Linking cut -- absolute vs adaptive:
      * ``link_quantile`` (when given, in [0, 1]) makes the cut ADAPTIVE: the
        threshold is the ``link_quantile``-th percentile of the buffer's own
        pairwise-cosine distribution, so it tracks each corpus/encoder's cosine
        band instead of a hand-tuned absolute (``link_quantile=0.9`` keeps the top
        ~10% strongest relations). The chosen value is returned as
        ``link_threshold`` in the summary.
      * Otherwise the fixed ``link_threshold`` is used.

    ``link_center=True`` by default: pooled recurrent signatures share a dominant
    direction (anisotropy), so linking on the centered residual avoids connecting
    everything to everything. Returns a summary: replayed / consolidated counts,
    resulting node and edge counts, the key dimensionality, and the effective
    ``link_threshold``. A no-op all-zero summary when there are no consolidatable
    sessions.
    """
    with SessionMemory(session_db) as mem:
        buffer = SessionReplayBuffer.from_session_memory(mem, device=device, seed=seed)

    if len(buffer) == 0:
        return {"replayed": 0, "consolidated": 0,
                "nodes": 0, "edges": 0, "key_dim": 0,
                "link_threshold": link_threshold}

    # Adaptive cut: derive the threshold from the corpus's own cosine band.
    if link_quantile is not None:
        link_threshold = calibrate_threshold_from_keys(
            buffer.keys, quantile=link_quantile,
            center=link_center, seed=seed,
        )

    graph = GraphKnowledgeMemory(
        key_dim=buffer.key_dim, db_path=graph_db, max_entries=max_entries,
        auto_link=True, link_threshold=link_threshold,
        link_top_k=link_top_k, link_center=link_center,
    )
    try:
        consolidator = SleepWakeConsolidator(seed=seed)
        result: ReplayConsolidationResult = consolidator.nrem_replay(
            buffer, graph,
            key_field=SessionReplayBuffer.KEY_FIELD,
            content_field=SessionReplayBuffer.CONTENT_FIELD,
            meta_field=SessionReplayBuffer.META_FIELD,
            salience_field=SessionReplayBuffer.SALIENCE_FIELD,
            consolidate_fraction=consolidate_fraction,
        )
        nodes = graph.n_nodes()
        edges = graph.n_edges()
    finally:
        graph.close()

    return {
        "replayed": result.replayed,
        "consolidated": result.consolidated,
        "nodes": nodes,
        "edges": edges,
        "key_dim": buffer.key_dim,
        "link_threshold": float(link_threshold),
    }
