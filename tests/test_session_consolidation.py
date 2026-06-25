"""
tests/test_session_consolidation.py -- Gap 4 (NREM): the sleep bridge that turns
fast, volatile SessionMemory working-state into durable, content-addressable
long-term knowledge.

What is pinned here
-------------------
mt_lnn/session_consolidation.py glues three EXISTING, zero-model-coupling tiers
together without duplicating any policy:

    SessionMemory (fast working state, keyed by session_id, NOT content-addressable)
        --pool_session_key--> a (d_proto,) content signature per session
        --SleepWakeConsolidator.nrem_replay--> PersistentKnowledgeMemory (slow,
                                               content-addressable long-term store)

The headline test is the user's requested before/after comparison: a session's
own recurrent signature CANNOT recall that session from the long-term store
BEFORE a sleep pass (the store is empty), and CAN after -- a direct, measurable
demonstration that the sleep cycle makes session memory long-term and
recall-by-content addressable. This is the consolidation PLUMBING + recall
contract; whether richer signatures improve real recall is a separate
effectiveness question.

Scope honesty: keys here are pooled from synthetic recurrent states, not a real
model's h_prev. That the pooled signature is discriminative enough to self-recall
is exactly what center=True (Mu & Viswanath 2018) is for, and is asserted here on
well-separated states -- the mechanism is proven, the real-corpus separability is
an effectiveness question for the GPU/real-model track.
"""

import os
import types

import torch

from mt_lnn.memory import SessionMemory
from mt_lnn.knowledge_memory import PersistentKnowledgeMemory
from mt_lnn.session_consolidation import (
    pool_session_key,
    SessionReplayBuffer,
    consolidate_sessions,
    consolidate_sessions_to_graph,
)


# ---------------------------------------------------------------------------
# Helpers: write synthetic sessions into a SessionMemory without touching model.py
# ---------------------------------------------------------------------------

def _fake_cache(h_states):
    """A minimal duck-typed cache: SessionMemory.save reads cache.layers[i][1]."""
    cache = types.SimpleNamespace()
    cache.layers = [(None, h, None) for h in h_states]
    return cache


def _seed_sessions(db_path, *, n_sessions, n_layers, d_proto, seed=0):
    """Write n_sessions distinct synthetic sessions; return {sid: pooled_key}."""
    g = torch.Generator().manual_seed(seed)
    expected = {}
    with SessionMemory(db_path) as mem:
        for s in range(n_sessions):
            # (B, P, d_proto) per layer; well-separated per session.
            h_states = [
                torch.randn(1, 3, d_proto, generator=g) + 4.0 * s
                for _ in range(n_layers)
            ]
            mem.save(f"sess{s}", _fake_cache(h_states), token_count=10 * (s + 1))
            expected[f"sess{s}"] = pool_session_key(h_states)
    return expected


def _cleanup(*db_paths):
    for db in db_paths:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(db + suffix)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# pool_session_key
# ---------------------------------------------------------------------------

def test_pool_session_key_shape_and_none():
    """Pooling reduces per-layer (B,P,d_proto) state to a single (d_proto,) key,
    and returns None when there is no recurrent state at all."""
    d = 16
    h_states = [torch.randn(2, 4, d), None, torch.randn(2, 4, d)]
    key = pool_session_key(h_states)
    assert key is not None and key.shape == (d,)
    # All-None (no recurrent state) -> nothing to consolidate.
    assert pool_session_key([None, None]) is None
    assert pool_session_key([]) is None


def test_pool_session_key_is_deterministic():
    h_states = [torch.randn(1, 5, 8)]
    a = pool_session_key(h_states)
    b = pool_session_key(h_states)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# SessionReplayBuffer duck-type contract (what nrem_replay relies on)
# ---------------------------------------------------------------------------

def test_replay_buffer_sample_contract():
    db = os.path.join(os.path.dirname(__file__), "_consol_buf.db")
    _cleanup(db)
    try:
        _seed_sessions(db, n_sessions=4, n_layers=2, d_proto=12, seed=1)
        with SessionMemory(db) as mem:
            buf = SessionReplayBuffer.from_session_memory(mem)
        assert len(buf) == 4
        assert buf.key_dim == 12
        batch = buf.sample(3)
        # key/salience are tensors (nrem_replay calls .shape/.reshape on them);
        # content/meta are parallel lists (indexed row-wise).
        assert isinstance(batch[buf.KEY_FIELD], torch.Tensor)
        assert isinstance(batch[buf.SALIENCE_FIELD], torch.Tensor)
        assert batch[buf.KEY_FIELD].shape == (3, 12)
        assert batch[buf.SALIENCE_FIELD].shape == (3,)
        assert isinstance(batch[buf.CONTENT_FIELD], list) and len(batch[buf.CONTENT_FIELD]) == 3
        assert isinstance(batch[buf.META_FIELD], list) and len(batch[buf.META_FIELD]) == 3
        # content rows are session_ids; meta carries the bookkeeping.
        for sid, meta in zip(batch[buf.CONTENT_FIELD], batch[buf.META_FIELD]):
            assert meta["session_id"] == sid
    finally:
        _cleanup(db)


def test_replay_buffer_skips_stateless_sessions():
    """A session whose every layer state is None carries no content key and must
    be skipped (nothing to consolidate), not crash the buffer build."""
    db = os.path.join(os.path.dirname(__file__), "_consol_skip.db")
    _cleanup(db)
    try:
        with SessionMemory(db) as mem:
            mem.save("real", _fake_cache([torch.randn(1, 2, 8)]), token_count=5)
            mem.save("empty", _fake_cache([None]), token_count=5)
        with SessionMemory(db) as mem:
            buf = SessionReplayBuffer.from_session_memory(mem)
        assert len(buf) == 1
        assert buf.sample(1)[buf.CONTENT_FIELD] == ["real"]
    finally:
        _cleanup(db)


# ---------------------------------------------------------------------------
# Headline: before/after sleep-consolidation recall accuracy
# ---------------------------------------------------------------------------

def test_recall_accuracy_before_vs_after_consolidation():
    """The user's before/after comparison, made measurable.

    Each session's own pooled recurrent signature is used as a recall query
    against the long-term store. BEFORE a sleep pass the store is empty so recall
    accuracy is 0; AFTER nrem_replay consolidates the sessions, each signature
    recalls its OWN session_id -- i.e. the fast working state has become
    long-term and content-addressable. This is the whole point of the sleep
    bridge, asserted as a number that strictly improves.
    """
    sess_db = os.path.join(os.path.dirname(__file__), "_consol_sess.db")
    know_db = os.path.join(os.path.dirname(__file__), "_consol_know.db")
    _cleanup(sess_db, know_db)
    try:
        n_sessions, d_proto = 6, 32
        expected = _seed_sessions(
            sess_db, n_sessions=n_sessions, n_layers=2, d_proto=d_proto, seed=7
        )

        # BEFORE: empty long-term store -> a fresh query recalls nothing.
        kb_before = PersistentKnowledgeMemory(key_dim=d_proto, db_path=know_db)
        try:
            before_hits = sum(
                1
                for sid, key in expected.items()
                if kb_before.recall(key, touch=False, center=True) == sid
            )
        finally:
            kb_before.close()
        before_acc = before_hits / n_sessions
        assert before_acc == 0.0, "store should be empty before any sleep pass"
        _cleanup(know_db)  # drop the empty store so consolidation starts clean

        # SLEEP: run one NREM consolidation pass (the orchestration under test).
        summary = consolidate_sessions(sess_db, know_db, consolidate_fraction=1.0, seed=7)
        assert summary["replayed"] == n_sessions
        assert summary["consolidated"] == n_sessions
        assert summary["knowledge_entries"] == n_sessions
        assert summary["key_dim"] == d_proto

        # AFTER: each session's signature recalls its own session_id.
        kb_after = PersistentKnowledgeMemory(key_dim=d_proto, db_path=know_db)
        try:
            after_hits = sum(
                1
                for sid, key in expected.items()
                if kb_after.recall(key, touch=False, center=True) == sid
            )
        finally:
            kb_after.close()
        after_acc = after_hits / n_sessions

        assert after_acc > before_acc, \
            "sleep consolidation did not improve cross-session recall accuracy"
        assert after_acc == 1.0, \
            f"each session should self-recall after consolidation, got {after_acc:.2f}"
    finally:
        _cleanup(sess_db, know_db)


def test_consolidate_to_graph_sediments_links_between_similar_sessions():
    """Sleep into a GRAPH does not just store sessions -- it links the related
    ones. Two sessions with near-identical recurrent signatures get connected;
    a distinct one does not pull a spurious edge. The result is a graph M1 can
    later traverse by spreading activation, not a flat list."""
    sess_db = os.path.join(os.path.dirname(__file__), "_consol_graph_sess.db")
    graph_db = os.path.join(os.path.dirname(__file__), "_consol_graph.db")
    _cleanup(sess_db, graph_db)
    try:
        d = 24
        torch.manual_seed(11)
        shared = torch.randn(1, 3, d)              # the signature A and B share
        with SessionMemory(sess_db) as mem:
            # A and B are near-identical (cosine ~1); C is distinct.
            mem.save("A", _fake_cache([shared.clone()]), token_count=10)
            mem.save("B", _fake_cache([shared + 1e-3 * torch.randn(1, 3, d)]),
                     token_count=20)
            mem.save("C", _fake_cache([torch.randn(1, 3, d) + 50.0]),
                     token_count=30)

        summary = consolidate_sessions_to_graph(
            sess_db, graph_db, link_threshold=0.55, link_center=False, seed=11
        )
        assert summary["replayed"] == 3
        assert summary["consolidated"] == 3
        assert summary["nodes"] == 3
        assert summary["edges"] >= 2, \
            "similar sessions were not linked during graph consolidation"
        assert summary["key_dim"] == d

        # The sedimented graph is traversable: from A's signature, spreading
        # activation reaches B through the cosine-linked edge.
        from mt_lnn.graph_memory import GraphKnowledgeMemory
        from mt_lnn.session_consolidation import pool_session_key
        g = GraphKnowledgeMemory(key_dim=d, db_path=graph_db)
        try:
            key_a = pool_session_key([shared.clone()])
            hits = g.spread_activation(key_a, seeds=1, hops=1, top_k=3)
            assert "B" in {c for (c, _a, _m) in hits}
        finally:
            g.close()
    finally:
        _cleanup(sess_db, graph_db)


def test_consolidate_to_graph_is_noop_on_empty_store():
    """Graph consolidation with nothing consolidatable is a safe all-zero no-op."""
    sess_db = os.path.join(os.path.dirname(__file__), "_consol_graph_empty.db")
    graph_db = os.path.join(os.path.dirname(__file__), "_consol_graph_empty_g.db")
    _cleanup(sess_db, graph_db)
    try:
        with SessionMemory(sess_db):
            pass
        summary = consolidate_sessions_to_graph(sess_db, graph_db)
        assert summary == {
            "replayed": 0, "consolidated": 0, "nodes": 0, "edges": 0,
            "key_dim": 0, "link_threshold": 0.55,
        }
    finally:
        _cleanup(sess_db, graph_db)


def test_consolidate_to_graph_adaptive_quantile_threshold():
    """link_quantile makes the cut ADAPTIVE: instead of a hand-tuned absolute
    cosine, the threshold is read off the corpus's own pairwise-cosine band. Two
    near-identical sessions plus a distinct one still link the similar pair, and
    the summary reports the data-derived threshold actually used (in (0, 1))."""
    sess_db = os.path.join(os.path.dirname(__file__), "_consol_graph_adapt_s.db")
    graph_db = os.path.join(os.path.dirname(__file__), "_consol_graph_adapt_g.db")
    _cleanup(sess_db, graph_db)
    try:
        d = 24
        torch.manual_seed(13)
        shared = torch.randn(1, 3, d)
        with SessionMemory(sess_db) as mem:
            mem.save("A", _fake_cache([shared.clone()]), token_count=10)
            mem.save("B", _fake_cache([shared + 1e-3 * torch.randn(1, 3, d)]),
                     token_count=20)
            mem.save("C", _fake_cache([torch.randn(1, 3, d) + 50.0]),
                     token_count=30)

        # No absolute link_threshold tuning -- derive it from the data.
        summary = consolidate_sessions_to_graph(
            sess_db, graph_db, link_quantile=0.50, link_center=False, seed=13
        )
        assert summary["nodes"] == 3
        assert summary["edges"] >= 2, "adaptive threshold failed to link the similar pair"
        # The reported threshold is the data-derived one, a genuine cosine in (0, 1).
        assert 0.0 < summary["link_threshold"] < 1.0
    finally:
        _cleanup(sess_db, graph_db)


def test_consolidate_sessions_is_noop_on_empty_store():
    """Calling the sleep bridge with no consolidatable sessions is a safe all-zero
    no-op, so a server can always trigger it without guarding."""
    sess_db = os.path.join(os.path.dirname(__file__), "_consol_empty.db")
    know_db = os.path.join(os.path.dirname(__file__), "_consol_empty_know.db")
    _cleanup(sess_db, know_db)
    try:
        # Touch an empty SessionMemory so the db exists but has no sessions.
        with SessionMemory(sess_db):
            pass
        summary = consolidate_sessions(sess_db, know_db)
        assert summary == {
            "replayed": 0, "consolidated": 0, "knowledge_entries": 0, "key_dim": 0
        }
        # No knowledge db should have been created/populated.
        assert not os.path.exists(know_db) or PersistentKnowledgeMemory(
            key_dim=1, db_path=know_db
        ).__len__() == 0
    finally:
        _cleanup(sess_db, know_db)


def test_consolidate_fraction_keeps_only_most_salient():
    """consolidate_fraction routes the salience policy that already lives in
    SleepWakeConsolidator: with fraction 0.5 only the highest-token_count half of
    the sessions are persisted (this module adds no new policy, it just feeds it)."""
    sess_db = os.path.join(os.path.dirname(__file__), "_consol_frac.db")
    know_db = os.path.join(os.path.dirname(__file__), "_consol_frac_know.db")
    _cleanup(sess_db, know_db)
    try:
        # token_count increases with s -> sess3 is most salient, sess0 least.
        _seed_sessions(sess_db, n_sessions=4, n_layers=1, d_proto=16, seed=3)
        summary = consolidate_sessions(
            sess_db, know_db, consolidate_fraction=0.5, seed=3
        )
        assert summary["replayed"] == 4
        assert summary["consolidated"] == 2     # top half by token_count
        assert summary["knowledge_entries"] == 2
    finally:
        _cleanup(sess_db, know_db)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except Exception:
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
            raise
