"""
tests/test_graph_memory.py -- associative graph memory for M1.

Pins mt_lnn.graph_memory.GraphKnowledgeMemory:

  * weighted typed edges over content-addressable nodes (borrowed substrate);
  * Awareness-style semantic-cosine auto-linking (link iff cosine >= threshold);
  * the M1-distinctive piece: multi-hop SPREADING ACTIVATION recall that surfaces
    a node only weakly similar to the query but strongly LINKED to a strong match
    -- exactly what a flat Top-K (and Awareness's RRF rerank) cannot do.

The headline test makes that difference measurable: a chain A->B->C where only A
matches the query. A flat cosine seeding (hops=0) returns only A; spreading
activation (hops>=2) reaches B and then C through the edges. That is the
associative recall, asserted as nodes a Top-K provably would not have returned.

Scope honesty: keys here are constructed orthogonal vectors so "similar vs
linked" is unambiguous. That the graph walk PROPAGATES correctly is the mechanism
under test; whether richer real-embedding graphs improve end-task recall is a
separate effectiveness question.
"""

import os

import torch

from mt_lnn.graph_memory import GraphKnowledgeMemory


def _e(dim, i, scale=1.0):
    """A one-hot-ish key along axis i (orthogonal across i)."""
    v = torch.zeros(dim)
    v[i] = scale
    return v


def _cleanup(db):
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db + suffix)
        except FileNotFoundError:
            pass


def _db(name):
    return os.path.join(os.path.dirname(__file__), name)


# ---------------------------------------------------------------------------
# Nodes + edges bookkeeping
# ---------------------------------------------------------------------------

def test_add_node_and_link_counts():
    db = _db("_graph_basic.db")
    _cleanup(db)
    try:
        with GraphKnowledgeMemory(key_dim=8, db_path=db) as g:
            a = g.add_node(_e(8, 0), "A")
            b = g.add_node(_e(8, 1), "B")
            assert isinstance(a, int) and isinstance(b, int) and a != b
            assert g.n_nodes() == 2
            g.link(a, b, weight=0.9)
            # bidirectional by default -> 2 edge rows.
            assert g.n_edges() == 2
            # Re-link overwrites weight, does not duplicate.
            g.link(a, b, weight=0.5)
            assert g.n_edges() == 2
            # Self-loops ignored.
            g.link(a, a, weight=1.0)
            assert g.n_edges() == 2
    finally:
        _cleanup(db)


# ---------------------------------------------------------------------------
# Semantic auto-linking (Awareness-style)
# ---------------------------------------------------------------------------

def test_auto_link_semantic_links_similar_only():
    db = _db("_graph_autolink.db")
    _cleanup(db)
    try:
        with GraphKnowledgeMemory(key_dim=8, db_path=db) as g:
            a = g.add_node(_e(8, 0), "A")
            # Near-duplicate of A (cosine ~1) -> should auto-link to A.
            near = _e(8, 0) + 0.01 * _e(8, 2)
            b = g.add_node(near, "B", auto_link=True, link_threshold=0.55)
            assert g.n_edges() >= 2, "semantically similar node was not linked"
            # An orthogonal node (cosine 0 < threshold) -> no new edge.
            before = g.n_edges()
            g.add_node(_e(8, 5), "C", auto_link=True, link_threshold=0.55)
            assert g.n_edges() == before, "orthogonal node should not be linked"
    finally:
        _cleanup(db)


# ---------------------------------------------------------------------------
# Headline: multi-hop spreading activation beats flat Top-K
# ---------------------------------------------------------------------------

def test_spreading_activation_reaches_what_topk_misses():
    db = _db("_graph_spread.db")
    _cleanup(db)
    try:
        dim = 8
        with GraphKnowledgeMemory(key_dim=dim, db_path=db) as g:
            # A matches the query; B and C are ORTHOGONAL to it (cosine 0), so
            # cosine seeding alone can never surface them.
            a = g.add_node(_e(dim, 0), "A")
            b = g.add_node(_e(dim, 1), "B")
            c = g.add_node(_e(dim, 2), "C")
            g.link(a, b, weight=0.9)   # A <-> B
            g.link(b, c, weight=0.9)   # B <-> C  (C only reachable via 2 hops)

            query = _e(dim, 0)         # == A's key

            # Flat cosine seeding (hops=0): only A has activation.
            flat = g.spread_activation(query, seeds=1, hops=0, top_k=5)
            flat_contents = {c0 for (c0, _a, _m) in flat}
            assert flat_contents == {"A"}, \
                f"hops=0 must reduce to flat Top-K, got {flat_contents}"

            # Spreading 2 hops: A -> B -> C all activated, ranked by reach.
            spread = g.spread_activation(
                query, seeds=1, hops=2, decay=0.8, top_k=5
            )
            ranked = [(c0, act) for (c0, act, _m) in spread]
            contents = [c0 for c0, _ in ranked]
            assert set(contents) == {"A", "B", "C"}, \
                f"spreading activation did not reach the chain, got {contents}"
            # Activation strictly decays along the chain A > B > C > 0.
            act = dict(ranked)
            assert act["A"] > act["B"] > act["C"] > 0.0
    finally:
        _cleanup(db)


def test_decay_and_hops_bound_the_reach():
    db = _db("_graph_reach.db")
    _cleanup(db)
    try:
        dim = 8
        with GraphKnowledgeMemory(key_dim=dim, db_path=db) as g:
            a = g.add_node(_e(dim, 0), "A")
            b = g.add_node(_e(dim, 1), "B")
            c = g.add_node(_e(dim, 2), "C")
            g.link(a, b, weight=0.9)
            g.link(b, c, weight=0.9)
            query = _e(dim, 0)

            # 1 hop reaches B but NOT C.
            one = {c0 for (c0, _a, _m) in g.spread_activation(query, seeds=1, hops=1, top_k=5)}
            assert one == {"A", "B"}, f"1 hop should reach A,B only, got {one}"

            # decay=0 kills all propagation -> only the seed.
            zero = {c0 for (c0, _a, _m) in g.spread_activation(query, seeds=1, hops=3, decay=0.0, top_k=5)}
            assert zero == {"A"}, f"decay=0 should not propagate, got {zero}"
    finally:
        _cleanup(db)


def test_spread_on_empty_graph_is_empty():
    db = _db("_graph_empty.db")
    _cleanup(db)
    try:
        with GraphKnowledgeMemory(key_dim=4, db_path=db) as g:
            assert g.spread_activation(_e(4, 0), seeds=3, hops=2) == []
    finally:
        _cleanup(db)


# ---------------------------------------------------------------------------
# Backward-compat of the return_ids extension on PersistentKnowledgeMemory
# ---------------------------------------------------------------------------

def test_knowledge_query_return_ids_is_backward_compatible():
    from mt_lnn.knowledge_memory import PersistentKnowledgeMemory
    kb = PersistentKnowledgeMemory(key_dim=4, db_path=":memory:")
    try:
        rid = kb.write(_e(4, 0), "X")
        # Default: 3-tuple (content, score, meta) -- unchanged contract.
        hit = kb.query(_e(4, 0), top_k=1)[0]
        assert len(hit) == 3 and hit[0] == "X"
        # Opt-in: 4-tuple prefixed with the id.
        hit_id = kb.query(_e(4, 0), top_k=1, return_ids=True)[0]
        assert len(hit_id) == 4 and hit_id[0] == rid and hit_id[1] == "X"
    finally:
        kb.close()


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
