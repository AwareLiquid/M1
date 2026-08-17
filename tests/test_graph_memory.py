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

from mt_lnn.graph_memory import (
    GraphKnowledgeMemory,
    calibrate_threshold_from_keys,
)


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
# Adaptive (quantile-based) link threshold -- removes the hand-tuned knob
# ---------------------------------------------------------------------------

def test_calibrate_threshold_is_a_quantile_of_pairwise_cosines():
    # Two tight clusters (axes 0 and 1). Within-cluster cosines ~1, cross-cluster
    # ~0, so the distribution is bimodal. A high quantile sits in the high mode
    # (links only the strong within-cluster relations); a low quantile drops into
    # the low mode (would link across clusters too).
    torch.manual_seed(0)
    dim = 16
    cluster0 = [_e(dim, 0) + 0.02 * torch.randn(dim) for _ in range(6)]
    cluster1 = [_e(dim, 1) + 0.02 * torch.randn(dim) for _ in range(6)]
    keys = torch.stack(cluster0 + cluster1, dim=0)

    hi = calibrate_threshold_from_keys(keys, quantile=0.90)
    lo = calibrate_threshold_from_keys(keys, quantile=0.10)
    assert hi > lo, "higher quantile must give a stricter (higher) threshold"
    assert hi > 0.8, f"top-decile cosine should land in the within-cluster mode, got {hi}"
    assert lo < 0.5, f"bottom-decile cosine should land in the cross-cluster mode, got {lo}"


def test_calibrate_threshold_degrades_safely_below_two_keys():
    assert calibrate_threshold_from_keys(torch.zeros(0, 8), quantile=0.9) == 1.0
    assert calibrate_threshold_from_keys(_e(8, 0).unsqueeze(0), quantile=0.9) == 1.0


def test_adaptive_threshold_links_only_strong_relations():
    # The data-derived threshold, used to build edges, should connect near nodes
    # but not an orthogonal outlier -- the knob now comes from the corpus.
    import math
    db = _db("_graph_adaptive.db")
    _cleanup(db)
    try:
        dim = 16
        # A fan of 8 keys spread evenly over the e0->e1 quarter-arc, so pairwise
        # cosines = cos(angle gap) span a smooth UNIMODAL range from ~1 (adjacent)
        # down to ~0 (the ends). A quantile then lands at a real intermediate
        # cosine, not stuck against a degenerate all-~1 cluster.
        n = 8
        fan = []
        for i in range(n):
            theta = i * (math.pi / 2) / (n - 1)
            fan.append(math.cos(theta) * _e(dim, 0) + math.sin(theta) * _e(dim, 1))
        keys = torch.stack(fan, dim=0)

        thr = calibrate_threshold_from_keys(keys, quantile=0.50)
        assert 0.0 < thr < 1.0, f"adaptive threshold should be a mid cosine, got {thr}"

        with GraphKnowledgeMemory(
            key_dim=dim, db_path=db,
            auto_link=True, link_threshold=thr, link_top_k=n,
        ) as g:
            for i, k in enumerate(fan):
                g.write(k, f"fan{i}")
            outlier_id = g.write(_e(dim, 9), "outlier")  # orthogonal to the fan

            assert g.n_edges() > 0, \
                "calibrated threshold formed no edges among genuinely-near nodes"
            # The orthogonal outlier (cosine ~0 << thr) links to nothing.
            assert g._neighbours(outlier_id) == [], \
                "an orthogonal outlier should not be linked under the adaptive cut"
    finally:
        _cleanup(db)


# ---------------------------------------------------------------------------
# Living-knowledge lifecycle: supersede / contradict + self-correcting recall
# ---------------------------------------------------------------------------

def test_supersede_hides_stale_from_recall_but_keeps_history():
    db = _db("_graph_supersede.db")
    _cleanup(db)
    try:
        dim = 8
        with GraphKnowledgeMemory(key_dim=dim, db_path=db) as g:
            old = g.add_node(_e(dim, 0), "deadline: Friday")
            new = g.add_node(_e(dim, 0) + 0.001 * _e(dim, 1), "deadline: Monday")
            assert g.status_of(old) == "active"

            g.supersede(old, new)
            assert g.status_of(old) == "superseded"
            assert g.n_superseded() == 1
            # A directed supersedes edge new -> old was recorded (provenance).
            assert (old, 1.0) in [(d, round(w)) for (d, w) in g._neighbours(new)]

            q = _e(dim, 0)
            # Live view (default): the stale node is gone, the current one remains.
            live = {c for (c, _a, _m) in g.spread_activation(q, seeds=5, hops=0)}
            assert "deadline: Friday" not in live
            assert "deadline: Monday" in live
            # Full history on demand: exclude_superseded=False brings the old back.
            full = {c for (c, _a, _m) in
                    g.spread_activation(q, seeds=5, hops=0, exclude_superseded=False)}
            assert "deadline: Friday" in full and "deadline: Monday" in full
    finally:
        _cleanup(db)


def test_auto_supersede_retires_older_near_duplicate_only():
    db = _db("_graph_autosupersede.db")
    _cleanup(db)
    try:
        dim = 8
        with GraphKnowledgeMemory(key_dim=dim, db_path=db) as g:
            v1 = g.add_node(_e(dim, 0), "fact v1")
            unrelated = g.add_node(_e(dim, 5), "unrelated")
            # A newer near-duplicate of v1 (cosine ~1) -> should retire v1 only.
            key_v2 = _e(dim, 0) + 0.001 * _e(dim, 1)
            v2 = g.add_node(key_v2, "fact v2")
            n = g.auto_supersede(v2, key_v2, threshold=0.95, top_k=5)
            assert n == 1, f"auto_supersede should retire exactly v1, got {n}"
            assert g.status_of(v1) == "superseded"
            assert g.status_of(unrelated) == "active", "orthogonal node wrongly retired"
            assert g.status_of(v2) == "active"

            # A node never retires something NEWER than itself (only_older).
            assert g.auto_supersede(v1, _e(dim, 0), threshold=0.95) == 0
    finally:
        _cleanup(db)


def test_mark_contradiction_is_asserted_and_resolves_by_recency():
    db = _db("_graph_contradict.db")
    _cleanup(db)
    try:
        dim = 8
        with GraphKnowledgeMemory(key_dim=dim, db_path=db) as g:
            # Two NON-duplicate nodes that the caller asserts conflict (the module
            # does not detect this -- different directions, cosine ~0).
            a = g.add_node(_e(dim, 0), "the sky is green")
            b = g.add_node(_e(dim, 1), "the sky is blue")
            superseded = g.mark_contradiction(a, b)   # resolve by recency
            assert superseded == a, "older claim should be the one retired"
            assert g.status_of(a) == "superseded"
            assert g.status_of(b) == "active"
            # The contradiction is on record (symmetric edge), not destroyed.
            assert b in [d for (d, _w) in g._neighbours(a)]
            assert a in [d for (d, _w) in g._neighbours(b)]
            # resolve=False records the conflict without retiring either side.
            c = g.add_node(_e(dim, 2), "claim c")
            d = g.add_node(_e(dim, 3), "claim d")
            assert g.mark_contradiction(c, d, resolve=False) is None
            assert g.status_of(c) == "active" and g.status_of(d) == "active"
    finally:
        _cleanup(db)


def test_supersede_and_contradict_reject_self():
    import pytest
    db = _db("_graph_lifecycle_self.db")
    _cleanup(db)
    try:
        with GraphKnowledgeMemory(key_dim=4, db_path=db) as g:
            a = g.add_node(_e(4, 0), "A")
            with pytest.raises(ValueError):
                g.supersede(a, a)
            with pytest.raises(ValueError):
                g.mark_contradiction(a, a)
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
