"""
tests/test_knowledge_memory.py — behavioural contract for the persistent
long-term declarative knowledge store (mt_lnn/knowledge_memory.py, 2026-06-15).

This is the content-addressable counterpart to SessionMemory (which stores
recurrent hidden state). We pin the properties a knowledge tier must guarantee:
  • write → query round-trip retrieves the right content by cosine similarity;
  • ranking is by cosine similarity (aligned key beats orthogonal key);
  • recall() returns the single best content, None on an empty store;
  • persistence: knowledge written, closed, and reopened from the same file is
    still queryable (survives process restart → can be synced across devices);
  • bounded footprint: max_entries evicts the least-recently-accessed record;
  • dimension mismatches raise instead of silently corrupting the index;
  • metadata round-trips.

Run:  python -m pytest tests/test_knowledge_memory.py -v
"""
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn import PersistentKnowledgeMemory                          # noqa: E402


def _basis(dim, i):
    """Unit basis vector e_i in R^dim (mutually orthogonal keys)."""
    v = torch.zeros(dim)
    v[i] = 1.0
    return v


# --------------------------------------------------------------------------- #
# write / query round-trip + cosine ranking                                    #
# --------------------------------------------------------------------------- #
def test_write_query_roundtrip_by_similarity():
    kb = PersistentKnowledgeMemory(key_dim=8, db_path=":memory:")
    kb.write(_basis(8, 0), "fact-A")
    kb.write(_basis(8, 1), "fact-B")
    kb.write(_basis(8, 2), "fact-C")

    # Query closest to e_1 → fact-B should rank first.
    hits = kb.query(_basis(8, 1), top_k=3)
    assert len(hits) == 3
    assert hits[0][0] == "fact-B"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)   # cosine of identical unit vectors
    # The orthogonal others score ~0.
    assert hits[1][1] == pytest.approx(0.0, abs=1e-5)


def test_query_ranks_aligned_above_orthogonal():
    kb = PersistentKnowledgeMemory(key_dim=4, db_path=":memory:")
    kb.write(torch.tensor([1.0, 1.0, 0.0, 0.0]), "diagonal")
    kb.write(torch.tensor([0.0, 0.0, 1.0, 0.0]), "orthogonal")
    hits = kb.query(torch.tensor([2.0, 2.0, 0.0, 0.0]), top_k=2)  # parallel to "diagonal"
    assert hits[0][0] == "diagonal"
    assert hits[0][1] > hits[1][1]


def test_recall_returns_best_or_none():
    kb = PersistentKnowledgeMemory(key_dim=4, db_path=":memory:")
    assert kb.recall(_basis(4, 0)) is None        # empty store
    kb.write(_basis(4, 3), "the-answer")
    assert kb.recall(_basis(4, 3)) == "the-answer"


# --------------------------------------------------------------------------- #
# persistence across "process restart"                                         #
# --------------------------------------------------------------------------- #
def test_persistence_across_reopen(tmp_path):
    db = str(tmp_path / "kb.db")
    kb = PersistentKnowledgeMemory(key_dim=6, db_path=db)
    kb.write(_basis(6, 2), {"text": "persistent knowledge", "n": 7})
    kb.close()

    # Re-open the same file in a fresh instance → knowledge survived.
    kb2 = PersistentKnowledgeMemory(key_dim=6, db_path=db)
    assert len(kb2) == 1
    recalled = kb2.recall(_basis(6, 2))
    assert recalled == {"text": "persistent knowledge", "n": 7}
    kb2.close()


# --------------------------------------------------------------------------- #
# bounded footprint — LRU eviction                                             #
# --------------------------------------------------------------------------- #
def test_capacity_evicts_least_recently_used():
    kb = PersistentKnowledgeMemory(key_dim=4, db_path=":memory:", max_entries=2)
    kb.write(_basis(4, 0), "A")
    kb.write(_basis(4, 1), "B")
    assert len(kb) == 2

    # Touch A via a query so B becomes the least-recently-accessed.
    kb.query(_basis(4, 0), top_k=1)
    # Insert C → overflow → evict the LRU record, which is B.
    kb.write(_basis(4, 2), "C")
    assert len(kb) == 2

    contents = {h[0] for h in kb.query(_basis(4, 0), top_k=2, touch=False)} \
        | {h[0] for h in kb.query(_basis(4, 2), top_k=2, touch=False)}
    assert "A" in contents
    assert "C" in contents
    assert "B" not in contents


# --------------------------------------------------------------------------- #
# metadata round-trips                                                         #
# --------------------------------------------------------------------------- #
def test_meta_roundtrip():
    kb = PersistentKnowledgeMemory(key_dim=4, db_path=":memory:")
    kb.write(_basis(4, 0), "content", meta={"source": "doc1", "page": 3})
    content, score, meta = kb.query(_basis(4, 0), top_k=1)[0]
    assert content == "content"
    assert meta == {"source": "doc1", "page": 3}


# --------------------------------------------------------------------------- #
# defensive validation                                                         #
# --------------------------------------------------------------------------- #
def test_dimension_mismatch_raises():
    kb = PersistentKnowledgeMemory(key_dim=8, db_path=":memory:")
    with pytest.raises(ValueError):
        kb.write(_basis(4, 0), "wrong-dim")
    kb.write(_basis(8, 0), "ok")
    with pytest.raises(ValueError):
        kb.query(_basis(4, 0), top_k=1)


def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        PersistentKnowledgeMemory(key_dim=0, db_path=":memory:")
    with pytest.raises(ValueError):
        PersistentKnowledgeMemory(key_dim=4, db_path=":memory:", max_entries=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
