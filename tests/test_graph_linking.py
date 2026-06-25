"""
tests/test_graph_linking.py -- reference extraction + matching (Awareness port)
and its integration into GraphKnowledgeMemory.

Pins the content-driven edge-building mechanism ported from the Awareness-SDK
memory graph (link-discovery.mjs):

  * extract_references: pull backtick / file-path / PascalCase references out of a
    text node, skip fenced code blocks, ignore common-word PascalCase, dedup;
  * match_references: match those references to known node titles with the
    Awareness confidence weights (path 1.0 / basename 0.6 / backtick 0.8 /
    pascal 0.7);
  * GraphKnowledgeMemory.link_by_reference: turn a node's text into typed
    'reference' edges, and show that spreading activation then traverses them --
    so a note recalls the entities it merely *mentions*, not only its
    embedding-neighbours.

Scope honesty: this is the wiring + matching MECHANISM (regex + string match,
zero LLM), faithful to the borrowed source. Whether reference edges improve a
real recall task is a separate effectiveness question.
"""

import os

import torch

from mt_lnn.graph_linking import extract_references, match_references
from mt_lnn.graph_memory import GraphKnowledgeMemory


# ---------------------------------------------------------------------------
# extract_references
# ---------------------------------------------------------------------------

def test_extract_backtick_path_pascal():
    text = (
        "The `Indexer` reads from src/core/indexer.mjs and calls `store.write()`.\n"
        "See WorkspaceScanner for details. API and JSON are ignored.\n"
    )
    refs = extract_references(text)
    by_type = {(r["name"], r["type"]) for r in refs}
    assert ("Indexer", "backtick") in by_type
    assert ("write", "backtick") in by_type          # store.write() -> write
    assert ("src/core/indexer.mjs", "path") in by_type
    assert ("WorkspaceScanner", "pascal") in by_type
    # IGNORE_PASCAL words are not references; single-word "API"/"JSON" are too.
    names = {r["name"] for r in refs}
    assert "API" not in names and "JSON" not in names


def test_extract_skips_code_fences_and_dedups():
    text = (
        "`Foo` appears once.\n"
        "```\n`Foo` inside a fence must be skipped\n```\n"
        "`Foo` again -> still deduped.\n"
    )
    refs = [r for r in extract_references(text) if r["name"] == "Foo"]
    assert len(refs) == 1
    assert refs[0]["line"] == 1


# ---------------------------------------------------------------------------
# match_references (confidence weights)
# ---------------------------------------------------------------------------

def test_match_confidence_weights():
    nodes = [
        {"id": 1, "title": "Indexer"},
        {"id": 2, "title": "src/core/indexer.mjs"},
        {"id": 3, "title": "indexer.mjs"},
        {"id": 4, "title": "WorkspaceScanner"},
    ]
    refs = [
        {"name": "Indexer", "type": "backtick", "line": 1},
        {"name": "src/core/indexer.mjs", "type": "path", "line": 2},
        {"name": "WorkspaceScanner", "type": "pascal", "line": 3},
    ]
    links = match_references(refs, nodes)
    conf = {(l["target_id"]): l["confidence"] for l in links}
    assert conf[1] == 0.8        # backtick
    assert conf[2] == 1.0        # exact path
    assert conf[4] == 0.7        # pascal

    # A path that only matches by basename -> 0.6.
    links2 = match_references(
        [{"name": "deep/nested/indexer.mjs", "type": "path", "line": 1}], nodes
    )
    assert links2 and links2[0]["target_id"] == 3 and links2[0]["confidence"] == 0.6


def test_match_empty_inputs():
    assert match_references([], [{"id": 1, "title": "X"}]) == []
    assert match_references([{"name": "X", "type": "backtick", "line": 1}], []) == []


# ---------------------------------------------------------------------------
# Integration: link_by_reference builds traversable edges
# ---------------------------------------------------------------------------

def _e(dim, i):
    v = torch.zeros(dim)
    v[i] = 1.0
    return v


def _cleanup(db):
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db + suffix)
        except FileNotFoundError:
            pass


def test_link_by_reference_creates_traversable_edges():
    db = os.path.join(os.path.dirname(__file__), "_graph_reflink.db")
    _cleanup(db)
    try:
        dim = 8
        with GraphKnowledgeMemory(key_dim=dim, db_path=db) as g:
            # Target nodes carry a title in meta so they are matchable.
            idx = g.add_node(_e(dim, 1), "indexer node", meta={"title": "Indexer"})
            scan = g.add_node(_e(dim, 2), "scanner node",
                              meta={"title": "WorkspaceScanner"})
            # A note node (orthogonal key) that MENTIONS both by name.
            note = g.add_node(
                _e(dim, 0), "design note",
                meta={"title": "DesignNote"},
            )
            n = g.link_by_reference(
                note, "The `Indexer` is driven by WorkspaceScanner."
            )
            assert n == 2, f"expected 2 reference edges, got {n}"

            # Reference edges are directional (note -> mentioned); spreading from
            # the note's key reaches both mentioned entities though their keys are
            # orthogonal to the query.
            spread = g.spread_activation(_e(dim, 0), seeds=1, hops=1, decay=0.9, top_k=5)
            contents = {c for (c, _a, _m) in spread}
            assert contents == {"design note", "indexer node", "scanner node"}, \
                f"reference edges were not traversed, got {contents}"

            # Directional: querying from the Indexer's key does NOT walk back to
            # the note (no reverse edge was created).
            back = g.spread_activation(_e(dim, 1), seeds=1, hops=1, decay=0.9, top_k=5)
            assert {c for (c, _a, _m) in back} == {"indexer node"}
    finally:
        _cleanup(db)


def test_link_by_reference_no_matches_is_zero():
    db = os.path.join(os.path.dirname(__file__), "_graph_reflink_zero.db")
    _cleanup(db)
    try:
        with GraphKnowledgeMemory(key_dim=4, db_path=db) as g:
            note = g.add_node(_e(4, 0), "note", meta={"title": "Note"})
            # No other titled nodes to match -> no edges, no crash.
            assert g.link_by_reference(note, "mentions `Nothing` here.") == 0
            assert g.n_edges() == 0
    finally:
        _cleanup(db)


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
