"""MemoryBroker contract tests — one workload, three backends.

The external backend is exercised against an in-process mock of the
Awareness-SDK local daemon's REST surface (the exact routes, statuses and
response shapes the adapter consumes) — no live daemon needed, and the
M1-repo-never-imports-SDK rule is trivially preserved.

The parametric/graph assertions re-state the invariants those backends are
sold on (bit-exact snapshots, O(1) state, single-binding forget) THROUGH the
broker face, so a protocol regression cannot hide behind the wrapped classes'
own test suites.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mt_lnn.memory_broker import (
    BrokerError,
    ExternalBroker,
    GraphBroker,
    MemoryBroker,
    ParametricBroker,
    create_broker,
)


# --------------------------------------------------------------------- fake

class _FakeDaemonHandler(BaseHTTPRequestHandler):
    """Minimal Awareness local daemon: the /api/v1 routes the adapter uses.

    Behavior mirrors the real surface's contract points the broker depends
    on: write returns {status, id} (incl. the duplicate short-circuit),
    search returns ranked items WITHOUT numeric scores and is workspace- (not
    session-) scoped, and the list endpoint pages via limit/offset.
    """

    stores: dict = {}          # class-level so every test request sees one store
    next_id: int = 0

    def log_message(self, *args):   # silence the test output
        pass

    def _json(self, payload, status=200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _seen(self):                # rows sorted newest-first, like the daemon
        return sorted(self.stores.values(), key=lambda r: r["seq"], reverse=True)

    def do_POST(self):
        if self.path != "/api/v1/memories":
            self._json({"error": "Not found"}, 404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        for row in self._seen():
            if row["content"] == body["content"]:
                self._json({"status": "duplicate", "id": row["id"]})
                return
        _FakeDaemonHandler.next_id += 1
        rid = f"mem_20260905_{_FakeDaemonHandler.next_id:04x}"
        self.stores[rid] = {"seq": _FakeDaemonHandler.next_id, "id": rid, **body}
        self._json({"status": "ok", "id": rid, "mode": "local"})

    def do_GET(self):
        if self.path.startswith("/api/v1/memories/search"):
            q = _param(self.path, "q")
            limit = int(_param(self.path, "limit", "20"))
            items = [dict(r, session_id=r["session_id"])
                     for r in self._seen() if q in r["content"]][:limit]
            self._json({"items": items, "total": len(items), "query": q})
            return
        if self.path.startswith("/api/v1/memories"):
            limit = int(_param(self.path, "limit", "50"))
            offset = int(_param(self.path, "offset", "0"))
            rows = self._seen()[offset:offset + limit]
            self._json({"items": rows, "total": len(self.stores),
                        "limit": limit, "offset": offset})
            return
        self._json({"error": "Not found"}, 404)


def _param(path, key, default=None):
    from urllib.parse import parse_qs, urlparse
    vals = parse_qs(urlparse(path).query).get(key)
    return vals[0] if vals else default


@pytest.fixture()
def fake_daemon():
    _FakeDaemonHandler.stores = {}
    _FakeDaemonHandler.next_id = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeDaemonHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


# ---------------------------------------------------------------- parametric

class TestParametricBroker:
    def test_is_broker_and_write_recall_roundtrip(self):
        with create_broker("parametric", d_mem=64) as b:
            assert isinstance(b, MemoryBroker)
            assert b.write("s1", "favorite color", "blue") == "favorite color"
            hits = b.recall("s1", "favorite color", top_k=3)
            assert hits[0].value == "blue" and hits[0].score > 0.9

    def test_empty_session_recalls_nothing(self):
        b = ParametricBroker(d_mem=64)
        assert b.recall("never-written", "anything") == []

    def test_forget_single_binding_then_whole_session(self):
        b = ParametricBroker(d_mem=64)
        b.write("s1", "k1", "v1")
        b.write("s1", "k2", "v2")
        assert b.forget("s1", "k1") is True
        assert b.recall("s1", "k1") == []          # "no memory of it"
        assert b.recall("s1", "k2")[0].value == "v2"   # neighbour survives
        assert b.forget("s1") is True
        assert b.recall("s1", "k2") == []
        assert b.forget("ghost") is False

    def test_snapshot_restore_bit_exact_through_broker(self):
        b = ParametricBroker(d_mem=64)
        b.write("s1", "k1", "v1")
        b.write("s1", "k2", "v2")
        snap = b.snapshot("s1")
        b.write("s1", "k1", "MUTATED")             # drift after snapshot
        b.restore("s1", snap)
        assert b.recall("s1", "k1")[0].value == "v1"   # byte-level rollback
        with pytest.raises(KeyError):
            b.snapshot("ghost")

    def test_state_bytes_constant_in_writes(self):
        b = ParametricBroker(d_mem=64)
        b.write("s1", "k1", "v1")
        size1 = b.state_bytes("s1")
        for i in range(50):
            b.write("s1", f"k{i}", f"v{i}")
        assert b.state_bytes("s1") == size1        # the O(1) claim, broker face
        assert b.state_bytes("ghost") == 0

    def test_meta_has_no_storage_field(self):
        b = ParametricBroker(d_mem=64)
        b.write("s1", "k", "v", meta={"will": "not persist"})
        assert b.recall("s1", "k")[0].meta is None


# --------------------------------------------------------------------- graph

class TestGraphBroker:
    def test_is_broker_and_write_recall_roundtrip(self, tmp_path):
        with create_broker("graph", key_dim=64,
                           root_dir=str(tmp_path)) as b:
            assert isinstance(b, MemoryBroker)
            b.write("s1", "favorite color", "the color blue")
            hits = b.recall("s1", "favorite color", top_k=3)
            assert hits[0].value == "the color blue"
            assert hits[0].score > 0.9             # accumulated activation

    def test_sessions_are_isolated_stores(self, tmp_path):
        b = GraphBroker(key_dim=64, root_dir=str(tmp_path))
        b.write("s1", "k", "v-one")
        b.write("s2", "k", "v-two")
        assert b.recall("s1", "k")[0].value == "v-one"
        assert b.recall("s2", "k")[0].value == "v-two"
        assert b.recall("s3", "k") == []           # unknown session
        b.close()

    def test_multi_hop_surfaces_linked_node(self, tmp_path):
        # a→b typed edge with positive weight: recalling a must surface b via
        # propagation (auto_link keeps weight=cosine, which feature-hash
        # neighbours give no right to be positive — the substrate's rule).
        b = GraphBroker(key_dim=64, root_dir=str(tmp_path))
        b.write("s1", "anchor", "anchor node")
        b.write("s1", "other", "linked node")
        b.graph("s1").link(1, 2, weight=0.9)
        values = [h.value for h in b.recall("s1", "anchor", top_k=5)]
        assert "linked node" in values             # reached over the edge
        b.close()

    def test_whole_session_forget_only(self, tmp_path):
        b = GraphBroker(key_dim=64, root_dir=str(tmp_path))
        b.write("s1", "k", "v")
        with pytest.raises(NotImplementedError):
            b.forget("s1", "k")
        assert b.forget("s1") is True
        assert b.recall("s1", "k") == []
        assert b.forget("s1") is False             # store file is gone

    def test_snapshot_restore_bit_exact_and_tamper_checked(self, tmp_path):
        b = GraphBroker(key_dim=64, root_dir=str(tmp_path))
        b.write("s1", "k1", "v1")
        snap = b.snapshot("s1")
        b.write("s1", "k2", "v2")                  # drift after snapshot
        b.restore("s1", snap)
        assert [h.value for h in b.recall("s1", "k1", top_k=5)] == ["v1"]
        # node k2 is gone from the store after rollback; a near-zero cosine
        # can still seed a ~1e-9 activation (honest substrate behavior), so
        # assert above the noise floor.
        assert [h for h in b.recall("s1", "k2") if h.score > 1e-6] == []
        # flip a MIDDLE b64 char: SQLite files can end in zero bytes, whose
        # b64 IS "AAAA" — tampering there would decode to the same bytes.
        bad = dict(snap, db_bytes_b64=snap["db_bytes_b64"][:8]
                   + ("B" if snap["db_bytes_b64"][8] != "B" else "C")
                   + snap["db_bytes_b64"][9:])
        with pytest.raises(BrokerError):
            b.restore("s1", bad)
        with pytest.raises(KeyError):
            b.snapshot("ghost")
        b.close()

    def test_state_bytes_is_file_size_o_n_honest(self, tmp_path):
        b = GraphBroker(key_dim=64, root_dir=str(tmp_path))
        assert b.state_bytes("s1") == 0
        b.write("s1", "k", "v" * 4096)
        before = b.state_bytes("s1")
        for i in range(8):                          # cross SQLite page boundaries
            b.write("s1", f"k{i}", "w" * 4096)
        assert b.state_bytes("s1") > before        # grows with content: O(n)
        b.close()

    def test_lifecycle_pass_through_via_graph(self, tmp_path):
        b = GraphBroker(key_dim=64, root_dir=str(tmp_path))
        b.write("s1", "old", "v1")
        b.write("s1", "new", "v2")
        g = b.graph("s1")
        g.link(1, 2)                               # an edge so activation conducts
        g.supersede(1, 2)                          # SUPERSEDES lifecycle intact
        # node 1 is retired from results but still conducts to node 2
        assert [h.value for h in b.recall("s1", "old", top_k=5)] == ["v2"]
        b.close()


# ------------------------------------------------------------------ external

class TestExternalBroker:
    def test_is_broker(self):
        assert isinstance(ExternalBroker(), MemoryBroker)

    def test_write_maps_content_session_tags_metadata(self, fake_daemon):
        b = ExternalBroker(base_url=fake_daemon)
        rid = b.write("sess-1", "favorite color", "blue",
                      meta={"origin": "unit-test"})
        assert rid.startswith("mem_")
        row = next(iter(_FakeDaemonHandler.stores.values()))
        assert row["content"] == "blue"
        assert row["session_id"] == "sess-1"
        assert row["tags"] == ["m1:key=favorite color"]
        assert row["metadata"]["m1_key"] == "favorite color"
        assert row["metadata"]["origin"] == "unit-test"

    def test_write_dedup_returns_existing_id(self, fake_daemon):
        b = ExternalBroker(base_url=fake_daemon)
        a = b.write("s", "k", "same content")
        c = b.write("s", "k", "same content")      # daemon dedups -> 200 dup
        assert a == c

    def test_recall_filters_to_session_no_score(self, fake_daemon):
        b = ExternalBroker(base_url=fake_daemon)
        b.write("s1", "k", "the answer is 42")
        b.write("s2", "k", "the answer is 42")     # same content, other session
        hits = b.recall("s1", "answer", top_k=5)
        assert len(hits) == 1 and hits[0].value == "the answer is 42"
        assert hits[0].id.startswith("mem_")
        assert hits[0].score is None               # cascade exposes no score
        assert b.recall("ghost", "answer") == []

    def test_unsupported_ops_raise_with_reason(self, fake_daemon):
        b = ExternalBroker(base_url=fake_daemon)
        for op in (lambda: b.forget("s"),
                   lambda: b.snapshot("s"),
                   lambda: b.restore("s", {})):
            with pytest.raises(NotImplementedError):
                op()

    def test_state_bytes_walks_pages_and_sums_content(self, fake_daemon):
        b = ExternalBroker(base_url=fake_daemon)
        expected = 0
        for i in range(7):
            content = f"x{i}" * (i + 1)
            expected += len(content.encode())
            b.write("s1", f"k{i}", content)
        b.write("s2", "k", "y" * 10)               # other session: excluded
        assert b.state_bytes("s1") == expected

    def test_daemon_down_raises_broker_error(self):
        # port 1 on localhost is never our mock; transport failure -> BrokerError
        b = ExternalBroker(base_url="http://127.0.0.1:1", timeout=0.2)
        with pytest.raises(BrokerError):
            b.write("s", "k", "v")

    def test_project_dir_header_sent(self, fake_daemon):
        seen = {}
        orig = _FakeDaemonHandler.do_POST

        def spy(self):
            seen["dir"] = self.headers.get("X-Awareness-Project-Dir")
            orig(self)

        _FakeDaemonHandler.do_POST = spy
        try:
            ExternalBroker(base_url=fake_daemon,
                           project_dir="/tmp/proj").write("s", "k", "v")
        finally:
            _FakeDaemonHandler.do_POST = orig
        assert seen["dir"] == "/tmp/proj"


# ------------------------------------------------------------------- factory

class TestFactory:
    def test_known_kinds_and_kwargs_forward(self, tmp_path):
        assert create_broker("parametric", d_mem=32).memory.d_mem == 32
        assert create_broker("graph", key_dim=16,
                             root_dir=str(tmp_path)).key_dim == 16
        assert create_broker("external").base_url.endswith(":37800")

    def test_unknown_kind_lists_choices(self):
        with pytest.raises(ValueError, match="parametric"):
            create_broker("redis")
