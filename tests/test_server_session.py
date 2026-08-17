"""
tests/test_server_session.py -- cross-session recurrent-state persistence (Gap 4)
wired into the inference server (serve/server.py), SMALL byte-level model.

Pins the persistence HTTP contract:

  * A completion with a session_id continues from (and writes back) that
    session's saved LNN h_prev, so token_count accumulates ACROSS requests --
    proving the recurrent state is persisted and resumed, not recomputed fresh.
  * Sessions are listable, inspectable, and deletable via /v1/sessions.
  * WITHOUT a session_id the server stays stateless (no session fields), even
    when SESSION_DB is configured -- the zero-regression default.

Scope honesty: this verifies the persistence PLUMBING (store -> resume ->
update -> forget). That h_prev actually changes the model's logits is proven at
the model level in tests/test_memory.py::test_persistent_state_continues_inference.

Skipped automatically if FastAPI's TestClient deps (httpx) are unavailable.
"""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

# Configure the server BEFORE importing the app so startup picks these up.
os.environ["SMALL"] = "1"
os.environ["MAX_NEW_TOKENS_CAP"] = "32"
os.environ["SESSION_DB"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_server_session_test.db"
)

from fastapi.testclient import TestClient  # noqa: E402

from serve.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Start clean: remove any stale db from a previous run.
    db = os.environ["SESSION_DB"]
    # Clean both the session db and the sibling knowledge db (SESSION_DB +
    # ".knowledge.db") the /v1/sleep bridge consolidates into, with WAL sidecars.
    bases = (db, db + ".knowledge.db", db + ".graph.db")
    def _wipe():
        for base in bases:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(base + suffix)
                except FileNotFoundError:
                    pass
    _wipe()
    with TestClient(app) as c:   # triggers startup (reads SESSION_DB)
        yield c
    _wipe()


def _complete(client, **body):
    body.setdefault("max_new_tokens", 4)
    body.setdefault("do_sample", False)   # greedy → deterministic
    r = client.post("/v1/completions", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Persistence: token_count accumulates across separate requests
# ---------------------------------------------------------------------------

def test_session_token_count_accumulates_across_requests(client):
    a = _complete(client, prompt="hello", session_id="convo1")
    assert a["session_id"] == "convo1"
    first = a["session_token_count"]
    assert first == len("hello") + a["n_new_tokens"]   # prompt bytes + new tokens

    b = _complete(client, prompt="world", session_id="convo1")
    second = b["session_token_count"]
    # The session RESUMED: the second turn's count includes the first turn's,
    # not a fresh recount -> strictly greater, by prompt2 + its new tokens.
    assert second == first + len("world") + b["n_new_tokens"], \
        "session did not resume from persisted state (token_count reset)"


def test_session_metadata_listed_and_inspectable(client):
    _complete(client, prompt="abc", session_id="convo2")
    listed = client.get("/v1/sessions").json()["sessions"]
    ids = {s["session_id"] for s in listed}
    assert "convo2" in ids
    info = client.get("/v1/sessions/convo2").json()
    assert info["session_id"] == "convo2"
    assert info["token_count"] > 0
    assert "updated_at" in info


def test_session_deletable(client):
    _complete(client, prompt="zzz", session_id="convo_del")
    d = client.delete("/v1/sessions/convo_del")
    assert d.status_code == 200 and d.json()["deleted"] is True
    # Now gone.
    assert client.get("/v1/sessions/convo_del").status_code == 404
    assert client.delete("/v1/sessions/convo_del").status_code == 404


# ---------------------------------------------------------------------------
# Zero-regression default: no session_id => stateless
# ---------------------------------------------------------------------------

def test_no_session_id_is_stateless(client):
    j = _complete(client, prompt="hi")           # no session_id
    assert "session_id" not in j
    assert "session_token_count" not in j
    # And nothing was persisted under an empty/None key.
    listed = client.get("/v1/sessions").json()["sessions"]
    assert all(s["session_id"] for s in listed)  # no blank-key rows


def test_sleep_consolidates_sessions_into_knowledge_store(client):
    """POST /v1/sleep replays the persisted sessions and consolidates them into
    the long-term knowledge store (Gap 4 NREM). After a turn under a fresh
    session_id, a sleep pass reports >=1 replayed/consolidated and a non-empty
    knowledge store keyed on a positive-dim signature -- the working->long-term
    transfer actually ran end-to-end through the HTTP surface."""
    _complete(client, prompt="remember me", session_id="sleep_src")
    r = client.post("/v1/sleep")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["replayed"] >= 1
    assert body["consolidated"] >= 1
    assert body["knowledge_entries"] >= 1
    assert body["key_dim"] > 0
    # Idempotent-safe: a second pass still succeeds (re-consolidates, no crash).
    assert client.post("/v1/sleep").status_code == 200
    # Default pass runs NREM only -- no SHY downscaling block.
    assert "downscale" not in body


def test_sleep_build_graph_sediments_relational_store(client):
    """POST /v1/sleep?build_graph=true additionally sediments sessions into the
    relational graph store. A couple of persisted sessions become nodes; the
    response carries a 'graph' block with the node count and a graph_db path.
    Default (build_graph omitted) leaves no graph block -- zero regression."""
    _complete(client, prompt="graph one", session_id="g_one")
    _complete(client, prompt="graph two", session_id="g_two")

    plain = client.post("/v1/sleep").json()
    assert "graph" not in plain

    r = client.post("/v1/sleep", params={"build_graph": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "graph" in body
    assert body["graph"]["nodes"] >= 2
    assert body["graph"]["consolidated"] >= 2
    assert body["graph"]["graph_db"].endswith(".graph.db")
    # link_threshold validation.
    bad = client.post("/v1/sleep", params={"build_graph": "true", "link_threshold": "1.5"})
    assert bad.status_code == 400


def test_sleep_build_graph_adaptive_link_quantile(client):
    """link_quantile makes the graph link cut ADAPTIVE -- derived from the
    sessions' own cosine band instead of a hand-tuned absolute. The response
    echoes the data-derived threshold actually used (a real cosine in [0, 1]),
    and an out-of-range quantile is rejected."""
    _complete(client, prompt="adaptive one", session_id="aq_one")
    _complete(client, prompt="adaptive two", session_id="aq_two")

    r = client.post("/v1/sleep", params={"build_graph": "true", "link_quantile": "0.5"})
    assert r.status_code == 200, r.text
    g = r.json()["graph"]
    assert g["nodes"] >= 2
    # The cut is a real cosine derived from the data -> range [-1, 1] (centered
    # signatures of a tiny corpus can be anti-correlated, so it may be negative).
    assert -1.0 <= g["link_threshold"] <= 1.0
    # Validation: quantile must be in [0, 1].
    bad = client.post("/v1/sleep", params={"build_graph": "true", "link_quantile": "1.5"})
    assert bad.status_code == 400


def test_sleep_shy_downscaling_is_optin_and_noop_without_adapters(client):
    """downscale_factor < 1.0 runs the SHY stage; the served small model has no
    MT adapters, so it must be a safe all-zero no-op block (proves the stage is
    wired without mutating a from-scratch model's weights), while < 0 or > 1 is
    rejected and the default (1.0) omits the stage entirely."""
    _complete(client, prompt="sleep tight", session_id="shy_src")
    r = client.post("/v1/sleep", params={"downscale_factor": 0.5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "downscale" in body
    # from-scratch served model has no MTResidualAdapter -> nothing rescaled.
    assert body["downscale"]["n_adapters"] == 0
    assert body["downscale"]["n_tensors"] == 0
    assert body["downscale"]["factor"] == 0.5
    # Validation: factor must be in (0, 1].
    assert client.post("/v1/sleep", params={"downscale_factor": 0.0}).status_code == 400
    assert client.post("/v1/sleep", params={"downscale_factor": 1.5}).status_code == 400


def test_sleep_requires_session_persistence(client, monkeypatch):
    """Without SESSION_DB the sleep bridge has nothing to consolidate and must
    404 rather than silently creating an empty knowledge store. monkeypatch
    restores the live session_db after the test, leaving the client usable."""
    import serve.server as srv
    monkeypatch.setitem(srv._STATE, "session_db", "")
    assert client.post("/v1/sleep").status_code == 404


def test_stream_supports_session(client):
    with client.stream("POST", "/v1/completions/stream",
                       json={"prompt": "hey", "max_new_tokens": 4,
                             "do_sample": False, "session_id": "stream1"}) as s:
        events = [ln for ln in s.iter_lines() if ln]
    assert events[-1].endswith("[DONE]")
    # The streamed turn persisted state too.
    info = client.get("/v1/sessions/stream1").json()
    assert info["token_count"] > 0
