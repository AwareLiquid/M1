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
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db + suffix)
        except FileNotFoundError:
            pass
    with TestClient(app) as c:   # triggers startup (reads SESSION_DB)
        yield c
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db + suffix)
        except FileNotFoundError:
            pass


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


def test_stream_supports_session(client):
    with client.stream("POST", "/v1/completions/stream",
                       json={"prompt": "hey", "max_new_tokens": 4,
                             "do_sample": False, "session_id": "stream1"}) as s:
        events = [ln for ln in s.iter_lines() if ln]
    assert events[-1].endswith("[DONE]")
    # The streamed turn persisted state too.
    info = client.get("/v1/sessions/stream1").json()
    assert info["token_count"] > 0
