"""
tests/test_server.py — smoke tests for the native inference server
(serve/server.py) in SMALL mode (fresh tiny byte-level model, no HF download).

Pins the HTTP contract used by deployments:

  • /health reports ok once startup ran.
  • /v1/model returns the architecture summary.
  • /v1/completions returns generated text + token bookkeeping and respects
    the max_new_tokens cap.
  • /v1/completions/stream emits one SSE event per token then [DONE].

Skipped automatically if FastAPI's TestClient deps (httpx) are unavailable.
"""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by starlette TestClient

os.environ["SMALL"] = "1"          # tiny byte-level model, no tokenizer download
os.environ["MAX_NEW_TOKENS_CAP"] = "32"

from fastapi.testclient import TestClient  # noqa: E402

from serve.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # triggers startup (model build)
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client):
    j = client.get("/v1/model").json()
    assert j["vocab_size"] == 256          # SMALL byte model
    assert j["tokenizer"] == "byte"
    assert j["n_params"] > 0


def test_completion_greedy_is_deterministic(client):
    body = {"prompt": "hello", "max_new_tokens": 8, "do_sample": False}
    a = client.post("/v1/completions", json=body).json()
    b = client.post("/v1/completions", json=body).json()
    assert a["n_new_tokens"] == 8
    assert a["tokens"] == b["tokens"], "greedy server output not deterministic"
    assert isinstance(a["text"], str)


def test_completion_respects_cap(client):
    r = client.post("/v1/completions",
                    json={"prompt": "x", "max_new_tokens": 999})
    assert r.status_code == 400


def test_completion_accepts_raw_token_ids(client):
    r = client.post("/v1/completions",
                    json={"input_ids": [104, 105, 106], "max_new_tokens": 4,
                          "do_sample": False})
    assert r.status_code == 200
    assert r.json()["n_new_tokens"] == 4


def test_stream_emits_token_events_then_done(client):
    with client.stream("POST", "/v1/completions/stream",
                       json={"prompt": "hi", "max_new_tokens": 5,
                             "do_sample": False}) as s:
        events = [ln for ln in s.iter_lines() if ln]
    assert events[-1].endswith("[DONE]")
    token_events = [e for e in events if '"token"' in e]
    assert 1 <= len(token_events) <= 5
