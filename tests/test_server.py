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


# ---------------------------------------------------------------------------
# /partners stats endpoint — PARTNER_STATS_TOKEN is fail-closed (P0-1).
# An unset token must NOT fall back to "no auth": the old `if token and ...`
# short-circuit published every partner's counts whenever the env var was
# missing or left as the compose-file default "".
# ---------------------------------------------------------------------------

def test_partner_stats_locked_without_token(client, monkeypatch, tmp_path):
    monkeypatch.delenv("PARTNER_STATS_TOKEN", raising=False)
    r = client.get("/partners")
    assert r.status_code == 503
    assert "counts" not in r.json()


def test_partner_stats_rejects_wrong_token(client, monkeypatch):
    monkeypatch.setenv("PARTNER_STATS_TOKEN", "s3cret")
    r = client.get("/partners", params={"token": "wrong"})
    assert r.status_code == 401


def test_partner_stats_serves_with_valid_token(client, monkeypatch, tmp_path):
    import json as _json

    import serve.server as srv

    monkeypatch.setenv("PARTNER_STATS_TOKEN", "s3cret")
    monkeypatch.setattr(srv, "_PARTNER_DIR", str(tmp_path))
    (tmp_path / "counts.json").write_text(_json.dumps({"clawhunt": 3}))
    r = client.get("/partners", params={"token": "s3cret"})
    assert r.status_code == 200
    body = r.json()
    assert body == {"counts": {"clawhunt": 3}, "total": 3}
