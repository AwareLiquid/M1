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

import json
import os

import pytest
import torch

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
    # Schema parity with server_hf.py — the demo frontend's applyModel()
    # switches on these three fields; they used to be absent entirely.
    assert j["base_model"] is None
    assert j["adapter_loaded"] is False
    assert j["is_baseline"] is True        # SMALL mode = fresh untrained model
    assert "checkpoint" in j


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
    import serve.server as srv

    monkeypatch.setenv("PARTNER_STATS_TOKEN", "s3cret")
    monkeypatch.setattr(srv, "_PARTNER_DIR", str(tmp_path))
    (tmp_path / "counts.json").write_text(json.dumps({"clawhunt": 3}))
    r = client.get("/partners", params={"token": "s3cret"})
    assert r.status_code == 200
    body = r.json()
    assert body == {"counts": {"clawhunt": 3}, "total": 3}


# ---------------------------------------------------------------------------
# _record_partner_hit — counts.json must be written atomically (P0-3).
# A truncate-in-place write interrupted by `docker stop` used to leave a file
# the next read rejects, silently zeroing all referral history.
# ---------------------------------------------------------------------------

def test_partner_hit_atomic_write_leaves_no_tmp_and_valid_json(monkeypatch, tmp_path):
    import serve.server as srv

    class _FakeRequest:
        headers: dict = {}

        class client:
            host = "127.0.0.1"

    monkeypatch.setattr(srv, "_PARTNER_DIR", str(tmp_path))
    srv._record_partner_hit("clawhunt", _FakeRequest())
    srv._record_partner_hit("clawhunt", _FakeRequest())
    counts = json.loads((tmp_path / "counts.json").read_text())
    assert counts == {"clawhunt": 2}
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"temp files leaked: {leftovers}"


# ---------------------------------------------------------------------------
# _gen_lock — bounded waiting (P0-3). No more hanging forever behind a
# long CPU generation: 503 after timeout, 503 immediately when queue full.
# ---------------------------------------------------------------------------

def test_gen_lock_times_out_with_503(client, monkeypatch):
    import serve.server as srv

    monkeypatch.setattr(srv, "_GEN_LOCK_TIMEOUT", 0.05)
    acquired = srv._MODEL_LOCK.acquire(timeout=5)
    assert acquired
    try:
        r = client.post("/v1/completions",
                        json={"prompt": "x", "max_new_tokens": 2})
        assert r.status_code == 503
    finally:
        srv._MODEL_LOCK.release()


def test_gen_lock_full_queue_rejects_immediately(client, monkeypatch):
    import serve.server as srv

    monkeypatch.setattr(srv, "_GEN_QUEUE_CAP", 0)
    r = client.post("/v1/completions",
                    json={"prompt": "x", "max_new_tokens": 2})
    assert r.status_code == 503
    assert "queue full" in r.json()["detail"]
    # A refused request must not leave the waiter counter negative/leaked.
    assert srv._GEN_WAITING == 0


# ---------------------------------------------------------------------------
# QUANTIZE_INT8 startup flag (P0-2) — weight-only int8 serving path.
# ---------------------------------------------------------------------------

def test_quantize_int8_startup_flag(monkeypatch):
    import serve.server as srv

    monkeypatch.setenv("QUANTIZE_INT8", "1")
    saved = dict(srv._STATE)
    # _startup() 会 torch.set_grad_enabled(False) —— 线程局部标志。经
    # TestClient lifespan 时跑在 worker 线程无碍；这里在主线程直接调用，
    # 必须保存/恢复，否则污染后续所有需要 autograd 的测试
    # （全量套件 17 个 "does not require grad" 失败的根因）。
    grad_was = torch.is_grad_enabled()
    srv._STATE.clear()
    srv._startup()
    try:
        assert srv._STATE["ready"]
        q = srv._STATE["quantization"]
        assert q["mode"] == "int8-weightonly"
        assert q["n_quantized"] > 0
        # 量化后模型照常生成（Int8Weight 在 forward 里惰性反量化）
        m = srv._STATE["model"]
        ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=4, do_sample=False)
        assert out.shape[1] >= 4
    finally:
        srv._STATE.clear()
        srv._STATE.update(saved)   # 恢复 module 级 client 的就绪状态
        torch.set_grad_enabled(grad_was)


# ---------------------------------------------------------------------------
# RAG integration (P2-9): BM25 index → completions prefix + search endpoints.
# ---------------------------------------------------------------------------

@pytest.fixture()
def rag_index(client, monkeypatch, tmp_path):
    """Load a tiny BM25 index into _STATE directly (no RAG_INDEX file needed)."""
    import serve.server as srv
    from mt_lnn.rag import BM25Index

    idx = BM25Index([
        "The mitochondrion is the powerhouse of the cell and produces ATP.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The French Revolution began in 1789 and toppled the monarchy.",
    ])
    monkeypatch.setitem(srv._STATE, "rag_index", idx)
    yield idx
    srv._STATE["rag_index"] = None


def test_rag_completions_prefix_and_report(client, rag_index):
    r = client.post("/v1/completions", json={
        "prompt": "What produces ATP in the cell?", "max_new_tokens": 4,
        "do_sample": False, "rag": True, "rag_top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["rag"]["used"] is True
    assert body["rag"]["n_hits"] == 2


def test_rag_disabled_by_default_in_request(client):
    r = client.post("/v1/completions", json={
        "prompt": "hello", "max_new_tokens": 2, "do_sample": False})
    assert r.status_code == 200
    assert "rag" not in r.json()


def test_rag_search_endpoint(client, rag_index):
    r = client.get("/v1/rag/search", params={"q": "ATP powerhouse", "top_k": 2})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert 1 <= len(hits) <= 2
    assert "mitochondrion" in hits[0]["passage"]
    assert hits[0]["score"] >= hits[-1]["score"]


def test_rag_status_and_disabled_404(client, rag_index):
    assert client.get("/v1/rag").json()["n_passages"] == 3
    client.app  # noqa: B018 - clarity
    import serve.server as srv
    saved = srv._STATE["rag_index"]
    srv._STATE["rag_index"] = None
    try:
        assert client.get("/v1/rag").status_code == 404
        assert client.get("/v1/rag/search",
                          params={"q": "x"}).status_code == 404
    finally:
        srv._STATE["rag_index"] = saved


def test_rag_stream_emits_rag_event_first(client, rag_index):
    with client.stream("POST", "/v1/completions/stream",
                       json={"prompt": "French Revolution year",
                             "max_new_tokens": 3, "do_sample": False,
                             "rag": True}) as s:
        events = [ln for ln in s.iter_lines() if ln]
    first = events[0]
    assert first.startswith("data: ") and '"rag"' in first
    assert events[-1].endswith("[DONE]")


# ---------------------------------------------------------------------------
# Auth config preflight (P2-10): strict + zero active keys fails fast.
# ---------------------------------------------------------------------------

def test_validate_auth_config_strict_empty_fails(tmp_path):
    from mt_lnn.api_auth import ApiKeyStore
    from serve.server import _validate_auth_config

    store = ApiKeyStore(str(tmp_path / "keys.db"))
    with pytest.raises(RuntimeError, match="no ACTIVE keys"):
        _validate_auth_config("strict", store)
    store.issue("ops")
    _validate_auth_config("strict", store)        # 有可用 key 即通过


def test_validate_auth_config_off_and_soft(tmp_path):
    from mt_lnn.api_auth import ApiKeyStore
    from serve.server import _validate_auth_config

    _validate_auth_config("off", None)            # off → 不动
    _validate_auth_config("soft", ApiKeyStore(str(tmp_path / "k.db")))


# ---------------------------------------------------------------------------
# RAG 边界路径 (P0-3 补缺): 截断 / 无命中 / RAG_INDEX pickle 启动加载。
# ---------------------------------------------------------------------------

def test_rag_context_truncated_to_max_chars(client, rag_index, monkeypatch):
    import serve.server as srv

    monkeypatch.setattr(srv, "RAG_MAX_CHARS", 40)
    req = srv.CompletionRequest(
        prompt="What produces ATP in the cell mitochondrion?",
        rag=True, rag_top_k=3)
    original = req.prompt
    info = srv._apply_rag(req)
    assert info["used"] is True
    # 上下文段被截到 ~40 字符: 改写后 prompt 不应比 原 prompt+模板+40 长多少
    budget = len(original) + len("[Retrieved context]\n\n\n\n") + 40 + 20
    assert len(req.prompt) <= budget, (len(req.prompt), budget)


def test_rag_no_hit_reports_unused_and_keeps_prompt(client, rag_index):
    import serve.server as srv

    req = srv.CompletionRequest(
        prompt="zzz qqq xxx unrelated tokens", rag=True, rag_top_k=2)
    original = req.prompt
    info = srv._apply_rag(req)
    assert info == {"used": False, "n_hits": 0, "top_k": 2}
    assert req.prompt == original          # 无命中不改写


def test_rag_index_pickle_loaded_at_startup(monkeypatch, tmp_path):
    import pickle

    import serve.server as srv
    from mt_lnn.rag import BM25Index

    idx = BM25Index(["alpha beta gamma", "delta epsilon zeta"])
    p = tmp_path / "rag.pkl"
    p.write_bytes(pickle.dumps(idx))
    monkeypatch.setenv("RAG_INDEX", str(p))

    saved, grad = dict(srv._STATE), torch.is_grad_enabled()
    srv._STATE.clear()
    srv._startup()
    try:
        assert srv._STATE["rag_index"] is not None
        assert len(srv._STATE["rag_index"]) == 2
    finally:
        srv._STATE.clear()
        srv._STATE.update(saved)
        torch.set_grad_enabled(grad)


def test_rag_index_garbage_pickle_degrades_to_disabled(monkeypatch, tmp_path):
    import pickle

    import serve.server as srv

    p = tmp_path / "not_an_index.pkl"
    p.write_bytes(pickle.dumps({"not": "a BM25Index"}))
    monkeypatch.setenv("RAG_INDEX", str(p))

    saved, grad = dict(srv._STATE), torch.is_grad_enabled()
    srv._STATE.clear()
    srv._startup()
    try:
        # 非 BM25Index: rag 优雅关闭, 服务照常就绪
        assert srv._STATE["rag_index"] is None
        assert srv._STATE["ready"]
    finally:
        srv._STATE.clear()
        srv._STATE.update(saved)
        torch.set_grad_enabled(grad)


# ---------------------------------------------------------------------------
# counts.json 原子性的本意场景 (P0-3 补缺): 写入中途失败 → 旧计数完好。
# ---------------------------------------------------------------------------

def test_partner_hit_interrupted_write_keeps_old_counts(monkeypatch, tmp_path):
    import serve.server as srv

    (tmp_path / "counts.json").write_text(json.dumps({"clawhunt": 41}))
    monkeypatch.setattr(srv, "_PARTNER_DIR", str(tmp_path))
    monkeypatch.setattr(srv.json, "dump",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))

    class _FakeRequest:
        headers: dict = {}

        class client:
            host = "127.0.0.1"

    with pytest.raises(RuntimeError, match="disk full"):
        srv._record_partner_hit("clawhunt", _FakeRequest())
    # 原子性: 旧文件原封不动, 临时文件被清理
    assert json.loads((tmp_path / "counts.json").read_text()) == {"clawhunt": 41}
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
