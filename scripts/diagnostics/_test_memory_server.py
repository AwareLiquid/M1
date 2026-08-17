"""End-to-end test of declarative-memory wiring in serve/server_hf.py.

Boots the real FastAPI app in-process (TestClient, no network) against the
cached Qwen2.5-0.5B-Instruct base with KB_PATH=:memory:, then exercises:
  - /v1/model + /v1/memory/stats report memory enabled
  - empty store: completion is a strict no-op (memory_used == False)
  - /v1/memory/write stores facts
  - /v1/memory/query retrieves the correct fact (centered scoring)
  - completion with use_memory injects the right fact and reports the hit
  - an off-topic query does NOT inject a wrong fact (honest floor)

Run:  PYTHONUTF8=1 python _test_memory_server.py
"""
import json
import os
import sys

os.environ["BASE_MODEL"] = "Qwen/Qwen2.5-0.5B-Instruct"
os.environ["DEVICE"] = "cpu"
os.environ["KB_PATH"] = ":memory:"
os.environ["KB_SCORE_FLOOR"] = "0.15"
os.environ.pop("ADAPTER_CKPT", None)   # bare base; memory is adapter-agnostic
os.environ["THINKING"] = "0"

from fastapi.testclient import TestClient
from serve.server_hf import app

R = {}
with TestClient(app) as client:          # triggers startup (loads model + store)
    info = client.get("/v1/model").json()
    R["model_memory_enabled"] = info.get("memory_enabled")
    R["stats_empty"] = client.get("/v1/memory/stats").json()

    # Empty store -> completion must be a strict no-op.
    c0 = client.post("/v1/completions", json={
        "prompt": "What is the capital of France?",
        "max_new_tokens": 8, "do_sample": False, "use_memory": True}).json()
    R["empty_store_memory_used"] = c0.get("memory_used")

    facts = [
        "The internal project codename for the 2026 launch is Project Halcyon.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Python was created by Guido van Rossum.",
    ]
    ids = [client.post("/v1/memory/write", json={"content": f}).json()["id"]
           for f in facts]
    R["written_ids"] = ids
    R["stats_after_write"] = client.get("/v1/memory/stats").json()

    # Query retrieves the right fact (a fact the base model cannot know).
    q = client.post("/v1/memory/query", json={
        "query": "What is the codename for the 2026 launch?", "top_k": 3}).json()
    R["query_top1"] = q["hits"][0] if q["hits"] else None
    R["query_correct"] = bool(q["hits"]) and "Halcyon" in q["hits"][0]["content"]

    # Completion with memory injects the codename fact and reports the hit.
    c1 = client.post("/v1/completions", json={
        "prompt": "What is the internal project codename for the 2026 launch?",
        "max_new_tokens": 24, "do_sample": False, "use_memory": True}).json()
    R["completion_memory_used"] = c1.get("memory_used")
    R["completion_hits"] = c1.get("memory_hits")
    R["completion_text"] = c1.get("text", "")[:160]
    R["completion_grounded"] = "Halcyon" in c1.get("text", "")

    # Off-topic query should not inject a spurious fact above the floor.
    c2 = client.post("/v1/completions", json={
        "prompt": "Compose a two-line poem about the ocean at dawn.",
        "max_new_tokens": 8, "do_sample": False, "use_memory": True}).json()
    R["offtopic_memory_used"] = c2.get("memory_used")
    R["offtopic_hits"] = c2.get("memory_hits")

print(json.dumps(R, ensure_ascii=False, indent=2))

ok = (
    R["model_memory_enabled"] is True
    and R["empty_store_memory_used"] is False
    and R["query_correct"] is True
    and R["completion_memory_used"] is True
    and R["completion_grounded"] is True
)
print("ALL_PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
