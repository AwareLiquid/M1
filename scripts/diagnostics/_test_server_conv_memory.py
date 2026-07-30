"""In-process integration test for episodic conversation memory in server_hf.

Boots the real FastAPI app with CONV_MEMORY=1 (baseline Qwen, no adapter, CPU,
1-token generation so it is fast) via TestClient, then drives a 3-turn flow:

  turn 1  user states a fact            -> stored, no recall yet
  turn 2  user asks about that fact     -> conv_memory_used, recalls turn 1
  turn 3  off-topic question            -> conv_memory_used False (honest floor)

Asserts the PLUMBING (observe on every turn, recall above floor prepended to the
prompt, off-topic recalls nothing). Does NOT assert generated answer quality --
a baseline 0.5B with 1 new token says little; the point is that the right memory
is retrieved and surfaced, which is the buildable "remembers you" slice.

Run:  PYTHONUTF8=1 python _test_server_conv_memory.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
os.environ["ADAPTER_CKPT"] = ""            # baseline base model (no checkpoint)
os.environ["DEVICE"] = "cpu"
os.environ["THINKING"] = "0"
os.environ["KB_PATH"] = ""                 # declarative KB off; isolate conv mem
os.environ["CONV_MEMORY"] = "1"
_dbf = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_dbf.close()
os.environ["CONV_DB_PATH"] = _dbf.name

from fastapi.testclient import TestClient   # noqa: E402
from serve.server_hf import app             # noqa: E402

R = {}
with TestClient(app) as client:             # triggers @app.on_event("startup")
    info = client.get("/v1/model").json()
    R["conv_enabled"] = info.get("conv_memory_enabled")

    def ask(prompt):
        return client.post("/v1/completions", json={
            "prompt": prompt, "max_new_tokens": 1, "do_sample": False,
        }).json()

    t1 = ask("My name is Alex and I work as a marine biologist.")
    R["t1_used"] = t1["conv_memory_used"]          # nothing stored yet -> False
    R["t1_entries_after"] = client.get("/v1/model").json()["conv_memory_entries"]

    t2 = ask("What is my name and profession?")    # a QUESTION: recalls, not stored
    R["t2_used"] = t2["conv_memory_used"]
    R["t2_hits"] = [h["content"] for h in t2["conv_memory_hits"]]
    R["t2_recalls_alex"] = any("Alex" in h for h in R["t2_hits"])

    t3 = ask("What is the theory of general relativity?")   # off-topic question
    R["t3_used"] = t3["conv_memory_used"]          # off-topic -> honest no-recall
    # Questions (t2, t3) must NOT have been stored -- still just the 1 statement.
    R["entries_after_questions"] = client.get("/v1/model").json()["conv_memory_entries"]

    # --- streaming endpoint must "remember you" too: the first SSE event is the
    # memory metadata. A NEW statement + a query about it proves recall flows in
    # the stream path and that observe() runs there as well.
    client.post("/v1/completions/stream", json={
        "prompt": "I'm allergic to peanuts.", "max_new_tokens": 1, "do_sample": False,
    })
    R["stream_obs_entries"] = client.get("/v1/model").json()["conv_memory_entries"]

    sq = client.post("/v1/completions/stream", json={
        "prompt": "What am I allergic to?", "max_new_tokens": 1, "do_sample": False,
    })
    first = sq.text.splitlines()[0]                 # "data: {...memory event...}"
    mem_event = json.loads(first[len("data: "):])
    R["stream_first_is_memory"] = mem_event.get("event") == "memory"
    R["stream_recalls_peanuts"] = any(
        "peanut" in h["content"].lower()
        for h in mem_event.get("conv_memory_hits", []))

print(json.dumps(R, ensure_ascii=False, indent=2))

ok = (
    R["conv_enabled"] is True
    and R["t1_used"] is False
    and R["t1_entries_after"] == 1
    and R["t2_used"] is True
    and R["t2_recalls_alex"]
    and R["t3_used"] is False
    and R["entries_after_questions"] == 1          # questions are not stored
    and R["stream_obs_entries"] == 2               # stream observed the new statement
    and R["stream_first_is_memory"]                # stream leads with memory event
    and R["stream_recalls_peanuts"]                # stream recalls it on next turn
)
print("SERVER_CONV_MEM_PASS" if ok else "SERVER_CONV_MEM_FAIL")

try:                                   # best-effort temp cleanup (Win file lock)
    os.unlink(_dbf.name)
except OSError:
    pass
sys.exit(0 if ok else 1)
