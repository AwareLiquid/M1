"""Verify PersistentKnowledgeMemory survives a process/connection restart.

This is the guarantee a Docker writable-volume deployment relies on: facts
written in one container lifetime must still be queryable after the container
(and thus the SQLite connection) is recreated against the same db file.

We use random unit vectors as keys (no LM needed) so the test is fast and
isolates the storage layer from the encoder. Steps:
  1. Open a fresh db file, write 3 facts, capture their ids + a query result.
  2. DROP the handle (simulating container stop), reopen the SAME file.
  3. Assert: entry count persisted, contents intact, and the centered-cosine
     query still returns the right fact across the restart.

Run:  PYTHONUTF8=1 python _test_kb_persist.py
"""
import os
import tempfile

import torch

from mt_lnn.knowledge_memory import PersistentKnowledgeMemory

KEY_DIM = 32
torch.manual_seed(0)

# Three distinct, well-separated unit-vector keys + a query close to key #2.
keys = torch.nn.functional.normalize(torch.randn(3, KEY_DIM), dim=-1)
facts = [
    "Fact A: the vault code is 1234.",
    "Fact B: the meeting is on Tuesday.",
    "Fact C: the server room is on floor 7.",
]
query_vec = torch.nn.functional.normalize(keys[1] + 0.05 * torch.randn(KEY_DIM), dim=-1)

R = {}
tmp = tempfile.mkdtemp()
db = os.path.join(tmp, "knowledge.db")

# --- lifetime 1: write, then close ---
kb1 = PersistentKnowledgeMemory(key_dim=KEY_DIM, db_path=db)
ids = [kb1.write(keys[i], facts[i]) for i in range(3)]
R["written_ids"] = ids
R["count_before"] = len(kb1)
hit1 = kb1.query(query_vec, top_k=1, center=True)   # [(content, score, meta)]
R["query_before"] = hit1[0][0] if hit1 else None
del kb1   # drop the connection -- simulates container stop

# --- lifetime 2: reopen the SAME file, must see persisted data ---
kb2 = PersistentKnowledgeMemory(key_dim=KEY_DIM, db_path=db)
R["count_after"] = len(kb2)
hit2 = kb2.query(query_vec, top_k=1, center=True)
R["query_after"] = hit2[0][0] if hit2 else None

import json
print(json.dumps(R, ensure_ascii=False, indent=2))

ok = (
    R["count_before"] == 3
    and R["count_after"] == 3                      # survived restart
    and R["query_before"] == facts[1]              # retrieval correct pre-restart
    and R["query_after"] == facts[1]               # retrieval correct post-restart
)
print("PERSIST_PASS" if ok else "PERSIST_FAIL")
import sys
sys.exit(0 if ok else 1)
