"""End-to-end test of EpisodicConversationMemory: does the assistant
"remember you" across turns?

Simulates a multi-turn chat where the user states facts about themselves, with
noise (a trivial "ok") and a repeat mixed in, then checks that:
  - worth-remembering filtering skips the trivial turn
  - dedup skips the repeated statement
  - later queries recall the RIGHT earlier statement (name/job vs hobby)
  - an off-topic query recalls nothing above the floor (honest, no fabrication)

Uses a dedicated sentence-embedding model (BAAI/bge-small-en-v1.5, CLS-pooled +
L2-normalized) as the encoder. The earlier mean-pooled Qwen-0.5B encoder could
NOT separate first-person paraphrases ("I love hiking" lost to "marine
biologist", the universal nearest neighbour) -- see _diag_conv_mem.py. bge is
isotropic enough to rank by genuine content, so centering is OFF here. This
proves the buildable slice of a "Her-like" experience -- continuity / recall --
NOT understanding or AGI.

Run:  PYTHONUTF8=1 python _test_conversation_memory.py
"""
import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from mt_lnn.knowledge_memory import PersistentKnowledgeMemory
from mt_lnn.conversation_memory import EpisodicConversationMemory

MID = "BAAI/bge-small-en-v1.5"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModel.from_pretrained(MID).eval()


@torch.no_grad()
def encode(text: str) -> torch.Tensor:
    ids = tok(text or " ", return_tensors="pt", truncation=True,
              max_length=128, padding=True)
    out = model(**ids)
    cls = out.last_hidden_state[:, 0]                     # CLS pooling (bge recipe)
    return F.normalize(cls, dim=-1)[0].float()            # (d,)


probe = encode("dimension probe")
store = PersistentKnowledgeMemory(key_dim=probe.numel(), db_path=":memory:")
# bge is isotropic -> no centering. Its raw cosine has a high baseline (~0.45
# even for unrelated text), so the honest floor must sit above that baseline:
# real hits here score 0.54-0.74, an off-topic query tops out at ~0.46. A floor
# of 0.5 cleanly separates "relevant" from "spurious".
mem = EpisodicConversationMemory(store, encode, score_floor=0.5, center=False)

R = {}

# --- a multi-turn conversation (user utterances only) ---
turns = [
    "My name is Alex and I work as a marine biologist.",
    "I love hiking in the mountains on weekends.",
    "ok",                                                  # trivial -> skip
    "My name is Alex and I work as a marine biologist.",   # duplicate -> skip
    "I'm allergic to peanuts.",
]
ids = [mem.observe(t) for t in turns]
R["observe_ids"] = ids
R["trivial_skipped"] = ids[2] is None
R["duplicate_skipped"] = ids[3] is None
R["n_stored"] = len(mem)

# --- later turns: does it recall the right thing? ---
name_hits = mem.recall("What is the user's name and profession?", top_k=1)
hobby_hits = mem.recall("What does the user enjoy doing in their free time?", top_k=1)
R["recall_name_top"] = name_hits[0][0] if name_hits else None
R["recall_hobby_top"] = hobby_hits[0][0] if hobby_hits else None
R["name_correct"] = bool(name_hits) and "Alex" in name_hits[0][0]
R["hobby_correct"] = bool(hobby_hits) and "hiking" in hobby_hits[0][0]

# --- off-topic query -> honest no-recall (nothing above floor) ---
off = mem.recall("Explain the theory of general relativity.", top_k=3)
R["offtopic_recall"] = [(c[:40], round(s, 3)) for c, s in off]

print(json.dumps(R, ensure_ascii=False, indent=2))

ok = (
    R["trivial_skipped"]
    and R["duplicate_skipped"]
    and R["n_stored"] == 3
    and R["name_correct"]
    and R["hobby_correct"]
    and R["offtopic_recall"] == []     # honest: off-topic recalls nothing
)
print("CONV_MEM_PASS" if ok else "CONV_MEM_FAIL")
sys.exit(0 if ok else 1)
