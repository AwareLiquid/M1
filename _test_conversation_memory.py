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

from mt_lnn.knowledge_memory import PersistentKnowledgeMemory
from mt_lnn.conversation_memory import EpisodicConversationMemory
from mt_lnn.sentence_encoder import SentenceEncoder

# Dedicated sentence embedder; statements encoded plain, queries with bge's
# retrieval instruction (asymmetric) -- widens the relevant/off-topic gap.
enc = SentenceEncoder()
store = PersistentKnowledgeMemory(key_dim=enc.dim, db_path=":memory:")
# bge is isotropic -> no centering. Raw cosine has a high baseline (~0.4 even
# for unrelated text); with the asymmetric query prompt real hits score ~0.50-
# 0.74 and an off-topic query tops out at ~0.39, so a 0.45 floor cleanly
# separates "relevant" from "spurious".
mem = EpisodicConversationMemory(
    store, enc.as_fn(is_query=False), score_floor=0.45, center=False,
    query_encode_fn=enc.as_fn(is_query=True))

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
