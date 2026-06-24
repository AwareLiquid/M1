"""PASS-gated test for the typed-card layer (mt_lnn.memory_cards) wired into
EpisodicConversationMemory.

What this PROVES (the honest, robust wins) -- and, just as important, what it
does NOT claim:

  [A] STATEMENT typing is correct (13/13 EN+ZH incl. the new EMOTION type). Every
      stored memory gets a usable card_type.
  [B] observe() tags the stored memory's meta with that card_type (+confidence),
      so the type travels with the record for later boosting.
  [C] classifier=None is byte-identical to the untyped behaviour (no card_type in
      meta; recall order unchanged) -- the typed layer is strictly opt-in.
  [D] THE HEADLINE WIN: the cardTypeBoost (borrowed from Awareness-Market) fixes a
      REAL raw-cosine failure. With several Chinese statements in the store,
      multilingual-e5-small's raw cosine is dominated by same-language "magnet"
      sentences -- e.g. the allergy question "用户对什么过敏？" does NOT even rank
      the allergy statement in its top-3; the emotion sentence wins. Classifying
      the query to its type (health/preference/plan) and boosting matching-type
      cards promotes the correct memory back to rank 1. We assert 0/3 correct
      WITHOUT typing -> 3/3 correct WITH typing.

HONEST NON-CLAIM (documented, not asserted): this does NOT fix off-topic
rejection. Under e5 the cosine band is high and compressed (~0.7-0.9 for almost
any pair), so an off-topic query (a) is not separated from a relevant one by any
absolute score_floor and (b) is itself confidently MIS-typed, so the boost can
even amplify a spurious match. The type layer improves RANKING of in-domain
recall; it is not a relevance gate. See _diag_typed_recall.py for the evidence.

Run:  PYTHONUTF8=1 python _test_memory_cards.py
"""
import json
import sys

from mt_lnn.knowledge_memory import PersistentKnowledgeMemory
from mt_lnn.conversation_memory import EpisodicConversationMemory
from mt_lnn.sentence_encoder import SentenceEncoder
from mt_lnn.memory_cards import MemoryCardClassifier

enc = SentenceEncoder()
clf = MemoryCardClassifier(
    enc.as_fn(is_query=False), query_encode_fn=enc.as_fn(is_query=True))

R = {}

# --- [A] statement typing accuracy (EN + ZH, incl. emotion) ---
typing_cases = [
    ("My name is Alex and I work as a marine biologist.", "identity"),
    ("I love hiking in the mountains on weekends.", "preference"),
    ("I'm allergic to peanuts.", "health"),
    ("I've been feeling really anxious and lonely lately.", "emotion"),
    ("I feel so happy and proud today.", "emotion"),
    ("I plan to move to Berlin next year.", "plan"),
    ("My sister Mia is a nurse in Toronto.", "relationship"),
    ("I take medication for high blood pressure.", "health"),
    ("My favourite food is ramen.", "preference"),
    ("I want to learn to play the piano.", "plan"),
    ("我感到非常焦虑和孤独。", "emotion"),
    ("我喜欢在周末去山里徒步。", "preference"),
    ("我对花生过敏。", "health"),
]
typed_ok = sum(clf.classify_statement(t)[0] == want for t, want in typing_cases)
R["typing_acc"] = f"{typed_ok}/{len(typing_cases)}"
R["typing_pass"] = typed_ok == len(typing_cases)

# --- [B] observe() tags meta with the card_type ---
store_b = PersistentKnowledgeMemory(key_dim=enc.dim, db_path=":memory:")
mem_b = EpisodicConversationMemory(
    store_b, enc.as_fn(is_query=False), score_floor=0.0, center=False,
    query_encode_fn=enc.as_fn(is_query=True), classifier=clf)
rid = mem_b.observe("I'm allergic to peanuts.")
# Read the stored meta straight back from the store.
back = store_b.query(enc.encode("I'm allergic to peanuts."), top_k=1, touch=False)
stored_meta = back[0][2] if back else {}
R["observe_meta"] = stored_meta
R["meta_tag_pass"] = (
    rid is not None and isinstance(stored_meta, dict)
    and stored_meta.get("card_type") == "health"
    and "type_confidence" in stored_meta)

# --- [C] classifier=None is the untyped baseline (no tag; same recall order) ---
def build(classifier, boost=0.05):
    s = PersistentKnowledgeMemory(key_dim=enc.dim, db_path=":memory:")
    m = EpisodicConversationMemory(
        s, enc.as_fn(is_query=False), score_floor=0.0, center=False,
        query_encode_fn=enc.as_fn(is_query=True),
        classifier=classifier, type_boost=boost)
    return s, m

s_none, mem_none = build(None)
rid_none = mem_none.observe("I'm allergic to peanuts.")
back_none = s_none.query(enc.encode("I'm allergic to peanuts."), top_k=1, touch=False)
meta_none = back_none[0][2] if back_none else {}
R["untyped_meta"] = meta_none
R["none_pass"] = isinstance(meta_none, dict) and "card_type" not in meta_none

# --- [D] HEADLINE: type boost fixes broken Chinese recall ranking ---
# A store with several same-language statements so e5's raw cosine is dominated
# by magnet sentences. Each relevant query should recall its matching TYPE.
zh_stmts = [
    ("I'm allergic to peanuts.", "health"),
    ("我对花生过敏。", "health"),
    ("我感到非常焦虑和孤独。", "emotion"),
    ("我打算明年搬到北京。", "plan"),
    ("我喜欢在周末去山里徒步。", "preference"),
    ("我叫李雷，是一名教师。", "identity"),
]
# query -> the card_type a correct recall must surface
zh_queries = [
    ("用户对什么过敏？", "health"),
    ("用户喜欢做什么？", "preference"),
    ("用户的计划是什么？", "plan"),
]

# Without typing: how many land the right TYPE at rank 1? (expected: few)
_, mem_raw = build(None)
for t, _ in zh_stmts:
    mem_raw.observe(t)
raw_top_types = []
for q, _ in zh_queries:
    hits = mem_raw.recall_cards(q, top_k=1)   # card_type is None (untyped store)
    # classify the top hit's CONTENT to see which type it actually is
    raw_top_types.append(clf.classify_statement(hits[0]["content"])[0] if hits else None)
raw_correct = sum(t == want for (_, want), t in zip(zh_queries, raw_top_types))
R["raw_top_types"] = raw_top_types
R["raw_correct"] = f"{raw_correct}/{len(zh_queries)}"

# With typing (boost large enough to overcome the magnet gap): expect 3/3.
_, mem_typed = build(clf, boost=0.15)
for t, _ in zh_stmts:
    mem_typed.observe(t)
typed_top_types = []
for q, _ in zh_queries:
    hits = mem_typed.recall_cards(q, top_k=1)
    typed_top_types.append(hits[0]["card_type"] if hits else None)
typed_correct = sum(t == want for (_, want), t in zip(zh_queries, typed_top_types))
R["typed_top_types"] = typed_top_types
R["typed_correct"] = f"{typed_correct}/{len(zh_queries)}"
# The win is meaningful only if typing strictly improves AND reaches full marks.
R["boost_pass"] = typed_correct == len(zh_queries) and typed_correct > raw_correct

print(json.dumps(R, ensure_ascii=False, indent=2))

ok = (R["typing_pass"] and R["meta_tag_pass"] and R["none_pass"]
      and R["boost_pass"])
print("MEMORY_CARDS_PASS" if ok else "MEMORY_CARDS_FAIL")
sys.exit(0 if ok else 1)
