"""Honest evidence for the typed-card recall layer: what it FIXES and what it
does NOT. Prints three blocks, no assertions -- this is the reproducible record
the test docstrings cite.

[1] THE FIX (ranking). With several same-language statements in the store,
    multilingual-e5-small's raw cosine is dominated by "magnet" sentences: a
    Chinese allergy/preference/plan question recalls the wrong memory (often the
    emotion sentence). Classifying the query to its card_type and boosting
    matching-type cards (the Awareness-Market cardTypeBoost) promotes the correct
    memory to rank 1. Shown: raw top-type vs typed top-type per query.

[2] THE NON-FIX (off-topic relevance gating). e5's cosines sit in a high,
    compressed band, so an off-topic query is NOT separated from a relevant one
    by any absolute floor -- and worse, off-topic queries are themselves
    confidently MIS-typed, so the boost can amplify a spurious match. Shown:
    off-topic vs relevant best-scores overlap, and off-topic query types.

[3] WHY e5 ANYWAY. The same block in English separates cleanly at floor ~0.75;
    the cost of bilingual coverage is a softer off-topic guarantee for Chinese.

Run:  PYTHONUTF8=1 python _diag_typed_recall.py
"""
from mt_lnn.knowledge_memory import PersistentKnowledgeMemory
from mt_lnn.conversation_memory import EpisodicConversationMemory
from mt_lnn.sentence_encoder import SentenceEncoder
from mt_lnn.memory_cards import MemoryCardClassifier

enc = SentenceEncoder()
clf = MemoryCardClassifier(
    enc.as_fn(is_query=False), query_encode_fn=enc.as_fn(is_query=True))


def build(classifier, boost=0.15):
    store = PersistentKnowledgeMemory(key_dim=enc.dim, db_path=":memory:")
    return EpisodicConversationMemory(
        store, enc.as_fn(is_query=False), score_floor=0.0, center=False,
        query_encode_fn=enc.as_fn(is_query=True),
        classifier=classifier, type_boost=boost)


STMTS = [
    "I'm allergic to peanuts.",
    "我对花生过敏。",
    "我感到非常焦虑和孤独。",
    "我打算明年搬到北京。",
    "我喜欢在周末去山里徒步。",
    "我叫李雷，是一名教师。",
]
ZH_Q = [("用户对什么过敏？", "health"),
        ("用户喜欢做什么？", "preference"),
        ("用户的计划是什么？", "plan")]

print("=== [1] RANKING: raw (untyped) vs typed top recall per Chinese query ===")
raw = build(None)
typed = build(clf, boost=0.15)
for m in (raw, typed):
    for s in STMTS:
        m.observe(s)
for q, want in ZH_Q:
    rh = raw.recall_cards(q, top_k=1)
    raw_t = clf.classify_statement(rh[0]["content"])[0] if rh else None
    th = typed.recall_cards(q, top_k=1)
    typed_t = th[0]["card_type"] if th else None
    print(f"  {q[:14]:14s} want={want:11s} raw->{str(raw_t):11s} "
          f"typed->{str(typed_t):11s} {'FIXED' if typed_t==want and raw_t!=want else ''}")

print("\n=== [2] OFF-TOPIC is NOT separable / is mis-typed (raw best-scores) ===")
probe = build(None)
for s in STMTS:
    probe.observe(s)
print("  relevant queries:")
for q, _ in ZH_Q:
    h = probe.recall(q, top_k=1)
    qt, qc = clf.classify_query(q)
    print(f"    {h[0][1]:.3f}  q_type={qt:11s}({qc:.2f})  {q}")
print("  off-topic queries (best-score overlaps relevant; type is spurious):")
for q in ["请解释量子力学。", "今天天气怎么样？", "Write a python function to sort a list."]:
    h = probe.recall(q, top_k=1)
    qt, qc = clf.classify_query(q)
    print(f"    {h[0][1]:.3f}  q_type={qt:11s}({qc:.2f})  {q}")

print("\n=== [3] English DOES separate at floor ~0.75 ===")
en = build(None)
for s in ["My name is Alex and I work as a marine biologist.",
          "I love hiking in the mountains on weekends.",
          "I'm allergic to peanuts."]:
    en.observe(s)
print("  relevant:")
for q in ["What is the user name and profession?",
          "What does the user enjoy doing in their free time?",
          "What food is the user allergic to?"]:
    print(f"    {en.recall(q, top_k=1)[0][1]:.3f}  {q}")
print("  off-topic:")
for q in ["Explain the theory of general relativity.",
          "What is the capital of France?"]:
    print(f"    {en.recall(q, top_k=1)[0][1]:.3f}  {q}")
