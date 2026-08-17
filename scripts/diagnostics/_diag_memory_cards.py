"""Validate the typed-card classifier really separates personal statement types
(esp. the new EMOTION type vs PREFERENCE), and that it is multilingual via the
shared bge space (Chinese statements route to the same type as English).

Prints per-statement predicted type + confidence and an accuracy. The point is
honesty: prove the tagger works before wiring it into recall -- not assume it.
"""
from mt_lnn.sentence_encoder import SentenceEncoder
from mt_lnn.memory_cards import MemoryCardClassifier

enc = SentenceEncoder()
clf = MemoryCardClassifier(
    enc.as_fn(is_query=False), query_encode_fn=enc.as_fn(is_query=True))

# (statement, expected_type)
cases = [
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
    # multilingual: same meanings in Chinese should hit the same types
    ("我感到非常焦虑和孤独。", "emotion"),
    ("我喜欢在周末去山里徒步。", "preference"),
    ("我对花生过敏。", "health"),
]

correct = 0
for text, want in cases:
    got, conf = clf.classify_statement(text)
    ok = got == want
    correct += ok
    mark = "OK " if ok else "XX "
    print(f"  {mark}{text[:46]:46s} want={want:12s} got={got:12s} conf={conf:.3f}")

print(f"\ncard-type acc={correct}/{len(cases)}")

# Also show the emotion-vs-preference margin on the trickiest pair ("I love ...":
# 'love' is emotional wording but the intent is a preference).
tricky = "I love hiking in the mountains on weekends."
sc = clf.scores(tricky)
top = sorted(sc.items(), key=lambda kv: -kv[1])[:3]
print(f"\ntricky '{tricky[:30]}...' top3 = "
      + ", ".join(f"{k}:{v:.3f}" for k, v in top))
