"""Does bge's asymmetric query instruction widen the relevant/off-topic gap?

Compares symmetric (no prefix) vs asymmetric (query gets the bge retrieval
instruction, statements do not) on the same 3 statements. Reports, for each
query, the top relevant score and the best OFF-TOPIC (relativity) score -- a
bigger gap means a safer honest floor.
"""
from mt_lnn.sentence_encoder import SentenceEncoder

statements = [
    "My name is Alex and I work as a marine biologist.",
    "I love hiking in the mountains on weekends.",
    "I'm allergic to peanuts.",
]
queries = [
    ("What is the user's name and profession?", 0),
    ("What does the user enjoy doing in their free time?", 1),
    ("What food is the user allergic to?", 2),
]
OFFTOPIC = "Explain the theory of general relativity."


def evaluate(use_query_instr: bool):
    enc = SentenceEncoder(
        query_instruction=None if not use_query_instr else
        "Represent this sentence for searching relevant passages: ")
    keys = [enc.encode(s, is_query=False) for s in statements]
    correct = 0
    rows = []
    for qtext, want in queries:
        qv = enc.encode(qtext, is_query=use_query_instr)
        sc = [float(k @ qv) for k in keys]
        pick = max(range(3), key=lambda i: sc[i])
        correct += (pick == want)
        rows.append((sc[want], pick == want))
    # off-topic: best score against any stored statement
    ov = enc.encode(OFFTOPIC, is_query=use_query_instr)
    off_best = max(float(k @ ov) for k in keys)
    min_true = min(r[0] for r in rows)
    return correct, min_true, off_best


for tag, flag in (("symmetric ", False), ("asymmetric", True)):
    acc, min_true, off_best = evaluate(flag)
    print(f"{tag}: acc={acc}/3  min_true_hit={min_true:.3f}  "
          f"best_offtopic={off_best:.3f}  gap={min_true - off_best:+.3f}")
