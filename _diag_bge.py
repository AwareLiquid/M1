"""Check whether a dedicated sentence-embedding model (BAAI/bge-small-en-v1.5,
~33M) solves the paraphrase recall case that mean-pooled Qwen-0.5B failed.

Same 3 statements / 3 queries as _diag_conv_mem.py. bge uses CLS pooling +
L2 normalize (its documented recipe). Prints the cosine matrix and per-query
argmax correctness. If this is 3/3 it is the honest "good encoder" fix.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MID = "BAAI/bge-small-en-v1.5"
tok = AutoTokenizer.from_pretrained(MID)
m = AutoModel.from_pretrained(MID).eval()

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


@torch.no_grad()
def enc(text):
    ids = tok(text or " ", return_tensors="pt", truncation=True,
              max_length=128, padding=True)
    out = m(**ids)
    cls = out.last_hidden_state[:, 0]           # CLS pooling (bge recipe)
    return F.normalize(cls, dim=-1)[0]


keys = torch.stack([enc(s) for s in statements])
correct = 0
for qtext, want in queries:
    qv = enc(qtext)
    sc = keys @ qv
    pick = int(sc.argmax())
    correct += (pick == want)
    mark = "OK " if pick == want else "XX "
    print(f"  {mark}{qtext:42s} want={want} pick={pick} "
          f"scores={[round(float(x),3) for x in sc]}")
print(f"\nbge-small acc={correct}/3")
