"""Sweep text->key encoders for knowledge_memory retrieval accuracy.

The mean-pool-last-layer encoder mislabels 3/6 queries (anisotropy makes one
fact a universal nearest neighbor). Try alternatives and measure top-1 accuracy:
does each on-topic query retrieve its INTENDED fact?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

mid = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()

facts = [
    "Paris is the capital of France",
    "Water boils at 100 C at sea level",
    "Python was created by Guido van Rossum",
    "The Great Wall of China is over 13000 miles long",
]
# (query, intended_fact_index)
queries = [
    ("What is the capital of France?", 0),
    ("At what temperature does water boil?", 1),
    ("Who created Python?", 2),
    ("How long is the Great Wall?", 3),
]


@torch.no_grad()
def hidden(text, layer=-1):
    ids = tok(text or " ", return_tensors="pt", truncation=True, max_length=128).input_ids
    o = m(input_ids=ids, output_hidden_states=True, use_cache=False)
    return o.hidden_states[layer][0]   # (T, D)


def enc_mean_last(t):
    return hidden(t, -1).mean(0)

def enc_last_tok(t):
    return hidden(t, -1)[-1]

def enc_mean_mid(t):
    return hidden(t, len(m.model.layers)//2 if hasattr(m,'model') else -1).mean(0)

def enc_norm_mean(t):
    h = F.normalize(hidden(t, -1), dim=-1)
    return h.mean(0)

ENCODERS = {
    "mean_last": enc_mean_last,
    "last_tok": enc_last_tok,
    "mean_mid": enc_mean_mid,
    "norm_mean_last": enc_norm_mean,
}


def accuracy(encoder, center=False):
    keys = torch.stack([encoder(f).float() for f in facts])     # (N, D)
    qs = [(encoder(q).float(), idx) for q, idx in queries]
    if center:
        mu = keys.mean(0)
        keys = keys - mu
        qs = [(qv - mu, idx) for qv, idx in qs]
    keys = F.normalize(keys, dim=-1)
    correct = 0
    detail = []
    for qv, idx in qs:
        qv = F.normalize(qv, dim=-1)
        scores = keys @ qv
        top = int(scores.argmax())
        correct += (top == idx)
        detail.append((top, idx, round(float(scores.max()), 3)))
    return correct, len(qs), detail


for name, fn in ENCODERS.items():
    for center in (False, True):
        c, n, det = accuracy(fn, center=center)
        tag = name + ("+center" if center else "")
        print(f"{tag:22s} acc={c}/{n}  picks={[d[0] for d in det]} want={[d[1] for d in det]}")
