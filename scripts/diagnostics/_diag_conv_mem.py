"""Diagnose why episodic recall picks the wrong statement, and find an encoder
that separates first-person conversational statements.

Prints, for each candidate encoder x {centered?}, the full query->statement
cosine matrix and whether each query's argmax is the intended statement. The
3 statements are deliberately similar in form ("I ...") -- the hard case that
mean-pool + plain cosine collapses.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MID)
m = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).eval()

statements = [
    "My name is Alex and I work as a marine biologist.",
    "I love hiking in the mountains on weekends.",
    "I'm allergic to peanuts.",
]
# (query, intended statement index)
queries = [
    ("What is the user's name and profession?", 0),
    ("What does the user enjoy doing in their free time?", 1),
    ("What food is the user allergic to?", 2),
]


@torch.no_grad()
def hidden(text, layer=-1):
    ids = tok(text or " ", return_tensors="pt", truncation=True,
              max_length=128).input_ids
    o = m(input_ids=ids, output_hidden_states=True, use_cache=False)
    return o.hidden_states[layer][0]


def enc_mean_last(t): return hidden(t, -1).mean(0)
def enc_last_tok(t):  return hidden(t, -1)[-1]
def enc_norm_mean(t): return F.normalize(hidden(t, -1), dim=-1).mean(0)
def enc_mean_mid(t):  return hidden(t, len(m.model.layers) // 2).mean(0)


ENC = {
    "mean_last": enc_mean_last,
    "last_tok": enc_last_tok,
    "norm_mean_last": enc_norm_mean,
    "mean_mid": enc_mean_mid,
}


def run(enc, center):
    keys = torch.stack([enc(s).float() for s in statements])
    qs = [(enc(q).float(), idx) for q, idx in queries]
    if center:
        mu = keys.mean(0)
        keys = keys - mu
        qs = [(qv - mu, idx) for qv, idx in qs]
    keys = F.normalize(keys, dim=-1)
    correct = 0
    rows = []
    for (qv, idx), (qtext, _) in zip(qs, queries):
        qv = F.normalize(qv, dim=0)
        sc = (keys @ qv)
        pick = int(sc.argmax())
        correct += (pick == idx)
        rows.append((qtext[:38], idx, pick, [round(float(x), 3) for x in sc]))
    return correct, rows


for name, fn in ENC.items():
    for center in (False, True):
        c, rows = run(fn, center)
        tag = name + ("+center" if center else "")
        print(f"\n=== {tag:22s} acc={c}/3 ===")
        for qtext, want, pick, sc in rows:
            mark = "OK " if want == pick else "XX "
            print(f"  {mark}{qtext:38s} want={want} pick={pick} scores={sc}")
