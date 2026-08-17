"""Evidence check: does a MULTILINGUAL encoder (intfloat/multilingual-e5-small,
~118M, the model Awareness-Market uses) fix the Chinese card-typing failures
bge-small-en-v1.5 had, WITHOUT regressing English?

e5 recipe differs from bge: MEAN pooling (masked) + "query: "/"passage: "
prefixes. Implemented inline here so the check is self-contained before deciding
to generalize SentenceEncoder. Prints English + Chinese card-type accuracy and
the recall hard case (name/hobby/allergy).
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MID = "intfloat/multilingual-e5-small"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModel.from_pretrained(MID).eval()


@torch.no_grad()
def embed(text, kind):  # kind: "query" | "passage"
    s = f"{kind}: {text or ' '}"
    enc = tok(s, return_tensors="pt", truncation=True, max_length=256, padding=True)
    out = model(**enc)
    # masked mean pooling
    mask = enc.attention_mask.unsqueeze(-1).float()
    summed = (out.last_hidden_state * mask).sum(1)
    mean = summed / mask.sum(1).clamp(min=1e-9)
    return F.normalize(mean, dim=-1)[0]


# --- card typing: anchors per type (statement side = passage) ---
ANCHORS = {
    "identity": "My name is. I work as. I am a. I live in. My profession is.",
    "preference": "I love. I like. I enjoy. I prefer. My favourite is. My hobby is.",
    "emotion": "I feel. I am happy. I am sad. I am anxious. I am lonely. My mood is.",
    "plan": "I plan to. I am going to. I want to. My goal is. I intend to.",
    "relationship": "My friend. My partner. My sister. My mother. My colleague. My family.",
    "health": "I am allergic to. I take medication for. I cannot eat. My health condition.",
    "detail": "An important detail about me. Something to remember about me.",
}
ids = list(ANCHORS)
akeys = torch.stack([embed(ANCHORS[i], "passage") for i in ids])


def classify(text):
    v = embed(text, "passage")
    sc = akeys @ v
    j = int(sc.argmax())
    return ids[j], float(sc[j])


cases = [
    ("My name is Alex and I work as a marine biologist.", "identity"),
    ("I love hiking in the mountains on weekends.", "preference"),
    ("I'm allergic to peanuts.", "health"),
    ("I've been feeling really anxious and lonely lately.", "emotion"),
    ("I plan to move to Berlin next year.", "plan"),
    ("My sister Mia is a nurse in Toronto.", "relationship"),
    ("我感到非常焦虑和孤独。", "emotion"),
    ("我喜欢在周末去山里徒步。", "preference"),
    ("我对花生过敏。", "health"),
    ("我叫李雷，是一名教师。", "identity"),
    ("我打算明年搬到北京。", "plan"),
]
correct = 0
for text, want in cases:
    got, conf = classify(text)
    ok = got == want
    correct += ok
    print(f"  {'OK ' if ok else 'XX '}{text[:40]:40s} want={want:12s} got={got:12s} {conf:.3f}")
print(f"\ne5 card-type acc={correct}/{len(cases)}")

# --- recall hard case (query vs statement) ---
statements = [
    "My name is Alex and I work as a marine biologist.",
    "I love hiking in the mountains on weekends.",
    "I'm allergic to peanuts.",
]
skeys = torch.stack([embed(s, "passage") for s in statements])
queries = [("What is the user's name and profession?", 0),
           ("What does the user enjoy doing in their free time?", 1),
           ("What food is the user allergic to?", 2),
           ("用户叫什么名字，做什么工作？", 0),
           ("用户对什么过敏？", 2)]
rc = 0
for q, want in queries:
    v = embed(q, "query")
    sc = skeys @ v
    pick = int(sc.argmax())
    rc += pick == want
    print(f"  {'OK ' if pick==want else 'XX '}{q[:34]:34s} want={want} pick={pick} {[round(float(x),3) for x in sc]}")
print(f"e5 recall acc={rc}/{len(queries)}")
