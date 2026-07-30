"""Calibrate the causal-consistency detector so it DISCRIMINATES genuine
trajectory breaks from routine token-by-token generation.

The first e2e run flagged 36/40 tokens as breaks at floor=0.3 (subspace) -- not
discriminative. Here we measure the per-step consistency-score distribution on a
COHERENT continuation vs a deliberate TOPIC-SWITCH continuation, for both
detector methods and a couple of windows, and report separation so we can pick a
floor (or method) that fires on the switch but not on coherent text.
"""
import json
import statistics as st

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mt_lnn.causality import CausalConsistencyChecker

mid = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
model.config.use_cache = True


@torch.no_grad()
def per_step_scores(prompt, method, window, n=50):
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
    chk = CausalConsistencyChecker(window=window, method=method)
    past, cur = None, ids
    scores = []
    for _ in range(n):
        out = model(input_ids=cur, past_key_values=past, use_cache=True,
                    output_hidden_states=True)
        past = out.past_key_values
        h = out.hidden_states[-1][:, -1, :]
        scores.append(float(chk.update(h)))
        nxt = int(out.logits[:, -1, :].argmax(-1).item())
        cur = torch.tensor([[nxt]])
        if nxt == tok.eos_token_id:
            break
    return scores


coherent = "Explain in a few sentences why the sky appears blue during the day."
switch = "List three prime numbers, then suddenly describe the taste of the color blue, then explain tax law."

report = {}
for method in ("cosine", "subspace"):
    for window in (8, 16):
        sc_c = per_step_scores(coherent, method, window)
        sc_s = per_step_scores(switch, method, window)
        # Skip the warmup steps (window fills) for a fair steady-state read.
        warm = window
        c = sc_c[warm:] or sc_c
        s = sc_s[warm:] or sc_s
        report[f"{method}_w{window}"] = {
            "coherent_min": round(min(c), 3),
            "coherent_mean": round(st.mean(c), 3),
            "coherent_p10": round(sorted(c)[max(0, len(c)//10 - 1)], 3),
            "switch_min": round(min(s), 3),
            "switch_mean": round(st.mean(s), 3),
            "switch_p10": round(sorted(s)[max(0, len(s)//10 - 1)], 3),
        }

print(json.dumps(report, ensure_ascii=False, indent=2))

# A good floor sits BELOW coherent_p10 (so coherent text rarely fires) but ABOVE
# switch_min (so the switch does fire). Report the best discriminating config.
print("\n-- discrimination (coherent_p10 - switch_min, want > 0 and a gap) --")
for k, v in report.items():
    gap = round(v["coherent_p10"] - v["switch_min"], 3)
    print(f"{k:14s} coherent_p10={v['coherent_p10']:.3f} switch_min={v['switch_min']:.3f} gap={gap}")
