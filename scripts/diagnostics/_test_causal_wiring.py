"""Local validation for the causal-consistency wiring into self-thinking.

Three checks, increasing in integration:
  1. DETECTOR fires: feed CausalConsistencyChecker a smooth trajectory then an
     abrupt orthogonal jump -> subspace score must drop below the floor.
  2. ROUTER converts: a low consistency_signal must force SELF_CRITIQUE with
     reason "causal_break" regardless of (low) token entropy.
  3. END-TO-END: generate_with_thinking(consistency_check=True) on a real HF
     model runs without error, populates per-step consistency scores, and the
     summary surfaces the causal view. Proves the wiring is live, not inert.

Run:  PYTHONUTF8=1 python _test_causal_wiring.py
"""
import json
import sys

import torch

from mt_lnn.causality import CausalConsistencyChecker
from mt_lnn.deliberation import DeliberationRouter, RouterThresholds, Route
from mt_lnn.thinking import generate_with_thinking

results = {}

# ---------------------------------------------------------------------------
# 1) Detector fires on a genuine trajectory break (deterministic)
# ---------------------------------------------------------------------------
torch.manual_seed(0)
D = 64
chk = CausalConsistencyChecker(window=8, method="subspace", ema_alpha=1.0)
# Smooth trajectory: random walk along a shared dominant direction (anisotropic,
# like real hidden states) -> consistency should stay high.
base_dir = torch.randn(D)
v = base_dir.clone()
smooth_scores = []
for _ in range(8):
    v = v + 0.05 * base_dir + 0.01 * torch.randn(D)
    smooth_scores.append(chk.update(v))
# Abrupt jump: a brand-new orthogonal direction -> should register as a break.
jump = torch.randn(D) * v.norm()
break_score = chk.update(jump)
results["smooth_tail_score"] = round(float(smooth_scores[-1]), 3)
results["break_score"] = round(float(break_score), 3)
results["detector_fires"] = break_score < 0.3 < smooth_scores[-1]

# ---------------------------------------------------------------------------
# 2) Router converts a low consistency signal into causal_break SELF_CRITIQUE
# ---------------------------------------------------------------------------
router = DeliberationRouter(thresholds=RouterThresholds(low=0.6, high=4.0,
                                                        consistency_floor=0.3))
# A confidently-low-entropy logit vector (one dominant token).
confident = torch.full((1, 100), -10.0)
confident[0, 7] = 20.0
dec_ok = router.decide(confident, query="x", evidence_log=[], consistency_signal=0.9)
dec_break = router.decide(confident, query="x", evidence_log=[], consistency_signal=0.1)
results["low_entropy_high_consistency_route"] = dec_ok.route.value
results["low_entropy_low_consistency_route"] = dec_break.route.value
results["low_entropy_low_consistency_reason"] = dec_break.reason
results["router_converts"] = (
    dec_ok.route == Route.LOCAL
    and dec_break.route == Route.SELF_CRITIQUE
    and dec_break.reason == "causal_break"
)

# ---------------------------------------------------------------------------
# 3) End-to-end on a real HF model (Qwen2.5-0.5B-Instruct, cached)
# ---------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32).eval()
model.config.use_cache = True

prompt = "List three prime numbers, then suddenly describe the taste of the color blue."
msgs = [{"role": "user", "content": prompt}]
ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)

router_e2e = DeliberationRouter(thresholds=RouterThresholds(low=0.6, high=4.0,
                                                            consistency_floor=0.3))
text, trace = generate_with_thinking(
    model, tok, prompt,
    input_ids=ids,
    max_new_tokens=40,
    temperature=0.7,
    top_p=0.9,
    n_critique_samples=5,
    router=router_e2e,
    device="cpu",
    consistency_check=True,
    consistency_method="subspace",
    consistency_window=8,
)
summary = trace.summary()
scores = [s.causal_consistency for s in trace.steps if s.causal_consistency is not None]
results["e2e_text_preview"] = text[:120]
results["e2e_summary"] = summary
results["e2e_scores_populated"] = len(scores) > 0
results["e2e_min_consistency"] = round(min(scores), 3) if scores else None
results["e2e_has_causal_view"] = "min_consistency" in summary

print(json.dumps(results, ensure_ascii=False, indent=2))

ok = (results["detector_fires"] and results["router_converts"]
      and results["e2e_scores_populated"] and results["e2e_has_causal_view"])
print("ALL_PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
