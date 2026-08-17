"""Quick CPU perf smoke for the default ~125M config (2026-07-15 fix batch).

Measures single-thread-pool CPU numbers as a regression reference:
train-step tokens/s at T=256, decode tokens/s (prefill 128 + 32 new).
"""
import sys
import time

import torch

sys.path.insert(0, ".")
from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel

torch.manual_seed(0)
cfg = MTLNNConfig()
model = MTLNNModel(cfg)
n = sum(p.numel() for p in model.parameters()) / 1e6
print(f"params {n:.1f}M  d_model={cfg.d_model} layers={cfg.n_layers} threads={torch.get_num_threads()}")

# train step (B=1, T=256)
model.train()
x = torch.randint(0, cfg.vocab_size, (1, 256))
out = model(x, labels=x); out["loss"].backward(); model.zero_grad(set_to_none=True)  # warmup
t0 = time.perf_counter()
iters = 3
for _ in range(iters):
    out = model(x, labels=x)
    out["loss"].backward()
    model.zero_grad(set_to_none=True)
dt = time.perf_counter() - t0
print(f"train B=1 T=256: {256 * iters / dt:,.0f} tok/s  (loss {out['loss'].item():.2f})")

# decode
model.eval()
x = torch.randint(0, cfg.vocab_size, (1, 128))
with torch.no_grad():
    o = model(x, use_cache=True)
    cache, nxt = o["cache"], o["logits"][:, -1:].argmax(-1)
    t0 = time.perf_counter()
    N = 32
    for _ in range(N):
        o = model(nxt, cache=cache, use_cache=True)
        cache, nxt = o["cache"], o["logits"][:, -1:].argmax(-1)
    dt = time.perf_counter() - t0
print(f"decode (prefill 128 + {N} new): {N / dt:,.1f} tok/s")
