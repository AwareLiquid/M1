"""E4 measurement: old defaults (decay_wm=T, predictive_coding=T) vs new lean
defaults (both False) — train tok/s, decode tok/s, CPU. Same seed, same data.
"""
import sys
import time

import torch

sys.path.insert(0, ".")
from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel


def bench(tag, **flags):
    torch.manual_seed(0)
    cfg = MTLNNConfig(**flags)
    model = MTLNNModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 256))
    model.train()
    out = model(x, labels=x); out["loss"].backward(); model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for _ in range(3):
        out = model(x, labels=x)
        out["loss"].backward()
        model.zero_grad(set_to_none=True)
    train_tps = 256 * 3 / (time.perf_counter() - t0)

    model.eval()
    p = torch.randint(0, cfg.vocab_size, (1, 128))
    with torch.no_grad():
        o = model(p, use_cache=True)
        cache, nxt = o["cache"], o["logits"][:, -1:].argmax(-1)
        t0 = time.perf_counter()
        N = 32
        for _ in range(N):
            o = model(nxt, cache=cache, use_cache=True)
            cache, nxt = o["cache"], o["logits"][:, -1:].argmax(-1)
        dec_tps = N / (time.perf_counter() - t0)
    print(f"{tag}: train {train_tps:,.0f} tok/s | decode {dec_tps:,.1f} tok/s "
          f"(loss {out['loss'].item():.2f})")
    return train_tps, dec_tps


old = bench("old defaults (dwm=T, pc=T)", use_decay_wm=True, use_predictive_coding=True)
new = bench("new lean defaults        ", )  # shipped defaults after E4
print(f"delta: train +{(new[0]/old[0]-1)*100:.1f}% | decode +{(new[1]/old[1]-1)*100:.1f}%")
