#!/usr/bin/env python3
"""
scripts/export_torchscript.py — export an MT-LNN checkpoint (or a fresh model)
to a TorchScript artifact and benchmark on-device latency.

Examples
--------
# Export a trained checkpoint at the deployment shape and benchmark on CPU:
python scripts/export_torchscript.py --ckpt checkpoints/final.pt \
    --batch 1 --seq_len 256 --optimize --out mt_lnn_ts.pt

# Quick smoke on a small fresh model:
python scripts/export_torchscript.py --small --seq_len 64

The artifact is the LOGITS path only (see mt_lnn/export.py for why). A traced
file is valid for the (batch, seq_len) it was traced at; pass the shape you will
serve. The on-device target is < 50 ms/token.
"""
import argparse
import dataclasses
import json
import os
import sys

# Allow `python scripts/export_torchscript.py` from anywhere: put the repo root
# (this file's parent's parent) on sys.path so `mt_lnn` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.export import export_and_benchmark
from mt_lnn.model import MTLNNModel


def _build_model(args) -> MTLNNModel:
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        cfg_dict = ckpt.get("config")
        if cfg_dict is None:
            raise SystemExit(
                f"{args.ckpt} has no embedded config; cannot reconstruct the model."
            )
        # Keep only fields the current dataclass knows (forward/backward compat).
        valid = {f.name for f in dataclasses.fields(MTLNNConfig)}
        cfg = MTLNNConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
        model = MTLNNModel(cfg)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if missing:
            print(f"[export] missing keys (defaulted): {missing}")
        if unexpected:
            print(f"[export] unexpected keys (ignored): {unexpected}")
        print(f"[export] loaded checkpoint {args.ckpt} (step {ckpt.get('step', '?')})")
        return model.eval()

    if args.small:
        cfg = MTLNNConfig(
            vocab_size=256, d_model=104, n_layers=2, n_heads=13, n_kv_heads=1,
            d_head=8, max_seq_len=max(64, args.seq_len), gwtb_n_heads=1,
            use_competitive_gwtb=True, use_world_model=True, use_hebbian=True,
        )
    else:
        # 125M-class default (matches the AwareLiquid-M2 backbone shape).
        cfg = MTLNNConfig(
            vocab_size=50257, d_model=832, n_layers=12, n_heads=13, n_kv_heads=1,
            d_head=64, max_seq_len=max(512, args.seq_len), gwtb_n_heads=1,
            use_competitive_gwtb=True, use_world_model=True, use_hebbian=True,
        )
    return MTLNNModel(cfg).eval()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint .pt with embedded config (else build fresh).")
    p.add_argument("--small", action="store_true",
                   help="Build a tiny model instead of the 125M-class default.")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--optimize", action="store_true",
                   help="Run torch.jit.optimize_for_inference on the trace.")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None, help="Save the artifact here.")
    p.add_argument("--no_eager", action="store_true",
                   help="Skip the eager baseline timing.")
    args = p.parse_args(argv)

    model = _build_model(args)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[export] model: {n_params/1e6:.1f}M params | device={args.device} | "
          f"shape=({args.batch},{args.seq_len})")

    report = export_and_benchmark(
        model, batch=args.batch, seq_len=args.seq_len, optimize=args.optimize,
        iters=args.iters, warmup=args.warmup, device=args.device,
        save_path=args.out, compare_eager=not args.no_eager,
    )
    report["n_params"] = n_params

    print("\n" + json.dumps(report, indent=2))
    mpt = report["scripted_ms_per_token"]
    target = 50.0
    verdict = "PASS" if mpt < target else "OVER"
    print(f"\n[latency] scripted {mpt:.3f} ms/token  (target < {target} ms/token) "
          f"→ {verdict}")
    if "speedup_vs_eager" in report:
        print(f"[latency] speedup vs eager: x{report['speedup_vs_eager']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
