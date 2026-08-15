"""P1 效率曲线：O(1) 状态 vs KV cache 的多长度 sweep。

BEAT_TRANSFORMER_PLAN.md P1 —— 把 8063×@1M 的单一数据点扩展成完整成本曲线。
本地 8GB 可跑 ≤8k；32k-1M 需云 GPU（1×A100）。

复用 state_only_streaming 的 kv_stream / state_only_stream 测量（cache 字节），
sweep 多个上下文长度输出对比表。

运行（云 GPU）:
  py -3.11 benchmarks/efficiency_curve.py \
      --lengths 512 2048 8192 32768 131072 524288 1048576
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.state_only_streaming import (
    build_config, kv_stream, state_only_stream,
)  # noqa: E402


def _base_args(seq_len: int, d_model: int):
    """构造 build_config 依赖的完整参数集（默认值对齐 state_only_streaming）。"""
    return argparse.Namespace(
        fixed_window=False,
        vocab_size=128,
        max_seq_len=seq_len,
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        gwtb_compression_ratio=4,
        gwtb_n_heads=2,
        coherence_heads=2,
        disable_dynamic_scale_gates=False,
        scale_gate_init_bias=2.0,
        scale_gate_active_threshold=0.5,
        scale_gate_skip_threshold=0.0,
        compute_skip_threshold=0.0,
        sparse_resonance_kernel=False,
        sparse_resonance_top_k=1,
    )


def measure_length(seq_len: int, d_model: int, device: str) -> dict:
    """单个上下文长度下的 KV vs state-only 字节对比。"""
    args = _base_args(seq_len, d_model)
    cfg = build_config(args, max_steps=seq_len)
    model = MTLNNModel(cfg).to(device)
    model.eval()

    torch.manual_seed(0)
    tokens = torch.randint(0, 128, (1, seq_len), device=device)

    with torch.no_grad():
        kv = kv_stream(model, tokens)
        st = state_only_stream(model, tokens)

    kv_bytes = kv["peak_cache_bytes"]
    st_bytes = st["peak_cache_bytes"]
    return {
        "seq_len": seq_len,
        "kv_cache_bytes": kv_bytes,
        "state_only_bytes": st_bytes,
        "ratio": round(kv_bytes / max(st_bytes, 1), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[512, 2048, 8192, 32768, 131072, 524288, 1048576])
    p.add_argument("--d_model", type=int, default=104)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="benchmarks/efficiency_curve.json")
    args = p.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        args.device if args.device != "auto" else "cpu")

    rows = []
    for L in args.lengths:
        t0 = time.time()
        try:
            r = measure_length(L, args.d_model, device)
            r["wall_s"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"  L={L:>8}: KV={r['kv_cache_bytes']:>12}B  "
                  f"state={r['state_only_bytes']:>6}B  ratio={r['ratio']}x  "
                  f"({r['wall_s']}s)")
        except torch.cuda.OutOfMemoryError:
            print(f"  L={L:>8}: OOM (context too large for this GPU)")
            break

    out = {"args": vars(args), "rows": rows}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}")

    if rows:
        print("\n=== COST CURVE (state-only bytes flat vs KV O(T)) ===")
        print(f"{'len':>8} | {'KV bytes':>12} | {'state bytes':>10} | ratio")
        for r in rows:
            print(f"{r['seq_len']:>8} | {r['kv_cache_bytes']:>12} | "
                  f"{r['state_only_bytes']:>10} | {r['ratio']}x")


if __name__ == "__main__":
    main()
