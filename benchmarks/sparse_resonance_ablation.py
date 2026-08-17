"""Ablate dense vs sparse-resonance tau-scale computation.

This script compares the default dense resonance path against optional
``sparse_resonance_kernel`` runs using the same randomly initialized weights
and the same synthetic token stream. It is meant to answer two questions:

- How many tau scales were actually computed?
- How much output drift does sparse top-k introduce relative to dense?

Example:
    python benchmarks/sparse_resonance_ablation.py --steps 100 --top_k 1 2 3
"""

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings
from statistics import mean, pstdev
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _elapsed(device: str, fn):
    _sync(device)
    start = time.perf_counter()
    result = fn()
    _sync(device)
    return result, time.perf_counter() - start


def _timed_repeats(device: str, fn, repeats: int, warmup: int) -> Dict:
    result = None
    for _ in range(warmup):
        result = fn()
    times = []
    for _ in range(repeats):
        result, elapsed = _elapsed(device, fn)
        times.append(elapsed)
    return {
        "result": result,
        "elapsed_s": mean(times),
        "elapsed_min_s": min(times),
        "elapsed_std_s": pstdev(times) if len(times) > 1 else 0.0,
        "repeats": repeats,
        "warmup": warmup,
    }


def make_config(args, *, sparse: bool, top_k: int) -> MTLNNConfig:
    return MTLNNConfig(
        vocab_size=args.vocab_size,
        max_seq_len=max(args.max_seq_len, args.steps + 1),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_head=args.d_model // args.n_heads,
        dropout=0.0,
        attention_dropout=0.0,
        gwtb_compression_ratio=args.gwtb_compression_ratio,
        gwtb_n_heads=args.gwtb_n_heads,
        coherence_heads=args.coherence_heads,
        dynamic_scale_gates=True,
        sparse_resonance_kernel=sparse,
        sparse_resonance_top_k=top_k,
    )


@torch.no_grad()
def run_full(model: MTLNNModel, tokens: torch.Tensor) -> torch.Tensor:
    return model(tokens, use_lnn_recurrence=True)["logits"].detach().cpu()


def divergence(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    diff = (reference - candidate).abs()
    return {
        "mean_abs": float(diff.mean().item()),
        "max_abs": float(diff.max().item()),
    }


def gate_diagnostics(model: MTLNNModel) -> Dict[str, float]:
    diag = model.get_mt_diagnostics()
    return {
        key: value for key, value in diag.items()
        if key.startswith("scale_gate") or key.startswith("sparse_resonance")
    }


def render_markdown(results: Dict) -> str:
    lines = [
        "# Sparse Resonance Ablation",
        "",
        "Compares dense tau-scale computation against optional sparse top-k resonance.",
        "",
        "| Mode | Top-k | Selected scale ratio | Mean time (s) | Min time (s) | Std (s) | Tok/s | Mean div vs dense | Max div vs dense |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["runs"]:
        div = row.get("divergence_vs_dense", {})
        lines.append(
            f"| `{row['mode']}` | {row['top_k']} | "
            f"{row['diagnostics'].get('sparse_resonance_scale_ratio', 1.0):.3f} | "
            f"{row['elapsed_s']:.4f} | {row['elapsed_min_s']:.4f} | "
            f"{row['elapsed_std_s']:.4f} | {row['tokens_per_s']:.1f} | "
            f"{div.get('mean_abs', 0.0):.3g} | {div.get('max_abs', 0.0):.3g} |"
        )

    lines.extend([
        "",
        "## Caveats",
        "",
        "- Sparse top-k selection is intentionally approximate and can change outputs.",
        "- The current implementation uses PyTorch tensor indexing and top-k over tau scales, not a custom CUDA/Triton kernel.",
        "- Speedups depend on sequence length, device, and whether top-k selection overhead dominates the skipped scale work.",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--json_output", type=Path, default=Path("benchmarks/sparse_resonance_ablation.json"))
    parser.add_argument("--markdown_output", type=Path, default=Path("benchmarks/sparse_resonance_ablation.md"))

    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=104)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--gwtb_compression_ratio", type=int, default=4)
    parser.add_argument("--gwtb_n_heads", type=int, default=2)
    parser.add_argument("--coherence_heads", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    dense_cfg = make_config(args, sparse=False, top_k=args.top_k[0])
    dense_model = MTLNNModel(dense_cfg).to(device).eval()
    dense_state = dense_model.state_dict()
    tokens = torch.randint(0, dense_cfg.vocab_size, (args.batch, args.steps), device=device)

    dense_timed = _timed_repeats(
        device, lambda: run_full(dense_model, tokens), args.repeats, args.warmup
    )
    dense_logits = dense_timed["result"]
    dense_diag = gate_diagnostics(dense_model)

    results: Dict[str, object] = {
        "benchmark": "sparse_resonance_ablation",
        "device": device,
        "torch": torch.__version__,
        "config": dataclasses.asdict(dense_cfg),
        "runs": [
            {
                "mode": "dense",
                "top_k": dense_cfg.n_time_scales,
                "elapsed_s": dense_timed["elapsed_s"],
                "elapsed_min_s": dense_timed["elapsed_min_s"],
                "elapsed_std_s": dense_timed["elapsed_std_s"],
                "repeats": args.repeats,
                "warmup": args.warmup,
                "tokens_per_s": args.steps / max(dense_timed["elapsed_s"], 1e-9),
                "diagnostics": dense_diag,
                "divergence_vs_dense": {"mean_abs": 0.0, "max_abs": 0.0},
            }
        ],
    }

    for top_k in args.top_k:
        sparse_cfg = make_config(args, sparse=True, top_k=top_k)
        sparse_model = MTLNNModel(sparse_cfg).to(device).eval()
        sparse_model.load_state_dict(dense_state, strict=False)
        sparse_timed = _timed_repeats(
            device, lambda: run_full(sparse_model, tokens), args.repeats, args.warmup
        )
        sparse_logits = sparse_timed["result"]
        sparse_diag = gate_diagnostics(sparse_model)
        results["runs"].append({
            "mode": "sparse",
            "top_k": top_k,
            "elapsed_s": sparse_timed["elapsed_s"],
            "elapsed_min_s": sparse_timed["elapsed_min_s"],
            "elapsed_std_s": sparse_timed["elapsed_std_s"],
            "repeats": args.repeats,
            "warmup": args.warmup,
            "tokens_per_s": args.steps / max(sparse_timed["elapsed_s"], 1e-9),
            "diagnostics": sparse_diag,
            "divergence_vs_dense": divergence(dense_logits, sparse_logits),
        })

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.markdown_output.write_text(render_markdown(results), encoding="utf-8")

    print(f"wrote {args.json_output}")
    print(f"wrote {args.markdown_output}")
    for row in results["runs"]:
        div = row["divergence_vs_dense"]
        print(
            f"{row['mode']} top_k={row['top_k']} "
            f"{row['tokens_per_s']:.1f} tok/s "
            f"mean={row['elapsed_s']:.4f}s min={row['elapsed_min_s']:.4f}s "
            f"selected={row['diagnostics'].get('sparse_resonance_scale_ratio', 1.0):.3f} "
            f"mean_div={div['mean_abs']:.3g} max_div={div['max_abs']:.3g}"
        )


if __name__ == "__main__":
    main()
