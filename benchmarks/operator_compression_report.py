"""Generate a Markdown report for MT-LNN operator-compression features.

The report is intentionally lightweight and CPU-friendly. It runs the
state-only streaming benchmark, records dynamic scale-gate diagnostics, and
writes both JSON and Markdown artifacts.

Example:
    python benchmarks/operator_compression_report.py --steps 100 1000
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from benchmarks import state_only_streaming


def _mode(run: dict, name: str) -> dict:
    for row in run["modes"]:
        if row.get("mode") == name:
            return row
    raise KeyError(name)


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _fmt_float(x: float) -> str:
    return f"{x:.3g}"


def run_benchmark(args) -> dict:
    bench_args = argparse.Namespace(
        steps=args.steps,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        output=str(args.json_output),
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        gwtb_compression_ratio=args.gwtb_compression_ratio,
        gwtb_n_heads=args.gwtb_n_heads,
        coherence_heads=args.coherence_heads,
        max_prefix_replay_steps=args.max_prefix_replay_steps,
        disable_dynamic_scale_gates=args.disable_dynamic_scale_gates,
        scale_gate_init_bias=args.scale_gate_init_bias,
        scale_gate_active_threshold=args.scale_gate_active_threshold,
        scale_gate_skip_threshold=args.scale_gate_skip_threshold,
        compute_skip_threshold=args.compute_skip_threshold,
        sparse_resonance_kernel=args.sparse_resonance_kernel,
        sparse_resonance_top_k=args.sparse_resonance_top_k,
    )

    torch.manual_seed(args.seed)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    cfg = state_only_streaming.build_config(bench_args, max_steps=max(args.steps))
    model = state_only_streaming.MTLNNModel(cfg).to(device).eval()

    results = {
        "benchmark": "operator_compression_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "torch": torch.__version__,
        "config": state_only_streaming.dataclasses.asdict(cfg),
        "runs": [],
    }

    for steps in args.steps:
        tokens = torch.randint(0, cfg.vocab_size, (args.batch, steps), device=device)
        results["runs"].append(state_only_streaming.run_one(model, tokens, bench_args, device))

    return results


def render_markdown(results: dict) -> str:
    lines = [
        "# MT-LNN Operator Compression Report",
        "",
        f"Generated: `{results['created_at']}`",
        f"Device: `{results['device']}`",
        f"PyTorch: `{results['torch']}`",
        "",
        "## Summary",
        "",
        "State-only streaming preserves recurrent `h_prev` while dropping historical KV tensors. "
        "This is compressed memory, not lossless transcript recall.",
        "",
        "| Steps | Mode | Tok/s | Final cache | Peak cache | Mean div vs full | Max div vs full |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]

    for run in results["runs"]:
        for mode_name in ["full_sequence_oracle", "kv_cache_stream", "state_only_stream", "prefix_replay_stream"]:
            row = _mode(run, mode_name)
            if row.get("skipped"):
                lines.append(
                    f"| {run['steps']} | `{mode_name}` | skipped | - | - | - | - |"
                )
                continue
            div = row.get("divergence_vs_full_sequence", {})
            lines.append(
                f"| {run['steps']} | `{mode_name}` | {row['tokens_per_s']:.1f} | "
                f"{_fmt_bytes(row['final_cache_bytes'])} | {_fmt_bytes(row['peak_cache_bytes'])} | "
                f"{_fmt_float(div.get('mean_abs', 0.0))} | {_fmt_float(div.get('max_abs', 0.0))} |"
            )

    lines.extend([
        "",
        "## Scale-Gate Diagnostics",
        "",
        "| Steps | Gate mean | Active ratio | Nonzero ratio | Sparse selected ratio | Per-scale means |",
        "|---:|---:|---:|---:|---:|---|",
    ])

    for run in results["runs"]:
        diag = run.get("scale_gate_diagnostics", {})
        per_scale = [
            (k, v) for k, v in sorted(diag.items())
            if k.startswith("scale_gate_s")
        ]
        per_scale_text = ", ".join(f"{k.replace('scale_gate_', '')}={v:.3f}" for k, v in per_scale)
        lines.append(
            f"| {run['steps']} | {diag.get('scale_gate_mean', 0.0):.3f} | "
            f"{diag.get('scale_gate_active_ratio', 0.0):.3f} | "
            f"{diag.get('scale_gate_nonzero_ratio', 0.0):.3f} | "
            f"{diag.get('sparse_resonance_scale_ratio', 0.0):.3f} | {per_scale_text} |"
        )

    lines.extend([
        "",
        "## Caveats",
        "",
        "- KV streaming remains the exact incremental path; its divergence from the full-sequence oracle should stay near numerical noise.",
        "- State-only streaming is intentionally lossy relative to full attention because it keeps only recurrent state.",
        "- Current scale-gate masking affects the blend weights and diagnostics. It is not yet a custom sparse kernel that avoids the upstream resonance matrix multiply.",
        "- Optional `--sparse_resonance_kernel` skips inactive tau-scale projection/scan work, but top-k scale selection is chunk-dependent and can change KV parity relative to dense full-sequence execution.",
        "- `benchmarks/state_only_streaming.py --metrics_jsonl <path>` emits structured JSONL metric events for observability plumbing.",
        "- Use `requirements.lock` to reproduce the local smoke benchmark environment; `requirements.txt` intentionally keeps flexible research-time lower bounds.",
        "- SQLite session persistence is local-first and should not be treated as a distributed high-QPS state store.",
        "- Claims in external material should report measured cache size and latency separately from future compute-skipping hypotheses.",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--json_output", type=Path, default=Path("benchmarks/operator_compression_report.json"))
    parser.add_argument("--markdown_output", type=Path, default=Path("benchmarks/operator_compression_report.md"))

    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=104)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--gwtb_compression_ratio", type=int, default=4)
    parser.add_argument("--gwtb_n_heads", type=int, default=2)
    parser.add_argument("--coherence_heads", type=int, default=2)
    parser.add_argument("--max_prefix_replay_steps", type=int, default=128)

    parser.add_argument("--disable_dynamic_scale_gates", action="store_true")
    parser.add_argument("--scale_gate_init_bias", type=float, default=2.0)
    parser.add_argument("--scale_gate_active_threshold", type=float, default=0.5)
    parser.add_argument("--scale_gate_skip_threshold", type=float, default=0.0)
    parser.add_argument("--compute_skip_threshold", type=float, default=0.0)
    parser.add_argument("--sparse_resonance_kernel", action="store_true")
    parser.add_argument("--sparse_resonance_top_k", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_benchmark(args)
    markdown = render_markdown(results)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.markdown_output.write_text(markdown, encoding="utf-8")

    print(f"wrote {args.json_output}")
    print(f"wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
