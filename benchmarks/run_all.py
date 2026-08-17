#!/usr/bin/env python3
"""
benchmarks/run_all.py — run the MT-LNN benchmark suite and emit a single
consolidated Markdown report (``benchmarks/report.md``).

Each benchmark is an independent CLI; this driver runs them as subprocesses,
captures their output, and stitches the results into one report so a reviewer
(or the deck) can cite a single artifact. A failure in one benchmark never
aborts the others — it is recorded as FAIL with its traceback.

Most benchmarks train their own tiny models (selective-copy) or measure memory
on a fresh model (operator-compression), so they need no checkpoint. The
TorchScript latency section is the one that benefits from a real trained model;
pass ``--ckpt checkpoints/m2_final.pt`` to include it.

Usage
-----
    python benchmarks/run_all.py                       # full suite, no ckpt
    python benchmarks/run_all.py --ckpt checkpoints/m2_final.pt
    python benchmarks/run_all.py --quick               # skip the long ones
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
MAX_CAPTURE = 12_000  # chars of captured output kept per section


def _run(title: str, argv: list[str], timeout: int) -> dict:
    print(f"\n=== {title} ===\n  $ {' '.join(argv)}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PY, *argv], cwd=str(ROOT), capture_output=True, text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
    except subprocess.TimeoutExpired:
        out = f"timed out after {timeout}s"
        status = "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        out = f"{type(e).__name__}: {e}"
        status = "ERROR"
    dt = time.time() - t0
    if len(out) > MAX_CAPTURE:
        out = out[:MAX_CAPTURE] + f"\n... [truncated {len(out) - MAX_CAPTURE} chars]"
    print(f"  -> {status} in {dt:.1f}s")
    return {"title": title, "argv": argv, "status": status,
            "seconds": round(dt, 1), "output": out}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, default=None,
                   help="Trained checkpoint .pt to include in the latency section.")
    p.add_argument("--out", type=str, default=str(ROOT / "benchmarks" / "report.md"))
    p.add_argument("--quick", action="store_true",
                   help="Skip long-running benchmarks (long_context, end_to_end).")
    args = p.parse_args(argv)

    plan: list[tuple[str, list[str], int, bool]] = [
        # (title, argv, timeout_secs, quick_skip)
        ("Operator compression (O(1) memory)",
         ["benchmarks/operator_compression_report.py"], 600, False),
        ("Sparse-resonance ablation (CPU speed)",
         ["benchmarks/sparse_resonance_ablation.py"], 900, False),
        ("End-to-end: selective-copy + AVP",
         ["benchmarks/run_benchmark.py"], 1200, True),
        ("Long-context needle-in-a-haystack",
         ["benchmarks/long_context.py"], 1800, True),
    ]

    results = []
    for title, cmd, timeout, quick_skip in plan:
        if args.quick and quick_skip:
            results.append({"title": title, "argv": cmd, "status": "SKIPPED (--quick)",
                            "seconds": 0, "output": ""})
            continue
        results.append(_run(title, cmd, timeout))

    # TorchScript latency only makes sense with a real (or fresh) model.
    if args.ckpt:
        if not os.path.exists(args.ckpt):
            results.append({"title": "TorchScript export + latency",
                            "argv": ["--ckpt", args.ckpt], "status": "ERROR",
                            "seconds": 0, "output": f"checkpoint not found: {args.ckpt}"})
        else:
            results.append(_run(
                "TorchScript export + latency",
                ["scripts/export_torchscript.py", "--ckpt", args.ckpt,
                 "--seq_len", "256", "--optimize", "--no_eager"], 900))

    # --- write the consolidated Markdown report ---------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# MT-LNN Benchmark Report",
        "",
        f"- Generated: {ts}",
        f"- Checkpoint: `{args.ckpt}`" if args.ckpt else "- Checkpoint: _(none — fresh/self-trained models)_",
        f"- Python: `{sys.version.split()[0]}`",
        "",
        "## Summary",
        "",
        "| Benchmark | Status | Time (s) |",
        "| --- | --- | --- |",
    ]
    for r in results:
        lines.append(f"| {r['title']} | {r['status']} | {r['seconds']} |")
    lines.append("")
    for r in results:
        lines += [f"## {r['title']}", "",
                  f"**Status:** {r['status']} &nbsp; **Time:** {r['seconds']}s", "",
                  "```", r["output"].rstrip() or "(no output)", "```", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")

    n_ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n[run_all] {n_ok}/{len(results)} OK → report written to {out_path}")
    # Non-zero exit if anything hard-failed (not counting skips).
    bad = [r for r in results if r["status"] not in ("OK",) and "SKIP" not in r["status"]]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
