"""bench_trace_audit.py — quantify how "self-sufficient" an AwareLiquid trace is.

Reads a ``*.trace.jsonl`` produced by :class:`mt_lnn.reasoning_trace.ReasoningTrace`
and emits a report:

  - token route breakdown (local / self_critique / cloud)
  - cloud-inject count + bytes absorbed
  - entropy stats (mean, p50, p95, max)
  - Φ̂ stats if sampled
  - **self_sufficiency**: 1 - (cloud_tokens / total_tokens) — the key
    AwareLiquid claim: "we only phone home when we have to"
  - **estimated_cloud_cost_saved_usd**: rough $ saved vs always-cloud, given
    typical frontier API pricing of ~$3 / 1M input tokens, ~$15 / 1M output

Usage::

    python scripts/bench_trace_audit.py demo_trace.jsonl
    python scripts/bench_trace_audit.py *.trace.jsonl --format json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Dict, List


PRICE_IN_PER_MTOK = 3.0
PRICE_OUT_PER_MTOK = 15.0


def load_trace(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def audit_one(path: Path) -> Dict:
    rows = load_trace(path)
    tokens = [r for r in rows if r.get("event") == "token"]
    injects = [r for r in rows if r.get("event") == "cloud_inject"]
    if not tokens:
        return {"path": str(path), "error": "no token events"}

    by_route = {"local": 0, "self_critique": 0, "cloud": 0}
    entropies, phis = [], []
    for t in tokens:
        r = t.get("route", "local")
        by_route[r] = by_route.get(r, 0) + 1
        if isinstance(t.get("entropy"), (int, float)):
            entropies.append(float(t["entropy"]))
        if isinstance(t.get("phi"), (int, float)):
            phis.append(float(t["phi"]))

    total = len(tokens)
    cloud_tok = by_route.get("cloud", 0)
    self_suff = 1.0 - (cloud_tok / total) if total else 0.0
    bytes_in = sum(int(i.get("fact_len", 0)) for i in injects)

    # Cost: if all generation had gone to cloud, every output token bills out
    # rate + the prompt context bills in rate once per request. We model
    # "saved" as the fraction of tokens we kept local.
    saved_output_usd = (total - cloud_tok) * PRICE_OUT_PER_MTOK / 1e6
    spent_input_usd = bytes_in / 4.0 * PRICE_IN_PER_MTOK / 1e6  # ~4 bytes/token

    def pctl(xs, p):
        return float(statistics.quantiles(xs, n=100)[p - 1]) if len(xs) >= 100 else (max(xs) if xs else 0.0)

    return {
        "path": str(path),
        "session_id": tokens[0].get("session_id"),
        "total_tokens": total,
        "routes": by_route,
        "cloud_injects": len(injects),
        "absorbed_bytes": bytes_in,
        "self_sufficiency": round(self_suff, 4),
        "entropy": {
            "mean": round(sum(entropies) / len(entropies), 3) if entropies else 0,
            "p50": round(statistics.median(entropies), 3) if entropies else 0,
            "p95": round(pctl(entropies, 95), 3) if len(entropies) >= 100 else (round(max(entropies), 3) if entropies else 0),
            "max": round(max(entropies), 3) if entropies else 0,
        },
        "phi_hat": (
            {
                "samples": len(phis),
                "mean": round(sum(phis) / len(phis), 4),
                "max": round(max(phis), 4),
            }
            if phis
            else {"samples": 0}
        ),
        "estimated_cost_usd": {
            "saved_vs_full_cloud_output": round(saved_output_usd, 6),
            "spent_on_cloud_inject_input": round(spent_input_usd, 6),
            "net_savings": round(saved_output_usd - spent_input_usd, 6),
        },
    }


def format_human(report: Dict) -> str:
    if "error" in report:
        return f"[{report['path']}] ERROR: {report['error']}"
    r = report
    lines = [
        f"\n=== {Path(r['path']).name} (session: {r['session_id']}) ===",
        f"Total tokens         : {r['total_tokens']}",
        f"  LOCAL              : {r['routes']['local']:>5}  ({pct(r['routes']['local'], r['total_tokens'])})",
        f"  SELF_CRITIQUE      : {r['routes']['self_critique']:>5}  ({pct(r['routes']['self_critique'], r['total_tokens'])})",
        f"  CLOUD              : {r['routes']['cloud']:>5}  ({pct(r['routes']['cloud'], r['total_tokens'])})",
        f"Cloud injects        : {r['cloud_injects']}  ({r['absorbed_bytes']}B absorbed)",
        f"Self-sufficiency     : {r['self_sufficiency'] * 100:.2f}%   (1 - cloud/total)",
        f"Entropy mean/max     : {r['entropy']['mean']} / {r['entropy']['max']}",
    ]
    if r["phi_hat"]["samples"]:
        lines.append(f"Phi-hat samples/mean : {r['phi_hat']['samples']} / {r['phi_hat']['mean']}")
    c = r["estimated_cost_usd"]
    lines.append(
        f"Cost vs full-cloud   : saved ${c['saved_vs_full_cloud_output']:.6f} output, "
        f"spent ${c['spent_on_cloud_inject_input']:.6f} input "
        f"-> net ${c['net_savings']:.6f}"
    )
    return "\n".join(lines)


def pct(a: int, b: int) -> str:
    return f"{(100 * a / b):.1f}%" if b else "0%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="trace JSONL files (globs ok)")
    p.add_argument("--format", choices=["human", "json"], default="human")
    args = p.parse_args()

    expanded: List[Path] = []
    for pat in args.paths:
        matched = [Path(x) for x in glob.glob(pat)]
        expanded.extend(matched if matched else [Path(pat)])
    expanded = [p for p in expanded if p.exists()]
    if not expanded:
        print("no matching trace files")
        return

    reports = [audit_one(p) for p in expanded]
    if args.format == "json":
        print(json.dumps(reports, indent=2, ensure_ascii=False))
    else:
        for r in reports:
            print(format_human(r))


if __name__ == "__main__":
    main()
