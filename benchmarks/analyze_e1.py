"""E1 result analysis: parity d16 sel vs stock separation verdict (decision gate G1).

Usage: py -3.11 benchmarks/analyze_e1.py [--difficulty 16] [--tag-sel e1-d16-sel] [--tag-stock e1-d16-stock]

Reads reasoning_depth.jsonl rows for two tags, extracts per-k acc (nested dict
from mix mode), runs protocol_report on the max k (= difficulty), prints G1.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.exp_protocol import (
    load_jsonl_rows,
    protocol_report,
)

RESULTS = "benchmarks/results/reasoning_depth.jsonl"


def extract_max_k_accs(rows, tag, difficulty):
    """Extract per-seed acc at the max k (= difficulty) for a tag.

    mtlnn_acc_by_depth is like {"1": {"1": 0.99, "2": 0.98, ..., "16": 1.0}}
    (mix-mode per-k eval, depth key="1"). Pick the value at k == difficulty.
    """
    sel_rows = sorted([r for r in rows if r.get("tag") == tag],
                      key=lambda r: r.get("seed", 0))
    accs = []
    for r in sel_rows:
        by_depth = r.get("mtlnn_acc_by_depth", {})
        d1 = by_depth.get("1", by_depth)
        if isinstance(d1, dict):
            val = None
            for k, v in d1.items():
                if int(k) == difficulty:
                    val = v
                    break
            if val is None and d1:
                val = max(float(v) for v in d1.values())
            accs.append(float(val))
        else:
            accs.append(float(d1))
    return accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--difficulty", type=int, default=16)
    p.add_argument("--tag-sel", default="e1-d16-sel")
    p.add_argument("--tag-stock", default="e1-d16-stock")
    p.add_argument("--results", default=RESULTS)
    args = p.parse_args()

    rows = load_jsonl_rows(args.results)
    sel = extract_max_k_accs(rows, args.tag_sel, args.difficulty)
    stock = extract_max_k_accs(rows, args.tag_stock, args.difficulty)

    print(f"== E1 verdict k={args.difficulty} ==")
    print(f"sel   accs: {[f'{a:.4f}' for a in sel]}")
    print(f"stock accs: {[f'{a:.4f}' for a in stock]}")

    if not sel or not stock:
        print("ERROR: insufficient data - E1 may not be finished or tag mismatch")
        return

    rep = protocol_report(sel, stock)
    print(f"\nsel   grok rate: {rep['sel_grok_rate']:.3f}  ({rep['sel_grok']}/{rep['n_sel']})")
    print(f"stock grok rate: {rep['stock_grok_rate']:.3f}  ({rep['stock_grok']}/{rep['n_stock']})")
    print(f"Fisher p: {rep['fisher_p']:.4f}")
    print(f"sign-test p: {rep['sign_test_p']:.4f}  (pos={rep['sign_test_pos']}, neg={rep['sign_test_neg']})")
    print(f"sel   bimodality: {rep['sel_bimodality']['verdict']}")
    print(f"stock bimodality: {rep['stock_bimodality']['verdict']}")
    print(f"\nverdict: {rep['verdict']}")
    print(f"decision gate G1: {'PASSED (separation holds)' if rep['decision_gate_passed'] else 'NOT passed'}")


if __name__ == "__main__":
    main()
