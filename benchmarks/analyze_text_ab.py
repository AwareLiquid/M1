"""Analyze text_selective_ab.jsonl: paired selective vs stock comparison.

Protocol (see text_selective_ab.py): for each budget (steps) and seed,
two matched models (same init seed, same data) differ ONLY in
selective_decay. Rows are appended per budget run; the same (seed, steps)
appears once for selective=False and once for selective=True.

This script:
  1. Groups rows by budget (steps).
  2. For the newest budget, computes the PAIRED delta per seed
     (delta = stock_ppl - selective_ppl; positive => selective better).
  3. Reports mean delta, per-seed wins, and the sign-test p-value
     (exact binomial, P(get >= w wins | p=0.5)).
  4. Optionally prints the v3-vs-v4 trajectory per seed (does the
     selective edge grow with budget?).

Usage:
  python benchmarks/analyze_text_ab.py [--all]
  --all: print every budget's paired table, not just the newest.
"""
import json, os, math, sys
from itertools import groupby

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "text_selective_ab.jsonl")


def load():
    rows = []
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sign_test(w, n):
    """Exact two-sided binomial sign test: P(X >= w | n, p=0.5)."""
    p = 0.0
    for k in range(w, n + 1):
        p += math.comb(n, k) * (0.5 ** n)
    return min(1.0, 2 * p)  # two-sided


def table(rows):
    # key: (steps, seed) -> dict(selective: ppl, stock: ppl)
    cells = {}
    for r in rows:
        key = (r["steps"], r["seed"])
        arm = "selective" if r["selective"] else "stock"
        cells.setdefault(key, {})[arm] = r["val_ppl"]
    budgets = sorted({k[0] for k in cells})
    return cells, budgets


def report(cells, budgets, newest_only):
    budgets = budgets if not newest_only else budgets[-1:]
    print(f"{'budget':>7} {'seed':>4} {'stock':>10} {'selective':>10} "
          f"{'delta':>9} {'win?':>5}")
    print("-" * 52)
    for b in budgets:
        deltas, wins, n = [], 0, 0
        for seed in sorted({k[1] for k in cells if k[0] == b}):
            c = cells.get((b, seed))
            if not c or "stock" not in c or "selective" not in c:
                continue
            d = c["stock"] - c["selective"]
            deltas.append(d)
            n += 1
            if d > 0:
                wins += 1
            print(f"{b:>7} {seed:>4} {c['stock']:>10.3f} {c['selective']:>10.3f} "
                  f"{d:>+9.3f} {'SEL' if d > 0 else '--'}")
        if deltas:
            mean = sum(deltas) / len(deltas)
            p = sign_test(wins, n) if n else 1.0
            print(f"  mean delta {mean:+.3f} | selective wins {wins}/{n} | "
                  f"sign-test p={p:.3f}")
        print()


def trajectory(cells, budgets):
    """Per-seed stock & selective PPL across budgets (does the edge grow?)."""
    print("trajectory (val PPL, lower=better):")
    print(f"{'seed':>4} {'budget':>7} {'stock':>10} {'selective':>10} "
          f"{'delta':>9}")
    print("-" * 48)
    for seed in sorted({k[1] for k in cells}):
        for b in budgets:
            c = cells.get((b, seed))
            if not c or "stock" not in c or "selective" not in c:
                continue
            d = c["stock"] - c["selective"]
            print(f"{seed:>4} {b:>7} {c['stock']:>10.3f} {c['selective']:>10.3f} "
                  f"{d:>+9.3f}")
        print()


if __name__ == "__main__":
    rows = load()
    if not rows:
        print(f"no rows in {OUT}")
        sys.exit(1)
    cells, budgets = table(rows)
    print(f"{len(rows)} rows, budgets={budgets}\n")
    report(cells, budgets, newest_only=("--all" not in sys.argv))
    if "--all" in sys.argv:
        trajectory(cells, budgets)
