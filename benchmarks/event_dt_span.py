"""event_dt_span.py — Δt 跨度压力测试 + τ 阶梯交互（本分支独有科学问题）。

唯一要回答的问题：Δt 分布跨度从 ~1 个数量级拉到 ~4 个数量级时，各架构
退化曲线的斜率谁最小？若 mt_lnn 优势随跨度扩大而扩大 → 连续时间主张
成立；若打平 → 如实记 Null。

【判定标准（跑之前预写死，不可事后移动）】
  PROVEN("优势随 Δt 跨度扩大") iff 同时满足：
    (a) 最宽跨度档：mt_lnn vs 最强离散基线的 publishable() 通过
        （≥3 seeds、双臂非双峰、配对符号检验 p<0.05，且 mt_lnn 更优——
        state 任务 RMSE 更低）；
    (b) gap(基线 − mt_lnn) 在最宽档 > 最窄档（差距随跨度拉大）。
  任一不满足 → Null，RESULTS.md 记 Null 区。
  τ 阶梯交互（次级，同门槛）：mt_lnn vs mt_lnn_tau1（τ 阶梯压到 1 个
  时间尺度）在最宽档过 publishable 且 τ1 更差 → 多尺度 τ 阶梯是跨度
  鲁棒性的来源之一；否则记 Null。

任务：state（状态估计）——Δt 最敏感的任务（要从事件流+真时间差回归
末时刻潜信号）。Δt 喂所有架构（护栏沿用 event_bench 口径）。

跨度档 [0, 1.5, 3, 6] → realized（p98/p5，排除同格同时事件）约
1.4 / 1.8 / 2.3 / 4.0 个数量级（以 JSON 实测为准）。事件总量跨档
大体不变（gap 拉长不稀释事件密度）。CPU 可跑，resume-safe。

    python benchmarks/event_dt_span.py --smoke
    python benchmarks/event_dt_span.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from benchmarks.event_bench import _build_event, task_views, train_eval
from benchmarks.event_stream import make_dataset
from benchmarks.experiment_protocol import publishable

SPANS = [0.0, 1.5, 3.0, 6.0]
ARCHS = ["mt_lnn", "mt_lnn_tau1", "lstm", "gru", "transformer"]


def main():
    args = _cli()
    spans, seeds = ([SPANS[-1]], [0]) if args.smoke else (SPANS, args.seeds)
    if args.smoke:
        args.n_train, args.n_test, args.epochs = 200, 50, 6
    res = _load(args.out)
    for span in spans:
        tr = make_dataset(args.n_train, args.channels, args.seq_len,
                          args.theta, span, args.bandwidth, seed=100)
        te = make_dataset(args.n_test, args.channels, args.seq_len,
                          args.theta, span, args.bandwidth, seed=900)
        _run_span(res, args, span, seeds, tr, te)
    _analyze(res, spans)
    _save(args.out, res, args)
    return 0


def _run_span(res, args, span, seeds, tr, te):
    """一个跨度档：全部架构 × seeds（resume-safe，key 含 span）。"""
    Xtr, ytr = task_views(tr, "state")
    Xte, yte = task_views(te, "state")
    res.setdefault("realized", {})[str(span)] = tr["dt_span_decades"]
    print(f"\n== state | span={span} | realized dt span "
          f"{tr['dt_span_decades']:.2f} dec ==")
    for arch in ARCHS:
        vals = []
        for seed in seeds:
            key = f"state|{span}|{arch}|{seed}"
            if key in res["runs"]:
                vals.append(res["runs"][key]["metric"])
                continue
            r = train_eval(arch, "state", Xtr, ytr, Xte, yte, args, seed)
            res["runs"][key] = r
            vals.append(r["metric"])
        print(f"  {arch:<14} mean {np.mean(vals):.4f} ± "
              f"{np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f}")


def _analyze(res, spans):
    """汇总 + 预注册判定（a/b 两关 + τ 交互）——只写档，不美化。"""
    rows = _collect(res, spans)
    if not rows:
        return
    print("\n" + "=" * 72)
    realized = [np.mean([v["span_dec"] for v in rows[s]["meta"]])
                for s in spans if s in rows]
    print("state normalized RMSE（越低越好）vs realized Δt span:")
    table = {}
    for s in spans:
        if s not in rows:
            continue
        for arch, vals in rows[s]["arms"].items():
            table.setdefault(arch, []).append(np.mean(vals))
            print(f"  span={s} ({np.mean([m['span_dec'] for m in rows[s]['meta']]):.2f} dec) "
                  f"{arch:<14} {np.mean(vals):.4f}")
    verdict = _verdict(rows, spans, table)
    res["verdict"] = verdict
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print(f"slope per decade: " + ", ".join(
        f"{a}={_slope(realized, ys):+.4f}" for a, ys in table.items()))
    res["slopes"] = {a: _slope(realized, ys) for a, ys in table.items()}


def _verdict(rows, spans, table):
    """预注册规则的两关 + τ 交互关。返回可直接入 JSON 的裁决。"""
    lo, hi = spans[0], spans[-1]
    discrete = ["lstm", "gru", "transformer"]
    hi_arms = {a: rows[hi]["arms"][a] for a in discrete if a in rows[hi]["arms"]}
    best = min(hi_arms, key=lambda a: np.mean(hi_arms[a]))
    mt = rows[hi]["arms"]["mt_lnn"]
    ok, rep = publishable(mt, hi_arms[best],
                          tag=f"span{hi}|state|mt_lnn vs {best}")
    gate_a = ok and np.mean(mt) < np.mean(hi_arms[best])
    gate_b = (np.mean(hi_arms[best]) - np.mean(mt)
              > np.mean(rows[lo]["arms"][best]) - np.mean(rows[lo]["arms"]["mt_lnn"]))
    tau = _tau_verdict(rows, hi)
    return {"primary": "PROVEN" if (gate_a and gate_b) else "NULL",
            "gate_a_widest_span_significant": bool(gate_a),
            "gate_b_gap_widens": bool(gate_b),
            "strongest_discrete_at_widest": best,
            "sign_test": rep.get("sign_test", {}),
            "gate_reasons": rep.get("reasons", []),
            "tau_ladder_interaction": tau}


def _tau_verdict(rows, hi):
    """τ 交互：mt_lnn vs mt_lnn_tau1 在最宽档。"""
    if "mt_lnn_tau1" not in rows[hi]["arms"]:
        return "n/a"
    ok, rep = publishable(rows[hi]["arms"]["mt_lnn"],
                          rows[hi]["arms"]["mt_lnn_tau1"],
                          tag=f"span{hi}|state|mt_lnn vs mt_lnn_tau1")
    better = np.mean(rows[hi]["arms"]["mt_lnn"]) < \
        np.mean(rows[hi]["arms"]["mt_lnn_tau1"])
    return {"status": "PROVEN" if (ok and better) else "NULL",
            "sign_test": rep.get("sign_test", {}),
            "reasons": rep.get("reasons", [])}


def _collect(res, spans):
    """runs → {span: {arms: {arch: [metrics]}, meta: [{span_dec}]}}。"""
    rows = {}
    for k, v in res["runs"].items():
        _, sp, arch, _ = k.split("|")
        if v.get("stable") and float(sp) in spans:
            r = rows.setdefault(float(sp), {"arms": {}, "meta": []})
            r["arms"].setdefault(arch, []).append(v["metric"])
            span_dec = res.get("realized", {}).get(str(sp), np.nan)
            r["meta"].append({"span_dec": span_dec})
    return rows


def _slope(xs, ys):
    """最小二乘斜率（metric 对 realized 跨度的退化速率）。"""
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": {}, "gru_d": "pending merge (iter/irregular-streaming-edge)"}


def _save(path, res, args):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    res.setdefault("realized", {})
    res["config"] = {"d_model": args.d_model, "n_layers": args.n_layers,
                     "epochs": args.epochs, "seq_len": args.seq_len,
                     "theta": args.theta, "channels": args.channels}
    with open(path, "w") as f:
        json.dump(res, f, indent=2)


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--bandwidth", type=float, default=2.0)
    ap.add_argument("--theta", type=float, default=0.35)
    ap.add_argument("--d_model", type=int, default=52)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="benchmarks/results/event_dt_span.json")
    args = ap.parse_args()
    args.seeds = [int(s) for s in (args.seeds or "0,1,2").split(",")]
    args.dev = "cpu"
    return args


if __name__ == "__main__":
    raise SystemExit(main())
