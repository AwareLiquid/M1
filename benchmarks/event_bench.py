"""event_bench.py — 事件流任务族：液态核心 vs 离散基线（CPU, E0 协议）。

任务（数据来自 benchmarks/event_stream.py 的 DVS 物理生成器）：
  * next_channel — 流式预测：看前 T-1 个事件，预测下一个事件的通道
    （C 类分类，chance = 1/C）
  * state        — 状态估计：看整条事件流，回归末时刻潜信号通道均值
    （回归，y 用训练集统计标准化 → normalized RMSE，常数基线 = 1.0）

基线经 benchmarks/battery_soh_edge.py 的 build() 工厂（模块级 import
不触发数据下载）：mt_lnn / lstm / gru / transformer，仅替换输入投影与
输出头以适配任务。**Δt 通道喂给所有架构**（公平护栏，口径沿用
battery_irregular_sampling：不规则信息不独占给任何一方）。

gru_d：属 iter/irregular-streaming-edge 分支，未合入 main——本脚本不
本地重写，JSON 里记 "gru_d: pending merge"。

E0 协议：每 θ 档每任务 mt_lnn vs 每个基线过 publishable()
（≥3 seeds、非双峰、配对符号检验 p<0.05），不达标只入档不引用。

θ 扫 3 档（事件密度），seeds ≥3，resume-safe（重跑跳过已完成 key）。

    python benchmarks/event_bench.py --smoke          # <2min 全链路
    python benchmarks/event_bench.py                  # 全量
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from benchmarks.battery_soh_edge import build
from benchmarks.event_stream import make_dataset
from benchmarks.experiment_protocol import publishable

THETAS = [0.15, 0.35, 0.8]          # 事件密度 3 档（burst 内每步事件率 0.4/0.22/0.06）
TASKS = ["next_channel", "state"]


def main():
    args = _cli()
    thetas = [args.theta] if args.smoke else THETAS
    seeds = [0] if args.smoke else [int(s) for s in args.seeds.split(",")]
    if args.smoke:                      # <2min 预算的缩减配置
        args.n_train, args.n_test, args.epochs = 200, 50, 6
    path = args.out
    res = _load(path)
    for theta in thetas:
        tr = make_dataset(args.n_train, args.channels, args.seq_len, theta,
                          args.span, args.bandwidth, seed=100)
        te = make_dataset(args.n_test, args.channels, args.seq_len, theta,
                          args.span, args.bandwidth, seed=900)
        for task in (TASKS if not args.smoke else ["next_channel"]):
            _run_task_grid(res, args, task, theta, seeds, tr, te)
            _save(path, res, args)
    _report(res, thetas)
    _save(path, res, args)               # gates 在 _report 里算，必须再落盘
    return 0


def _run_task_grid(res, args, task, theta, seeds, tr, te):
    """一个 (task, θ) 格子：全部架构 × 全部 seeds，逐 key resume。"""
    Xtr, ytr = task_views(tr, task)
    Xte, yte = task_views(te, task)
    print(f"\n== {task} | theta={theta} | "
          f"dt span {tr['dt_span_decades']:.2f} dec | chance "
          f"{_chance(task, args.channels):.3f} ==")
    for arch in [a for a in args.archs.split(",") if a]:
        vals = []
        for seed in seeds:
            key = f"{task}|{theta}|{arch}|{seed}"
            if key in res["runs"]:
                vals.append(res["runs"][key]["metric"])
                continue
            r = train_eval(arch, task, Xtr, ytr, Xte, yte, args, seed)
            res["runs"][key] = r
            vals.append(r["metric"])
        print(f"  {arch:<12} mean {np.mean(vals):.4f} ± "
              f"{np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f}  "
              f"n={len(vals)}")


def train_eval(arch, task, Xtr, ytr, Xte, yte, args, seed):
    """训练一个模型并返回指标（acc 或 normalized RMSE）+ 元信息。"""
    torch.manual_seed(seed)
    n_in, n_out = Xtr.shape[-1], (args.channels if task == "next_channel" else 1)
    m = _build_event(arch, n_in, n_out, args).to(args.dev)
    n_params = sum(p.numel() for p in m.parameters())
    lossf = nn.CrossEntropyLoss() if task == "next_channel" else nn.MSELoss()
    t = _train(m, torch.from_numpy(Xtr).to(args.dev),
               torch.from_numpy(ytr).to(args.dev), lossf, args)
    if t is not None:
        return {"metric": float("nan"), "stable": False,
                "params": n_params, "seed": seed}
    return _evaluate(m, task, torch.from_numpy(Xte).to(args.dev), yte,
                     n_params, seed)


def _train(m, Xtr_t, ytr_t, lossf, args):
    """AdamW + 梯度裁剪训练循环；损失非有限返回 True（调用方记 unstable）。"""
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    m.train()
    for _ in range(args.epochs):
        perm = torch.randperm(len(Xtr_t), device=args.dev)
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(m(Xtr_t[idx]), ytr_t[idx])
            if not torch.isfinite(loss):
                return True
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return None


def _build_event(arch, n_in, n_out, args):
    """build() 工厂 + 任务适配：换输入投影（n_in 维）与输出头（n_out 维）。

    Δt 是输入张量的最后一维，对所有架构一视同仁地过 inp 线性层。
    mt_lnn_tau1：液态核心但内部 τ 阶梯压到单一时间尺度（Task 3 的
    τ 阶梯交互消融臂）——只换 MTLNNLayer 配置，不改工厂其他部件。
    """
    m = build("mt_lnn" if arch == "mt_lnn_tau1" else arch,
              args.d_model, args.n_layers, args.seq_len)
    if arch == "mt_lnn_tau1":
        _swap_tau_scales(m, 1, args)
    m.inp = nn.Linear(n_in, args.d_model)
    m.head = nn.Linear(args.d_model, n_out)
    return m


def _swap_tau_scales(m, n_scales, args):
    """把 LiquidRegressor 的 MTLNNLayer 栈换成 n_scales 时间尺度的版本。"""
    from mt_lnn.config import MTLNNConfig
    from mt_lnn.mt_lnn_layer import MTLNNLayer
    cfg = MTLNNConfig(vocab_size=2, d_model=args.d_model,
                      n_layers=args.n_layers, n_heads=13,
                      d_head=max(1, args.d_model // 13), n_protofilaments=13,
                      n_time_scales=n_scales, max_seq_len=4096,
                      dropout=0.0, attention_dropout=0.0)
    m.layers = nn.ModuleList([MTLNNLayer(cfg)
                              for _ in range(args.n_layers)])


def task_views(ds, task):
    """从事件张量取 (X, y)：预测任务滑一位，状态任务用标准化目标。"""
    X = ds["X"]
    if task == "next_channel":
        C = X.shape[-1] - 2
        return (X[:, :-1, :].astype(np.float32),
                X[:, -1, :C].argmax(axis=1).astype(np.int64))
    y = ds["y"]
    return X, ((y - y.mean()) / (y.std() + 1e-8)).astype(np.float32)


def _evaluate(m, task, Xte_t, yte, n_params, seed):
    m.eval()
    with torch.no_grad():
        out = m(Xte_t).cpu().numpy()
    if task == "next_channel":
        metric = float((out.argmax(axis=1) == yte).mean())
    else:
        metric = float(np.sqrt(((out.reshape(-1) - yte) ** 2).mean()))
    return {"metric": metric, "stable": True, "params": n_params,
            "seed": seed}


def _report(res, thetas):
    """按 (task, θ, arch) 汇总 + mt_lnn vs 每基线的 E0 可引用判定。"""
    print("\n" + "=" * 72)
    for task in sorted({k.split("|")[0] for k in res["runs"]}):
        for theta in thetas:
            rows = _collect(res, task, theta)
            if not rows:
                continue
            print(f"\n{task} | theta={theta}")
            for arch, vals in rows.items():
                print(f"  {arch:<12} {np.mean(vals):>8.4f} ± "
                      f"{np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f} "
                      f"(n={len(vals)})")
            _gate(res, rows, task, theta)


def _gate(res, rows, task, theta):
    """mt_lnn vs 每个基线的 publishable() 判定，写回 res['gates']。"""
    if "mt_lnn" not in rows:
        return
    for arch, vals in rows.items():
        if arch == "mt_lnn" or len(vals) != len(rows["mt_lnn"]):
            continue
        ok, rep = publishable(_better(rows["mt_lnn"], task),
                              _better(vals, task),
                              tag=f"{task}|θ{theta}|mt_lnn vs {arch}")
        res.setdefault("gates", {})[rep["tag"]] = {
            "ok": ok, "reasons": rep["reasons"],
            "sign_test": rep.get("sign_test", {})}
        print(f"  E0 {arch:<10} {'CITABLE' if ok else 'archive-only'} "
              f"{rep['reasons'] or ''}")


def _collect(res, task, theta):
    rows = {}
    for k, v in res["runs"].items():
        t, th, arch, _ = k.split("|")
        if t == task and float(th) == theta and v.get("stable"):
            rows.setdefault(arch, []).append(v["metric"])
    return rows


def _better(vals, task):
    """统一方向：publishable 检 a>b；指标越小越好时取负。"""
    return vals if task == "next_channel" else [-v for v in vals]


def _chance(task, n_channels):
    return 1.0 / n_channels if task == "next_channel" else 1.0


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": {}, "gru_d": "pending merge (iter/irregular-streaming-edge)"}


def _save(path, res, args):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    res["config"] = {"d_model": args.d_model, "n_layers": args.n_layers,
                     "epochs": args.epochs, "seq_len": args.seq_len,
                     "n_train": args.n_train, "span": args.span,
                     "channels": args.channels}
    with open(path, "w") as f:
        json.dump(res, f, indent=2)


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", default="mt_lnn,lstm,gru,transformer")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--bandwidth", type=float, default=2.0)
    ap.add_argument("--span", type=float, default=2.0)
    ap.add_argument("--theta", type=float, default=0.35)
    ap.add_argument("--d_model", type=int, default=52)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="benchmarks/results/event_theta_sweep.json")
    args = ap.parse_args()
    args.dev = "cuda" if torch.cuda.is_available() else "cpu"
    return args


if __name__ == "__main__":
    raise SystemExit(main())
