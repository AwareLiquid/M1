"""Task 2 裁决驱动 — core vs stack 潜空间循环 (iter/latent-recursion)。

假设 H (预注册，依据 HANDOFF §2.5 教训与 docs/LATENT_RECURSION.md):
  整块循环 (stack_iterations，循环体 = 注意力+LNN) 使深度-准确率曲线单调上升，
  核心循环 (core_iterations，循环体 = LNN 子层) 不上升。
判负标准 (写死于 judge_decision，不事后移动):
  H 成立 ⇔ mean(stack) 随深度严格单调上升
           且 mean(stack,d8) − mean(stack,d1) ≥ 2·σ_d8 (σ ddof=1)
           且 E0 门槛通过 (publishable: ≥3 seeds + 非双峰 + 配对 p<0.05)。
  否则 Null，诊断四选一 (数据驱动): budget_wall / bimodal_grokking_zone /
  flat_iterations_ignored / direction_but_underpowered。
配对符号检验 3 seeds 的最小 p = 0.25 > 0.05，故可引用判决默认 6 seeds
(seeds 0..5，双峰掷硬币区也容得下 2 个坏种子)。

协议: pointer_chase 单环 mix 课程 (difficulty=8, n=16)，每配置全新模型
固定深度训练 (HRM 式)，per-k 评估；headline = mean_k(acc_k)。

Resume-safe: 每配置 (mode,depth,seed) 完成即原子写
  <out_dir>/latent_recursion_{mode}_d{depth}_s{seed}.json，
重启自动跳过已完成配置；全部完成后写
  <out_dir>/latent_recursion_verdict.json。
日志: train_model 的 step/loss 行 tee 到
  <out_dir>/latent_recursion_{mode}_d{depth}_s{seed}.log (带时间戳)。

用法:
  py benchmarks/latent_recursion.py --probe --device cpu   # 测速 + 全协议 ETA
  py benchmarks/latent_recursion.py --steps 3000 --seeds 0 1 2
  py benchmarks/latent_recursion.py                        # 30k 步 × 6 seeds 全协议
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.experiment_protocol import (bimodality_1d, paired_sign_test,
                                            publishable)
from benchmarks.reasoning_depth import (build_mtlnn, evaluate_per_k,
                                        make_mix_generator, train_model)
from benchmarks.reasoning_tasks import make_generator

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")
TASK = "pointer_chase"
DEPTHS = (1, 2, 4, 8)
MODES = ("core", "stack")
BUDGET_WALL_ACC = 0.25   # 低于此 (chance=1/16≈0.06) 视为预算墙
FLAT_SPREAD = 0.05       # 各档均值极差低于此视为"无视迭代"平坦


def main():
    args = _parse_args()
    args.out_dir = args.out_dir or RESULTS_DIR
    os.makedirs(args.out_dir, exist_ok=True)
    args.device = _resolve_device(args.device)
    if args.probe:
        _probe_timing(args)
        return
    todo = pending_configs(args)
    print(f"[plan] device={args.device} steps={args.steps} "
          f"待跑 {len(todo)} 配置 (已完成 "
          f"{len(MODES) * len(DEPTHS) * len(args.seeds) - len(todo)})",
          flush=True)
    for mode, depth, seed in todo:
        run_config(args, mode, depth, seed)
    write_verdict(args)


def _probe_timing(args):
    """每配置跑 probe_steps 步测 s/step，打印全协议 ETA 表 (选型用)。"""
    print(f"probe: {args.probe_steps} 步/配置, device={args.device}")
    protocol_label = f"全协议(x{len(args.seeds)} seeds)"
    print(f"{'config':<12} {'s/step':>8} {'单配置30k':>12} {protocol_label:>16}")
    sweep = 0.0
    for mode, depth in itertools.product(MODES, DEPTHS):
        sps = _measure_sps(args, mode, depth)
        sweep += sps
        print(f"{mode + '_d' + str(depth):<12} {sps:8.3f} "
              f"{sps * 30000 / 60:10.1f}m {sweep * 30000 * len(args.seeds) / 3600:14.1f}h")
    total = sweep * args.steps * len(args.seeds)
    print(f"\n全协议估计: {len(MODES)}x{len(DEPTHS)}x{len(args.seeds)} 配置 "
          f"× {args.steps} 步 ≈ {total / 3600:.1f}h (不含评估/IO)")


def _measure_sps(args, mode, depth):
    """单配置 s/step (build 开销摊到 probe_steps 上, 步数够则误差 <5%)。"""
    model, mix = _build_for(args, mode, depth, seed=0)
    _set_depth(model, mode, depth)
    t0 = time.time()
    train_model(model, mix, args.device, args.probe_steps, args.batch,
                args.lr, seed=0, depth_choices=[depth], depth_setter=mode,
                log_every=args.probe_steps + 1)
    return (time.time() - t0) / args.probe_steps


def run_config(args, mode, depth, seed):
    """训练+评估一个配置: tee 日志 → 原子落盘。中断则重启重跑本配置。"""
    t0, tag = time.time(), f"{mode}_d{depth}_s{seed}"
    log = _open_log(args, tag)
    model, mix = _build_for(args, mode, depth, seed)
    _set_depth(model, mode, depth)
    _log(log, tag, f"start steps={args.steps} device={args.device} "
                   f"params={model.get_num_params()} lr={args.lr}")
    with contextlib.redirect_stdout(Tee(sys.stdout, log)):
        train_model(model, mix, args.device, args.steps, args.batch,
                    args.lr, seed, depth_choices=[depth], depth_setter=mode,
                    log_every=args.log_every)
    acc = evaluate_per_k(model, TASK, args.difficulty, args.n_values,
                         np.random.default_rng(10_000 + seed), args.device)
    row = _make_row(args, mode, depth, seed, acc, time.time() - t0)
    save_row(row_path(args, mode, depth, seed), row)
    _log(log, tag, f"done mean_acc={row['mean_acc']:.4f} "
                   f"wall_s={row['wall_s']} s/step={row['s_per_step']}")
    log.close()


def _build_for(args, mode, depth, seed):
    """固定深度训练的模型 + mix 生成器 (与 run_fixed_sweep 同构的建法)。"""
    gen, vocab, _ = make_generator(TASK, args.difficulty, args.n_values, seed=0)
    probe = gen(1, np.random.default_rng(0))
    # core 需在构建时带门控参数 (max_depth=2), stack 无参数 (max_depth=1) —
    # 与 reasoning_depth.run_fixed_sweep 完全一致的位等价建法。
    model = build_mtlnn(vocab, probe.tokens.shape[1],
                        2 if mode == "core" else 1, seed)
    mix = make_mix_generator(TASK, args.difficulty, args.n_values)
    return model, mix


def _make_row(args, mode, depth, seed, acc, wall_s):
    return {
        "task": TASK, "protocol": "latent_recursion_task2",
        "mode": mode, "depth": depth, "seed": seed,
        "steps": args.steps, "batch": args.batch, "lr": args.lr,
        "device": args.device, "difficulty": args.difficulty,
        "acc_by_k": {str(k): round(float(v), 4) for k, v in acc.items()},
        "mean_acc": float(np.mean(list(acc.values()))),
        "wall_s": round(wall_s, 1),
        "s_per_step": round(wall_s / max(args.steps, 1), 4),
        "git_rev": _git_rev(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_verdict(args):
    rows = _load_rows(args)
    if not rows:
        print("[verdict] 无已完成配置, 跳过判决")
        return
    verdict = judge_decision(rows, DEPTHS, MODES)
    path = os.path.join(args.out_dir, "latent_recursion_verdict.json")
    save_row(path, verdict)
    print(f"\n[verdict] h_supported={verdict['h_supported']} "
          f"diagnosis={verdict['diagnosis']['kind']} → {path}")
    for m in MODES:
        for d in DEPTHS:
            t = verdict["tiers"][m][str(d)]
            print(f"  {m:5s} d{d}: {t['mean']} ± {t['std']} (n={t['n']})")


def judge_decision(rows, depths=DEPTHS, modes=MODES, alpha=0.05):
    """预注册判决 (纯函数, 可单测)。标准见模块 docstring。"""
    tiers = {m: _tier_stats(rows, m, depths) for m in modes}
    # stack 是裁决对象 (红线: core 只是阴性对照) — 无 stack 臂时 (如
    # core 单模式进程) 不得产 headline verdict, 2026-08-31 A100 事故
    # (KeyError: 'stack', verdict 崩成 n_rows=1 残留) 的根因。
    has_stack = "stack" in tiers and \
        any(t["n"] for t in tiers["stack"].values())
    if not has_stack:
        gate = _gate_headline(rows, depths, alpha)
        return {
            "h_supported": False,
            "h_components": {"monotonic": None,
                             "gain_d_last_minus_d1": None,
                             "sigma_d_last": None,
                             "threshold_2sigma": None},
            "tiers": tiers, "gate": gate,
            "diagnosis": {"kind": "no_stack_arms",
                          "note": "无 stack 臂 — 裁决对象缺席, "
                                  "core-only 运行不构成判决",
                          "stack_means": []},
            "n_rows": len(rows), "depths": list(depths),
        }
    means = [tiers["stack"][str(d)]["mean"] for d in depths]
    sigma = tiers["stack"][str(depths[-1])]["std"] or 0.0
    monotonic = all(x is not None for x in means) and \
        all(means[i] < means[i + 1] for i in range(len(means) - 1))
    gate = _gate_headline(rows, depths, alpha)
    h_supported = bool(monotonic and means[-1] - means[0] >= 2.0 * sigma
                       and gate["ok"])
    return {
        "h_supported": h_supported,
        "h_components": {"monotonic": monotonic,
                         "gain_d_last_minus_d1": round(means[-1] - means[0], 4),
                         "sigma_d_last": sigma,
                         "threshold_2sigma": round(2.0 * sigma, 4)},
        "tiers": tiers, "gate": gate,
        "diagnosis": _diagnose(tiers, depths, h_supported, gate),
        "n_rows": len(rows), "depths": list(depths),
    }


def _tier_stats(rows, mode, depths):
    """每深度档: per-seed headline 的 mean/std(ddof=1)/双峰标记。"""
    out = {}
    for d in depths:
        vals = _accs(rows, mode, d)
        out[str(d)] = {
            "n": len(vals), "per_seed": [round(v, 4) for v in vals],
            "mean": round(float(np.mean(vals)), 4) if vals else None,
            "std": (round(float(np.std(vals, ddof=1)), 4)
                    if len(vals) > 1 else None),
            "bimodal": (bimodality_1d(vals)["bimodal"]
                        if len(vals) >= 4 else None)}
    return out


def _gate_headline(rows, depths, alpha):
    """E0 门槛: stack 最深档 vs 最浅档 publishable + 逐档 core-vs-stack 配对。"""
    hi, lo = depths[-1], depths[0]
    ok, report = publishable(_accs(rows, "stack", hi), _accs(rows, "stack", lo),
                             tag=f"stack_d{hi}_vs_d{lo}", alpha=alpha)
    report["ok"] = ok
    report["core_vs_stack_by_depth"] = {
        str(d): _paired_if_paired(rows, d) for d in depths}
    return report


def _paired_if_paired(rows, d):
    """两臂 seeds 齐才配对 (部分完成的中途判决容忍缺臂)。"""
    a, b = _accs(rows, "stack", d), _accs(rows, "core", d)
    if not a or not b or len(a) != len(b):
        return {"p": None, "wins": 0, "losses": 0, "n_effective": 0}
    return paired_sign_test(a, b)


def _diagnose(tiers, depths, h_supported, gate):
    """判负时的数据诊断: 循环体该改哪里 (顺序预注册)。"""
    if h_supported:
        return {"kind": "H_supported",
                "note": "整块循环买到深度; 对照看 tiers.core 是否平坦"}
    means = [tiers["stack"][str(d)]["mean"] for d in depths]
    reasons = "".join(gate.get("reasons", []))
    if max(means) < BUDGET_WALL_ACC:
        return {"kind": "budget_wall",
                "note": f"全程 <{BUDGET_WALL_ACC} — 训练预算墙, 先加 steps/调课程再谈深度效应",
                "stack_means": means}
    if "双峰" in reasons:
        return {"kind": "bimodal_grokking_zone",
                "note": "双峰掷硬币区 — 加 seeds 重跑, 本轮不可引用",
                "stack_means": means}
    if max(means) - min(means) < FLAT_SPREAD:
        return {"kind": "flat_iterations_ignored",
                "note": "各深度档持平 — 模型学会无视迭代 (§4.5 round-1 退化解); "
                        "下一步: deep_supervision(逐迭代 CE) 或 poisson 深度课程",
                "stack_means": means}
    return {"kind": "direction_but_underpowered",
            "note": "方向上升但未过 2σ/E0 — 加 seeds 或 steps",
            "stack_means": means}


def pending_configs(args):
    """枚举 mode×depth×seed, 跳过"同预算已完成"的配置 (resume 语义)。

    行 JSON 的 steps 必须与本次 --steps 一致才跳过 — 否则低预算验证行
    (如 1000 步 Stage-1) 会让正式 30k 跑静默跳过该配置。"""
    todo = []
    for mode, depth, seed in itertools.product(MODES, DEPTHS, args.seeds):
        r = _load_json(row_path(args, mode, depth, seed))
        if r is not None and r.get("steps") == args.steps:
            print(f"[resume] 跳过已完成 {mode} d{depth} s{seed} "
                  f"mean_acc={r['mean_acc']:.4f}")
            continue
        todo.append((mode, depth, seed))
    return todo


def _load_rows(args):
    """只收同预算 (steps 匹配) 的行 — 判决不得混合不同训练预算。"""
    rows = []
    for mode, depth, seed in itertools.product(MODES, DEPTHS, args.seeds):
        r = _load_json(row_path(args, mode, depth, seed))
        if r is not None and r.get("steps") == args.steps:
            rows.append(r)
    return rows


def row_path(args, mode, depth, seed):
    return os.path.join(args.out_dir,
                        f"latent_recursion_{mode}_d{depth}_s{seed}.json")


def save_row(path, row):
    """原子写 (tmp + os.replace) — 中断不产生半截 JSON。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=1)
    os.replace(tmp, path)


def _accs(rows, mode, d):
    return [r["mean_acc"] for r in rows
            if r["mode"] == mode and r["depth"] == d]


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _set_depth(model, mode, depth):
    if mode == "stack":
        model.set_stack_iterations(depth)
    else:
        model.set_core_iterations(depth)


def _open_log(args, tag):
    path = os.path.join(args.out_dir, f"latent_recursion_{tag}.log")
    return open(path, "a", encoding="utf-8")


def _log(log, tag, msg):
    line = f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}"
    print(line, flush=True)
    print(line, file=log, flush=True)


def _resolve_device(name):
    if name != "auto":
        return name
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _git_rev():
    # 容器/非 git 沙箱里 subprocess 不可用 — 允许启动脚本 export GIT_COMMIT
    # 兜底 (2026-08-31 批次行 JSON git_rev=unknown 的 provenance 缺口)。
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"][:12]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=root,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Tee:
    """同时写 console 与日志文件 (逐段 flush, 训练行不丢)。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--difficulty", type=int, default=8)
    p.add_argument("--n_values", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--device", default="auto",
                   help="auto|cpu|mps|cuda (auto: cuda→mps→cpu)")
    p.add_argument("--out_dir", default=None,
                   help="默认 benchmarks/results/")
    p.add_argument("--probe", action="store_true",
                   help="每配置跑 probe_steps 步测速, 打印 ETA 后退出")
    p.add_argument("--probe_steps", type=int, default=10)
    p.add_argument("--log_every", type=int, default=200)
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
