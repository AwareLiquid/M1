"""Task 3 — Anytime 前沿: 潜空间深度 N vs CoT-token 基线 (FLOPs 对齐)。

同一任务 (pointer_chase 单环 mix 课程, difficulty=8, n=16) 两条路:
  a) 潜空间: MT-LNN stack 深度 mix 训练 (poisson 采样, 反制"学会无视
     迭代"的均匀随机深度 — §4.5 round-1 教训), 同一权重在深度
     {1,2,4,8} 评估 → anytime 曲线 (可提前停)。
  b) CoT 基线: ModernCausalTransformer, 确定性中间符号 (口径写死于
     reasoning_tasks.gen_pointer_chase_cot): 粒度 g 的链在 g,2g,...,k
     跳处显式落节点, 链尾即答案; 每个 g 单独训一个模型, 贪心自回归
     评估 (错一步链全错 — 对 CoT 诚实的口径)。
横轴 = Task 1 账本 FLOPs (口径冻结, benchmarks/compute_accounting.py):
  潜空间 N → account_mtlnn(T=54, stack, N); CoT g → E_k[account_tfm
  (T=53, cot, m(k)-1)], m(k)=链长。纵轴 = mean_k accuracy (mix 口径)。

预注册读数规则 (dominance, 不事后移动): 潜空间点 (N) 被压制 ⇔ 存在
CoT 点 flops ≤ 且 seed 均值 acc ≥; 全部潜空间点被压制 → 如实记
"anytime 前沿被 CoT 基线压制" (负结果同样是结果)。

Resume-safe: 每配置原子写
  results/anytime_{cot_g{g}|latent}_s{seed}.json (steps 匹配才跳过)。
判决级预算需 GPU (30k 步); 本地 MPS 只做短步验证 (--probe 测速)。

用法:
  py benchmarks/anytime_frontier.py --probe --device mps
  py benchmarks/anytime_frontier.py --steps 1000 --seeds 0   # 验证跑
  py benchmarks/anytime_frontier.py                          # 30k × 6 seeds
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.compute_accounting import PROBE, account_mtlnn, account_tfm
from benchmarks.latent_recursion import Tee, save_row, _load_json, _log
from benchmarks.reasoning_depth import (build_mtlnn, build_transformer,
                                        evaluate_per_k, make_mix_generator,
                                        train_model)
from benchmarks.reasoning_tasks import (THINK, gen_pointer_chase_cot,
                                        make_generator, vocab_size)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")
TASK = "pointer_chase"
DIFFICULTY = 8
N_VALUES = 16
GRANULARITIES = (1000, 4, 2, 1)        # 1000 = ∞ (直接作答); g=8 在
                                       # difficulty=8 下与 ∞ 同链长 (m=1), 删
LATENT_DEPTHS = (1, 2, 4, 8)
PROMPT_T_LATENT = 54                 # [BOS] edges START s k THINK ANS
PROMPT_T_COT = 53                    # [BOS] edges START s k THINK (链为生成)


def main():
    args = _parse_args()
    args.out_dir = args.out_dir or RESULTS_DIR
    os.makedirs(args.out_dir, exist_ok=True)
    if args.probe:
        _probe_timing(args)
        return
    if args.frontier_only:
        write_frontier(args)
        return
    for seed in args.seeds:
        run_latent_config(args, seed)
        for g in GRANULARITIES:
            run_cot_config(args, g, seed)
    write_frontier(args)


def run_latent_config(args, seed):
    """潜空间臂: 一份权重, 深度 mix 训练, 各深度 per-k 评估 (anytime)。"""
    path = _row_path(args, "latent", seed)
    if _done(args, path):
        return
    t0, tag = time.time(), f"latent_s{seed}"
    log = open(os.path.join(args.out_dir, f"anytime_{tag}.log"), "a",
               encoding="utf-8")
    model = _build_latent(args, seed)
    mix = make_mix_generator(TASK, DIFFICULTY, N_VALUES)
    _log(log, tag, f"start poisson depth-mix steps={args.steps} "
                   f"params={model.get_num_params()}")
    with contextlib.redirect_stdout(Tee(sys.stdout, log)):
        train_model(model, mix, args.device, args.steps, args.batch, args.lr,
                    seed, depth_choices=list(LATENT_DEPTHS),
                    depth_setter="stack", depth_sampler="poisson",
                    poisson_lambda=3.0, log_every=args.log_every)
    acc_by_depth = {}
    for n in LATENT_DEPTHS:
        model.set_stack_iterations(n)
        acc_by_depth[n] = evaluate_per_k(model, TASK, DIFFICULTY, N_VALUES,
                                         np.random.default_rng(20_000 + seed),
                                         args.device)
        _log(log, tag, f"depth {n}: mean_k="
                       f"{np.mean(list(acc_by_depth[n].values())):.4f}")
    row = {"arm": "latent", "seed": seed, "steps": args.steps,
           "acc_by_depth": {str(n): _round(acc) for n, acc in
                            acc_by_depth.items()},
           "wall_s": round(time.time() - t0, 1), "git_rev": _git_rev(),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_row(path, row)
    _log(log, tag, f"done wall_s={row['wall_s']}")
    log.close()


def run_cot_config(args, g, seed):
    """CoT 臂: 粒度 g 的确定性中间符号链, teacher-forced 训练, 自回归评估。"""
    path = _row_path(args, f"cot_g{g}", seed)
    if _done(args, path):
        return
    t0, tag = time.time(), f"cot_g{g}_s{seed}"
    log = open(os.path.join(args.out_dir, f"anytime_{tag}.log"), "a",
               encoding="utf-8")
    model = _build_cot(args, seed)
    _log(log, tag, f"start granularity={g} steps={args.steps} "
                   f"params={model.get_num_params()}")
    train_cot_model(model, g, args, tag, log)
    acc = eval_cot_arm(model, g, args, seed)
    row = {"arm": "cot", "granularity": g, "seed": seed, "steps": args.steps,
           "acc_by_k": _round(acc),
           "mean_acc": round(float(np.mean(list(acc.values()))), 4),
           "wall_s": round(time.time() - t0, 1), "git_rev": _git_rev(),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_row(path, row)
    _log(log, tag, f"done mean_acc={row['mean_acc']} wall_s={row['wall_s']}")
    log.close()


def train_cot_model(model, g, args, tag, log):
    """CoT 训练循环 — 镜像 reasoning_depth.train_model 的优化器卫生
    (AdamW β2=0.95, cosine, clip 1.0), 唯一差异: 标签覆盖整条链而非仅答案位。"""
    model.to(args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = np.random.default_rng(args.seed_of(tag))
    with contextlib.redirect_stdout(Tee(sys.stdout, log)):
        for step in range(args.steps):
            ids, labels = make_cot_batch(g, args.batch, rng, args.device)
            out = model(ids, labels=labels)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % args.log_every == 0 or step == args.steps - 1:
                print(f"    step {step:5d}  loss {out['loss'].item():.4f}",
                      flush=True)


def make_cot_batch(g, batch, rng, device):
    """mix 口径 CoT 批: k~U{1..D}, 标签 = 链区所有位置 (-100 其余)。"""
    k = int(rng.integers(1, DIFFICULTY + 1))
    b = gen_pointer_chase_cot(batch, N_VALUES, k, g, rng)
    ids = torch.from_numpy(b.tokens).to(device)
    labels = torch.full_like(ids, -100)
    labels[:, -len(_cot_marks(k, g)):] = ids[:, -len(_cot_marks(k, g)):]
    return ids, labels


@torch.no_grad()
def eval_cot_arm(model, g, args, seed):
    """贪心自回归: 逐 token 生成整条链, 链尾判分 (错一步链全错)。"""
    model.eval()
    accs = {}
    for k in range(1, DIFFICULTY + 1):
        rng = np.random.default_rng(30_000 + 100 * seed + k)
        b = gen_pointer_chase_cot(args.eval_batch, N_VALUES, k, g, rng)
        ids = torch.from_numpy(b.tokens[:, :b.ans_pos]).to(args.device)
        for _ in _cot_marks(k, g):
            nxt = model(ids)["logits"][:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        correct = (ids[:, -1].cpu().numpy() == b.answer).sum().item()
        accs[k] = correct / args.eval_batch
    model.train()
    return accs


# ── 前沿装配 ─────────────────────────────────────────────────────────────────

def write_frontier(args):
    rows = _load_all(args)
    if not rows:
        print("[frontier] 无已完成行")
        return
    points = frontier_points(rows)
    dominated = {p["label"]: _dominated(p, points) for p in points
                 if p["arm"] == "latent"}
    payload = {"points": points, "latent_points_dominated": dominated,
               "preregistered_rule": "潜空间点被压制 ⇔ ∃CoT点 flops≤ 且 "
                                     "seed均值acc≥; 全部被压制 → 前沿被CoT压制",
               "steps": args.steps}
    path = os.path.join(args.out_dir, "anytime_frontier.json")
    save_row(path, payload)
    print(_markdown(points, dominated))


def frontier_points(rows):
    """行 → 前沿点: (arm, knob) → {mflops, acc 均值±std, per-k}。"""
    sh = dict(PROBE, T=PROMPT_T_COT, vocab=vocab_size(N_VALUES))
    pts = []
    for n in LATENT_DEPTHS:
        accs = [_mean_k(r, n) for r in rows
                if r["arm"] == "latent" and str(n) in r["acc_by_depth"]]
        if accs:
            pts.append(_point("latent", f"latent_N{n}", sh,
                              account_mtlnn(_t(sh, PROMPT_T_LATENT), n,
                                            "stack")["flops"], accs))
    for g in GRANULARITIES:
        accs = [r["mean_acc"] for r in rows
                if r["arm"] == "cot" and r["granularity"] == g]
        if accs:
            pts.append(_point("cot", f"cot_g{'inf' if g >= 1000 else g}", sh,
                              cot_mean_flops(sh, g), accs))
    return sorted(pts, key=lambda p: p["mflops"])


def cot_mean_flops(sh, g):
    """CoT g 的期望 FLOPs: E_k~U{1..D}[account_tfm(T=53, N=m(k)-1)]。"""
    total = 0
    for k in range(1, DIFFICULTY + 1):
        m = len(_cot_marks(k, g))
        total += account_tfm(sh, m - 1, "cot")["flops"]
    return total / DIFFICULTY


def _dominated(p, points):
    """预注册压制规则 (seed 均值, 无平局容忍 — 平局=压制, 对潜空间从严)。"""
    return any(q["arm"] == "cot" and q["mflops"] <= p["mflops"]
               and q["acc_mean"] >= p["acc_mean"] for q in points)


def _markdown(points, dominated):
    lines = ["| point | MFLOPs | acc (mean±std, seeds) | dominated |",
             "|---|---|---|---|"]
    for p in points:
        dom = dominated.get(p["label"], "")
        lines.append(f"| {p['label']} | {p['mflops']:.2f} | "
                     f"{p['acc_mean']:.3f} ± {p['acc_std']:.3f} (n={p['n']})"
                     f" | {dom} |")
    return "\n".join(lines)


def _point(arm, label, sh, flops, accs):
    return {"arm": arm, "label": label, "mflops": round(flops / 1e6, 3),
            "acc_mean": round(float(np.mean(accs)), 4),
            "acc_std": round(float(np.std(accs, ddof=1)) if len(accs) > 1
                             else 0.0, 4), "n": len(accs)}


def _mean_k(r, n):
    return float(np.mean(list(r["acc_by_depth"][str(n)].values())))


# ── 构建/杂务 ────────────────────────────────────────────────────────────────

def _build_latent(args, seed):
    gen, vocab, _ = make_generator(TASK, DIFFICULTY, N_VALUES, seed=0)
    T = gen(1, np.random.default_rng(0)).tokens.shape[1]
    return build_mtlnn(vocab, T, 1, seed)


def _build_cot(args, seed):
    vocab = vocab_size(N_VALUES)
    return build_transformer(vocab, PROMPT_T_COT + DIFFICULTY, seed)


def _probe_timing(args):
    """两臂各测 probe_steps 步 → 全协议 ETA。"""
    args.device = args.device if args.device != "auto" else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    model = _build_cot(args, 0)
    train_cot_model(model, 1, _probe_ns(args), "probe", _devnull())
    cot_sps = (time.time() - t0) / args.probe_steps
    t0 = time.time()
    model = _build_latent(args, 0)
    mix = make_mix_generator(TASK, DIFFICULTY, N_VALUES)
    train_model(model, mix, args.device, args.probe_steps, args.batch,
                args.lr, 0, depth_choices=list(LATENT_DEPTHS),
                depth_setter="stack", depth_sampler="poisson",
                log_every=args.probe_steps + 1)
    lat_sps = (time.time() - t0) / args.probe_steps
    per_seed = (cot_sps * len(GRANULARITIES) + lat_sps) * args.steps
    print(f"cot {cot_sps:.3f} s/step (×{len(GRANULARITIES)} g) + "
          f"latent {lat_sps:.3f} s/step")
    print(f"每 seed ≈ {per_seed / 3600:.1f}h × {len(args.seeds)} seeds "
          f"≈ {per_seed * len(args.seeds) / 3600:.1f}h (不含评估)")


def _probe_ns(args):
    return argparse.Namespace(steps=args.probe_steps, batch=args.batch,
                              lr=args.lr, device=args.device, seed_of=lambda
                              t: 0, log_every=args.probe_steps + 1)


def _cot_marks(k, g):
    from benchmarks.reasoning_tasks import _cot_hop_marks
    return _cot_hop_marks(k, g)


def _done(args, path):
    r = _load_json(path)
    if r is not None and r.get("steps") == args.steps:
        print(f"[resume] 跳过 {os.path.basename(path)}")
        return True
    return False


def _row_path(args, arm_knob, seed):
    return os.path.join(args.out_dir, f"anytime_{arm_knob}_s{seed}.json")


def _load_all(args):
    rows = []
    for seed in args.seeds:
        for arm, knob in ([("latent", "latent")] +
                          [("cot", f"cot_g{g}") for g in GRANULARITIES]):
            r = _load_json(_row_path(args, knob, seed))
            if r is not None and r.get("steps") == args.steps:
                rows.append(r)
    return rows


def _round(acc):
    return {str(k): round(float(v), 4) for k, v in acc.items()}


def _t(sh, T):
    return dict(sh, T=T)


def _git_rev():
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"][:12]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=root,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _devnull():
    return open(os.devnull, "w")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval_batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--frontier_only", action="store_true",
                   help="跳过训练, 只从已完成行重建 anytime_frontier.json"
                        " (12h 会话在最后一臂被杀时的兜底)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--probe_steps", type=int, default=10)
    p.add_argument("--log_every", type=int, default=200)
    args = p.parse_args(argv)
    args.seed_of = lambda tag: int(tag.split("_s")[-1]) if "_s" in tag else 0
    if args.device == "auto":
        args.device = ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    return args


if __name__ == "__main__":
    main()
