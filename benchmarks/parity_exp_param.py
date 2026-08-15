"""parity 长度外推：exp 参数化 vs tanh 参数化 对照验证（E5d）。

根因假设（2026-08-15，来自 E5/E5c）：
  分支 consciousness-m1-v2 的 both_khavari 达到 extrap 1.000，main 的
  selective_decay 只有 ~0.2。组件消融（lateral/MAP）已排除。代码对比发现
  参数化根本不同：
    分支:  lam_t = 2*exp(-softplus(delta(x))/tau) - 1   ∈ (-1, 1)
           delta 是线性层输出，softplus 无上界 → 输入可以让 lam 精确逼近
           +1（delta→0）或 -1（delta→∞），flip/hold 二值语义可精确实现。
    main:  lam_t = decay * tanh(W_sel·x + b_sel)          ∈ (-decay, decay)
           tanh 饱和需要 |W·x+b|>>3，饱和区梯度消失；decay<1 使 |lam|<1，
           长序列上每步泄漏，误差累积破坏外推。

本脚本复刻分支参数化的最小层（exp 式），与 main 的 tanh 式同场对比。
运行: py -3.11 benchmarks/parity_exp_param.py --steps 4000 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt_lnn.config import MTLNNConfig
from mt_lnn.mt_lnn_layer import MTLNNLayer


class ExpParityProbe(nn.Module):
    """分支式参数化的最小液核：h_t = lam_t·h_{t-1} + (1-decay)·A_t,
    lam_t = 2·exp(-softplus(Wd·x_t + bd)/tau) - 1 —— 输入相关、负特征值、
    指数参数化。单 proto、单 scale 维度最小化，与分支公式逐项对应。"""

    def __init__(self, d_model, tau_max=200.0):
        super().__init__()
        self.d_model = d_model
        self.tau_max = tau_max
        self.embed = nn.Embedding(2, d_model)
        # 输入门 A_t = sigmoid(W_in x + b_in)
        self.W_in = nn.Parameter(torch.empty(d_model, d_model))
        self.b_in = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.W_in, std=0.02)
        # tau: 固定几何多尺度（不学习，专注参数化对比）
        self.tau = torch.tensor([1.0, 4.0, 16.0, 64.0, tau_max])
        self.S = self.tau.numel()
        # 输入相关 delta 投影：每 scale 一个线性读数
        self.Wd = nn.Parameter(torch.empty(d_model, self.S))
        self.bd = nn.Parameter(torch.zeros(self.S))
        nn.init.normal_(self.Wd, std=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        h = self.embed(x)                     # (B,T,D)
        B, T, D = h.shape
        tau = self.tau.to(h.device)
        decay = torch.exp(-1.0 / tau)         # (S,) 固定
        states = []
        state = torch.zeros(B, self.S, D, device=h.device)
        for t in range(T):
            xt = h[:, t]                      # (B,D)
            # 分支式 delta：delta_s = softplus(Wd·x + bd)_s
            delta = F.softplus(xt @ self.Wd + self.bd)      # (B,S)
            # lam_t = 2·exp(-delta/tau) - 1 ∈ (-1,1)
            lam = 2.0 * torch.exp(-delta / tau.view(1, self.S)) - 1.0  # (B,S)
            A = torch.sigmoid(xt @ self.W_in + self.b_in)   # (B,D)
            # 逐 scale 更新：h_s = lam_s·h_s + (1-decay_s)·A
            state = state * lam.unsqueeze(-1) + \
                (1.0 - decay).view(1, self.S, 1) * A.unsqueeze(1)
            # 各 scale 的 max-能量混合（分支 blend 的极端简化：取 lam 最大者）
            mix_w = F.softmax(lam.abs(), dim=-1)            # (B,S)
            states.append((state * mix_w.unsqueeze(-1)).sum(dim=1))
        out = torch.stack(states, dim=1)                    # (B,T,D)
        return self.head(self.norm(out))


class TanhParityProbe(nn.Module):
    """main 式参数化对照：lam_t = decay·tanh(W_sel·x + b_sel)（同样最小化）。"""

    def __init__(self, d_model, tau_max=200.0):
        super().__init__()
        self.d_model = d_model
        self.tau_max = tau_max
        self.embed = nn.Embedding(2, d_model)
        self.W_in = nn.Parameter(torch.empty(d_model, d_model))
        self.b_in = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.W_in, std=0.02)
        self.tau = torch.tensor([1.0, 4.0, 16.0, 64.0, tau_max])
        self.S = self.tau.numel()
        self.sel_w = nn.Parameter(torch.empty(d_model, self.S))
        self.sel_b = nn.Parameter(torch.ones(self.S))
        nn.init.normal_(self.sel_w, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        h = self.embed(x)
        B, T, D = h.shape
        tau = self.tau.to(h.device)
        decay = torch.exp(-1.0 / tau)
        states = []
        state = torch.zeros(B, self.S, D, device=h.device)
        for t in range(T):
            xt = h[:, t]
            sel = xt @ self.sel_w + self.sel_b
            lam = decay.view(1, self.S) * torch.tanh(sel)   # main 式
            A = torch.sigmoid(xt @ self.W_in + self.b_in)
            state = state * lam.unsqueeze(-1) + \
                (1.0 - decay).view(1, self.S, 1) * A.unsqueeze(1)
            mix_w = F.softmax(lam.abs(), dim=-1)
            states.append((state * mix_w.unsqueeze(-1)).sum(dim=1))
        out = torch.stack(states, dim=1)
        return self.head(self.norm(out))


def make_batch(bs, lo, hi, device, gen):
    t = int(torch.randint(lo, hi + 1, (1,), generator=gen).item())
    x = torch.randint(0, 2, (bs, t), generator=gen)
    y = torch.cumsum(x, dim=1) % 2
    return x.to(device), y.to(device)


def run_one(model, args, seed, device):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for _ in range(args.steps):
        x, y = make_batch(args.batch, 1, args.train_len, device, gen)
        loss = F.cross_entropy(model(x).reshape(-1, 2), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    accs = {}
    with torch.no_grad():
        for name, (lo, hi) in {
            "in_dist": (1, args.train_len),
            "extrapolate": (args.train_len + 1, args.test_len),
        }.items():
            correct = total = 0
            for _ in range(args.eval_batches):
                x, y = make_batch(args.batch, lo, hi, device, gen)
                pred = model(x).argmax(-1)
                correct += (pred == y).all(dim=1).sum().item()
                total += x.shape[0]
            accs[name] = correct / total
    return accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_len", type=int, default=24)
    p.add_argument("--test_len", type=int, default=64)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--tau_max", type=float, default=200.0)
    p.add_argument("--eval_batches", type=int, default=8)
    p.add_argument("--out", type=str,
                   default="benchmarks/results/parity_exp_param.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  train<=({args.train_len})  "
          f"test=({args.train_len + 1}..{args.test_len})")
    print("per-sequence chance ~ 0.0\n")

    ARMS = [("exp_param", lambda: ExpParityProbe(args.d_model, args.tau_max)),
            ("tanh_param", lambda: TanhParityProbe(args.d_model, args.tau_max))]
    results = {}
    for arm, factory in ARMS:
        runs = []
        for s in range(args.seeds):
            model = factory().to(device)
            t0 = time.time()
            r = run_one(model, args, s, device)
            r["seed"] = s
            r["secs"] = round(time.time() - t0, 1)
            runs.append(r)
            print(f"[{arm}] seed={s} in_dist={r['in_dist']:.3f} "
                  f"extrap={r['extrapolate']:.3f} ({r['secs']}s)")
        mean = lambda k: sum(x[k] for x in runs) / len(runs)
        results[arm] = {"runs": runs,
                        "in_dist_mean": mean("in_dist"),
                        "extrapolate_mean": mean("extrapolate")}
        print(f"[{arm}] MEAN in_dist={mean('in_dist'):.3f} "
              f"extrap={mean('extrapolate'):.3f}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"wrote {out}")

    print("\n=== VERDICT (parameterisation A/B, full-sequence extrap) ===")
    for arm, _ in ARMS:
        r = results[arm]
        print(f"  {arm:12s} in_dist={r['in_dist_mean']:.3f}  "
              f"extrap={r['extrapolate_mean']:.3f}")
    exp = results["exp_param"]["extrapolate_mean"]
    tanh = results["tanh_param"]["extrapolate_mean"]
    if exp > tanh + 0.2:
        print("  -> EXP PARAMETERISATION WINS: the tanh saturation is the "
              "extrapolation blocker (branch vs main gap explained)")
    elif tanh > exp + 0.2:
        print("  -> UNEXPECTED: tanh wins - hypothesis falsified, re-derive")
    else:
        print("  -> NO DECISIVE DIFFERENCE at this budget - inconclusive")


if __name__ == "__main__":
    main()
