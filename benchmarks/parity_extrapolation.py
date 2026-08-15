"""parity 长度外推协议（E5 铺垫）——隔离液核，验证 selective_decay 的长度外推。

为什么存在
----------
main 的 reasoning_depth.py 是固定 seq_len 训练、无外推评估。本脚本复刻
consciousness-m1-v2 分支的 state_tracking_parity.py 协议（Delétang et al.
ICLR 2023 风格）到 main 的 MTLNNLayer：
  - 训练 U(1, N_train)，评估 U(N_train+1, N_test)，full-sequence acc
  - 3 cell 消融：legacy（都关）/ signed（常数负特征值）/ selective（输入相关+负）
  - 特征值诊断：验证负特征值机制真的激活（而非准确率噪声）

理论依据（Khavari et al. arXiv 2508.07395）：parity 需要输入相关 AND 负特征值
两者缺一不可。legacy 对角正特征值做不了 parity（Sarrof Thm 2）；signed 只给
(-1)^t 位置奇偶（非比特奇偶）；selective（λ_t = decay·tanh(W_sel·x_t+b_sel)）
两者合体，应做到长度外推。

运行：py -3.11 benchmarks/parity_extrapolation.py --steps 4000 --seeds 3
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


class ParityProbe(nn.Module):
    """embedding -> MTLNNLayer(s) -> LayerNorm -> linear readout。隔离液核。"""

    def __init__(self, cfg: MTLNNConfig, n_layers: int = 1):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(2, cfg.d_model)
        self.layers = nn.ModuleList([MTLNNLayer(cfg) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 2)

    def forward(self, x):                       # x: (B,T) int64 in {0,1}
        h = self.embed(x)                        # (B,T,d_model)
        for layer in self.layers:
            # MTLNNLayer.forward(x:(B,T,d_model)) -> (out:(B,T,d_model), h_last)
            out, _ = layer(h, None, use_scan=True)
            h = h + out
        return self.head(self.norm(h))           # (B,T,2)


def make_batch(bs, lo, hi, device, gen):
    """随机比特串；label_t = x[:t+1] 的 parity（running XOR）。"""
    t = int(torch.randint(lo, hi + 1, (1,), generator=gen).item())
    x = torch.randint(0, 2, (bs, t), generator=gen)
    y = torch.cumsum(x, dim=1) % 2
    return x.to(device), y.to(device)


def _read_eig_range(model, args, device, gen):
    """读训练后液核的特征值范围（验证负特征值机制是否激活）。"""
    res = model.layers[0].resonance            # VectorizedMultiScaleResonance
    tau = F.softplus(res.log_tau) + res.tau_min
    tau = tau.clamp(res.tau_min, res.tau_max)
    decay = torch.exp(-res.dt / tau)            # (P,S) ∈ (0,1)
    with torch.no_grad():
        if res.sel_w is not None:
            # selective：在真实数据上采样 realised λ_t = decay·tanh(W_sel·x+b)
            x, _ = make_batch(args.batch, 1, args.train_len, device, gen)
            B, T = x.shape
            layer = model.layers[0]
            x_proto = layer.in_proj(model.embed(x)).view(B, T, layer.n_proto, -1)
            sel = torch.einsum("btpd,pkd->btpk", x_proto, res.sel_w)
            lam_t = decay.view(1, 1, res.P, res.S) * torch.tanh(sel + res.sel_b)
            return float(lam_t.min()), float(lam_t.max())
        if res.decay_sign_raw is not None:
            lam = decay * torch.tanh(res.decay_sign_raw)
            return float(lam.min()), float(lam.max())
        return float(decay.min()), float(decay.max())


def run_one(signed: bool, selective: bool, seed: int, args, device):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    # d_model 必须是 n_protofilaments * (8 的倍数) —— config 校验 d_proto % 8 == 0
    cfg = MTLNNConfig(
        vocab_size=2, d_model=args.d_model, n_layers=1, n_heads=4, n_kv_heads=2,
        d_head=args.d_model // 4, max_seq_len=args.test_len + 8,
        n_protofilaments=args.n_proto, n_time_scales=args.n_scales,
        signed_decay=signed, selective_decay=selective,
        # 特征值幅值须能逼近 1，否则 parity 在序列结束前衰减掉：
        # |decay| = exp(-dt/tau)，默认 tau_max=10 封顶 0.905，0.905^64 ≈ 0.0016
        tau_max=args.tau_max,
        # E5c component ablation switches
        use_lateral_coupling=not args.no_lateral,
        use_map_gate=not args.no_map_gate,
        # E5e transition parameterisation
        selective_decay_mode=args.sel_mode,
    )
    model = ParityProbe(cfg, n_layers=args.probe_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(args.steps):
        x, y = make_batch(args.batch, 1, args.train_len, device, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1))
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
                correct += (pred == y).all(dim=1).sum().item()  # FULL-sequence
                total += x.shape[0]
            accs[name] = correct / total
    eig_min, eig_max = _read_eig_range(model, args, device, gen)
    accs["eig_min"] = eig_min
    accs["eig_max"] = eig_max
    accs["final_loss"] = float(loss)
    return accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_len", type=int, default=24)
    p.add_argument("--test_len", type=int, default=64)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--n_proto", type=int, default=4)
    p.add_argument("--n_scales", type=int, default=4)
    p.add_argument("--probe_layers", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--tau_max", type=float, default=200.0)
    p.add_argument("--eval_batches", type=int, default=8)
    p.add_argument("--arms", nargs="+", default=None,
                   help="subset of legacy/signed_only/selective (default: all)")
    p.add_argument("--no-lateral", action="store_true",
                   help="disable LateralCoupling (E5c component ablation)")
    p.add_argument("--no-map-gate", action="store_true",
                   help="disable MAPGate (E5c component ablation)")
    p.add_argument("--sel-mode", choices=["tanh", "exp"], default="tanh",
                   help="selective transition parameterisation (E5e); "
                        "exp = 2*exp(-softplus(Wx+b)/tau)-1")
    p.add_argument("--out", type=str,
                   default="benchmarks/results/parity_extrapolation.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  train<=({args.train_len})  "
          f"test=({args.train_len + 1}..{args.test_len})")
    print("random baseline (full-sequence acc) ~ 0.0\n")

    # 3 cell 消融。Khavari：parity 需输入相关 + 负特征值同时；signed 单独不足。
    ARMS = [
        ("legacy",        False, False),
        ("signed_only",   True,  False),
        ("selective",     False, True),   # selective 含 signed（superset）
    ]
    if args.arms is not None:
        ARMS = [a for a in ARMS if a[0] in args.arms]
    if not ARMS:
        raise SystemExit(f"--arms {args.arms} matched nothing; "
                         f"choose from legacy/signed_only/selective")
    results = {}
    for arm, signed, selective in ARMS:
        runs = []
        for s in range(args.seeds):
            t0 = time.time()
            r = run_one(signed, selective, s, args, device)
            r["seed"] = s
            r["secs"] = round(time.time() - t0, 1)
            runs.append(r)
            print(f"[{arm}] seed={s} in_dist={r['in_dist']:.3f} "
                  f"extrap={r['extrapolate']:.3f} "
                  f"eig=[{r['eig_min']:+.3f},{r['eig_max']:+.3f}] ({r['secs']}s)")
        mean = lambda k: sum(x[k] for x in runs) / len(runs)
        results[arm] = {
            "runs": runs,
            "in_dist_mean": mean("in_dist"),
            "extrapolate_mean": mean("extrapolate"),
            "signed": signed, "selective": selective,
        }
        print(f"[{arm}] MEAN in_dist={mean('in_dist'):.3f} "
              f"extrapolate={mean('extrapolate'):.3f}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"wrote {out}")

    print("\n=== VERDICT (length extrapolation, full-sequence acc) ===")
    for arm, _, _ in ARMS:
        r = results[arm]
        print(f"  {arm:14s} in_dist={r['in_dist_mean']:.3f}  "
              f"extrap={r['extrapolate_mean']:.3f}")
    sel = results["selective"]["extrapolate_mean"]
    other_arms = [a for a, _, _ in ARMS if a != "selective"]
    if other_arms:
        others = max(results[a]["extrapolate_mean"] for a in other_arms)
    else:
        others = 0.0
    if sel > others + 0.1:
        print("  -> CONSISTENT: only selective (input-dep + negative eig) generalises")
    elif sel <= 0.05 and others <= 0.05:
        print("  -> ALL CELLS FAIL: eigenvalue story not the binding constraint "
              "(or probe/budget too weak). Check eig ranges - if selective's eig "
              "never went negative, the mechanism was inert.")
    else:
        print("  -> MIXED / NOT REPRODUCED: re-derive before claiming anything.")


if __name__ == "__main__":
    main()
