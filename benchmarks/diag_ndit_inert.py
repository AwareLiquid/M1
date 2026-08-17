"""NDIT 旋转学惰诊断 — 训练后检查 Q_t 是否真的非对角。

判据：
  1. off-diag energy = ||hh_w - diag(hh_w)||_F / ||hh_w||_F —— 0 表示旋转
     学惰为纯对角（近似恒等/纯缩放），无非对角表达力。
  2. v_t 输入相关性：不同输入 token 的 v_t 方差 —— 0 表示 v_t ≈ 常数
     （b_h 主导），旋转与输入无关（那 A5 学不会是必然）。
运行: py -3.11 benchmarks/diag_ndit_inert.py
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt_lnn.config import MTLNNConfig
from mt_lnn.mt_lnn_layer import MTLNNLayer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.state_tracking_a5 import build_a5, verify_group, make_batch  # noqa: E402


class LiquidProbe(nn.Module):
    def __init__(self, cfg, n_classes):
        super().__init__()
        self.embed = nn.Embedding(n_classes, cfg.d_model)
        self.layer = MTLNNLayer(cfg)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, n_classes)

    def forward(self, x):
        h = self.embed(x)
        out, _ = self.layer(h, None, position_offset=0, use_scan=True)
        return self.head(self.norm(h + out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_len", type=int, default=16)
    p.add_argument("--test_len", type=int, default=48)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_proto", type=int, default=4)
    p.add_argument("--n_scales", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    args = p.parse_args()

    elems, table = build_a5()
    verify_group(elems, table)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    n_classes = table.shape[0]

    cfg = MTLNNConfig(
        vocab_size=n_classes, d_model=args.d_model, n_layers=1, n_heads=4,
        n_kv_heads=2, d_head=args.d_model // 4, max_seq_len=args.test_len + 8,
        n_protofilaments=args.n_proto, n_time_scales=args.n_scales,
        tau_max=200.0, selective_decay=True, selective_decay_mode="exp",
        use_householder_transition=True,
    )
    model = LiquidProbe(cfg, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for _ in range(args.steps):
        x, y = make_batch(args.batch, 1, args.train_len, table, device, gen)
        loss = F.cross_entropy(model(x).reshape(-1, n_classes), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    res = model.layer.resonance
    with torch.no_grad():
        hh_w = res.hh_w                              # (P,K,D,D)
        # off-diag energy 用第一反射（k 个反射的能量平均）
        off_energy = 0.0
        for r in range(res.hh_rank):
            w_r = hh_w[:, r]                          # (P,D,D)
            diag = torch.diagonal(w_r, dim1=-2, dim2=-1)
            off = w_r - torch.diag_embed(diag)
            off_energy += (off.norm() / w_r.norm()).item()
        off_energy /= res.hh_rank
        # v_t 输入相关性：对 8 个随机 batch 采样 v_t 的跨 batch 方差
        vs = []
        for _ in range(8):
            x, _ = make_batch(args.batch, 1, args.train_len, table, device, gen)
            h = model.embed(x).view(x.shape[0], x.shape[1], cfg.n_protofilaments, -1)
            v = torch.einsum("bpd,pde->bpe", h[:, 0], hh_w[:, 0]) + res.hh_b[:, 0]
            v = F.normalize(v, dim=-1)
            vs.append(v)
        vs = torch.stack(vs)                          # (8,B,P,D)
        v_spread = vs.std(dim=0).mean()               # 跨 batch 方向方差
        b_ratio = res.hh_b.norm() / (hh_w.norm() + res.hh_b.norm() + 1e-9)

        # 旋转正交性：Q_t 的 off-diag 作用——用单位基检查 Q e_i 的混合度
        # （off_energy 已足够回答学惰问题，此处省略显式构造）

    print(f"off-diag energy: {off_energy:.4f}  (0 = inert diagonal rotation)")
    print(f"v_t cross-batch std: {v_spread:.4f}  (0 = v_t input-independent)")
    print(f"bias fraction: {b_ratio:.4f}  (1 = b_h dominates, v_t ~ constant)")
    verdict = (
        "INERT: rotation learned no off-diagonal structure"
        if off_energy < 0.1 or v_spread < 0.02
        else "ACTIVE: off-diagonal structure present - failure is elsewhere")
    print(f"diagnosis: {verdict}")


if __name__ == "__main__":
    main()
