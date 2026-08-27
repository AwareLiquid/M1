"""battery_demo.py — 电池 SoH 产品演示：训练 → 预测未见过的电池 → 曲线图

一次跑通"客户会看到的东西"：
  1. 在 B0005/6/7 三块电池上训练 MT-LNN 液态回归器（CPU 分钟级）
  2. 预测 HELD-OUT 的 B0018 全寿命容量衰减曲线（跨电芯泛化 = 部署真实场景）
  3. 输出: 文本对比 (vs LSTM/基线) + 预测曲线 CSV + 手写 SVG 图

用法:
  py -3.11 benchmarks/battery_demo.py
  py -3.11 benchmarks/battery_demo.py --epochs 80 --out artifacts/battery_demo

诚实边界: 这是研究级演示——单一数据集 (NASA PCoE) 交叉电芯验证;
RMSE 数值的意义以电池容量尺度 (Ah) 为参照, 不代表量产 BMS 精度。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from benchmarks.battery_soh_edge import (LiquidRegressor, RNNRegressor,
                                         load_cell)

CELLS = ["B0005", "B0006", "B0007", "B0018"]


def standardize(Xtr, Xte):
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd


def train(model, X, y, epochs, batch, lr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    n = len(Xt)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def svg_chart(cycles, actual, pred, path, rmse, test_cell):
    """手写 SVG 折线图（无 matplotlib 依赖，官网同款风格）。"""
    w, h, m = 720, 360, 50
    x0, x1 = 0, len(cycles) - 1
    ymin = min(actual.min(), pred.min()) - 0.05
    ymax = max(actual.max(), pred.max()) + 0.05

    def px(i):
        return m + (i - x0) * (w - 2 * m) / max(1, x1 - x0)

    def py(v):
        return h - m - (v - ymin) * (h - 2 * m) / (ymax - ymin)

    pts_a = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(actual))
    pts_p = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(pred))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"
  viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">
  <rect width="{w}" height="{h}" fill="#0e0e10"/>
  <line x1="{m}" y1="{h-m}" x2="{w-m}" y2="{h-m}" stroke="#3a3a41" stroke-width="1"/>
  <line x1="{m}" y1="{m}" x2="{m}" y2="{h-m}" stroke="#3a3a41" stroke-width="1"/>
  <polyline points="{pts_a}" fill="none" stroke="#9a9aa5" stroke-width="2"/>
  <polyline points="{pts_p}" fill="none" stroke="#4f9dff" stroke-width="2"/>
  <text x="{w-m}" y="{h-14}" fill="#9a9aa5" font-size="13" text-anchor="end">actual capacity</text>
  <text x="{w-m}" y="{h-32}" fill="#4f9dff" font-size="13" text-anchor="end">MT-LNN predicted</text>
  <text x="{m}" y="30" fill="#f5f5f7" font-size="15">
    {test_cell} held-out: cycle count vs capacity (Ah) — RMSE {rmse:.4f} Ah</text>
</svg>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-cell", default="B0018")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=128)
    # 78 = 13×6: d_head=6 偶数, 满足 RoPE 校验 (历史 10-seed 结果用 65,
    # 在 even-d_head 校验加入之前; 65/13=5 现在会直接 ValueError)
    ap.add_argument("--d-model", type=int, default=78)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--out", default="artifacts/battery_demo")
    args = ap.parse_args()

    train_cells = [c for c in CELLS if c != args.test_cell]
    print(f"加载 NASA 电芯: train={train_cells} test=[{args.test_cell}]")
    Xs, ys = [], []
    for c in train_cells:
        x, y = load_cell(c, args.seq_len)
        print(f"  {c}: {len(x)} 次放电, 容量 {y.max():.3f} -> {y.min():.3f} Ah")
        Xs.append(x)
        ys.append(y)
    Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
    Xte, yte = load_cell(args.test_cell, args.seq_len)
    print(f"  {args.test_cell}: {len(Xte)} 次放电 (HELD OUT), "
          f"容量 {yte.max():.3f} -> {yte.min():.3f} Ah")

    Xtr_s, Xte_s, _, _ = standardize(Xtr, Xte)

    t0 = time.time()
    m = LiquidRegressor(args.d_model, args.n_layers)
    train(m, Xtr_s, ytr, args.epochs, args.batch, args.lr, args.seed)
    print(f"\nMT-LNN 训练完成 ({time.time()-t0:.0f}s), "
          f"参数 {sum(p.numel() for p in m.parameters()):,}")

    lstm = RNNRegressor("lstm", args.d_model, args.n_layers)
    train(lstm, Xtr_s, ytr, args.epochs, args.batch, args.lr, args.seed)
    print(f"LSTM 基线训练完成, 参数 {sum(p.numel() for p in lstm.parameters()):,}")

    with torch.no_grad():
        pred_lnn = m(torch.from_numpy(Xte_s)).numpy()
        pred_lstm = lstm(torch.from_numpy(Xte_s)).numpy()

    rmse = lambda p: float(np.sqrt(((p - yte) ** 2).mean()))
    base = float(np.sqrt(((yte - ytr.mean()) ** 2).mean()))
    print("\n══════════ 电池 SoH 演示 (held-out 跨电芯) ══════════")
    print(f"常数基线  RMSE = {base:.4f} Ah")
    print(f"LSTM      RMSE = {rmse(pred_lstm):.4f} Ah")
    print(f"MT-LNN    RMSE = {rmse(pred_lnn):.4f} Ah")

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "soh_mtlnn.pt")
    torch.save({"model": m.state_dict(), "config": {
        "d_model": args.d_model, "n_layers": args.n_layers,
        "seq_len": args.seq_len, "test_cell": args.test_cell,
    }}, model_path)
    csv_path = os.path.join(args.out, "prediction.csv")
    with open(csv_path, "w") as f:
        f.write("cycle,actual_ah,pred_ah\n")
        for i, (a, p) in enumerate(zip(yte, pred_lnn)):
            f.write(f"{i+1},{a:.5f},{p:.5f}\n")
    svg_path = svg_chart(np.arange(len(yte)), yte, pred_lnn,
                         os.path.join(args.out, "soh_curve.svg"),
                         rmse(pred_lnn), args.test_cell)
    print(f"\n产物: {model_path}\n      {csv_path}\n      {svg_path}")


if __name__ == "__main__":
    sys.exit(main())
