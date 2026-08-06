"""Generate deck figures: cost-vs-capability positioning + long-context
same-size comparison. All numbers sourced from BENCHMARKS.md / RESULTS.md.

Fig 1: cost-per-task (x, log) vs capability (y) -- AA 'slay-line' methodology.
  - DeepSeek-V4-Flash reference: ~$0.025-0.03/task, AA Intelligence Index 50
  - MT-LNN (M1/O1): long-context task capability (Selective Copy seq-exact)
    89.5% @T=32 (200K params, matched size) at edge-class cost (est. $0.0005-0.002/task)
  - Note: axes are task-specific; explicitly framed as positioning, not a
    same-benchmark ranking.
Fig 2: Selective Copy whole-sequence recall, same 200K params, official table
  (BENCHMARKS.md:536-540): T=37/101/229, Transformer/LNN/MT-LNN.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\AwareLiquid-Web\figures"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- Fig 1
fig, ax = plt.subplots(figsize=(9.6, 6.0), dpi=200)

# Slay zone (upper-left = cheap & capable; lower-right = slayed)
ax.add_patch(Rectangle((3e-4, 45), 1e2, 60, facecolor="#fdecea", edgecolor="none", zorder=0))
ax.text(0.06, 97, "“斩杀区”= 更贵且更弱\n(Deployable)", fontsize=9, color="#b03a2e", ha="center")

# DeepSeek-V4-Flash reference
ax.scatter([0.028], [50], s=420, marker="*", c="#f0a500", edgecolors="#8a5a00",
           zorder=5, label="DeepSeek-V4-Flash (AA 智能指数 50, ~\$0.025–0.03/任务)")
ax.annotate("DeepSeek-V4-Flash\nAA 智能指数 50 · \$0.028/任务",
            xy=(0.028, 50), xytext=(0.09, 55), fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#8a5a00"))

# MT-LNN: long-context task capability at edge cost
for (x, y, tag, dx, dy) in [
    (0.0016, 89.5, "M1/O1 @ T=32\n89.5% seq-exact", -0.35, 3),
    (0.0009, 21.9, "M1/O1 @ T=229\n21.9% (×2.0 vs Transformer)", -0.5, -9),
]:
    ax.scatter([x], [y], s=520, marker="D", c="#00468b", edgecolors="#002a56",
               zorder=5, label=tag if "T=32" in tag else None)
    ax.annotate(tag, xy=(x, y), xytext=(x * (1 + dx), y + dy), fontsize=10,
                fontweight="bold", color="#00468b",
                arrowprops=dict(arrowstyle="->", color="#00468b"))

# Transformer same-size reference (cheap but weaker on long context)
ax.scatter([0.0016], [67.6], s=300, marker="o", c="#999999", edgecolors="#555",
           zorder=4, label="同尺寸 Transformer @ T=32 (67.6%)")
ax.annotate("同尺寸 Transformer\nT=32 67.6%", xy=(0.0016, 67.6), xytext=(0.0045, 62),
            fontsize=9, color="#666", arrowprops=dict(arrowstyle="->", color="#999"))

ax.set_xscale("log")
ax.set_xlim(3e-4, 3)
ax.set_ylim(0, 105)
ax.set_xlabel("单任务推理成本 (USD, 对数刻度)", fontsize=12)
ax.set_ylabel("任务能力分 (Capability)", fontsize=12)
ax.set_title("成本-能力定位：我们在长上下文赛道，而非通用智能赛道\n"
             "(AA 斩杀线方法论；纵轴为各自任务的能力分，非同一基准排名)",
             fontsize=12, fontweight="bold")
ax.grid(True, which="both", ls="--", alpha=0.3)
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.text(0.01, 0.02, "数据来源: BENCHMARKS.md (Selective Copy, 200K 同尺寸) · "
        "AA 参考点: 用户提供 (2026-07-31 榜单) · 成本为估算",
        transform=ax.transAxes, fontsize=7, color="#888")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_cost_capability_positioning.png"), bbox_inches="tight")
plt.close(fig)
print("fig1 ->", os.path.join(OUT, "fig_cost_capability_positioning.png"))

# ---------------------------------------------------------------- Fig 2
fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)
T = [37, 101, 229]
trans = [0.672, 0.570, 0.109]
lnn = [0.703, 0.727, 0.172]
mt = [0.883, 0.742, 0.219]
x = np.arange(len(T))
w = 0.26
b1 = ax.bar(x - w, trans, w, label="Transformer", color="#c9c9c9")
b2 = ax.bar(x, lnn, w, label="LNN", color="#8ab4d8")
b3 = ax.bar(x + w, mt, w, label="MT-LNN", color="#00468b")
for bars in (b1, b2, b3):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015,
                f"{r.get_height():.1%}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"T={t}" for t in T])
ax.set_ylabel("整序列完全匹配率 (seq-exact)", fontsize=12)
ax.set_title("官方同尺寸对比：选择性拷贝 · 整序列召回 (200K 参数, 公平全序列解码)",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(axis="y", ls="--", alpha=0.3)
ax.text(0.01, 0.02, "数据来源: BENCHMARKS.md (Long-context sweep, 1500 步等预算)",
        transform=ax.transAxes, fontsize=7, color="#888")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_selective_copy_samesize.png"), bbox_inches="tight")
plt.close(fig)
print("fig2 ->", os.path.join(OUT, "fig_selective_copy_samesize.png"))

