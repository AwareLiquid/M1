"""
Nature-figure style charts for AwareLiquid website (appended to gen_charts.py).
Backend: Python / matplotlib. Style: monochrome, no top/right spines, SVG text.

Fig A: Official same-size Selective Copy (BENCHMARKS.md:536-540) - seq-exact
       recall at matched 200K params, fair full-sequence decode.
Fig B: Cost-per-task vs capability positioning (AA slay-line methodology) -
       honest framing: we compete on long-context + edge cost, not AGI indices.

Usage: python charts/gen_website_charts.py  (append-only; re-runs safe)
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

P = {
    "baseline_dark": "#0d0d0d",
    "baseline_mid":  "#6b6b80",
    "baseline_soft": "#c9c9d2",
    "ours":          "#0d0d0d",
    "delta_up":      "#0d0d0d",
    "delta_down":    "#6b6b80",
    "neutral":       "#a8a8b0",
    "neutral_light": "#e8e8ef",
    "bg_panel":      "#f7f7f9",
}

OUT = os.path.dirname(os.path.abspath(__file__))


def save_svg(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", format="svg")
    print(f"Saved {path}")
    plt.close(fig)


# ── Fig A: Official same-size Selective Copy (BENCHMARKS.md) ────────────────
def fig_selective_copy():
    T = ["T=37", "T=101", "T=229"]
    trans = [0.672, 0.570, 0.109]
    lnn = [0.703, 0.727, 0.172]
    mt = [0.883, 0.742, 0.219]

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = np.arange(len(T))
    w = 0.26
    b1 = ax.bar(x - w, trans, w, color=P["neutral_light"],
                edgecolor=P["baseline_mid"], linewidth=0.8, label="Transformer")
    b2 = ax.bar(x, lnn, w, color=P["baseline_soft"],
                edgecolor=P["baseline_mid"], linewidth=0.8, hatch="//",
                label="LNN")
    b3 = ax.bar(x + w, mt, w, color=P["ours"],
                edgecolor="white", linewidth=0.8, label="MT-LNN")
    for bars in (b1, b2, b3):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02,
                    f"{r.get_height():.0%}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#222")
    # ratio annotation at T=229
    ax.annotate("×2.0", xy=(2 + w, 0.219 + 0.09), xytext=(2 + w, 0.40),
                ha="center", fontsize=8, fontweight="bold", color=P["ours"],
                arrowprops=dict(arrowstyle="->", color=P["ours"], lw=1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(T)
    ax.set_ylabel("Whole-sequence recall (seq-exact)")
    ax.set_title("Selective Copy: matched 200K params, fair full-sequence decode",
                 fontsize=8.5, fontweight="bold", pad=8)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", ncol=3, fontsize=6.5)
    ax.spines["left"].set_color("#ccc")
    ax.tick_params(axis="x", bottom=False)
    ax.grid(axis="y", lw=0.5, alpha=0.4, color="#ddd", zorder=0)
    ax.text(0.005, 0.02, "Source: BENCHMARKS.md (Long-context sweep, equal 1500-step budget)",
            transform=ax.transAxes, fontsize=5.8, color="#999")
    save_svg(fig, "selective_copy_samesize.svg")


# ── Fig B: Cost vs capability positioning (honest, AA methodology) ──────────
def fig_cost_capability():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    # Slay zone shading (upper-left cheap & capable is the good zone; the
    # "slay line" concept: models more expensive AND weaker get slayed)
    ax.axvspan(2e-4, 3, ymin=0.0, ymax=0.35, color=P["neutral_light"], alpha=0.6, zorder=0)
    ax.text(0.5, 8, "slay zone\n(costlier & weaker)", ha="center", fontsize=6.5,
            color=P["baseline_mid"])

    # DeepSeek-V4-Flash reference (user-provided, 2026-07-31 AA index)
    ax.scatter([0.028], [50], s=90, marker="*", c=P["baseline_mid"],
               edgecolors=P["baseline_dark"], zorder=5)
    ax.annotate("DeepSeek-V4-Flash\nAA index 50 · ~$0.028/task",
                xy=(0.028, 50), xytext=(0.10, 55), fontsize=7,
                arrowprops=dict(arrowstyle="->", color=P["baseline_mid"], lw=0.8))

    # MT-LNN points: long-context capability at edge cost (est.)
    ax.scatter([0.0016], [89.5], s=110, marker="D", c=P["ours"], edgecolors="white", zorder=5)
    ax.annotate("M1/O1 @ T=32\n89.5% seq-exact", xy=(0.0016, 89.5),
                xytext=(0.0042, 92), fontsize=7, fontweight="bold",
                color=P["ours"],
                arrowprops=dict(arrowstyle="->", color=P["ours"], lw=0.8))
    ax.scatter([0.0009], [21.9], s=90, marker="D", c=P["baseline_mid"],
               edgecolors="white", zorder=5)
    ax.annotate("M1/O1 @ T=229\n21.9% (×2.0 vs Transformer)", xy=(0.0009, 21.9),
                xytext=(0.0042, 16), fontsize=7, color=P["baseline_mid"],
                arrowprops=dict(arrowstyle="->", color=P["baseline_mid"], lw=0.8))

    # Same-size Transformer reference
    ax.scatter([0.0016], [67.6], s=70, marker="o", c="white",
               edgecolors=P["baseline_mid"], zorder=5)
    ax.annotate("same-size Transformer\nT=32 67.6%", xy=(0.0016, 67.6),
                xytext=(0.0042, 62), fontsize=6.5, color=P["baseline_mid"],
                arrowprops=dict(arrowstyle="->", color=P["baseline_mid"], lw=0.7))

    ax.set_xscale("log")
    ax.set_xlim(3e-4, 3)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Cost per task (USD, log)")
    ax.set_ylabel("Capability score")
    ax.set_title("Positioning: long-context + edge cost, not AGI indices",
                 fontsize=8.5, fontweight="bold", pad=8)
    ax.spines["left"].set_color("#ccc")
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.3, color="#ddd", zorder=0)
    ax.text(0.005, 0.02,
            "Honest: axes are task-specific; we do not claim AGI-benchmark parity "
            "(48M–1.1B models score near random on MMLU/Agent suites). "
            "Cost is an estimate; capability = Selective Copy (BENCHMARKS.md).",
            transform=ax.transAxes, fontsize=5.8, color="#999")
    save_svg(fig, "cost_capability_positioning.svg")


if __name__ == "__main__":
    fig_selective_copy()
    fig_cost_capability()
    print("done")
