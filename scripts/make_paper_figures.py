import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "assets", "figures")
DATA = os.path.join(ROOT, "benchmarks", "long_context_results.json")
os.makedirs(OUT, exist_ok=True)

# --- Enhanced Publication style ---
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",     
    "pdf.fonttype": 42,         
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": True,
    "legend.edgecolor": "#cccccc",
    "legend.fancybox": False,
    "figure.dpi": 300,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "grid.color": "#999999",
    "grid.linestyle": "--"
})

# Refined Academic Palette (Colorblind friendly)
C_NEUTRAL = "#999999"   # generic baseline
C_METHOD  = "#0072B2"   # rich blue
C_ACCENT  = "#E69F00"   # accent orange
C_GAIN    = "#009E73"   # green
C_DROP    = "#D55E00"   # deep orange/red
C_MUTED   = "#56B4E9"   # light blue

COL = 3.34   # single-column width (inches)

def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"[fig] wrote {name}.pdf / .png")

def format_ax(ax):
    # Just to ensure grid is below elements
    ax.set_axisbelow(True)

def fig_subspace():
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    format_ax(ax)
    vals = [0.000, 0.414]
    
    bars = ax.bar(["Cosine", "Subspace"], vals,
                  width=0.5, color=[C_NEUTRAL, C_METHOD], edgecolor="#333333", linewidth=1.0, zorder=3)
    ax.set_ylabel(r"Switch sensitivity $\Delta c$")
    ax.set_ylim(0, 0.5)
    for b, v in zip(bars, vals):
        if v == 0:
            ax.text(b.get_x() + b.get_width() / 2, 0.02, "0.000", ha="center", va="bottom", fontsize=7.5, zorder=4)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, zorder=4)
    ax.annotate("anisotropy-masked", xy=(0, 0.0), xytext=(0, 0.15),
                ha="center", fontsize=7, color=C_DROP,
                arrowprops=dict(arrowstyle="->", color=C_DROP, lw=1.0, shrinkA=2, shrinkB=2))
    _save(fig, "fig_subspace")

def fig_gwt_bids():
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    format_ax(ax)
    labels = ["bid0", "bid1", "bid2", "ext"]
    vals = [0.25, 0.25, 0.25, 0.25]
    cols = [C_METHOD, C_METHOD, C_METHOD, C_ACCENT]
    ax.bar(labels, vals, width=0.5, color=cols, edgecolor="#333333", linewidth=1.0, zorder=3)
    ax.axhline(0.25, ls="--", lw=1.2, color=C_DROP, zorder=4)
    ax.text(3.3, 0.26, r"/K$", color=C_DROP, fontsize=7.5, va="bottom", ha="right")
    ax.set_ylabel("Weight share")
    ax.set_ylim(0, 0.4)
    ax.text(3, 0.10, "external\nworld", ha="center", fontsize=7, color="white", fontweight="bold", zorder=4)
    _save(fig, "fig_gwt_bids")

def fig_world_model():
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    format_ax(ax)
    steps = [5, 10, 15, 20]
    wm = [0.9649, 0.3854, 0.1981, 0.1022]
    pc = [0.9589, 0.9365, 0.9141, 0.8907]
    ax.plot(steps, wm, "-o", color=C_METHOD, lw=1.8, ms=5, label="World-model loss", zorder=3)
    ax.plot(steps, pc, "-s", color=C_ACCENT, lw=1.8, ms=5, label="Predictive-coding", zorder=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(steps)
    ax.legend(loc="upper right", framealpha=0.9)
    _save(fig, "fig_world_model")

def fig_long_context():
    with open(DATA, "r", encoding="utf-8") as f:
        rows = json.load(f)
    T = [r["T_total"] for r in rows]
    order = ("Transformer", "LNN", "MT-LNN")
    style = {
        "Transformer": (C_NEUTRAL, "^", "--"),
        "LNN": (C_MUTED, "D", "--"),
        "MT-LNN": (C_METHOD, "o", "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(COL * 2.02, 2.0))
    for metric, ax, ylab, ymax in (
        ("seq_exact", axes[0], "Sequence-exact", 0.6),
        ("tok_acc", axes[1], "Token accuracy", 1.0),
    ):
        format_ax(ax)
        for name in order:
            c, mk, ls = style[name]
            y = [r["models"][name][metric] for r in rows]
            ax.plot(T, y, ls, marker=mk, color=c, lw=1.8, ms=4.5, label=name, zorder=3)
        ax.set_xlabel(r"Sequence length {\mathrm{total}}$")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, ymax)
        ax.set_xticks(T)
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].set_title("a) Sequence Level", loc="left", fontsize=9, fontweight="bold")
    axes[1].set_title("b) Token Level", loc="left", fontsize=9, fontweight="bold")
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig_long_context")

def fig_halluc():
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    format_ax(ax)
    vals = [0.989, 0.575]
    bars = ax.bar(["Normal", "Logic-break"], vals,
                  width=0.5, color=[C_GAIN, C_DROP], edgecolor="#333333", linewidth=1.0, zorder=3)
    ax.set_ylabel("Self-detection score")
    ax.set_ylim(0, 1.15)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, zorder=4)
        
    ax.annotate("", xy=(0.5, 0.61), xytext=(0.5, 0.98),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2, shrinkA=0, shrinkB=0), zorder=4)
    ax.text(0.56, 0.80, r"$\Delta=-0.414$", fontsize=7.5, color="#333333", va="center", ha="left", fontweight="bold")
    _save(fig, "fig_halluc")

if __name__ == "__main__":
    fig_subspace()
    fig_gwt_bids()
    fig_world_model()
    fig_long_context()
    fig_halluc()
    print(f"[done] figures in {OUT}")
