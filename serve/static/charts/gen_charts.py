"""
Nature-figure style charts for AwareLiquid website.
Backend: Python / matplotlib
Style: NMI pastel palette, no top/right spines, editable SVG text
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.dirname(__file__)

# ── Style setup (nature-figure API) ──────────────────────────────────────────
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

# Monochrome palette, matching the site. The site runs no accent colour at all
# (a deliberate split from the violet the nearest-neighbour project uses), so
# charts separate series by VALUE and by line style -- solid vs dashed, filled
# vs hollow markers -- rather than by hue. This also survives greyscale printing
# and is readable with any form of colour blindness.
P = {
    "baseline_dark": "#0d0d0d",
    "baseline_mid":  "#6b6b80",
    "baseline_soft": "#c9c9d2",
    "ours":          "#0d0d0d",
    "ours_dark":     "#000000",
    "delta_up":      "#0d0d0d",
    "delta_down":    "#6b6b80",
    "neutral":       "#a8a8b0",
    "neutral_light": "#e8e8ef",
    "accent_green":  "#0d0d0d",
    "accent_amber":  "#6b6b80",
    "accent_red":    "#0d0d0d",
    "bg_panel":      "#f7f7f9",
}

def save_svg(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", format="svg")
    print(f"Saved {path}")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — EWC Forgetting Reduction (hero chart, landscape)
# Claim: EWC consolidation reduces cross-domain forgetting by 85% vs dense baseline
# ═══════════════════════════════════════════════════════════════════════════════
def fig_ewc():
    conditions = ["Dense\n(no defense)", "Liquid\nbackbone", "EWC\nconsolidation"]
    forgetting  = [10562, 9544, 1633]
    colors      = [P["baseline_soft"], P["baseline_mid"], P["accent_green"]]
    hatches     = ["", "", ""]

    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    bars = ax.bar(conditions, forgetting, color=colors, width=0.52,
                  edgecolor="white", linewidth=0.8, zorder=3)

    # value labels
    for bar, val in zip(bars, forgetting):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                f"{val:,}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#222")

    # reduction annotation
    ax.annotate("", xy=(2, forgetting[2] + 600), xytext=(0, forgetting[0] + 600),
                arrowprops=dict(arrowstyle="-|>", color=P["accent_green"],
                                lw=1.2, mutation_scale=10))
    ax.text(1, max(forgetting)*0.72, "−85% forgetting",
            ha="center", va="center", fontsize=7.5, color=P["accent_green"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=P["accent_green"],
                      lw=0.8, alpha=0.9))

    ax.set_ylabel("PPL rise on domain A (↓ better)", fontsize=8)
    ax.set_title("Cross-domain forgetting by condition", fontsize=8.5, fontweight="bold",
                 pad=8)
    ax.set_ylim(0, 13000)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax.spines["left"].set_color("#ccc")
    ax.tick_params(axis="x", bottom=False)
    ax.grid(axis="y", lw=0.5, alpha=0.4, color="#ddd", zorder=0)
    ax.set_axisbelow(True)

    # caption note
    fig.text(0.05, -0.06,
             "3 seeds · 60.7M model · WikiText-103 → TinyStories · 1,200 steps/domain",
             fontsize=6.5, color=P["neutral"], style="italic")

    fig.tight_layout()
    save_svg(fig, "ewc_forgetting.svg")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Hebbian Ablation with error bars
# Claim: Hebbian term is effectively inert at tested scales (PPL change < seed noise)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_hebbian():
    import json, pathlib
    data_path = pathlib.Path(OUT).parent.parent.parent / "experiments" / "report_ablation_hebbian_lr.json"
    with open(data_path) as f:
        d = json.load(f)

    settings = d["settings"]
    labels   = ["off\n(baseline)", "1e-4", "1e-2", "1e-1"]
    keys     = ["off", "1e-4", "1e-2", "1e-1"]
    means    = [settings[k]["ppl_mean"] for k in keys]
    stds     = [settings[k]["ppl_std"]  for k in keys]
    fracs    = [settings[k]["hebb_frac_mean"] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.6),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # Left: PPL with error bars
    colors_bar = [P["neutral_light"]] + [P["baseline_soft"]]*3
    ax1.bar(labels, means, yerr=stds, color=colors_bar, width=0.5,
            capsize=3, error_kw=dict(lw=0.8, color="#888"),
            edgecolor="white", linewidth=0.8, zorder=3)

    ax1.axhline(means[0], color=P["neutral"], lw=0.8, ls="--", alpha=0.7, zorder=2)
    ax1.set_ylabel("Validation PPL", fontsize=8)
    ax1.set_title("PPL vs Hebbian LR", fontsize=8.5, fontweight="bold", pad=6)
    ax1.set_ylim(310, 365)
    ax1.grid(axis="y", lw=0.5, alpha=0.4, color="#ddd", zorder=0)
    ax1.set_axisbelow(True)
    ax1.text(0.97, 0.06, "changes within\nseed noise",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=6.5, color=P["neutral"], style="italic")

    # Right: gradient share (log scale)
    frac_nonzero = [1e-12] + fracs[1:]   # replace 0 for log scale
    ax2.barh(labels, frac_nonzero, color=colors_bar, height=0.5,
             edgecolor="white", linewidth=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Gradient share (log)", fontsize=8)
    ax2.set_title("Hebbian gradient contribution", fontsize=8.5, fontweight="bold", pad=6)
    ax2.axvline(1e-4, color=P["accent_amber"], lw=0.8, ls="--", alpha=0.8)
    ax2.text(1.5e-4, 2.8, "meaningful\nthreshold?",
             fontsize=6, color=P["accent_amber"], va="top")
    ax2.grid(axis="x", lw=0.5, alpha=0.4, color="#ddd")
    ax2.set_axisbelow(True)

    fig.suptitle("Hebbian plasticity ablation — 2 seeds × 150 steps · WikiText-103",
                 fontsize=7.5, color=P["neutral"], y=1.02)
    fig.tight_layout()
    save_svg(fig, "hebbian_ablation.svg")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Sparse Resonance: speed vs divergence trade-off
# Claim: top-k=2 gives +24% speed with <0.34% mean divergence
# ═══════════════════════════════════════════════════════════════════════════════
def fig_sparse_resonance():
    import json, pathlib
    data_path = pathlib.Path(OUT).parent.parent.parent / "benchmarks" / "sparse_resonance_ablation.json"
    with open(data_path) as f:
        d = json.load(f)

    runs = d["runs"]
    labels    = ["dense\n(k=5)", "sparse\nk=1", "sparse\nk=2", "sparse\nk=3"]
    tok_s     = [r["tokens_per_s"] for r in runs]
    div_mean  = [r["divergence_vs_dense"]["mean_abs"] for r in runs]
    colors_pt = [P["neutral_light"], P["ours_dark"], P["ours"], P["baseline_mid"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.6))

    # Speed bar
    bars = ax1.bar(labels, tok_s, color=colors_pt, width=0.5,
                   edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(bars, tok_s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax1.set_ylabel("Tokens / second", fontsize=8)
    ax1.set_title("Throughput by resonance mode", fontsize=8.5, fontweight="bold", pad=6)
    ax1.set_ylim(0, 8000)
    ax1.grid(axis="y", lw=0.5, alpha=0.4, color="#ddd", zorder=0)
    ax1.set_axisbelow(True)

    # Divergence bar (skip dense = 0 exactly)
    div_display = [max(v, 1e-5) for v in div_mean]
    bars2 = ax2.bar(labels, div_display, color=colors_pt, width=0.5,
                    edgecolor="white", linewidth=0.8, zorder=3, log=True)
    ax2.set_ylabel("Mean abs divergence vs dense (log)", fontsize=8)
    ax2.set_title("Logit fidelity vs dense", fontsize=8.5, fontweight="bold", pad=6)
    ax2.grid(axis="y", lw=0.5, alpha=0.4, color="#ddd", zorder=0)
    ax2.set_axisbelow(True)
    ax2.text(0.5, 0.06, "lower = closer to dense",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=6.5, color=P["neutral"], style="italic")

    fig.suptitle("Sparse resonance: throughput vs fidelity",
                 fontsize=7.5, color=P["neutral"], y=1.02)
    fig.tight_layout()
    save_svg(fig, "sparse_resonance.svg")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Memory scaling: KV-cache vs state-only streaming
# Claim: state-only streaming keeps constant O(1) memory vs O(n) KV cache
# ═══════════════════════════════════════════════════════════════════════════════
def fig_memory_scaling():
    """O(1) carried state vs a matched KV-cache, out to 1M tokens.

    Replaces an earlier version that plotted a 100-1000 token run from
    operator_compression_report.json (state = 4,160 B) while the caption
    beneath it quoted the 1M-token result (0.381 MB). Chart and caption
    described two different experiments, which read as a 100x contradiction.
    This plots the experiment the caption describes.

    ARR state is a measured snapshot (prime the full context, then weigh the
    carried tensors). The KV line is the exact analytic size for the matched
    Llama config -- 3 GB of cache cannot be materialised on the test machine,
    and quoting an estimate as a measurement would be worse than saying so.
    Source: benchmarks/scaling_comparison.py --mode decode, 832x12, GQA=1.
    """
    ctx = np.array([512, 2048, 8192, 32768, 131072, 524288, 1048576], dtype=float)
    kv_mb = np.array([1.5, 6.0, 24.0, 96.0, 384.0, 1536.0, 3072.0])
    state_mb = np.full_like(ctx, 0.381)

    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    # Series separate by line style and marker fill, not hue: the site runs no
    # accent colour, and this stays legible in greyscale.
    ax.plot(ctx, kv_mb, color=P["baseline_dark"], lw=1.4, ls="--",
            marker="o", ms=3.5, mfc="white", mew=1.0,
            label="KV-cache  O(n)")
    ax.plot(ctx, state_mb, color=P["baseline_dark"], lw=1.8, ls="-",
            marker="s", ms=3.5,
            label="O-series carried state  O(1)")
    ax.fill_between(ctx, state_mb, kv_mb, alpha=0.06, color=P["baseline_dark"])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Context length (tokens)", fontsize=8)
    ax.set_ylabel("Inference memory (MB, log)", fontsize=8)
    ax.set_title("Carried state vs KV-cache, 512 to 1M tokens", fontsize=8.5,
                 fontweight="bold", pad=8)
    ax.set_xticks([512, 8192, 131072, 1048576])
    ax.set_xticklabels(["512", "8K", "128K", "1M"], fontsize=7)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.grid(lw=0.5, alpha=0.35, color="#ddd", zorder=0)
    ax.set_axisbelow(True)

    # Annotations sit INSIDE the axes; at log scale the previous positions
    # collided with the x-axis title (bottom) and escaped the frame (top).
    ax.set_ylim(0.08, 20000)
    ax.annotate("0.381 MB, flat", xy=(ctx[2], 0.381), xytext=(2500, 3.2),
                fontsize=6.5, color=P["baseline_dark"], ha="center",
                arrowprops=dict(arrowstyle="-|>", color=P["baseline_dark"],
                                lw=0.8, mutation_scale=8))
    ax.annotate("3,072 MB", xy=(ctx[-1], 3072.0), xytext=(150000, 1400),
                fontsize=6.5, color=P["baseline_dark"], ha="right")

    fig.tight_layout()
    save_svg(fig, "memory_scaling.svg")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Architecture overview schematic (horizontal pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.5, 1.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    boxes = [
        (0.3, "Input\nTokens",   P["neutral_light"],   "#888"),
        (1.8, "Embedding\nLayer",P["baseline_soft"],   P["baseline_dark"]),
        (3.3, "LTC Core\n(τ-gates)", P["baseline_mid"], "#fff"),
        (4.8, "Liquid Core\nModules", P["ours"],       "#fff"),
        (6.3, "Output\nHead",    P["baseline_soft"],   P["baseline_dark"]),
        (7.8, "Continual\nLearning",  P["accent_green"], "#fff"),
    ]

    bw, bh = 1.15, 0.78
    y0 = 0.62

    for x, label, fc, tc in boxes:
        rect = mpatches.FancyBboxPatch((x, y0), bw, bh,
                                        boxstyle="square,pad=0.06",
                                        facecolor=fc, edgecolor="#d0d0d8",
                                        linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + bw/2, y0 + bh/2, label, ha="center", va="center",
                fontsize=7, color=tc, fontweight="bold", linespacing=1.35)

    # arrows
    for i in range(len(boxes)-1):
        x_start = boxes[i][0] + bw
        x_end   = boxes[i+1][0]
        xm = (x_start + x_end) / 2
        ax.annotate("", xy=(x_end, y0 + bh/2),
                    xytext=(x_start, y0 + bh/2),
                    arrowprops=dict(arrowstyle="-|>", color="#aaa",
                                    lw=0.8, mutation_scale=7))

    # sub-labels for liquid core modules
    modules = ["GWT-B", "World\nModel", "Hebbian", "Pred.\nCoding"]
    for i, m in enumerate(modules):
        xm = 4.8 + (i % 2) * 0.58
        # Row gap must exceed the within-label line gap, otherwise the bottom
        # line of the top row ("Model") collides with the top line of the
        # bottom row ("Pred."). Push top row up / bottom row down to separate.
        ym = 0.02 if i >= 2 else 0.26
        ax.text(xm + bw/2 - 0.29, ym, m, ha="center", va="bottom",
                fontsize=5.5, color=P["ours_dark"], style="italic")

    ax.set_title("MT-LNN architecture pipeline", fontsize=8, fontweight="bold",
                 pad=4, loc="left", x=0.02)

    fig.tight_layout(pad=0.3)
    save_svg(fig, "architecture_pipeline.svg")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — PPL Ablation (Kaggle GPU run)
# Claim: MT-LNN adapter reduces perplexity by 28.5% over base LLM
# ═══════════════════════════════════════════════════════════════════════════════
def fig_ppl_ablation():
    labels = ["Base LLM\n(TinyLlama-1.1B)", "MT-LNN adapter\n(+LoRA, 1k steps)"]
    ppl    = [9.161, 6.553]
    colors = [P["baseline_soft"], P["ours"]]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    bars = ax.bar(labels, ppl, color=colors, width=0.45,
                  edgecolor="white", linewidth=0.8, zorder=3)

    for bar, val in zip(bars, ppl):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#222")

    # reduction arrow
    ax.annotate("", xy=(1, ppl[1] + 0.35), xytext=(0, ppl[0] + 0.35),
                arrowprops=dict(arrowstyle="-|>", color=P["accent_green"],
                                lw=1.2, mutation_scale=10))
    ax.text(0.5, (ppl[0] + ppl[1]) / 2 + 0.45, "−28.5% PPL",
            ha="center", va="bottom", fontsize=7.5, color=P["accent_green"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=P["accent_green"],
                      lw=0.8, alpha=0.9))

    ax.set_ylabel("Perplexity  (↓ better)", fontsize=8)
    ax.set_title("Language modelling PPL after MT-LNN adaptation", fontsize=8.5,
                 fontweight="bold", pad=8)
    ax.set_ylim(0, 11.5)
    ax.grid(axis="y", lw=0.5, alpha=0.4, color="#ddd", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", bottom=False)

    fig.text(0.05, -0.04,
             "Kaggle P100 GPU · 38,400 tokens · 50 batches · trainable 2.3M / 1.17B params",
             fontsize=6.5, color=P["neutral"], style="italic")

    fig.tight_layout()
    save_svg(fig, "ppl_ablation.svg")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Context injection accuracy + needle-in-haystack (corrected 2026-06-26)
# Left: +13.3pp accuracy uplift with MT-LNN context injection
# Right: needle-in-haystack — near-perfect within the base 2048 window; both collapse
#        at 4096 (exceeds TinyLlama RoPE window). Old "0%" was a chat-template harness bug.
# ═══════════════════════════════════════════════════════════════════════════════
def fig_context_and_needle():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.8),
                                    gridspec_kw={"width_ratios": [1.1, 1]})

    # ── Left: context injection accuracy ─────────────────────────────────────
    cond   = ["No injection\n(baseline)", "With MT-LNN\ninjection"]
    acc    = [83.33, 96.67]
    colors = [P["baseline_soft"], P["accent_green"]]

    bars = ax1.bar(cond, acc, color=colors, width=0.45,
                   edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(bars, acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color="#222")

    # uplift annotation
    ax1.annotate("", xy=(1, acc[1] + 2.5), xytext=(0, acc[0] + 2.5),
                arrowprops=dict(arrowstyle="-|>", color=P["accent_green"],
                                lw=1.2, mutation_scale=10))
    ax1.text(0.5, 91, "+13.3 pp", ha="center", va="bottom",
             fontsize=7.5, fontweight="bold", color=P["accent_green"],
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=P["accent_green"],
                       lw=0.8, alpha=0.9))

    ax1.set_ylabel("Accuracy  (↑ better)", fontsize=8)
    ax1.set_ylim(0, 107)
    ax1.set_title("Context injection uplift\n30 QA questions · Qwen-3B", fontsize=8,
                  fontweight="bold", pad=6)
    ax1.axhline(100, color="#ccc", lw=0.6, ls="--")
    ax1.grid(axis="y", lw=0.5, alpha=0.4, color="#ddd", zorder=0)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="x", bottom=False)

    # ── Right: needle-in-haystack (corrected, chat-template re-test) ──────────
    # Exact-match accuracy averaged over depths {0.1, 0.5, 0.9}, 5 samples/cell.
    # Source: bench_needle_m1_faithful.py -> benchmarks/needle_m1_chat_template.json
    base_acc    = [0.867, 1.000, 0.000]   # 1K / 2K / 4K
    adapter_acc = [1.000, 1.000, 0.000]

    x = np.arange(3)
    width = 0.35

    b1 = ax2.bar(x - width/2, base_acc, width, label="Base LLM",
                 color=P["baseline_soft"], edgecolor="white", zorder=3)
    b2 = ax2.bar(x + width/2, adapter_acc, width, label="MT-LNN adapter",
                 color=P["ours"], edgecolor="white", zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                         f"{h:.2f}".rstrip("0").rstrip("."), ha="center",
                         va="bottom", fontsize=6.5, fontweight="bold", color="#222")

    ax2.set_xticks(x)
    ax2.set_xticklabels(["1 K", "2 K", "4 K"])
    ax2.set_xlabel("Context length (tokens)", fontsize=8)
    ax2.set_ylabel("Exact match (↑)", fontsize=8)
    ax2.set_ylim(0, 1.12)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax2.set_title("Needle-in-haystack\n(chat template, re-test)", fontsize=8,
                  fontweight="bold", pad=6)
    ax2.legend(loc="lower left", fontsize=6.5)
    ax2.grid(axis="y", lw=0.5, alpha=0.4, color="#ddd", zorder=0)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="x", bottom=False)

    # Honest annotation: 4K collapse is a base-window limit, not an adapter failure
    ax2.text(2, 0.06, "exceeds base\n2048 window",
             ha="center", va="bottom", fontsize=5.8, color=P["accent_red"],
             style="italic")

    fig.tight_layout()
    save_svg(fig, "context_needle.svg")

# ── Run all ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating AwareLiquid charts (nature-figure style)…")
    fig_ewc()
    fig_hebbian()
    fig_sparse_resonance()
    fig_memory_scaling()
    fig_architecture()
    fig_ppl_ablation()
    fig_context_and_needle()
    print("Done.")
