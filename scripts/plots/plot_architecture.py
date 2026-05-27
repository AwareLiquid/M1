import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

# Nature-style configuration
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42
})

# NMI Pastel Palette + custom
PALETTE = {
    "baseline_dark": "#484878",
    "baseline_mid":  "#7884B4",
    "bg_lilac": "#E0E0F0",
    "gold":   "#FFD700",
    "teal":   "#42949E",
    "magenta":"#EA84DD",
    "green_1": "#DDF3DE",
    "neutral_mid":   "#767676",
    "neutral_light": "#CFCECE",
}

fig, ax = plt.subplots(figsize=(7.2, 3.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

def add_block(x, y, w, h, text, color, text_color='black', fontsize=8, alpha=1.0):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1,rounding_size=2", 
                                  linewidth=1, edgecolor=PALETTE['neutral_mid'], facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, color=text_color, fontweight='bold')

def add_arrow(p1, p2, color=PALETTE['baseline_dark'], style='-|>', lw=1.5):
    arrow = patches.FancyArrowPatch(posA=p1, posB=p2, arrowstyle=style, color=color, mutation_scale=10, linewidth=lw)
    ax.add_patch(arrow)

# 1. Input Layer
add_block(2, 2, 96, 5, "Input Token Embedding & States ($X_t$)", PALETTE['neutral_light'])

# 2. MT-DL Layer (Microtubule Dynamic Layer)
add_block(2, 10, 96, 20, "", PALETTE['bg_lilac'], alpha=0.5)
ax.text(50, 27, "Microtubule Liquid Neural Layer (Parallel Scan)", fontweight='bold', color=PALETTE['baseline_dark'], ha='center')

# Inside MT-DL: Protofilaments with Dynamic Gating & Predictive Coding
for i in range(13):
    add_block(5 + i*7.2, 18, 5.5, 6, f"$\\tau_{i+1}$", PALETTE['green_1'], fontsize=7)

ax.text(50, 14, "13 Parallel Channels ($\\kappa$ Dynamic Gated & Compute Skip)", ha='center', fontsize=7, color=PALETTE['neutral_mid'])
# Predictive coding arrows inside MT-DL
add_arrow((65.2, 20.5), (55.2, 20.5), color=PALETTE['magenta'], lw=1)
ax.text(60.2, 21.5, "Pred Loss", ha='center', fontsize=5, color=PALETTE['magenta'])

# 3. GWTB & Infinite Cache
add_block(5, 33, 42, 8, "Global Workspace Bottleneck\n(Compress & Broadcast)", PALETTE['gold'])
add_block(53, 33, 42, 8, "$O(1)$ Working Memory\n(Exponential Decay)", PALETTE['teal'], text_color='white')

# 4. Output Heads
add_block(5, 46, 28, 8, "Next-Token\n(Target Head)", PALETTE['neutral_mid'], text_color='white')
add_block(36, 46, 28, 8, "Causal Chain\nExtraction", PALETTE['magenta'], text_color='white')
add_block(67, 46, 28, 8, "Self-Monitor\nExtraction", PALETTE['baseline_dark'], text_color='white')

# Arrows
add_arrow((50, 7), (50, 10))  # Input to MT-DL
add_arrow((26, 30), (26, 33))  # MT-DL to GWTB
add_arrow((74, 30), (74, 33))  # MT-DL to Memory
add_arrow((47, 37), (53, 37), style='<|-|>', color=PALETTE['neutral_mid']) # GWTB <-> Memory
add_arrow((26, 41), (19, 46)) # to Target
add_arrow((50, 41), (50, 46)) # to Causal
add_arrow((74, 41), (81, 46)) # to Monitor

plt.tight_layout()
plt.savefig('fig_architecture.pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig_architecture.svg', bbox_inches='tight', dpi=300)
plt.savefig('fig_architecture.png', bbox_inches='tight', dpi=300)
print("Saved fig_architecture.pdf, .svg, and .png")
