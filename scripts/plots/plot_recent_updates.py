import json
import matplotlib.pyplot as plt
import numpy as np

# Nature Style Rules
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'

PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1":   "#F6CFCB",
    "red_2":   "#E9A6A1",
    "red_strong": "#B64342",
    "baseline_mid":  "#7884B4",
    "ours_large": "#F0C0CC",
    "neutral_mid":   "#767676",
    "neutral_dark":  "#4D4D4D",
    "neutral_black": "#272727"
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'wspace': 0.3})

# --- Panel A: Sparse Resonance Ablation ---
with open('benchmarks/sparse_resonance_ablation.json', 'r') as f:
    sparse_data = json.load(f)

runs = sparse_data['runs']
top_k_vals = [str(r['top_k']) + ("\n(Dense)" if r['mode'] == 'dense' else "\n(Sparse)") for r in runs]
tok_per_s = [r['tokens_per_s'] for r in runs]

ax = axes[0]
bars = ax.bar(top_k_vals, tok_per_s, color=[PALETTE["baseline_mid"] if r['mode'] == 'dense' else PALETTE["ours_large"] for r in runs], edgecolor=PALETTE['neutral_dark'], linewidth=1)

ax.set_ylabel("Inference Speed (Tokens / s)", fontsize=10, fontweight='bold', color=PALETTE['neutral_black'])
ax.set_xlabel("Top-K Resonance Scales Retained", fontsize=10, fontweight='bold', color=PALETTE['neutral_black'])
ax.set_title("a  Sparse Resonance Top-K Acceleration", loc='left', fontsize=12, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f'{int(yval)}', ha='center', va='bottom', fontsize=9, color=PALETTE['neutral_dark'])

# --- Panel B: Operator Compression Cache ---
with open('benchmarks/operator_compression_report.json', 'r') as f:
    comp_data = json.load(f)

steps = [100, 1000]
kv_cache_kb = []
state_cache_kb = []

for r in comp_data['runs']:
    for m in r['modes']:
        if m['mode'] == 'kv_cache_stream':
            kv_cache_kb.append(m['final_cache_bytes'] / 1024)
        elif m['mode'] == 'state_only_stream':
            state_cache_kb.append(m['final_cache_bytes'] / 1024)

ax2 = axes[1]
x = np.arange(len(steps))
width = 0.35

rects1 = ax2.bar(x - width/2, kv_cache_kb, width, label='KV Cache', color=PALETTE["baseline_mid"], edgecolor=PALETTE['neutral_dark'], linewidth=1)
rects2 = ax2.bar(x + width/2, state_cache_kb, width, label='State-only (Ours)', color=PALETTE["ours_large"], edgecolor=PALETTE['neutral_dark'], linewidth=1)

ax2.set_ylabel("Memory Consumption (KB) - Log Scale", fontsize=10, fontweight='bold', color=PALETTE['neutral_black'])
ax2.set_xlabel("Sequence Length (Tokens)", fontsize=10, fontweight='bold', color=PALETTE['neutral_black'])
ax2.set_title("b  Operator Compression: $O(1)$ Memory", loc='left', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(steps)
ax2.set_yscale('log')
ax2.legend(frameon=False, fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add text labels
for r1, r2 in zip(rects1, rects2):
    ax2.text(r1.get_x() + r1.get_width()/2., r1.get_height() * 1.2, f'{r1.get_height():.0f}', ha='center', va='bottom', fontsize=9)
    ax2.text(r2.get_x() + r2.get_width()/2., r2.get_height() * 1.2, f'{r2.get_height():.1f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('fig_operator_compression_updates.svg', dpi=300)
plt.savefig('fig_operator_compression_updates.png', dpi=300)
print("Saved fig_operator_compression_updates.svg/.png")
plt.savefig('fig_operator_compression_updates.pdf', dpi=300)


plt.savefig('fig_operator_compression_updates.pdf', dpi=300)
