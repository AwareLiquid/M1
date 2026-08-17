import matplotlib.pyplot as plt
import numpy as np

# Nature-style configuration
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42
})

PALETTE = {
    "baseline_dark": "#484878",
    "baseline_mid":  "#7884B4",
    "ours_base":  "#E4CCD8",
    "ours_large": "#F0C0CC",
    "red_strong": "#B64342",
    "green_strong": "#2E9E44",
}

colors = [PALETTE['baseline_dark'], PALETTE['baseline_mid'], PALETTE['ours_base'], PALETTE['ours_large']]

fig = plt.figure(figsize=(7.2, 2.5))

# Subplot 1: WikiText-103 PPL vs Phi_hat
ax1 = fig.add_subplot(131)
models = ['Transformer', 'LNN', 'MT-LNN-nGWT', 'MT-LNN']
ppl = [22.4, 21.1, 20.0, 19.1]
phi_hat = [0.17, 0.24, 0.32, 0.38]

ax1.scatter(phi_hat[0], ppl[0], color=colors[0], s=60, label=models[0], zorder=3)
ax1.scatter(phi_hat[1], ppl[1], color=colors[1], s=60, label=models[1], zorder=3)
ax1.scatter(phi_hat[2], ppl[2], color=colors[2], s=60, label=models[2], zorder=3)
ax1.scatter(phi_hat[3], ppl[3], color=colors[3], s=100, label=models[3], marker='*', zorder=3)

# Connect with line
ax1.plot(phi_hat, ppl, '--', color='#A8A8A8', zorder=1)

ax1.set_xlabel('Integration Metric ($\\hat{\\Phi}$ $\\uparrow$)')
ax1.set_ylabel('WikiText-103 PPL $\\downarrow$')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.05, 1.05))

# Subplot 2: LRA Accuracy
ax2 = fig.add_subplot(132)
tasks = ['Pathfinder', 'ListOps', 'Text', 'Avg']
tf = [64.2, 36.9, 64.3, 55.1]
lnn = [68.9, 37.8, 65.7, 57.5]
mt_ngwt = [71.4, 39.1, 67.2, 59.2]
mt_lnn = [73.1, 40.3, 68.9, 60.8]

x = np.arange(len(tasks))
width = 0.2

ax2.bar(x - 1.5*width, tf, width, label='Transformer', color=colors[0])
ax2.bar(x - 0.5*width, lnn, width, label='LNN', color=colors[1])
ax2.bar(x + 0.5*width, mt_ngwt, width, label='MT-LNN-nGWT', color=colors[2])
ax2.bar(x + 1.5*width, mt_lnn, width, label='MT-LNN', color=colors[3])

ax2.set_ylabel('LRA Accuracy (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(tasks)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(30, 80)
# legend not needed, ax1 has it
# ax2.legend(frameon=False)

# Subplot 3: Selective Copy Advantage
ax3 = fig.add_subplot(133)
seq_lengths = [37, 69, 101]
tf_acc = [52.1, 38.4, 22.0]
lnn_acc = [74.5, 68.2, 60.1]
mt_lnn_acc = [88.2, 85.1, 83.4]

# Advantage is MT-LNN / Transformer
advantage_lnn = np.array(lnn_acc) / np.array(tf_acc)
advantage_mt_lnn = np.array(mt_lnn_acc) / np.array(tf_acc)

ax3.plot(seq_lengths, advantage_lnn, '-o', label='LNN / TF', color=colors[1])
ax3.plot(seq_lengths, advantage_mt_lnn, '-P', label='MT-LNN / TF', color=colors[3])

ax3.set_xlabel('Sequence Length $T$')
ax3.set_ylabel('Advantage Ratio over Transformer')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_xticks(seq_lengths)
ax3.legend(frameon=False)

plt.tight_layout()
plt.savefig('fig_experiments.pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig_experiments.png', bbox_inches='tight', dpi=300)
print("Saved fig_experiments.pdf and fig_experiments.png")
