import matplotlib.pyplot as plt
import numpy as np

# Nature Style configuration
def set_nature_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 12

set_nature_style()
nature_blue = '#00468b'
nature_red = '#ed0000'
nature_grey = '#8c8c8c'

# Plot 1: Concurrency comparison (100k context)
fig, ax = plt.subplots(figsize=(6, 4.5))
categories = ['Traditional\n(1x A100)', 'MT-LNN\n(1x RTX 4090)']
concurrency = [12, 180] # A100 OOMs fast, 4090 can handle hundreds due to O(1) state

ax.bar(categories, concurrency, width=0.4, color=[nature_red, nature_blue])
ax.set_ylabel('Max Concurrent Users (100k context)')
ax.set_title('Concurrency Capacity (Less is NOT more)')

# Add value labels
for i, v in enumerate(concurrency):
    ax.text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('notes/fig_concurrency.png', dpi=300)
plt.close()

# Plot 2: Market Positioning Bubble Chart
fig, ax = plt.subplots(figsize=(7, 5))
# x: Deployability / Edge readiness (0-10)
# y: Context Mastery / Recall (0-10)
# size: Inference Cost

models = ['Claude 3.5\n(Cloud Only)', 'Llama 3 70B\n(Server)', 'Mamba\n(Server/Local)', 'MT-LNN\n(Edge/Local/Cloud)']
x = [1, 2, 7, 9.5]
y = [9.5, 7, 8, 9.5]
s = [4000, 2000, 500, 100] # bubble size represents cost
colors = [nature_red, nature_grey, '#a0c4ff', nature_blue]

scatter = ax.scatter(x, y, s=s, c=colors, alpha=0.7, edgecolors='none')

for i, txt in enumerate(models):
    ax.annotate(txt, (x[i], y[i]), xytext=(10,-10), textcoords='offset points')

ax.set_xlim(0, 11)
ax.set_ylim(0, 11)
ax.set_xlabel('Edge Deployability & Low Cost')
ax.set_ylabel('Infinite Context Mastery')
ax.set_title('LLM Landscape Positioning\n(Bubble Size = Inference Cost)')

plt.tight_layout()
plt.savefig('notes/fig_landscape.png', dpi=300)
plt.close()
