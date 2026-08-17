import matplotlib.pyplot as plt
import numpy as np

# 1. Concurrency Capacity
fig, ax = plt.subplots(figsize=(6, 5))
models = ['Traditional\n(1x A100)', 'MT-LNN\n(1x RTX 4090)']
concurrency = [12, 180]
colors = ['red', '#004c99']

bars = ax.bar(models, concurrency, color=colors, width=0.4)
ax.set_ylabel('Max Concurrent Users (100k context)')
ax.set_title('Concurrency Capacity (Less is NOT more)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 2, int(yval), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('concurrency.png', dpi=300)
plt.close()

# 2. LLM Landscape Positioning
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 11)

# X: Edge Deployability & Low Cost
# Y: Infinite Context Mastery
# Bubble Size: Inference Cost
points = [
    {'name': 'Claude 3.5\n(Cloud Only)', 'x': 1.5, 'y': 9.5, 's': 3000, 'c': '#ff4d4d'},
    {'name': 'Llama 3 70B\n(Server)', 'x': 2.5, 'y': 7.0, 's': 1500, 'c': 'gray'},
    {'name': 'Mamba\n(Server/Local)', 'x': 7.0, 'y': 8.0, 's': 800, 'c': '#b3d9ff'},
    {'name': 'MT-LNN\n(Edge/Local/Cloud)', 'x': 10.0, 'y': 9.5, 's': 200, 'c': '#0066cc'},
]

for p in points:
    ax.scatter(p['x'], p['y'], s=p['s'], c=p['c'], alpha=0.9)
    ax.text(p['x'] + 0.3 + (p['s']**0.5)/150, p['y'], p['name'], va='center')

ax.set_xlabel('Edge Deployability & Low Cost')
ax.set_ylabel('Infinite Context Mastery')
ax.set_title('LLM Landscape Positioning\n(Bubble Size = Inference Cost)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('landscape.png', dpi=300)
plt.close()

print("Charts generated")
