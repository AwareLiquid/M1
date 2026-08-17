import matplotlib.pyplot as plt
import numpy as np

# Data
context_length = np.linspace(0, 1000, 100) # x1000 Tokens
# Transformer cost (O(N^2) or slightly quadratic/linear mix)
# The chart shows a quadratic-like curve starting flat then shooting up
transformer_cost = context_length**1.8 * 0.004

# MT-LNN cost (O(1) memory)
mt_lnn_cost = np.ones_like(context_length) * 10

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(context_length, transformer_cost, 'r-', linewidth=2.5, label='Transformer (KV Cache)')
ax.plot(context_length, mt_lnn_cost, '#1f77b4', linewidth=2.5, label='MT-LNN (Working Memory)') # standard blue line

ax.set_title('Cost Explosion vs. Perfect Scaling')
ax.set_xlabel('Context Length (x1000 Tokens)')
ax.set_ylabel('Memory / Compute Cost')

# Adjust ticks to match screenshot: 0, 200, 400, 600, 800, 1000
ax.set_xticks([0, 200, 400, 600, 800, 1000])

# Adjust y limits to look similar
ax.set_ylim(-50, 1050)

# Spines removal like seaborn
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend without box
ax.legend(frameon=False, loc='upper left')

plt.tight_layout()
plt.savefig('cost_explosion.png', dpi=300, bbox_inches='tight')
