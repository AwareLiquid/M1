import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Nature-style configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.linewidth': 1,
    'text.color': '#333333',
})

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Colors (NMI pastel)
c_brain = '#E8EBF2'
c_mt = '#DCE8D8'
c_lnn = '#EEDCD2'
c_edge = '#72819E'

# Function to draw a node
def draw_node(ax, x, y, width, height, text, color, text_color='black', fontsize=12, fontweight='bold'):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2",
                                  edgecolor=c_edge, facecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    # Handle multiline text with bold tags properly (matplotlib doesn't support html bold easily, so we just use fontweight for the whole text or standard string)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=fontsize, color=text_color, fontweight=fontweight)

# Draw arrows
def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='#72819E', edgecolor='#72819E', width=2, headwidth=8, shrink=0.05))

# Draw the logical flow
draw_node(ax, 0.5, 4.5, 3, 1, "Human Brain\n(Macroscopic)", c_brain)
draw_node(ax, 4.5, 4.5, 4, 1, "Neurons & Synapses\n(Network Level)", c_brain)

draw_arrow(ax, 3.5, 5, 4.5, 5)

draw_node(ax, 4.5, 2.5, 4, 1, "Microtubules\n(Sub-cellular Cytoskeleton)", c_mt)
draw_arrow(ax, 6.5, 4.5, 6.5, 3.5)

draw_node(ax, 0.5, 0.5, 3, 1, "Liquid Neural Network\n(MT-LNN)", c_lnn)
draw_arrow(ax, 4.5, 3, 2.0, 1.5)

# Add descriptions
ax.text(2.5, 2.4, "Bio-inspiration:\nDynamic State Flow", ha='center', va='center', fontsize=9, fontstyle='italic', color='#555555')
ax.text(8.7, 3.8, "Deep down to\nquantum/molecular\nprocessing", ha='left', va='center', fontsize=9, fontstyle='italic', color='#555555')

plt.tight_layout()
plt.savefig('fig_microtubules.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_microtubules.png', dpi=300, bbox_inches='tight')
