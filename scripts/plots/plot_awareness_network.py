import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Arial']

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis('off')

cloud_color = "#e0f2fe"
cloud_edge = "#0284c7"
cloud_box = "#bae6fd"
cloud_text = "#0369a1"

edge_color = "#fdf2f8"
edge_edge = "#be185d"
edge_box = "#fce7f3"
edge_text = "#831843"
m1_box = "#ffe4e6"
m1_edge = "#e11d48"

cloud = patches.FancyBboxPatch((7.0, 3.2), 3.5, 2.5, boxstyle="round,pad=0.2", ec=cloud_edge, fc=cloud_color, alpha=0.9, lw=2)
ax.add_patch(cloud)
ax.text(8.75, 5.5, "Cloud (Knowledge Oracle)", ha="center", va="center", fontsize=14, fontweight='bold', color=cloud_text)

db = patches.FancyBboxPatch((7.45, 4.3), 2.6, 0.7, boxstyle="round,pad=0.08", ec=cloud_edge, fc=cloud_box, lw=1.5)
ax.add_patch(db)
ax.text(8.75, 4.65, "Vector DB (Facts / RAG)", ha="center", va="center", fontsize=11, color=cloud_text)

cloud_model = patches.FancyBboxPatch((7.45, 3.3), 2.6, 0.7, boxstyle="round,pad=0.08", ec=cloud_edge, fc=cloud_box, lw=1.5)
ax.add_patch(cloud_model)
ax.text(8.75, 3.65, "Awareness Cloud\n(MT-LNN 70B+)", ha="center", va="center", fontsize=11, color=cloud_text)

edge = patches.FancyBboxPatch((0.5, 0.5), 5.0, 5.0, boxstyle="round,pad=0.2", ec=edge_edge, fc=edge_color, alpha=0.9, lw=2)
ax.add_patch(edge)
ax.text(3.0, 5.2, "Edge / Local Device (Air-gapped)", ha="center", va="center", fontsize=14, fontweight='bold', color=edge_text)

m1_panel = patches.FancyBboxPatch((0.8, 0.8), 4.4, 4.0, boxstyle="round,pad=0.1", ec=m1_edge, fc=edge_box, lw=1.5)
ax.add_patch(m1_panel)
ax.text(3.0, 4.5, "AwareLiquid (MT-LNN) Engine", ha="center", va="center", fontsize=12, fontweight='bold', color=edge_text)

capsule = patches.FancyBboxPatch((1.05, 3.4), 3.9, 0.75, boxstyle="round,pad=0.05", ec=m1_edge, fc=m1_box, lw=1.5)
ax.add_patch(capsule)
ax.text(3.0, 3.77, "Personal State Capsule (4.1KB $h_{prev}$)\nLifelong memory logic", ha="center", va="center", fontsize=10.5, color=edge_text)

monitor = patches.FancyBboxPatch((1.05, 2.65), 3.9, 0.6, boxstyle="round,pad=0.05", ec=m1_edge, fc=m1_box, lw=1.5)
ax.add_patch(monitor)
ax.text(3.0, 2.95, "Predictive Error Monitor\n(Active UI / UX)", ha="center", va="center", fontsize=10.5, color=edge_text)

loop = patches.FancyBboxPatch((1.05, 1.2), 1.85, 1.3, boxstyle="round,pad=0.05", ec=m1_edge, fc=m1_box, lw=1.5)
ax.add_patch(loop)
ax.text(1.97, 1.85, "Latent Loop\n\n10s Silent\nDeduction", ha="center", va="center", fontsize=10.5, color=edge_text)

router = patches.FancyBboxPatch((3.1, 1.2), 1.85, 1.3, boxstyle="round,pad=0.05", ec=m1_edge, fc=m1_box, lw=1.5)
ax.add_patch(router)
ax.text(4.02, 1.85, "Cloud Oracle\nRouter\n\n(Fact Queries)", ha="center", va="center", fontsize=10.5, color=edge_text)

ax.text(3.0, 1.0, "CPU / NPU inference only", ha="center", va="center", fontsize=9.5, style='italic', color=edge_text)

ax.annotate("", xy=(7.0, 3.65), xytext=(4.95, 1.85), arrowprops=dict(arrowstyle="<|-|>", color="#94a3b8", lw=3.0, ls='dashed', shrinkA=10, shrinkB=5))
ax.text(5.95, 2.75, "1. Submits concise query\n2. Consumes raw facts", ha="center", va="center", fontsize=9.5, rotation=41, color="#334155", fontweight='bold')

plt.savefig("fig_awareness_network.pdf", bbox_inches="tight")
plt.savefig("fig_awareness_network.png", bbox_inches="tight", dpi=300)
print("Saved fig_awareness_network")