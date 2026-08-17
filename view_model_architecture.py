"""
生成MT-LNN架构的可视化摘要
无需ONNX，直接展示模型结构
"""
from mt_lnn.model import MTLNNModel, MTLNNConfig
import torch

print("=" * 70)
print(" " * 20 + "MT-LNN Architecture Viewer")
print("=" * 70)

# 创建模型
config = MTLNNConfig(
    vocab_size=200,
    n_layers=2,
    d_model=832,
    n_protofilaments=13,
    n_time_scales=5,
    max_seq_len=128
)

model = MTLNNModel(config)
model.eval()

# 统计参数
def count_parameters(module, name=""):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

print("\n📊 Model Overview")
print("-" * 70)
print(f"Model Type:       MT-LNN (Microtubule Liquid Neural Network)")
print(f"Layers:           {config.n_layers}")
print(f"D_model:          {config.d_model} = {config.n_protofilaments} protos × {config.d_proto}")
print(f"Protofilaments:   {config.n_protofilaments} (biological constant)")
print(f"Time Scales:      {config.n_time_scales}")
print(f"Vocab Size:       {config.vocab_size}")
print(f"Max Seq Length:   {config.max_seq_len}")

total, trainable = count_parameters(model)
print(f"\nTotal Parameters: {total:,}")
print(f"Trainable:        {trainable:,}")
print(f"Model Size:       {total * 4 / 1024 / 1024:.2f} MB (fp32)")

print("\n" + "=" * 70)
print("📦 Layer-by-Layer Breakdown")
print("=" * 70)

# 1. Embedding
emb_total, _ = count_parameters(model.embedding)
print(f"\n[1] Embedding Layer")
print(f"    Token Embed:    ({config.vocab_size}, {config.d_model})")
print(f"    Parameters:     {emb_total:,}")

# 2. Transformer Blocks
print(f"\n[2] MTLNNBlock × {config.n_layers}")
for i, block in enumerate(model.blocks):
    block_total, _ = count_parameters(block)
    attn_total, _ = count_parameters(block.attn)
    lnn_total, _ = count_parameters(block.lnn)

    print(f"\n    Block {i+1}:")
    print(f"      Attention:        {attn_total:,} params")
    print(f"        ├─ Q/K/V proj:  {config.n_heads} heads × {config.d_head}d")
    print(f"        └─ Output proj: {config.d_model}d")

    print(f"      MT-DL:            {lnn_total:,} params")
    print(f"        ├─ Resonance:   {config.n_protofilaments}×{config.n_time_scales} LTC channels")
    print(f"        ├─ Lateral:     RMC-style coupling")
    print(f"        └─ MAP Gates:   {config.n_protofilaments} MLPs")

    print(f"      Total:            {block_total:,} params")

# 3. GWTB
if hasattr(model, 'gwtb') and model.gwtb is not None:
    gwtb_total, _ = count_parameters(model.gwtb)
    d_gw = config.d_model // config.gwtb_compression_ratio
    print(f"\n[3] GWTB (Global Workspace)")
    print(f"    Compress:       {config.d_model} → {d_gw} (÷{config.gwtb_compression_ratio})")
    print(f"    Workspace SA:   {config.gwtb_n_heads} heads")
    print(f"    Broadcast:      {d_gw} → {config.d_model}")
    print(f"    Parameters:     {gwtb_total:,}")

# 4. Coherence
if hasattr(model, 'coherence') and model.coherence is not None:
    coh_total, _ = count_parameters(model.coherence)
    print(f"\n[4] Global Coherence (O(1) WM)")
    print(f"    Sparsity:       {config.coherence_sparsity:.1%} (top-k attention)")
    print(f"    Decay WM:       {'Enabled' if config.use_decay_wm else 'Disabled'}")
    print(f"    Parameters:     {coh_total:,}")

# 5. LM Head
lm_head_total, _ = count_parameters(model.lm_head)
print(f"\n[5] LM Head")
print(f"    Linear:         ({config.d_model}, {config.vocab_size})")
print(f"    Weight Tied:    Yes (shares with embedding)")
print(f"    Parameters:     {lm_head_total:,}")

print("\n" + "=" * 70)
print("🔬 Architectural Highlights")
print("=" * 70)

highlights = [
    ("Liquid Neural Network", f"{config.n_protofilaments}×{config.n_time_scales} = {config.n_protofilaments * config.n_time_scales} parallel LTC channels"),
    ("Time Constant Range", f"τ ∈ [{config.tau_min}, {config.tau_max}], dt={config.dt}"),
    ("GTP Hydrolysis", f"Period = {config.gtp_period} tokens, γ={config.gamma_init}"),
    ("Parallel Scan", "O(log N) recurrence via tree-based scan"),
    ("Lateral Coupling", "B-lattice bonds between protofilaments"),
    ("GWTB Bottleneck", f"Consciousness-inspired 8× compression"),
    ("O(1) Working Memory", "Exponential decay KV cache"),
]

for name, desc in highlights:
    print(f"  • {name:20s} {desc}")

print("\n" + "=" * 70)
print("🎨 Visualization Options")
print("=" * 70)
print("Since ONNX export has compatibility issues with parallel scan,")
print("here are alternative visualization methods:\n")

print("1. ✅ View Static Diagrams (Already Generated)")
print("   » fig_architecture.png      - Complete architecture")
print("   » fig_microtubules.png      - Biological mapping")
print("   » fig_awareness_network.png - System architecture")
print()

print("2. 🔍 TorchInfo Summary")
print("   » pip install torchinfo")
print("   » python view_model_torchinfo.py")
print()

print("3. 📊 Manual GraphViz")
print("   » pip install graphviz")
print("   » python generate_architecture_graph.py")
print()

print("4. 🌐 llm-viz (Requires TypeScript Modifications)")
print("   » See: llm-viz-integration/README.md")

print("\n" + "=" * 70)
print("💡 Recommendation: View the generated PNG diagrams!")
print("=" * 70)
print("\nRun: start fig_architecture.png  (Windows)")
print("Or:  open fig_architecture.png   (Mac/Linux)")
print()
