"""
使用torchinfo生成详细的MT-LNN模型结构
这是Netron的最佳替代方案
"""
from mt_lnn.model import MTLNNModel, MTLNNConfig
import torch
from torchinfo import summary

print("=" * 70)
print("MT-LNN Model Architecture Visualization")
print("=" * 70)

# 创建模型配置
config = MTLNNConfig(
    vocab_size=200,
    n_layers=2,          # 2层便于查看
    d_model=832,
    n_protofilaments=13,
    n_time_scales=5,
    max_seq_len=128
)

print("\nModel Configuration:")
print(f"  Vocab Size:      {config.vocab_size}")
print(f"  Layers:          {config.n_layers}")
print(f"  D_model:         {config.d_model}")
print(f"  Protofilaments:  {config.n_protofilaments}")
print(f"  Time Scales:     {config.n_time_scales}")

# 创建模型
model = MTLNNModel(config)
model.eval()

# 创建示例输入
batch_size = 2
seq_len = 10
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

print(f"\nInput Shape: {tuple(input_ids.shape)}")
print("\n" + "=" * 70)
print("Detailed Model Summary (torchinfo)")
print("=" * 70 + "\n")

# 使用torchinfo生成详细摘要
try:
    summary(
        model,
        input_data=input_ids,
        col_names=["input_size", "output_size", "num_params", "params_percent"],
        row_settings=["var_names"],
        depth=4,
        verbose=1
    )
except Exception as e:
    print(f"Full summary failed: {e}")
    print("\nGenerating simplified summary...\n")

    # 简化版本
    summary(
        model,
        input_size=(batch_size, seq_len),
        dtypes=[torch.long],
        col_names=["output_size", "num_params"],
        depth=3,
        verbose=0
    )

print("\n" + "=" * 70)
print("Key Architectural Features")
print("=" * 70)

features = [
    "Liquid Neural Network: 13x5 = 65 parallel LTC channels",
    "Time Constants: tau in [0.01, 10.0]",
    "GTP Hydrolysis: Periodic renewal every 256 tokens",
    "Lateral Coupling: RMC-style attention between protos",
    "GWTB Bottleneck: 8x compression for consciousness simulation",
    "O(1) Working Memory: Exponential decay KV cache",
    "Parallel Scan: O(log N) recurrence computation",
]

for i, feature in enumerate(features, 1):
    print(f"  {i}. {feature}")

print("\n" + "=" * 70)
print("Generated Visualizations")
print("=" * 70)
print("\n  Available in project root:")
print("    - fig_architecture.png      (Complete architecture)")
print("    - fig_microtubules.png      (Biological mapping)")
print("    - fig_awareness_network.png (System design)")
print("\n  View with:")
print("    Windows: start fig_architecture.png")
print("    Mac:     open fig_architecture.png")
print("    Linux:   xdg-open fig_architecture.png")

print("\n" + "=" * 70)
