"""
快速导出MT-LNN为ONNX格式，供Netron可视化
"""
from mt_lnn.model import MTLNNModel, MTLNNConfig
import torch

print("=" * 60)
print("MT-LNN ONNX Export for Netron Visualization")
print("=" * 60)

# 创建简化的MT-LNN模型（2层，适合可视化）
config = MTLNNConfig(
    vocab_size=200,
    n_layers=2,          # 2层足够展示架构
    d_model=832,
    n_protofilaments=13,
    n_time_scales=5,
    max_seq_len=128
)

print(f"\nModel Configuration:")
print(f"  Vocab size:      {config.vocab_size}")
print(f"  Layers:          {config.n_layers}")
print(f"  D_model:         {config.d_model}")
print(f"  Protofilaments:  {config.n_protofilaments}")
print(f"  Time scales:     {config.n_time_scales}")
print(f"  Max seq len:     {config.max_seq_len}")

print("\nCreating model...")
model = MTLNNModel(config)
model.eval()

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Parameters:")
print(f"  Total:      {total_params:,}")
print(f"  Trainable:  {trainable_params:,}")
print(f"  Size:       {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")

# 创建dummy input
batch_size = 1
seq_len = 10
dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
print(f"\nDummy Input Shape: {tuple(dummy_input.shape)}")

# 导出为ONNX
output_file = "mt_lnn_structure.onnx"
print(f"\nExporting to ONNX: {output_file}")

torch.onnx.export(
    model,
    dummy_input,
    output_file,
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'logits': {0: 'batch', 1: 'sequence'}
    },
    opset_version=14,
    do_constant_folding=True,
    verbose=False
)

import os
file_size_mb = os.path.getsize(output_file) / 1024 / 1024

print(f"\n{'=' * 60}")
print(f"✅ Export Complete!")
print(f"{'=' * 60}")
print(f"  File: {output_file}")
print(f"  Size: {file_size_mb:.2f} MB")
print(f"\nNext Steps:")
print(f"  1. Run: netron {output_file}")
print(f"  2. Or visit: https://netron.app")
print(f"     (drag and drop {output_file})")
print(f"{'=' * 60}")
