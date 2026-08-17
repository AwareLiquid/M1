"""
MT-LNN权重导出脚本 for llm-viz可视化
将MT-LNN模型导出为llm-viz兼容的JSON格式

使用方法:
    python export_mt_lnn_for_viz.py [--checkpoint PATH] [--output PATH]
"""

import torch
import numpy as np
import json
import argparse
import os
from pathlib import Path

def export_mt_lnn_weights(model, output_path="mt_lnn_weights.json", simplify=True):
    """
    导出MT-LNN权重为JSON格式，兼容llm-viz的数据加载器

    Args:
        model: MT-LNN模型实例
        output_path: 输出JSON文件路径
        simplify: 是否简化权重（仅保留关键参数，减小文件大小）
    """
    weights = {}

    print("🔄 Exporting MT-LNN weights...")

    # 1. Embedding层
    print("  ✓ Exporting embedding layer...")
    weights['embedding'] = {
        'token': model.embedding.token_embed.weight.detach().cpu().numpy().tolist(),
        'pos': None  # RoPE不需要位置嵌入表
    }

    # 2. MTLNNBlock层
    print(f"  ✓ Exporting {len(model.blocks)} transformer blocks...")
    weights['blocks'] = []
    for i, block in enumerate(model.blocks):
        print(f"    - Block {i+1}/{len(model.blocks)}")

        block_weights = {
            'layer_idx': i,

            # Attention部分（简化）
            'attention': {
                'n_heads': model.config.n_heads,
                'd_head': model.config.d_head,
                # 如果简化，只保存形状信息
                'q_proj_shape': list(block.attn.q_proj.weight.shape) if simplify else block.attn.q_proj.weight.detach().cpu().numpy().tolist(),
                'k_proj_shape': list(block.attn.k_proj.weight.shape) if simplify else block.attn.k_proj.weight.detach().cpu().numpy().tolist(),
                'v_proj_shape': list(block.attn.v_proj.weight.shape) if simplify else block.attn.v_proj.weight.detach().cpu().numpy().tolist(),
                'o_proj_shape': list(block.attn.o_proj.weight.shape) if simplify else block.attn.o_proj.weight.detach().cpu().numpy().tolist(),
            },

            # MT-DL (Microtubule Dynamic Layer) - 关键可视化数据
            'mt_dl': {
                'n_protofilaments': model.config.n_protofilaments,
                'n_time_scales': model.config.n_time_scales,

                # 多尺度共振 - 保留用于可视化
                'resonance': {
                    'log_tau': block.mt_dl.resonance.log_tau.detach().cpu().numpy().tolist(),  # (13,5)
                    'blend_weights': block.mt_dl.resonance.blend_weights.detach().cpu().numpy().tolist(),  # (13,5)
                    'tau_values': torch.nn.functional.softplus(block.mt_dl.resonance.log_tau).detach().cpu().numpy().tolist(),
                    # W_in太大，只保存统计信息
                    'W_in_stats': {
                        'mean': float(block.mt_dl.resonance.W_in.mean()),
                        'std': float(block.mt_dl.resonance.W_in.std()),
                        'shape': list(block.mt_dl.resonance.W_in.shape)
                    } if simplify else block.mt_dl.resonance.W_in.detach().cpu().numpy().tolist()
                },

                # 运行时状态（如果有缓存）
                'runtime_stats': {
                    'last_scale_gate_mean': block.mt_dl.resonance.last_scale_gate_mean.detach().cpu().numpy().tolist()
                        if hasattr(block.mt_dl.resonance, 'last_scale_gate_mean') else None,
                    'last_active_scale_ratio': float(block.mt_dl.resonance.last_active_scale_ratio)
                        if hasattr(block.mt_dl.resonance, 'last_active_scale_ratio') else None
                }
            },

            # Layer Norm（保留）
            'ln1': {
                'weight': block.ln1.weight.detach().cpu().numpy().tolist(),
                'bias': block.ln1.bias.detach().cpu().numpy().tolist() if block.ln1.bias is not None else None
            },
            'ln2': {
                'weight': block.ln2.weight.detach().cpu().numpy().tolist(),
                'bias': block.ln2.bias.detach().cpu().numpy().tolist() if block.ln2.bias is not None else None
            }
        }
        weights['blocks'].append(block_weights)

    # 3. GWTB (Global Workspace Theory Bottleneck)
    if hasattr(model, 'gwtb') and model.gwtb is not None:
        print("  ✓ Exporting GWTB layer...")
        weights['gwtb'] = {
            'compression_ratio': model.config.gwtb_compression_ratio,
            'd_gw': model.config.d_model // model.config.gwtb_compression_ratio,
            'has_workspace': True
        }

    # 4. Global Coherence Layer
    if hasattr(model, 'coherence') and model.coherence is not None:
        print("  ✓ Exporting global coherence layer...")
        weights['coherence'] = {
            'sparsity': model.config.coherence_sparsity,
            'use_decay_wm': model.config.use_decay_wm,
            'o1_memory': True
        }

    # 5. LM Head（形状信息）
    print("  ✓ Exporting LM head...")
    weights['lm_head'] = {
        'vocab_size': model.config.vocab_size,
        'weight_shape': list(model.lm_head.weight.shape),
        'weight_tied': True  # MT-LNN使用weight tying
    }

    # 6. 配置元数据 - 最重要，用于llm-viz布局计算
    print("  ✓ Exporting config metadata...")
    weights['config'] = {
        # 基础配置
        'vocab_size': model.config.vocab_size,
        'd_model': model.config.d_model,
        'n_layers': model.config.n_layers,
        'n_heads': model.config.n_heads,
        'd_head': model.config.d_head,
        'max_seq_len': model.config.max_seq_len,

        # MT-LNN特有
        'n_protofilaments': model.config.n_protofilaments,
        'n_time_scales': model.config.n_time_scales,
        'd_proto': model.config.d_proto,

        # 时间常数范围
        'tau_min': model.config.tau_min,
        'tau_max': model.config.tau_max,
        'dt': model.config.dt,

        # GTP参数
        'gamma_init': model.config.gamma_init,
        'gtp_period': model.config.gtp_period,

        # GWTB配置
        'gwtb_compression_ratio': model.config.gwtb_compression_ratio,
        'gwtb_per_block': model.config.gwtb_per_block,

        # 架构标识
        'model_type': 'MT-LNN',
        'architecture_version': '2.0'
    }

    # 7. 可视化提示（给llm-viz的渲染提示）
    weights['viz_hints'] = {
        'proto_circular_layout': True,
        'proto_radius': 5.0,
        'scale_height': 0.5,
        'layer_spacing': 8.0,
        'show_lateral_coupling': True,
        'show_gtp_pulse': True,
        'color_scheme': 'viridis'  # tau值颜色映射
    }

    # 保存为JSON
    print(f"\n💾 Saving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)

    file_size_mb = output_path.stat().st_size / 1024 / 1024

    print(f"\n✅ Export complete!")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Vocab: {model.config.vocab_size}")
    print(f"   Layers: {model.config.n_layers}")
    print(f"   D_model: {model.config.d_model}")
    print(f"   Protofilaments: {model.config.n_protofilaments} × {model.config.n_time_scales} scales")

    return output_path

def main():
    parser = argparse.ArgumentParser(description='Export MT-LNN weights for llm-viz')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to MT-LNN checkpoint (.pt file)')
    parser.add_argument('--output', type=str, default='llm-viz-integration/mt_lnn_weights.json',
                       help='Output JSON path')
    parser.add_argument('--vocab-size', type=int, default=200,
                       help='Vocabulary size for demo model')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Number of layers (use 2 for visualization, full model is 12)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Export full weights (warning: large file!)')

    args = parser.parse_args()

    print("=" * 60)
    print("MT-LNN → llm-viz Weight Exporter")
    print("=" * 60)

    # 导入MT-LNN模块
    try:
        from mt_lnn.model import MTLNN, MTLNNConfig
        print("✓ MT-LNN modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MT-LNN: {e}")
        print("   Make sure you're running from the MT-LNN project root")
        return 1

    # 创建或加载模型
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n📂 Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        # 尝试从checkpoint提取配置
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            print("⚠️  No config in checkpoint, using defaults")
            config = MTLNNConfig(
                vocab_size=args.vocab_size,
                n_layers=args.n_layers,
                d_model=832,
                n_protofilaments=13,
                n_time_scales=5
            )

        model = MTLNN(config)

        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
            print("✓ Weights loaded from checkpoint")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("✓ Weights loaded from checkpoint (state_dict)")
        else:
            print("⚠️  No weights in checkpoint, using random initialization")
    else:
        print("\n🎲 Creating demo model with random weights...")
        print(f"   (Use --checkpoint to load trained weights)")
        config = MTLNNConfig(
            vocab_size=args.vocab_size,
            n_layers=args.n_layers,
            d_model=832,
            n_protofilaments=13,
            n_time_scales=5
        )
        model = MTLNN(config)

    model.eval()

    # 导出
    simplify = not args.no_simplify
    if simplify:
        print("\n📦 Using simplified export (shape info only for large tensors)")
        print("   Use --no-simplify for full weights (warning: ~100MB+ per layer)")

    output_path = export_mt_lnn_weights(model, args.output, simplify=simplify)

    # 给出下一步提示
    print("\n" + "=" * 60)
    print("🚀 Next Steps:")
    print("=" * 60)
    print("1. Install llm-viz:")
    print("   git clone https://github.com/bbycroft/llm-viz.git")
    print("   cd llm-viz && npm install")
    print()
    print("2. Copy weights file:")
    print(f"   cp {output_path} llm-viz/public/")
    print()
    print("3. Start llm-viz dev server:")
    print("   cd llm-viz && npm run dev")
    print()
    print("4. Visit: http://localhost:3002/llm")
    print()
    print("5. Follow integration guide in:")
    print("   llm-viz-integration/README.md")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    exit(main())
