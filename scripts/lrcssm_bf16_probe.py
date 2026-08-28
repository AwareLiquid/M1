"""lrcssm_bf16_probe.py — 验证 LrcSSM 评估文档的 bf16 失效假设（CPU）

假设: bf16 下训练 PPL 恶化 3-7.5× 源于液态递推的数值敏感路径。
本脚本直接测量: 同一层 fp32 vs bf16 前向的状态轨迹分歧, 并关联 tau。

用法: py -3.11 scripts/lrcssm_bf16_probe.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    ckpt = torch.load("checkpoints/hybrid_125m_serve.pt", map_location="cpu",
                      mmap=True, weights_only=False)
    import dataclasses
    from mt_lnn.config import MTLNNConfig
    from mt_lnn.model import MTLNNModel

    cfg_dict = ckpt["config"]
    valid = {f.name for f in dataclasses.fields(MTLNNConfig) if f.init}
    cfg = MTLNNConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    model = MTLNNModel(cfg)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    layer = model.blocks[0].lnn
    res = layer.resonance

    # 1. tau 分布
    tau = F.softplus(res.log_tau) + res.tau_min
    tau_c = tau.clamp(res.tau_min, res.tau_max)
    print(f"[tau] min={tau_c.min():.3f} max={tau_c.max():.3f} "
          f"中位={tau_c.median():.3f} shape={tuple(tau.shape)}")
    print(f"  慢通道 (tau>10 帧) 占比: {(tau_c>10).float().mean():.1%}")

    # 2. 前向: fp32 vs bf16, 测输出与状态轨迹分歧
    d_model = cfg.d_model
    x = torch.randn(2, 64, d_model)  # (B, T, d_model)

    import copy
    layer16 = copy.deepcopy(layer).to(torch.bfloat16)

    with torch.no_grad():
        out32, h32 = layer(x, None, position_offset=0, use_scan=True)
        out16, h16 = layer16(x.to(torch.bfloat16), None, position_offset=0,
                             use_scan=True)

    rel_out = (out32 - out16).norm() / out32.norm()
    rel_h = 0.0
    if isinstance(h32, torch.Tensor) and isinstance(h16, torch.Tensor):
        rel_h = (h32 - h16).norm() / h32.norm()
    print(f"\n[前向分歧] 输出相对误差 {rel_out.item():.4f} | "
          f"末态 h 相对误差 {rel_h:.4f}")

    # 3. 更细: 误差 vs 序列长度 (bf16 漂移是否随 T 累积)
    drift = []
    with torch.no_grad():
        for T in (4, 16, 64):
            xT = torch.randn(2, T, d_model)
            o32, _ = layer(xT, None, position_offset=0, use_scan=True)
            o16, _ = layer16(xT.to(torch.bfloat16), None, position_offset=0,
                             use_scan=True)
            err = ((o32.float() - o16.float()).norm() / o32.norm()).item()
            drift.append((T, err))
    print("[误差 vs 序列长度] " + " | ".join(f"T={t}: {e:.4f}" for t, e in drift))

    # 4. bf16 下的梯度检查 (LrcSSM 稳定性三保证对应项):
    #    单步收缩性: |h_{t+1} - h_t| 是否随 t 递减
    with torch.no_grad():
        o, h_last = layer(x, None, position_offset=0, use_scan=True)
    print(f"\n[末态 h] norm={h_last.norm():.4f} "
          f"(有界性检查: 有限={torch.isfinite(h_last).all().item()})")

    print("\n结论提示: 若输出相对误差 >0.05, 说明 bf16 数值路径确实严重损伤")
    print("液态递推; 下一步 = lam_t/衰减计算局部 fp32 修复 (见 M3 docs/lrcssm-evaluation.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
