"""fast_weight_core.py — 每层第二记忆通道（DEEP_INTEGRATION_PLAN 1a 核心化）

现状与目标
----------
fast-weight 目前是 adapter 的附加件（llama_adapter 的 (F,z)，挂在 frozen
基座外层）。P4 决策门证明"adapter 附加"形式的 LM 增量 ≈ 0，而机制本身有效
（跨会话 recall 0.56 vs 0.000）。1a 的方向：**核心化** —— 每个 MTLNNBlock
自带一个 fast-weight 记忆通道，与 KV cache（工作记忆）平级。

本模块的实现（受 DeltaNet/Titans 路线启发的低秩因果 fast-weight）：

.. math::

    F_t = \\sum_{s \\le t} \\gamma^{t-s} (x_s W_k)(x_s W_v)^\\top   (rank-r)
    y_t = \\tanh(g) \\cdot W_o (F_t (x_t W_q))

* **低秩控制**：k/v 投影到 r 维（默认 16）再外积，写/读都是 O(D·r)，
  不做 O(D²) 全秩 —— 这是 1a 明确要求的成本控制；
* **因果**：F_t 只含 ≤t 的 token（cumsum 实现），无未来泄漏；
* **零门控输出**：`g` 初始化为 0 → 开启但未训练时贡献恰为 +0.0，
  残差流逐位不变（与 rhythm/global-coherence 同款契约，不伤现有
  checkpoint；梯度仍可流入 gate，训练时能自己学会开多大）；
* **窗口内记忆**：F 在每个窗口内从零累积（训练/并行友好）。跨会话持久
  （(F,z) snapshot 进 LayerCache）是下一步 —— 决策门（recall ≥ 0.56 且
  PPL 不退）先要本钩子在 LM 训练里被验证。

边界（诚实）：本模块是架构钩子 + 契约测试；1a 的验收（跨会话记忆基准
recall 0.56+ 不伤 PPL）需要 GPU 训练实验，不是本文件能自证的。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoreFastWeight(nn.Module):
    """Block 级低秩因果 fast-weight 记忆通道。

    Args:
        d_model: 残差流宽度。
        rank: 低秩维度 r（k/v 投影目标）。16 → 每层额外 ~3·D·r 参数。
        decay: 幂衰减 γ（0 < γ ≤ 1；1 = 无衰减的累加外积）。
    """

    def __init__(self, d_model: int, rank: int = 16, decay: float = 0.99):
        super().__init__()
        self.rank = int(rank)
        self.decay = float(decay)
        self.k_proj = nn.Linear(d_model, rank, bias=False)
        self.v_proj = nn.Linear(d_model, rank, bias=False)
        self.q_proj = nn.Linear(d_model, rank, bias=False)
        self.o_proj = nn.Linear(rank, d_model, bias=False)
        # 零门控：g=0 → 贡献恰为 +0.0（位等价）；tanh 有界防早期爆炸。
        # 注意：即便 g=0 也走完整计算路径 —— 否则 gate 梯度恒为 0，
        # 永远学不开（这是零门控模块的通用陷阱）。
        self.gate = nn.Parameter(torch.zeros(()))
        # init_weights(apply) 跳过标记: 本模块的投影由独立生成器初始化
        # (见 MTLNNBlock.__init__ 的 RNG 隔离), 全局 apply 重初始化会覆盖
        # 专用初始化并移位全局 RNG 流 —— 同 seed 的 on/off A/B 就不可比了。
        self._fw_init_done = True
        for _m in (self.k_proj, self.v_proj, self.q_proj, self.o_proj):
            _m._fw_init_done = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → 残差增量 (B, T, D)。gate=0 时为全零张量。"""
        g = torch.tanh(self.gate)
        B, T, _ = x.shape
        k = self.k_proj(x)                                  # (B, T, r)
        v = self.v_proj(x)                                  # (B, T, r)
        q = self.q_proj(x)                                  # (B, T, r)
        # F: (B, r_k, T, r_v) —— 外积 k_t ⊗ v_t 沿时间的因果累积
        if self.decay >= 1.0:
            kv = torch.einsum("btk,btv->bktv", k, v)         # 无衰减
            F = kv.cumsum(dim=2)                            # 因果: 只含 s≤t
        else:
            # 几何衰减: S_t = γ^t · cumsum(γ^{-s} k_s v_s^T)
            # （γ^t·γ^{-s} 在 fp32 下对 t ≤ 2^23 精确）
            t_idx = torch.arange(T, device=x.device, dtype=x.dtype)
            w_fwd = torch.pow(self.decay, -t_idx)           # γ^{-s}, (T,)
            w_back = torch.pow(self.decay, t_idx)           # γ^{t}, (T,)
            kv = torch.einsum("btk,btv->bktv",
                              k * w_fwd[None, :, None], v)
            F = kv.cumsum(dim=2) * w_back[None, None, :, None]
        # 读出: y_t = F[:, :, t, :] @ q_t → (B, T, r_v)
        y = torch.einsum("bktr,btr->btr", F, q)
        return g * self.o_proj(y)

    def extra_repr(self) -> str:
        return (f"rank={self.rank}, decay={self.decay}, "
                f"gate={float(torch.tanh(self.gate).detach()):.4f}")
