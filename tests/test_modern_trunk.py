"""tests/test_modern_trunk.py — modern-trunk 三缺失件的零回归与数学性质.

覆盖契约（Clean-code 契约: 每组 位等价 / 数学性质 / 易回归点, 全 CPU <2min）:
  TestFfnSwiglu          — Task A1 ffn_swiglu:  off 参数集逐位不变 + on 同 seed
                           trunk/forward 逐位等价 (w2 零初始化恒等式)、d_ff
                           取整规则、活梯度、FFN 真实贡献回归点
  TestQKNorm             — Task A2 qk_norm:     off 零参数、同 seed trunk 逐位
                           一致、极端输入 logit 上界 (不达 fp16 溢出量级)、
                           KV-cache prefill/decode parity 回归点
  TestScaledResidualInit — Task A3 scaled_residual_init: off 位等价、on 与
                           off 的精确倍率关系、RNG 隔离、默认路径易回归点

⚠ Phase A (worktree) 只写不跑 — 本文件在 Phase B 合并后于主 worktree 验证。
"""

import math

import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel

# fp16 最大可表示数 — QK logits 超过它即溢出为 inf (2026-07 事故的故障类)
FP16_MAX = 65504.0


def _cfg(**kw):
    base = dict(vocab_size=128, max_seq_len=64, d_model=104, n_layers=4,
                n_heads=4, n_kv_heads=2, d_head=26, gwtb_n_heads=1,
                dropout=0.0, attention_dropout=0.0)
    base.update(kw)
    return MTLNNConfig(**base)


def _ids(batch=2, seq=16):
    return torch.randint(0, 128, (batch, seq))


class TestFfnSwiglu:
    def test_off_default_builds_no_ffn(self):
        """off 位等价之一: 默认参数集里不存在任何 FFN 参数。"""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg())
        assert m.blocks[0].ffn is None and m.blocks[0].ffn_norm is None
        assert not any(".ffn" in n for n in m.state_dict())

    def test_on_off_same_seed_bitwise(self):
        """off 位等价之二 (任务指定的写法): 同 seed 构造 on/off 两份模型,
        共享 trunk 参数逐位一致; w2 零初始化 → init forward 输出逐位一致。"""
        torch.manual_seed(0)
        off = MTLNNModel(_cfg())
        torch.manual_seed(0)
        on = MTLNNModel(_cfg(ffn_swiglu=True))
        off_d, on_d = off.state_dict(), on.state_dict()
        ffn_keys = {k for k in on_d if ".ffn" in k}
        assert len(ffn_keys) == 4 * 4          # w1/w2/w3 + ffn_norm, ×4 层
        assert set(off_d) == set(on_d) - ffn_keys
        assert all(torch.equal(off_d[k], on_d[k]) for k in off_d)
        off.eval(); on.eval()
        x = _ids()
        with torch.no_grad():
            assert torch.equal(off(x)["logits"], on(x)["logits"])

    def test_d_ff_uses_baseline_rounding_rule(self):
        """数学性质: d_ff = round(expansion·d_model/256)·256, 与 modern
        baseline (scaling_comparison.py) 同规则; 832×8/3 → 2304。"""
        assert MTLNNConfig(d_model=832).d_ff == 2304
        assert MTLNNConfig(d_model=832, ffn_expansion=4.0).d_ff == 3328
        # d_head=8: 裸 d_model=104 会撞 config 的 d_head==d_model//n_heads 断言
        assert MTLNNConfig(d_model=104, d_head=8).d_ff == 256   # 8/3·104=277 → 256

    def test_w2_zero_init_with_live_gradient(self):
        """开关数学性质: w2 零初始化 (恒等残差) 但第一步就有非零梯度 —
        分支学得开, 不是死支路。"""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg(ffn_swiglu=True))
        for blk in m.blocks:
            assert torch.count_nonzero(blk.ffn.w2.weight).item() == 0
            assert torch.count_nonzero(blk.ffn.w1.weight).item() > 0
        x = _ids()
        m(x, labels=x)["loss"].backward()
        assert m.blocks[0].ffn.w2.weight.grad is not None
        assert m.blocks[0].ffn.w2.weight.grad.abs().sum().item() > 0

    def test_ffn_actually_contributes_once_open(self):
        """易回归点: FFN 被接线但永不贡献是最危险的静默回归 — 把 w2 填成
        非零后, on 模型输出必须偏离 off 模型。"""
        torch.manual_seed(0)
        off = MTLNNModel(_cfg())
        torch.manual_seed(0)
        on = MTLNNModel(_cfg(ffn_swiglu=True))
        with torch.no_grad():
            for blk in on.blocks:
                blk.ffn.w2.weight.normal_(mean=0.0, std=0.02)
        off.eval(); on.eval()
        x = _ids()
        with torch.no_grad():
            assert not torch.equal(off(x)["logits"], on(x)["logits"])


class TestQKNorm:
    def test_off_default_builds_no_qk_norm(self):
        """off 位等价: 默认无 q_norm/k_norm 参数, 路径关闭。"""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg())
        assert not m.blocks[0].attn.qk_norm
        assert not any("q_norm" in n or "k_norm" in n for n in m.state_dict())

    def test_same_seed_trunk_untouched_by_flag(self):
        """off 位等价之二: ones 初始化不消耗 RNG — 同 seed on/off 的共享
        trunk 参数逐位一致, on 仅多出 per-head d_head 维 norm 权重。"""
        torch.manual_seed(0)
        off = MTLNNModel(_cfg())
        torch.manual_seed(0)
        on = MTLNNModel(_cfg(qk_norm=True))
        off_d, on_d = off.state_dict(), on.state_dict()
        extra = {k for k in on_d if "q_norm" in k or "k_norm" in k}
        assert len(extra) == 4 * 2
        assert on_d["blocks.0.attn.q_norm.weight"].shape == (26,)
        assert set(off_d) == set(on_d) - extra
        assert all(torch.equal(off_d[k], on_d[k]) for k in off_d)

    def test_extreme_input_logits_stay_below_fp16_overflow(self):
        """数学性质 (任务指定): 构造极端 q/k — 未归一化 logits 达到 fp16
        溢出量级, 归一化后有上界 d_head·max|w|², 远离 65504。"""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg(qk_norm=True)).eval()
        attn = m.blocks[0].attn
        x = torch.randn(2, 16, 104) * 1e6      # 极端输入
        with torch.no_grad():
            Q = attn.q_proj(x).view(2, 16, 4, 26).transpose(1, 2)
            K = attn.k_proj(x).view(2, 16, 2, 26).transpose(1, 2)
            # GQA: K 只有 2 个 KV head, 对齐到 4 个 Q head 才能逐 head 点积
            K_rep = K.repeat_interleave(2, dim=1)
            raw = torch.einsum("bhtd,bhsd->bhts", Q, K_rep)
            Qn, Kn = attn.q_norm(Q), attn.k_norm(K)
            normed = torch.einsum("bhtd,bhsd->bhts", Qn,
                                  Kn.repeat_interleave(2, dim=1))
        assert raw.abs().max().item() > FP16_MAX          # 未封顶即事故量级
        w_bound = (attn.q_norm.weight.abs().max()
                   * attn.k_norm.weight.abs().max())
        assert normed.abs().max().item() <= 26 * w_bound + 1e-2
        # RMSNorm 输出的 L2 范数固定 ≤ max|w|·sqrt(d_head) — 上界的来源
        assert Qn.norm(dim=-1).max().item() <= w_bound * math.sqrt(26) + 1e-2

    def test_kv_cache_parity_with_qk_norm(self):
        """易回归点: 归一化后的 K 必须是进 cache 的那份 — prefill+增量
        decode 与全序列 forward 保持 parity (norm 位置放错即此处破裂)。"""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg(qk_norm=True)).eval()
        x = torch.randint(0, 128, (1, 12))
        with torch.no_grad():
            full = m(x)["logits"]
            pre = m(x[:, :8], use_cache=True)
            post = m(x[:, 8:], cache=pre["cache"], use_cache=True)
        assert pre["cache"].token_count == 8
        assert torch.allclose(full[:, 8:], post["logits"], atol=1e-4)


class TestScaledResidualInit:
    def test_off_default_init_untouched(self):
        """off 位等价: 默认两次构建逐位一致, lnn.out_proj 保持历史 std=0.01
        口径 (若误挂到默认路径会缩到 ≈0.0035, 此处即破裂)。"""
        torch.manual_seed(0)
        a = MTLNNModel(_cfg())
        torch.manual_seed(0)
        b = MTLNNModel(_cfg())
        a_d = a.state_dict()
        assert all(torch.equal(a_d[k], b.state_dict()[k]) for k in a_d)
        assert a.blocks[0].lnn.out_proj.weight.std().item() > 0.008

    def test_on_scales_residual_exits_exactly(self):
        """开关数学性质 (任务指定): on 模型的两个残差出口 = off 模型 ×
        1/sqrt(2·n_layers) 精确常数; bias 不动; 其余参数逐位一致。"""
        torch.manual_seed(0)
        off = MTLNNModel(_cfg())
        torch.manual_seed(0)
        on = MTLNNModel(_cfg(scaled_residual_init=True))
        s = 1.0 / math.sqrt(2 * 4)
        for i in range(4):
            assert torch.equal(on.blocks[i].lnn.out_proj.weight,
                               off.blocks[i].lnn.out_proj.weight * s)
            assert torch.equal(on.blocks[i].attn.out_proj.weight,
                               off.blocks[i].attn.out_proj.weight * s)
            assert torch.all(on.blocks[i].lnn.out_proj.bias == 0)
        on_d, off_d = on.state_dict(), off.state_dict()
        assert set(off_d) == set(on_d)          # A3 不新增参数
        # 注意后缀必须精确: lnn.lateral.out_proj (RMC 内部) 不是残差出口,
        # 不得被缩放 — endswith("lnn.out_proj.weight") 恰好把它排除在外。
        exit_weights = {k for k in on_d
                        if k.endswith("attn.out_proj.weight")
                        or k.endswith("lnn.out_proj.weight")}
        assert len(exit_weights) == 8           # 4 层 × (attn + lnn) 出口
        others = (k for k in off_d if k not in exit_weights)
        assert all(torch.equal(off_d[k], on_d[k]) for k in others)

    def test_flag_consumes_no_rng(self):
        """RNG 隔离 (任务指定: 复用 model.py 既有隔离模式): 缩放是确定性
        乘法, 构造 on 模型后全局 RNG 流与 off 完全一致 — 同 seed 后续
        初始化 (更大模型的后续 block/头) 不被移位。"""
        torch.manual_seed(0)
        _ = MTLNNModel(_cfg())
        state_off = torch.get_rng_state()
        torch.manual_seed(0)
        _ = MTLNNModel(_cfg(scaled_residual_init=True))
        assert torch.equal(state_off, torch.get_rng_state())

    def test_on_model_forward_backward_finite(self):
        """易回归点: 缩放后的 init 必须仍然 train 得动 — fwd/bwd 全有限。
        该旋钮的使命是稳住深残差流的 init, 而不是造出一个不收敛的死模型。"""
        torch.manual_seed(0)
        m = MTLNNModel(_cfg(scaled_residual_init=True))
        x = _ids()
        loss = m(x, labels=x)["loss"]
        assert torch.isfinite(loss).item()
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
