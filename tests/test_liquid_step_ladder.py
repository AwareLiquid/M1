"""液态步长 τ 阶梯 (liquid_step_ladder, iter/latent-recursion Task 4) 的契约测试。

锁定 (沿用仓库 N=1 位等价测试模式):
  1. 零新参数 — 开关不改变参数集 (对 A/B 可比性至关重要)。
  2. 默认 off 全路径不进 ladder 分支; ON + N=1 与 OFF 逐位一致
     (第 0 次迭代无偏置 — "off 时位等价" 的强形式)。
  3. ON + N>1 (core 与 stack 两循环体) 确实改变输出。
  4. ladder_scale_bias 纯函数精确数学 + 快/慢 τ 的方向性。
  5. config 校验: ladder 开启时 kappa 必须 > 0。

Run:  python -m pytest tests/test_liquid_step_ladder.py -v  (全 CPU, <1min)
"""

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn import MTLNNConfig, MTLNNModel

# ladder_scale_bias 随 PR #9(iter/latent-recursion)合入;test 文件先行入库(ca60978)。
# 源码缺席时跳过整个模块,PR #9 合入后自动恢复。
try:
    from mt_lnn.mt_lnn_layer import ladder_scale_bias
    _HAS_LADDER = True
except ImportError:
    _HAS_LADDER = False
    ladder_scale_bias = None

pytestmark = pytest.mark.skipif(
    not _HAS_LADDER,
    reason="ladder_scale_bias 不在 main(mt_lnn/mt_lnn_layer.py)——随 PR #9 合入后恢复",
)


def _cfg(**overrides):
    kw = dict(
        vocab_size=200, max_seq_len=64, d_model=128, n_layers=2,
        n_heads=4, n_kv_heads=2, d_head=32, dropout=0.0,
        attention_dropout=0.0,
    )
    kw.update(overrides)
    return MTLNNConfig(**kw)


def _ids(seed=0, B=2, T=32, vocab=200):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (B, T), generator=g)


def _pair(core_iters, **ladder_kw):
    """同 seed 的 off/on 模型对 (on 加载 off 的权重 — 零新参数所以 strict)。"""
    torch.manual_seed(7)
    off = MTLNNModel(_cfg(core_iterations=core_iters)).eval()
    torch.manual_seed(7)
    on = MTLNNModel(_cfg(core_iterations=core_iters,
                         **ladder_kw)).eval()
    on.load_state_dict(off.state_dict())
    return off, on


def test_ladder_adds_no_parameters():
    off, on = _pair(4, liquid_step_ladder=True)
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    assert n_off == n_on
    assert on.blocks[0].lnn.resonance.ladder is True
    assert off.blocks[0].lnn.resonance.ladder is False


def test_ladder_on_depth1_bit_identical_to_off():
    off, on = _pair(4, liquid_step_ladder=True)
    ids = _ids()
    off.set_core_iterations(1)
    on.set_core_iterations(1)
    with torch.no_grad():
        assert torch.equal(off(ids)["logits"], on(ids)["logits"]), \
            "第 0 次迭代无偏置 — ON + N=1 必须与 OFF 逐位一致"


def test_ladder_changes_output_core_depth4():
    off, on = _pair(4, liquid_step_ladder=True)
    ids = _ids()
    off.set_core_iterations(4)
    on.set_core_iterations(4)
    with torch.no_grad():
        assert not torch.equal(off(ids)["logits"], on(ids)["logits"]), \
            "τ 阶梯在 core 深度 4 必须改变输出"


def test_ladder_changes_output_stack_depth4():
    off = MTLNNModel(_cfg()).eval()
    torch.manual_seed(7)
    on = MTLNNModel(_cfg(liquid_step_ladder=True)).eval()
    on.load_state_dict(off.state_dict())
    ids = _ids()
    off.set_stack_iterations(4)
    on.set_stack_iterations(4)
    with torch.no_grad():
        o_off = off(ids)["logits"]
        o_on = on(ids)["logits"]
    assert not torch.equal(o_off, o_on), "τ 阶梯在 stack 深度 4 必须改变输出"
    # stack=1 时 ON 回到逐位一致 (pass 0 无偏置)
    off.set_stack_iterations(1)
    on.set_stack_iterations(1)
    with torch.no_grad():
        assert torch.equal(off(ids)["logits"], on(ids)["logits"])


def test_ladder_scale_bias_math():
    tau = torch.tensor([[1.0, 100.0]])
    assert torch.equal(ladder_scale_bias(tau, 1, 1.0),
                       torch.tensor([[-0.5, -50.0]]))
    assert torch.equal(ladder_scale_bias(tau, 0, 1.0),
                       torch.tensor([[-1.0, -100.0]]))  # 纯函数仍可算 k=0
    # 方向性: 快 τ 偏置浅 (多走), 慢 τ 偏置深 (少走); k 增大趋平 (回归 stock)
    fast = ladder_scale_bias(tau, 1, 1.0)[0, 0]
    slow = ladder_scale_bias(tau, 1, 1.0)[0, 1]
    assert fast > slow
    assert abs(ladder_scale_bias(tau, 99, 1.0)[0, 1]) < abs(slow)


def test_config_rejects_bad_kappa():
    with pytest.raises(ValueError):
        _cfg(liquid_step_ladder=True, liquid_step_kappa=0.0)
    _cfg(liquid_step_kappa=0.0)          # ladder 关闭时 kappa 不校验
    _cfg(liquid_step_ladder=True, liquid_step_kappa=0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
