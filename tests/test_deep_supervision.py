"""tests/test_deep_supervision.py — 深监督迭代训练栈（P1-5）

ROADMAP_M2 §4.5 round 1/2 的裁决：anytime(随机深度)训练教会模型"无视迭代"，
需要 Geiping/HRM 式深监督才有希望。本文件钉住三件套契约：
  1. 模型暴露逐 stack 迭代 logits（return_stack_iter_logits，默认 off 位等价）
  2. 泊松深度采样（1+Poisson(λ) 截断，低深度占质量，区别于均匀随机）
  3. 逐迭代 CE 深监督 loss（每个迭代都被直接监督）
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mt_lnn.config import MTLNNConfig  # noqa: E402
from mt_lnn.model import MTLNNModel  # noqa: E402
from benchmarks.reasoning_depth import (  # noqa: E402
    deep_supervision_loss, sample_depth,
)


def _tiny():
    torch.manual_seed(0)
    cfg = MTLNNConfig(vocab_size=32, d_model=104, n_layers=2, n_heads=13,
                      n_kv_heads=1, d_head=8, max_seq_len=64, gwtb_n_heads=1)
    return MTLNNModel(cfg)


def test_flag_off_is_bit_identical():
    """默认关闭 = 单收集路径，输出逐位一致（回归门）。"""
    m = _tiny().eval()
    ids = torch.randint(0, 32, (2, 32))
    with torch.no_grad():
        a = m(ids)["logits"]
        b = m(ids, return_stack_iter_logits=False)["logits"]
    assert torch.equal(a, b)


def test_stack_iter_logits_exposed():
    m = _tiny().eval()
    m.set_stack_iterations(3)
    ids = torch.randint(0, 32, (2, 32))
    with torch.no_grad():
        out = m(ids, return_stack_iter_logits=True)
    iters = out["stack_iter_logits"]
    assert iters.shape == (3, 2, 32, 32)   # (n_iter, B, T, V)
    # 读出在 stack 循环内、全局 GWTB 之前 —— 最后一项与最终 logits 高度相关
    # 但不相等（差一个 gated-residual GWTB）；方向必须一致
    cos = torch.nn.functional.cosine_similarity(
        iters[-1].flatten(0, 1), out["logits"].flatten(0, 1), dim=-1)
    assert float(cos.min()) > 0.95


def test_single_iter_returns_no_key():
    m = _tiny().eval()                       # stack_iterations=1 默认
    ids = torch.randint(0, 32, (1, 16))
    with torch.no_grad():
        out = m(ids, return_stack_iter_logits=True)
    assert "stack_iter_logits" not in out    # 单迭代没有深监督对象


def test_sample_depth_poisson_respects_bounds_and_mass():
    rng = np.random.default_rng(0)
    choices = list(range(1, 9))              # 1..8
    ds = [sample_depth(rng, choices, "poisson", 2.0) for _ in range(2000)]
    assert min(ds) >= 1 and max(ds) <= 8
    # λ=2 期望≈3：深度 1-4 应占大头（对比均匀随机的 50%）
    assert sum(1 for d in ds if d <= 4) / len(ds) > 0.7
    # uniform 采样器保持历史行为
    rng2 = np.random.default_rng(0)
    assert sample_depth(rng2, choices, "uniform") in choices


def test_deep_supervision_loss_values():
    m = _tiny().train()
    m.set_stack_iterations(2)
    ids = torch.randint(0, 32, (2, 32))
    labels = ids
    out = m(ids, labels=labels, return_stack_iter_logits=True)
    total, per_iter = deep_supervision_loss(out, labels)
    assert torch.isfinite(total)
    assert per_iter is not None and len(per_iter) == 2
    # 无 stack_iter_logits（如单迭代/core 模式）时回退 out["loss"]
    out1 = m(ids, labels=labels)
    fallback, none = deep_supervision_loss(out1, labels)
    assert none is None and torch.isfinite(fallback)


def test_deep_supervision_backward_flows():
    """深监督 loss 必须可反传（每个迭代都拿到梯度压力）。"""
    m = _tiny().train()
    m.set_stack_iterations(2)
    ids = torch.randint(0, 32, (2, 32))
    out = m(ids, labels=ids, return_stack_iter_logits=True)
    total, _ = deep_supervision_loss(out, ids)
    total.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
