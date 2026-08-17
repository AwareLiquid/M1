"""Householder NDIT 在 MTLNNLayer 中的集成测试（M2 里程碑前置）。

运行：py -3.11 -m pytest tests/test_ndit.py -v
"""
import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.mt_lnn_layer import MTLNNLayer


def _cfg(**kw):
    base = dict(
        vocab_size=16, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2,
        d_head=8, max_seq_len=32, n_protofilaments=4, n_time_scales=4,
        dropout=0.0, attention_dropout=0.0,  # deterministic for equality tests
    )
    base.update(kw)
    return MTLNNConfig(**base)


def test_ndit_off_is_zero_regression():
    # use_householder_transition=False → 与默认层输出一致
    torch.manual_seed(0)
    l1 = MTLNNLayer(_cfg())
    torch.manual_seed(0)
    l2 = MTLNNLayer(_cfg(use_householder_transition=False))
    l2.load_state_dict(l1.state_dict())
    x = torch.randn(2, 8, 32)
    o1, _ = l1(x, None, use_scan=True)
    o2, _ = l2(x, None, use_scan=True)
    assert (o1 - o2).abs().max() < 1e-6


def test_ndit_forward_finite():
    # NDIT 开启（+selective）→ 输出有限、形状正确
    torch.manual_seed(1)
    l = MTLNNLayer(_cfg(selective_decay=True, selective_decay_mode="exp",
                        use_householder_transition=True))
    x = torch.randn(2, 8, 32)
    out, h_last = l(x, None, use_scan=True)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()
    assert torch.isfinite(h_last).all()
    assert h_last.shape == (2, 4, 4, 8)   # (B, P, S, D)


def test_ndit_params_have_gradient():
    # NDIT 参数（hh_w/hh_b）有梯度流动
    torch.manual_seed(2)
    l = MTLNNLayer(_cfg(selective_decay=True, use_householder_transition=True))
    x = torch.randn(2, 8, 32)
    out, _ = l(x, None, use_scan=True)
    out.sum().backward()
    assert l.resonance.hh_w.grad is not None
    assert l.resonance.hh_w.grad.abs().sum() > 0
    assert l.resonance.hh_b.grad is not None


def test_ndit_rotation_is_unitary_per_step():
    # 每 token 的 Q_t 酉：|Q_t h| = |h|（Householder 保范，多反射积也酉）
    torch.manual_seed(3)
    l = MTLNNLayer(_cfg(selective_decay=True, use_householder_transition=True))
    res = l.resonance
    x = torch.randn(1, 4, 4, 8)          # (B,T,P,D) proto 输入
    with torch.no_grad():
        h = torch.randn(1, 4, 4, 8)
        rotated = h
        for r in range(res.hh_rank):
            v = torch.einsum("bpd,pde->bpe", x[:, 0], res.hh_w[:, r]) \
                + res.hh_b[:, r]
            v = torch.nn.functional.normalize(v, dim=-1)
            dots = torch.einsum("bpd,bpsd->bps", v, rotated)
            rotated = rotated - 2.0 * dots.unsqueeze(-1) * v.unsqueeze(2)
        # 保范：|Qh| == |h|（每次反射都保范）
        assert (rotated.norm(dim=-1) - h.norm(dim=-1)).abs().max() < 1e-5


def test_ndit_selective_only_requires_both_flags():
    # NDIT 无 selective_decay 时应等价于无 NDIT（hh 未使用）
    torch.manual_seed(4)
    l1 = MTLNNLayer(_cfg())
    torch.manual_seed(4)
    l2 = MTLNNLayer(_cfg(use_householder_transition=True))
    l2.load_state_dict(l1.state_dict(), strict=False)  # hh params extra
    x = torch.randn(2, 8, 32)
    o1, _ = l1(x, None, use_scan=True)
    o2, _ = l2(x, None, use_scan=True)
    assert (o1 - o2).abs().max() < 1e-6
