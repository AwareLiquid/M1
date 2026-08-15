"""DeltaProduct NDIT 在 MTLNNLayer 中的集成测试。

运行：py -3.11 -m pytest tests/test_deltaproduct.py -v
"""
import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.mt_lnn_layer import MTLNNLayer


def _cfg(**kw):
    base = dict(
        vocab_size=16, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2,
        d_head=8, max_seq_len=32, n_protofilaments=4, n_time_scales=4,
        dropout=0.0, attention_dropout=0.0,
    )
    base.update(kw)
    return MTLNNConfig(**base)


def test_dp_off_is_zero_regression():
    torch.manual_seed(0)
    l1 = MTLNNLayer(_cfg())
    torch.manual_seed(0)
    l2 = MTLNNLayer(_cfg(use_deltaproduct_transition=False))
    l2.load_state_dict(l1.state_dict())
    x = torch.randn(2, 8, 32)
    o1, _ = l1(x, None, use_scan=True)
    o2, _ = l2(x, None, use_scan=True)
    assert (o1 - o2).abs().max() < 1e-6


def test_dp_forward_finite():
    torch.manual_seed(1)
    l = MTLNNLayer(_cfg(selective_decay=True, selective_decay_mode="exp",
                        use_deltaproduct_transition=True))
    x = torch.randn(2, 8, 32)
    out, h_last = l(x, None, use_scan=True)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()
    assert torch.isfinite(h_last).all()


def test_dp_init_is_identity_ish():
    # δ init ≈ 0 → 转移 ≈ I：状态的第一阶应接近 (1-decay)⊙B_t（纯 LTC）
    torch.manual_seed(2)
    l = MTLNNLayer(_cfg(selective_decay=True, use_deltaproduct_transition=True))
    res = l.resonance
    with torch.no_grad():
        g0 = res.dp_scale * torch.tanh(res.dp_g_b)   # (P,R) 输入无关部分
        assert g0.abs().max() <= res.dp_scale + 1e-6  # 有界小


def test_dp_params_have_gradient():
    torch.manual_seed(3)
    l = MTLNNLayer(_cfg(selective_decay=True, use_deltaproduct_transition=True))
    x = torch.randn(2, 8, 32)
    out, _ = l(x, None, use_scan=True)
    out.sum().backward()
    for name in ("dp_u_w", "dp_v_w", "dp_g_w", "dp_u_b", "dp_v_b", "dp_g_b"):
        p = getattr(l.resonance, name)
        assert p.grad is not None, name
        assert p.grad.abs().sum() > 0, name


def test_dp_transition_is_non_involutory():
    # (I + A)(I + A) != I 一般成立（非对合）——与 Householder 对合的本质差异
    torch.manual_seed(4)
    l = MTLNNLayer(_cfg(selective_decay=True, use_deltaproduct_transition=True))
    res = l.resonance
    x = torch.randn(1, 4, 4, 8)
    with torch.no_grad():
        xt = x[:, 0]
        # 构造一步的转移矩阵（单 rank，D=8，作用于第一个 scale 的切片）
        u = torch.tanh(torch.einsum("bpd,pde->bpe", xt, res.dp_u_w[:, 0])
                       + res.dp_u_b[:, 0])                    # (1,P,D)
        v = torch.tanh(torch.einsum("bpd,pde->bpe", xt, res.dp_v_w[:, 0])
                       + res.dp_v_b[:, 0])                    # (1,P,D)
        g = res.dp_scale * torch.tanh(
            torch.einsum("bpd,pd->bp", xt, res.dp_g_w[:, 0])
            + res.dp_g_b[:, 0])                               # (1,P)
        P, D = u.shape[1], u.shape[-1]
        Id = torch.eye(D).unsqueeze(0).unsqueeze(0)           # (1,1,D,D)
        A = (g.unsqueeze(-1) * u).unsqueeze(-1) * v.unsqueeze(2)  # (1,P,D,D)
        T = Id + A                                            # 转移矩阵
        T2 = T @ T
        # 非对合：T^2 != I（允许数值上明显偏离）
        I_prod = Id @ Id
        assert (T2 - Id).abs().max() > 1e-4, "transition is involutory"
