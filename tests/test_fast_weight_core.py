"""tests/test_fast_weight_core.py — fast-weight 核心化契约（P1-4 / 1a）

DEEP_INTEGRATION_PLAN 1a 的架构钩子。硬契约：
  • config 默认 off → 模型参数集与 forward 输出**逐位不变**（零回归）
  • 开启但未训练（gate=0）→ 输出**逐位不变**（不伤现有 checkpoint 语义）
  • gate>0 → 输出改变，且**因果**（改 t 之后的 token 不影响 ≤t 的输出）
  • gate=0 时 gate 参数仍有梯度通路（零门控陷阱：early-return 会让
    gate 梯度恒 0，永远学不开）
"""

import pytest

torch = pytest.importorskip("torch")

from mt_lnn.config import MTLNNConfig  # noqa: E402
from mt_lnn.fast_weight_core import CoreFastWeight  # noqa: E402
from mt_lnn.model import MTLNNModel  # noqa: E402


def _cfg(**kw):
    base = dict(vocab_size=64, d_model=104, n_layers=2, n_heads=13,
                n_kv_heads=1, d_head=8, max_seq_len=64, gwtb_n_heads=1,
                dropout=0.0, attention_dropout=0.0)
    base.update(kw)
    return MTLNNConfig(**base)


def _ids(seed=1, b=2, t=32):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 64, (b, t), generator=g)


class TestModule:
    def test_zero_gate_output_is_exactly_zero(self):
        torch.manual_seed(0)
        fw = CoreFastWeight(104, rank=8)
        x = torch.randn(2, 16, 104)
        out = fw(x)
        assert out.shape == x.shape
        assert torch.count_nonzero(out) == 0

    def test_open_gate_changes_output(self):
        torch.manual_seed(0)
        fw = CoreFastWeight(104, rank=8)
        with torch.no_grad():
            fw.gate.fill_(1.0)
        x = torch.randn(2, 16, 104)
        assert torch.count_nonzero(fw(x)) > 0

    def test_causal_no_future_leakage(self):
        torch.manual_seed(0)
        fw = CoreFastWeight(104, rank=8, decay=0.9)
        with torch.no_grad():
            fw.gate.fill_(1.0)
        x = torch.randn(1, 24, 104)
        y_full = fw(x)
        # 篡改 t=15 之后的内容, t≤14 的读出必须不变 (因果)
        x2 = x.clone()
        x2[:, 15:] = torch.randn_like(x2[:, 15:])
        y_mut = fw(x2)
        assert torch.equal(y_full[:, :15], y_mut[:, :15])

    def test_zero_gate_has_gradient_path(self):
        """零门控陷阱: gate=0 时 d(out)/d(gate) 必须非零, 否则永远学不开。"""
        torch.manual_seed(0)
        fw = CoreFastWeight(104, rank=8)
        x = torch.randn(2, 8, 104)
        fw(x).sum().backward()
        assert fw.gate.grad is not None
        assert float(fw.gate.grad.abs()) > 0.0

    def test_decay_geometric_matches_explicit_loop(self):
        """cumsum-γ^t 实现与逐步循环参考一致 (数值护栏)。"""
        torch.manual_seed(3)
        r, d, T = 4, 0.8, 10
        fw = CoreFastWeight(16, rank=r, decay=d)
        with torch.no_grad():
            fw.gate.fill_(1.0)
        x = torch.randn(1, T, 16)
        k, v, q = fw.k_proj(x), fw.v_proj(x), fw.q_proj(x)
        got = torch.einsum("bktr,btr->btr", _F_reference(k, v, d), q)
        y = torch.einsum("bktr,btr->btr", _F_cumsum(k, v, d), q)
        assert torch.allclose(got, y, atol=1e-4)


def _F_reference(k, v, decay):
    """逐步循环参考实现: S_t = decay·S_{t-1} + k_t v_t^T。"""
    B, T, r = k.shape
    S = torch.zeros(B, r, r)
    out = []
    for t in range(T):
        S = decay * S + torch.einsum("br,bs->brs", k[:, t], v[:, t])
        out.append(S)
    return torch.stack(out, dim=2)          # (B, r, T, r)


def _F_cumsum(k, v, decay):
    t_idx = torch.arange(k.shape[1], dtype=k.dtype)
    kv = torch.einsum("btk,btv->bktv",
                      k * torch.pow(decay, -t_idx)[None, :, None], v)
    return kv.cumsum(dim=2) * torch.pow(decay, t_idx)[None, None, :, None]


class TestModelIntegration:
    def test_off_is_bit_identical(self):
        torch.manual_seed(0)
        a = MTLNNModel(_cfg())                       # 默认 off
        torch.manual_seed(0)
        b = MTLNNModel(_cfg(fast_weight_core=False))
        ids = _ids()
        with torch.no_grad():
            la = a(ids)["logits"]
            lb = b(ids)["logits"]
        assert torch.equal(la, lb)

    def test_on_untrained_is_bit_identical(self):
        """开启 + gate=0 → 参数集不同但输出逐位一致（残差 +0.0）。"""
        torch.manual_seed(0)
        base = MTLNNModel(_cfg())
        torch.manual_seed(0)
        cored = MTLNNModel(_cfg(fast_weight_core=True))
        ids = _ids()
        with torch.no_grad():
            la = base(ids)["logits"]
            lb = cored(ids)["logits"]
        assert torch.equal(la, lb)

    def test_on_changes_param_count(self):
        torch.manual_seed(0)
        base = MTLNNModel(_cfg())
        torch.manual_seed(0)
        cored = MTLNNModel(_cfg(fast_weight_core=True, fast_weight_rank=8))
        n_base = sum(p.numel() for p in base.parameters())
        n_core = sum(p.numel() for p in cored.parameters())
        assert n_core > n_base
        # 每层 4 个 (D×r) 投影 + 1 个标量 gate
        per_layer = 4 * 104 * 8 + 1
        assert n_core - n_base == per_layer * 2     # n_layers=2

    def test_open_gate_trained_output_diverges(self):
        torch.manual_seed(0)
        cored = MTLNNModel(_cfg(fast_weight_core=True))
        with torch.no_grad():
            for blk in cored.blocks:
                blk.fast_weight.gate.fill_(0.5)
        ids = _ids()
        torch.manual_seed(0)
        base = MTLNNModel(_cfg())
        with torch.no_grad():
            la = base(ids)["logits"]
            lb = cored(ids)["logits"]
        assert not torch.allclose(la, lb)

    def test_backward_reaches_gate(self):
        m = MTLNNModel(_cfg(fast_weight_core=True))
        out = m(_ids(b=1, t=8), labels=_ids(b=1, t=8))
        out["loss"].backward()
        for i, blk in enumerate(m.blocks):
            assert blk.fast_weight.gate.grad is not None, f"block {i}"
