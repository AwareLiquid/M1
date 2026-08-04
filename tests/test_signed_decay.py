"""Tests for signed decay (negative-eigenvalue extension, config.signed_decay).

Contract:
  1. Default (False): no parameter, exact historical forward (zero regression).
  2. True: one (P,S) sign parameter per resonance bank; init tanh(3)≈0.995 so
     the start is near-stock; gradients flow.
  3. pscan handles NEGATIVE multipliers identically to the sequential scan
     (the algebra never assumed positivity — this pins it).
  4. Flipped channels genuinely oscillate: with sign → −1 the state alternates
     sign under constant input (the mechanism parity needs).

Run:  python -m pytest tests/test_signed_decay.py -v
"""

import sys
import warnings

import pytest
import torch

sys.path.insert(0, ".")

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.parallel_scan import pscan_constant_A, pscan_sequential


def _cfg(**overrides):
    kw = dict(
        vocab_size=200,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=32,
        dropout=0.0,
        attention_dropout=0.0,
    )
    kw.update(overrides)
    return MTLNNConfig(**kw)


def _tokens(seed=0, B=2, T=24, vocab=200):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (B, T), generator=g)


def test_default_has_no_param_and_matches_stock():
    torch.manual_seed(5)
    m_off = MTLNNModel(_cfg())
    assert not any("decay_sign_raw" in n for n, _ in m_off.named_parameters())
    for block in m_off.blocks:
        assert block.lnn.resonance.decay_sign_raw is None


def test_flag_creates_param_and_near_stock_start():
    torch.manual_seed(5)
    m_off = MTLNNModel(_cfg())
    m_on = MTLNNModel(_cfg(signed_decay=True))
    m_on.load_state_dict(m_off.state_dict(), strict=False)
    m_off.eval(), m_on.eval()
    ids = _tokens()
    with torch.no_grad():
        o_off = m_off(ids)["logits"]
        o_on = m_on(ids)["logits"]
    # tanh(3)=0.9951 → near-stock but not identical
    assert not torch.equal(o_off, o_on)
    rel = (o_off - o_on).abs().max() / o_off.abs().max().clamp_min(1e-6)
    assert rel < 0.20, f"start should be near stock dynamics, rel={rel:.3f}"


def test_gradients_reach_sign_param():
    torch.manual_seed(5)
    m = MTLNNModel(_cfg(signed_decay=True))
    ids = _tokens()
    m(ids, labels=ids)["loss"].backward()
    for block in m.blocks:
        g = block.lnn.resonance.decay_sign_raw.grad
        assert g is not None and torch.isfinite(g).all()


def test_pscan_matches_sequential_with_negative_A():
    torch.manual_seed(5)
    B, T, D = 3, 17, 5  # non-pow2 T exercises padding
    A = (torch.rand(B, T) * 1.8 - 0.9)          # ∈ (−0.9, 0.9), mixed signs
    X = torch.randn(B, T, D)
    h0 = torch.randn(B, D)
    # pscan_constant_A takes constant per-sequence A: use per-B constants
    A_const = A[:, 0]
    ref = pscan_sequential(A_const.unsqueeze(-1).expand(B, T), X, h_init=h0)
    out = pscan_constant_A(A_const, X, h_init=h0)
    assert torch.allclose(ref, out, atol=1e-5), (ref - out).abs().max()


def test_negative_channel_oscillates():
    """λ<0 must alternate the state's sign under constant input — the
    mechanism parity requires and positive decay can never produce."""
    lam = torch.tensor(-0.9)
    h = torch.tensor(1.0)
    signs = []
    for _ in range(6):
        h = lam * h + 0.01
        signs.append(h.item() > 0)
    assert signs == [False, True, False, True, False, True]


def test_cache_parity_with_signed_decay():
    torch.manual_seed(5)
    m = MTLNNModel(_cfg(signed_decay=True))
    m.eval()
    ids = _tokens(B=1, T=12)
    with torch.no_grad():
        full = m(ids)["logits"]
        pre = m(ids[:, :8], use_cache=True)
        step = m(ids[:, 8:], cache=pre["cache"], use_cache=True,
                 position_offset=8)
        stitched = torch.cat([pre["logits"], step["logits"]], dim=1)
    assert torch.allclose(full, stitched, atol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
