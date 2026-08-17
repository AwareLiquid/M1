"""Tests for the global-head quota (M2 architecture principle #1).

Contract: n_global_heads=0 (default) reproduces the historical ALiBi-style
γ init bit-exactly; n_global_heads=k reserves k tail heads at γ≈1e-3 while
the remaining heads keep the geometric schedule.

Run:  python -m pytest tests/test_global_heads.py -v
"""

import sys
import warnings

import pytest
import torch

sys.path.insert(0, ".")

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

import torch.nn.functional as F

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.mt_attention import MicrotubuleAttention


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


def test_default_matches_historical_schedule():
    """n_global_heads=0 must be bit-exact vs the old geometric schedule."""
    old = 0.1 * (2.0 ** torch.linspace(3.0, -3.0, 4))
    new = MicrotubuleAttention._build_alibi_gamma(4, 0.1, n_global_heads=0)
    assert torch.equal(old, new)


def test_quota_reserves_tail_heads():
    g = MicrotubuleAttention._build_alibi_gamma(4, 0.1, n_global_heads=2)
    assert g.shape == (4,)
    # tail heads are truly global
    assert (g[2:] == MicrotubuleAttention._GLOBAL_HEAD_GAMMA).all()
    # local heads keep a geometric spread over the REMAINING slots
    expect_local = 0.1 * (2.0 ** torch.linspace(3.0, -3.0, 2))
    assert torch.allclose(g[:2], expect_local)


def test_all_global():
    g = MicrotubuleAttention._build_alibi_gamma(4, 0.1, n_global_heads=4)
    assert (g == MicrotubuleAttention._GLOBAL_HEAD_GAMMA).all()


def test_model_gamma_init_respects_quota():
    model = MTLNNModel(_cfg(n_global_heads=2))
    for block in model.blocks:
        gamma = F.softplus(block.attn.gtp_gamma)
        # softplus(raw) round-trips the target within fp tolerance
        assert gamma[2:].max().item() < 2e-3, "tail heads must be global"
        assert gamma[0].item() > 0.1, "head 0 must stay strongly local"


def test_forward_smoke_with_quota():
    torch.manual_seed(0)
    model = MTLNNModel(_cfg(n_global_heads=1))
    ids = torch.randint(0, 200, (2, 16))
    out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"])


def test_config_rejects_bad_quota():
    with pytest.raises(ValueError):
        _cfg(n_global_heads=5)  # > n_heads=4
    with pytest.raises(ValueError):
        _cfg(n_global_heads=-1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
