"""Tests for J-Space J1: workspace reverberation (docs/JSPACE_DESIGN.md §2).

Contract:
  1. workspace_iterations=1 (default) is bit-exact vs the pre-existing
     single-pass pipeline — zero regression, zero new parameters.
  2. N>1 reverberation changes the output (it is a real computation).
  3. Cache parity: prefill-then-decode still matches full-sequence forward
     at N>1 (the cache stores the final pass's K/V).
  4. set_workspace_iterations covers top-level AND per-block GWTB.

Run:  python -m pytest tests/test_jspace.py -v
"""

import sys
import warnings

import pytest
import torch

sys.path.insert(0, ".")

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel


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


def test_default_is_single_pass_and_paramfree():
    torch.manual_seed(3)
    m1 = MTLNNModel(_cfg())
    mN = MTLNNModel(_cfg(workspace_iterations=3))
    # identical parameter sets (reverberation adds none)
    n1 = sorted(n for n, _ in m1.named_parameters())
    nN = sorted(n for n, _ in mN.named_parameters())
    assert n1 == nN
    # N-built model at 1 pass == 1-built model given identical weights
    mN.load_state_dict(m1.state_dict())
    mN.set_workspace_iterations(1)
    m1.eval(), mN.eval()
    ids = _tokens()
    with torch.no_grad():
        assert torch.equal(m1(ids)["logits"], mN(ids)["logits"])


def test_reverberation_changes_output():
    torch.manual_seed(3)
    m = MTLNNModel(_cfg())
    m.eval()
    ids = _tokens()
    with torch.no_grad():
        o1 = m(ids)["logits"]
        m.set_workspace_iterations(4)
        o4 = m(ids)["logits"]
    assert not torch.equal(o1, o4), "4 reverberation passes should not be a no-op"


def test_cache_parity_at_n2():
    """prefill+decode must match full forward with reverberation on."""
    torch.manual_seed(3)
    m = MTLNNModel(_cfg(workspace_iterations=2))
    m.eval()
    ids = _tokens(B=1, T=12)
    with torch.no_grad():
        full = m(ids)["logits"]
        pre = m(ids[:, :8], use_cache=True)
        step = m(ids[:, 8:], cache=pre["cache"], use_cache=True,
                 position_offset=8)
        stitched = torch.cat([pre["logits"], step["logits"]], dim=1)
    assert torch.allclose(full, stitched, atol=1e-4), (
        (full - stitched).abs().max().item()
    )


def test_grads_flow_through_reverberation():
    torch.manual_seed(3)
    m = MTLNNModel(_cfg(workspace_iterations=3))
    ids = _tokens()
    out = m(ids, labels=ids)
    out["loss"].backward()
    g = m.gwtb.q_proj.weight.grad
    assert g is not None and torch.isfinite(g).all()


def test_setter_guards_and_per_block_coverage():
    m = MTLNNModel(_cfg(gwtb_per_block=True))
    m.set_workspace_iterations(2)
    for block in m.blocks:
        assert block.gwtb.workspace_iterations == 2
    with pytest.raises(ValueError):
        m.set_workspace_iterations(0)


def test_config_rejects_bad_value():
    with pytest.raises(ValueError):
        _cfg(workspace_iterations=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
