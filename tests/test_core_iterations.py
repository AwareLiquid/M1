"""Tests for latent recurrent depth ("thinking steps", M2 P0).

Contract (docs/ROADMAP_M2.md §4 P0-A):
  1. Default config (core_iterations=1) keeps a byte-identical parameter set
     and the exact single-call code path — zero regression.
  2. A model built with core_iterations=N>1 evaluated at depth 1 (via
     set_core_iterations) is bit-identical to the single-pass forward of the
     same weights — the gate parameter is unused at depth 1.
  3. Depth > 1 actually changes the output (state threading is ungated), and
     gradients flow to the feedback gate.
  4. Config validation rejects core_iterations < 1.

Run:  python -m pytest tests/test_core_iterations.py -v
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


def _tokens(seed=0, B=2, T=32, vocab=200):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (B, T), generator=g)


def test_default_config_has_no_gate_param():
    """core_iterations=1 must not add any parameter (byte-identical set)."""
    model = MTLNNModel(_cfg())
    names = [n for n, _ in model.named_parameters()]
    assert not any("core_iter_gate" in n for n in names)
    for block in model.blocks:
        assert block.core_iterations == 1
        assert block.core_iter_gate is None


def test_iter_model_has_one_gate_per_block():
    model = MTLNNModel(_cfg(core_iterations=4))
    gates = [n for n, _ in model.named_parameters() if "core_iter_gate" in n]
    assert len(gates) == model.config.n_layers
    for block in model.blocks:
        assert block.core_iterations == 4
        assert float(block.core_iter_gate.detach()) == 0.0  # zero-init


def test_depth1_parity_with_single_pass_model():
    """N-built model at depth 1 == 1-built model, given identical weights."""
    torch.manual_seed(7)
    m1 = MTLNNModel(_cfg())
    mN = MTLNNModel(_cfg(core_iterations=4))
    # copy shared weights (mN additionally has the unused-at-depth-1 gates)
    mN.load_state_dict(m1.state_dict(), strict=False)
    mN.set_core_iterations(1)
    m1.eval(), mN.eval()

    ids = _tokens()
    with torch.no_grad():
        o1 = m1(ids)["logits"]
        oN = mN(ids)["logits"]
    assert torch.equal(o1, oN), "depth-1 must be bit-identical to single-pass"


def test_depth_changes_output():
    """State threading is ungated, so depth>1 must differ from depth 1."""
    torch.manual_seed(7)
    model = MTLNNModel(_cfg(core_iterations=4))
    model.eval()
    ids = _tokens()
    with torch.no_grad():
        model.set_core_iterations(1)
        o1 = model(ids)["logits"]
        model.set_core_iterations(4)
        o4 = model(ids)["logits"]
    assert not torch.equal(o1, o4), "depth 4 should not be a no-op"


def test_gradients_flow_to_gate():
    torch.manual_seed(7)
    model = MTLNNModel(_cfg(core_iterations=3))
    ids = _tokens()
    out = model(ids, labels=ids)
    out["loss"].backward()
    for block in model.blocks:
        assert block.core_iter_gate.grad is not None
        assert torch.isfinite(block.core_iter_gate.grad)


def test_set_core_iterations_guards():
    model = MTLNNModel(_cfg())  # built WITHOUT iteration support
    with pytest.raises(RuntimeError):
        model.set_core_iterations(2)
    model.set_core_iterations(1)  # depth 1 always allowed
    with pytest.raises(ValueError):
        model.set_core_iterations(0)


def test_config_rejects_bad_depth():
    with pytest.raises(ValueError):
        _cfg(core_iterations=0)


def test_stack_iterations_depth1_is_identity_path():
    """stack_iterations=1 (default) must be the exact single-pass path."""
    torch.manual_seed(7)
    m = MTLNNModel(_cfg())
    assert m.stack_iterations == 1
    ids = _tokens()
    m.eval()
    with torch.no_grad():
        o1 = m(ids)["logits"]
        m.set_stack_iterations(1)
        o1b = m(ids)["logits"]
    assert torch.equal(o1, o1b)


def test_stack_iterations_changes_output_and_flows_grads():
    torch.manual_seed(7)
    m = MTLNNModel(_cfg())
    ids = _tokens()
    m.eval()
    with torch.no_grad():
        o1 = m(ids)["logits"]
        m.set_stack_iterations(3)
        o3 = m(ids)["logits"]
    assert not torch.equal(o1, o3), "3 stack passes should not be a no-op"
    m.train()
    out = m(ids, labels=ids)
    out["loss"].backward()  # must not raise; grads flow through tied passes
    assert any(p.grad is not None for p in m.parameters())


def test_stack_iterations_rejects_cache():
    torch.manual_seed(7)
    m = MTLNNModel(_cfg())
    m.set_stack_iterations(2)
    ids = _tokens(B=1, T=8)
    with pytest.raises(RuntimeError):
        m(ids, use_cache=True)


def test_depth_with_cache_generate_smoke():
    """generate() must still work on an iteration-enabled model (cache stores
    only the final iteration's h_last — API unchanged)."""
    torch.manual_seed(7)
    model = MTLNNModel(_cfg(core_iterations=2))
    model.eval()
    ids = _tokens(B=1, T=8)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=4)
    assert out.shape[1] == 12


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
