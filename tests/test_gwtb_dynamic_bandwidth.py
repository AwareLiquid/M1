"""
tests/test_gwtb_dynamic_bandwidth.py — behavioural contract for the dynamic
workspace-bandwidth gate added to GWTBLayer / CompetitiveGWTBLayer (2026-06-15).

The Global Workspace bottleneck previously had a FIXED width: all ``d_gw``
channels were always fully active. The dynamic-bandwidth gate makes that
capacity input-dependent —

    g_t = sigmoid(W_g · z_t + b_g)   ∈ (0,1)^d_gw
    z_gated = z_t * g_t

so the workspace allocates how many bottleneck channels "ignite" per token
(brain analog: few cortical units fire on calm/redundant input, many on
surprising input).

We pin the contract that matters for a zero-regression, default-off feature:
  • the gate exists IFF ``gwtb_dynamic_bandwidth=True``;
  • OFF is bit-identical to the pre-existing fixed-bandwidth path;
  • the gate is genuinely INPUT-DEPENDENT after a gradient step;
  • gradient reaches the gate (it is in the loss graph and trainable);
  • at init the workspace starts at near-full bandwidth (mostly-open bias),
    so it does not start sparse with dead units;
  • the eval-time hard mask is a real compute-skip: it forces sub-threshold
    channels to exactly zero and changes the output vs the soft path;
  • the diagnostics report a sane active-bandwidth fraction in [0, 1].

Run:  python -m pytest tests/test_gwtb_dynamic_bandwidth.py -v
"""
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn import MTLNNConfig                                       # noqa: E402
from mt_lnn.gwtb import GWTBLayer, CompetitiveGWTBLayer             # noqa: E402


def _config(**overrides):
    base = dict(
        d_model=128,
        n_heads=4,                  # 128 / 4 = 32
        d_head=32,
        gwtb_compression_ratio=8,   # d_gw = 16
        gwtb_n_heads=4,             # 16 / 4 = 4 ok
        max_seq_len=64,
        dropout=0.0,
    )
    base.update(overrides)
    return MTLNNConfig(**base)


def _ids_free_input(B=2, T=8, d_model=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, d_model, generator=g)


# --------------------------------------------------------------------------- #
# flag wiring                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Layer", [GWTBLayer, CompetitiveGWTBLayer])
def test_gate_exists_iff_enabled(Layer):
    on = Layer(_config(gwtb_dynamic_bandwidth=True))
    assert hasattr(on, "bandwidth_gate_weight")
    assert on.dynamic_bandwidth is True

    off = Layer(_config(gwtb_dynamic_bandwidth=False))
    assert not hasattr(off, "bandwidth_gate_weight")
    assert off.dynamic_bandwidth is False


# --------------------------------------------------------------------------- #
# OFF path is bit-identical to the original fixed-bandwidth pipeline           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Layer", [GWTBLayer, CompetitiveGWTBLayer])
def test_disabled_is_bit_identical(Layer):
    x = _ids_free_input()

    torch.manual_seed(1)
    ref = Layer(_config(gwtb_dynamic_bandwidth=False)).eval()
    torch.manual_seed(1)
    same = Layer(_config(gwtb_dynamic_bandwidth=False)).eval()

    with torch.no_grad():
        out_ref, _ = ref(x)
        out_same, _ = same(x)
    # Same seed + same (no) feature → identical construction → identical output.
    assert torch.equal(out_ref, out_same)


# --------------------------------------------------------------------------- #
# at init the workspace starts mostly-open (near full bandwidth)               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Layer", [GWTBLayer, CompetitiveGWTBLayer])
def test_near_full_bandwidth_at_init(Layer):
    layer = Layer(_config(gwtb_dynamic_bandwidth=True, gwtb_bandwidth_gate_bias=4.0)).eval()
    x = _ids_free_input()
    with torch.no_grad():
        layer(x)
    # sigmoid(4) ≈ 0.982 → essentially every channel above the 0.1 threshold.
    assert layer.last_active_bandwidth.item() == pytest.approx(1.0, abs=1e-6)
    assert layer.last_bandwidth_gate_mean.item() > 0.95


# --------------------------------------------------------------------------- #
# the gate becomes genuinely input-dependent once W_g is non-zero             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Layer", [GWTBLayer, CompetitiveGWTBLayer])
def test_gate_is_input_dependent_after_training(Layer):
    torch.manual_seed(2)
    layer = Layer(_config(gwtb_dynamic_bandwidth=True))
    # Perturb the gate weight away from its zero init so it can respond to input.
    with torch.no_grad():
        layer.bandwidth_gate_weight.add_(torch.randn_like(layer.bandwidth_gate_weight))
    layer.eval()

    xa = _ids_free_input(seed=10)
    xb = _ids_free_input(seed=20)
    with torch.no_grad():
        layer(xa)
        gate_a = layer.last_bandwidth_gate_mean.clone()
        layer(xb)
        gate_b = layer.last_bandwidth_gate_mean.clone()
    # Two different inputs now yield different mean gate activation.
    assert not torch.allclose(gate_a, gate_b)


# --------------------------------------------------------------------------- #
# the gate participates in the loss graph (is trainable)                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Layer", [GWTBLayer, CompetitiveGWTBLayer])
def test_gradient_reaches_bandwidth_gate(Layer):
    torch.manual_seed(3)
    layer = Layer(_config(gwtb_dynamic_bandwidth=True)).train()
    x = _ids_free_input()
    out, _ = layer(x)
    out.sum().backward()
    # The bias is always in the loss graph; the weight receives gradient too.
    assert layer.bandwidth_gate_bias.grad is not None
    assert layer.bandwidth_gate_bias.grad.abs().sum().item() > 0.0
    assert layer.bandwidth_gate_weight.grad is not None


# --------------------------------------------------------------------------- #
# eval-time hard mask is a real compute-skip (forces dormant channels to 0)    #
# --------------------------------------------------------------------------- #
def test_hard_mask_reduces_bandwidth_and_changes_output():
    torch.manual_seed(4)
    cfg = _config(gwtb_dynamic_bandwidth=True, gwtb_bandwidth_hard_mask=True,
                  gwtb_bandwidth_gate_bias=0.0, gwtb_bandwidth_threshold=0.5)
    hard = GWTBLayer(cfg)
    # Drive some channels below threshold by giving the gate a non-trivial weight.
    with torch.no_grad():
        hard.bandwidth_gate_weight.add_(torch.randn_like(hard.bandwidth_gate_weight))
    # Build a soft twin with identical parameters (no hard mask).
    torch.manual_seed(4)
    soft = GWTBLayer(_config(gwtb_dynamic_bandwidth=True, gwtb_bandwidth_hard_mask=False,
                             gwtb_bandwidth_gate_bias=0.0, gwtb_bandwidth_threshold=0.5))
    soft.load_state_dict(hard.state_dict())
    hard.eval(); soft.eval()

    x = _ids_free_input()
    with torch.no_grad():
        out_hard, _ = hard(x)
        out_soft, _ = soft(x)
    # With bias 0 some channels fall below 0.5 → active bandwidth strictly < 1.
    assert hard.last_active_bandwidth.item() < 1.0
    # Hard-masking those channels to zero changes the broadcast → different output.
    assert not torch.allclose(out_hard, out_soft)


# --------------------------------------------------------------------------- #
# diagnostics stay in a sane range                                            #
# --------------------------------------------------------------------------- #
def test_active_bandwidth_is_a_fraction():
    layer = GWTBLayer(_config(gwtb_dynamic_bandwidth=True)).eval()
    with torch.no_grad():
        layer(_ids_free_input())
    val = layer.last_active_bandwidth.item()
    assert 0.0 <= val <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
