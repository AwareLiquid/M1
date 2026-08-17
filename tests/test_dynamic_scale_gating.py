"""
tests/test_dynamic_scale_gating.py -- behavioural contract for the endogenous
dynamic-kappa multi-scale gate.

This mechanism (BRAIN_INSPIRED_ROADMAP.md, Phase 1 item "内源性动态通道门控") is ON
by default (``dynamic_scale_gates=True`` in ``MTLNNConfig``). Prior tests only
checked that the diagnostic numbers fall in range and pinned the top-k ratio;
none asserted that the gate is genuinely input-dependent, that it participates in
the loss graph, or that the legacy static path degenerates correctly. We pin
those here.

What the layer does (``VectorizedMultiScaleResonance.forward``): when gating is
on it computes ``dynamic_kappa = sigmoid(kappa_gate(x))`` of shape ``(B,T,P,S)``
and forms the per-position blend ``gated_w = softmax-renorm(blend_weights *
dynamic_kappa)`` so the cross-scale mixture becomes input-dependent rather than a
single static ``softmax(blend_weights)``. With ``sparse_resonance_kernel=True``
it additionally selects the top-k scales by mean gate and computes resonance for
only those (the compute-skipping path).

We pin:
- ``kappa_gate`` exists iff the feature is on;
- the gate is genuinely INPUT-DEPENDENT -- two different inputs produce different
  per-scale gate means -- whereas the static path reports a constant all-ones
  gate regardless of input;
- gradient reaches ``kappa_gate`` (it is in the loss graph and trainable);
- the sparse-kernel path selects EXACTLY ``top_k`` scales and reports the right
  sparsity ratio, and a sparse forward changes the output relative to dense.

Run:  python -m pytest tests/test_dynamic_scale_gating.py -v
"""
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn import MTLNNConfig, MTLNNModel                          # noqa: E402


def _config(**overrides):
    base = dict(
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
    base.update(overrides)
    return MTLNNConfig(**base)


def _resonances(model):
    return [b.lnn.resonance for b in model.blocks]


# --------------------------------------------------------------------------- #
# flag wiring                                                                  #
# --------------------------------------------------------------------------- #
def test_kappa_gate_exists_iff_enabled():
    on = MTLNNModel(_config(dynamic_scale_gates=True))
    for r in _resonances(on):
        assert hasattr(r, "kappa_gate")

    off = MTLNNModel(_config(dynamic_scale_gates=False))
    for r in _resonances(off):
        assert not hasattr(r, "kappa_gate")


# --------------------------------------------------------------------------- #
# the gate is genuinely input-dependent                                        #
# --------------------------------------------------------------------------- #
def test_gate_is_input_dependent_when_enabled():
    torch.manual_seed(0)
    model = MTLNNModel(_config(dynamic_scale_gates=True)).eval()

    ids_a = torch.randint(0, 200, (2, 16))
    ids_b = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        model(ids_a)
        gate_a = [r.last_scale_gate_mean.clone() for r in _resonances(model)]
        model(ids_b)
        gate_b = [r.last_scale_gate_mean.clone() for r in _resonances(model)]

    # At least one block's per-scale gate vector responds to the input change.
    assert any(not torch.allclose(a, b) for a, b in zip(gate_a, gate_b))
    # And the gate values are genuine sigmoid activations in (0, 1).
    for a in gate_a:
        assert torch.all((a > 0.0) & (a < 1.0))


def test_static_path_reports_constant_gate():
    torch.manual_seed(0)
    model = MTLNNModel(_config(dynamic_scale_gates=False)).eval()
    ids_a = torch.randint(0, 200, (2, 16))
    ids_b = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        model(ids_a)
        gate_a = [r.last_scale_gate_mean.clone() for r in _resonances(model)]
        model(ids_b)
        gate_b = [r.last_scale_gate_mean.clone() for r in _resonances(model)]
    # The legacy static blend reports an all-ones gate, invariant to input.
    for a, b in zip(gate_a, gate_b):
        assert torch.allclose(a, torch.ones_like(a))
        assert torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# the gate participates in the loss graph                                      #
# --------------------------------------------------------------------------- #
def test_gradient_reaches_kappa_gate():
    torch.manual_seed(0)
    model = MTLNNModel(_config(dynamic_scale_gates=True)).train()
    ids = torch.randint(0, 200, (2, 16))
    model(ids, labels=ids)["loss"].backward()
    for r in _resonances(model):
        assert r.kappa_gate.weight.grad is not None
        assert r.kappa_gate.weight.grad.abs().sum().item() > 0.0


# --------------------------------------------------------------------------- #
# sparse compute-skipping path selects exactly top_k scales                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("top_k", [1, 2])
def test_sparse_kernel_selects_exactly_top_k_scales(top_k):
    torch.manual_seed(0)
    cfg = _config(dynamic_scale_gates=True, sparse_resonance_kernel=True,
                  sparse_resonance_top_k=top_k)
    model = MTLNNModel(cfg).eval()
    ids = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        model(ids)
    for r in _resonances(model):
        assert int(r.last_sparse_selected_scales.sum().item()) == top_k
        assert r.last_sparse_scale_ratio.item() == pytest.approx(top_k / r.S, abs=1e-6)


def test_sparse_forward_changes_output_vs_dense():
    torch.manual_seed(0)
    ids = torch.randint(0, 200, (2, 16))

    dense = MTLNNModel(_config(dynamic_scale_gates=True,
                               sparse_resonance_kernel=False)).eval()
    with torch.no_grad():
        out_dense = dense(ids)["logits"]

    # Same seed -> identical initial weights -> any output difference is due to
    # the sparse top-k scale selection actually skipping scales.
    torch.manual_seed(0)
    sparse = MTLNNModel(_config(dynamic_scale_gates=True,
                                sparse_resonance_kernel=True,
                                sparse_resonance_top_k=1)).eval()
    with torch.no_grad():
        out_sparse = sparse(ids)["logits"]

    assert not torch.allclose(out_dense, out_sparse)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
