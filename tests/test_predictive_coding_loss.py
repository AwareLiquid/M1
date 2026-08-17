"""
tests/test_predictive_coding_loss.py -- behavioural contract for the multi-scale
predictive-coding auxiliary loss.

This mechanism (BRAIN_INSPIRED_ROADMAP.md, Phase 1 item "多尺度预测编码") is ON by
default (``use_predictive_coding=True``, ``predictive_loss_weight=0.05`` in
``MTLNNConfig``) yet, prior to this file, had NO test pinning its behaviour --
the rest of the suite only ever set ``use_predictive_coding=False`` to silence
it. We close that gap here.

What the layer actually does (``VectorizedMultiScaleResonance.forward``): in
training mode, with ``S > 1`` time-scales, each slower channel ``s`` predicts
the (detached) faster channel ``s-1`` through a learned ``W_pred`` of shape
``(P, S-1, D, D)``, and the per-block scalar
``last_pred_error = MSE(W_pred . h[1:], h[:-1].detach())`` is stored. The model
(``model.py``) sums those per-block scalars into ``result["pred_loss"]`` and adds
``predictive_loss_weight * pred_loss`` to the training loss (finiteness-guarded).

We pin, end to end:
- ``W_pred`` exists with shape ``(P, S-1, D, D)`` exactly when the feature is on
  AND ``S > 1``; it is absent when disabled or when ``S == 1``;
- the per-block surprise is a finite non-negative scalar, strictly positive in
  training on a generic input, and exactly zero in eval (no train-time leak);
- the model surfaces ``pred_loss`` and it equals the sum of the per-block
  surprises;
- the loss is genuinely wired: gradient reaches ``W_pred`` (which feeds NOTHING
  but the predictive loss), so a non-zero ``W_pred.grad`` proves the term is in
  the loss graph;
- ``predictive_loss_weight`` linearly scales the contribution: the total loss
  rises by exactly ``weight * pred_loss`` relative to weight ``0``.

Run:  python -m pytest tests/test_predictive_coding_loss.py -v
"""
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn import MTLNNConfig, MTLNNModel                          # noqa: E402


def _config(**overrides):
    """Tiny but multi-scale config (n_time_scales defaults to 5, so S > 1).

    This file TESTS the predictive-coding feature, so the flag is enabled here
    explicitly — since E4 (2026-07-15) the shipped default is
    use_predictive_coding=False (the switch-matrix ablation measured it
    PPL-negative), and a feature under test must opt in, not rely on defaults."""
    base = dict(
        vocab_size=200,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=32,
        use_predictive_coding=True,
        dropout=0.0,            # deterministic
        attention_dropout=0.0,
    )
    base.update(overrides)
    return MTLNNConfig(**base)


def _resonances(model):
    return [b.lnn.resonance for b in model.blocks]


# --------------------------------------------------------------------------- #
# W_pred existence / shape                                                     #
# --------------------------------------------------------------------------- #
def test_w_pred_exists_with_correct_shape_when_enabled():
    model = MTLNNModel(_config(use_predictive_coding=True))
    for r in _resonances(model):
        assert hasattr(r, "W_pred")
        assert r.S > 1
        assert tuple(r.W_pred.shape) == (r.P, r.S - 1, r.D, r.D)


def test_no_w_pred_when_disabled():
    model = MTLNNModel(_config(use_predictive_coding=False))
    for r in _resonances(model):
        assert not hasattr(r, "W_pred")


def test_no_w_pred_when_single_scale():
    # S == 1 has no faster/slower pair to predict, so the guard drops W_pred
    # even with the feature enabled.
    model = MTLNNModel(_config(use_predictive_coding=True, n_time_scales=1))
    for r in _resonances(model):
        assert r.S == 1
        assert not hasattr(r, "W_pred")


# --------------------------------------------------------------------------- #
# surprise scalar: positive in training, exactly zero in eval                 #
# --------------------------------------------------------------------------- #
def test_pred_error_is_positive_finite_scalar_in_training():
    torch.manual_seed(0)
    model = MTLNNModel(_config()).train()
    ids = torch.randint(0, 200, (2, 16))
    model(ids, labels=ids)
    for r in _resonances(model):
        e = r.last_pred_error
        assert e.dim() == 0                      # scalar
        assert torch.isfinite(e)
        assert e.item() > 0.0                     # generic input -> non-trivial


def test_pred_error_is_zero_in_eval():
    torch.manual_seed(0)
    model = MTLNNModel(_config()).eval()
    ids = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        model(ids, labels=ids)
    for r in _resonances(model):
        assert r.last_pred_error.item() == 0.0    # no train-time leakage


# --------------------------------------------------------------------------- #
# model surfaces pred_loss and it equals the per-block sum                     #
# --------------------------------------------------------------------------- #
def test_model_reports_pred_loss_equal_to_block_sum():
    torch.manual_seed(0)
    model = MTLNNModel(_config()).train()
    ids = torch.randint(0, 200, (2, 16))
    out = model(ids, labels=ids)
    assert "pred_loss" in out
    block_sum = sum(r.last_pred_error for r in _resonances(model))
    assert torch.allclose(out["pred_loss"], block_sum, atol=1e-9)
    assert out["pred_loss"].item() > 0.0


# --------------------------------------------------------------------------- #
# the loss is genuinely wired: gradient reaches W_pred                         #
# --------------------------------------------------------------------------- #
def test_gradient_reaches_w_pred_proving_loss_is_wired():
    # W_pred feeds the predictive loss and NOTHING else, so a non-zero grad on
    # it after backprop through the model loss proves pred_loss is in the graph.
    torch.manual_seed(0)
    model = MTLNNModel(_config()).train()
    ids = torch.randint(0, 200, (2, 16))
    out = model(ids, labels=ids)
    out["loss"].backward()
    for r in _resonances(model):
        assert r.W_pred.grad is not None
        assert r.W_pred.grad.abs().sum().item() > 0.0


def test_disabling_predictive_coding_removes_pred_loss_from_output():
    torch.manual_seed(0)
    model = MTLNNModel(_config(use_predictive_coding=False)).train()
    ids = torch.randint(0, 200, (2, 16))
    out = model(ids, labels=ids)
    assert "pred_loss" not in out


# --------------------------------------------------------------------------- #
# predictive_loss_weight linearly scales the contribution                     #
# --------------------------------------------------------------------------- #
def test_predictive_loss_weight_scales_the_contribution_linearly():
    # With dropout disabled and a fixed model/input, the predictive surprise is
    # deterministic. Turning the weight from 0 to w must raise the total loss by
    # exactly w * pred_loss -- proving both that the term is ADDED and that the
    # weight MULTIPLIES it.
    torch.manual_seed(0)
    model = MTLNNModel(_config()).train()
    ids = torch.randint(0, 200, (2, 16))

    model.config.predictive_loss_weight = 0.0
    out0 = model(ids, labels=ids)
    loss0, pred0 = out0["loss"].item(), out0["pred_loss"].item()

    model.config.predictive_loss_weight = 0.5
    out1 = model(ids, labels=ids)
    loss1, pred1 = out1["loss"].item(), out1["pred_loss"].item()

    assert pred0 == pytest.approx(pred1, abs=1e-6)        # surprise is invariant
    assert (loss1 - loss0) == pytest.approx(0.5 * pred1, abs=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
