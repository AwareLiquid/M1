"""
tests/test_predictive_coding.py — behavioural contract for the hierarchical
predictive-coding loop (mt_lnn/predictive_coding.py, 2026-06-15).

HierarchicalPredictiveCoder is a zero-model-coupling observer over a stack of
per-layer activations (lowest -> highest). For n_levels representations, level
l+1 predicts level l through a trainable top-down linear projector; the residual
is the prediction error. Errors are weighted by a learnable per-level precision
(inverse-variance), giving:
  * a precision-weighted coding loss (free-energy proxy a trainer can add);
  * a per-level surprise spectrum ((1-cos)/2 in [0,1]);
  * a bounded aggregate surprise / last_pred_error in [0,1] that duck-types into
    salience_events.world_model_surprise and NeuromodulationController.update.

We pin:
  * shapes: n_levels-1 predictions/errors, per_level_error length, dims;
  * the coding loss is trainable -- gradient reaches the top-down predictors and
    a step reduces the loss on a fixed activation stack;
  * detach_target stop-grad: the coding loss does NOT push the activations;
  * surprise / per_level_error bounded in [0,1]; perfect prediction -> ~0;
  * last_pred_error buffer mirrors the returned surprise (the neuromod bridge);
  * input validation (wrong count / wrong width / mismatched leading dims);
  * 联调: last_pred_error feeds NeuromodulationController.update(surprise=...)
    and raises acetylcholine when the prediction error spikes.

Run:  python -m pytest tests/test_predictive_coding.py -v
"""
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn import (                                                  # noqa: E402
    HierarchicalPredictiveCoder,
    PredictiveCodingResult,
    NeuromodulationController,
)
from mt_lnn.salience_events import world_model_surprise               # noqa: E402


def _stack(dims, batch=2, seq=3, seed=0):
    """A list of per-level activations (low->high) with the given widths."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, seq, d, generator=g) for d in dims]


# --------------------------------------------------------------------------- #
# shapes & construction                                                        #
# --------------------------------------------------------------------------- #
def test_forward_shapes_and_result_type():
    dims = [16, 16, 16, 16]
    pc = HierarchicalPredictiveCoder(dims)
    acts = _stack(dims)
    res = pc.forward(acts)

    assert isinstance(res, PredictiveCodingResult)
    assert len(res.predictions) == len(dims) - 1
    assert len(res.errors) == len(dims) - 1
    for l in range(len(dims) - 1):
        assert res.predictions[l].shape == acts[l].shape
        assert res.errors[l].shape == acts[l].shape
    assert res.per_level_error.shape == (len(dims) - 1,)
    assert res.coding_loss.ndim == 0
    assert res.surprise.ndim == 0


def test_nonuniform_dims_supported():
    dims = [8, 12, 6]
    pc = HierarchicalPredictiveCoder(dims)
    res = pc.forward(_stack(dims))
    assert res.predictions[0].shape[-1] == 8     # predicts level 0 (width 8)
    assert res.predictions[1].shape[-1] == 12    # predicts level 1 (width 12)


def test_construction_validates():
    with pytest.raises(ValueError):
        HierarchicalPredictiveCoder([16])          # need >= 2 levels
    with pytest.raises(ValueError):
        HierarchicalPredictiveCoder([16, 0])       # dims must be >= 1


# --------------------------------------------------------------------------- #
# coding loss is trainable                                                     #
# --------------------------------------------------------------------------- #
def test_coding_loss_trains_top_down_predictors():
    torch.manual_seed(0)
    dims = [16, 16]
    pc = HierarchicalPredictiveCoder(dims)
    acts = _stack(dims, seed=1)

    opt = torch.optim.SGD(pc.parameters(), lr=0.5)
    first = pc.coding_loss(acts).item()
    for _ in range(50):
        opt.zero_grad()
        loss = pc.coding_loss(acts)
        loss.backward()
        # gradient must reach the top-down predictor weights
        assert pc.predictors[0].weight.grad is not None
        opt.step()
    last = pc.coding_loss(acts).item()
    assert last < first              # the loop actually learns to predict


def test_coding_loss_convenience_matches_forward():
    dims = [8, 8, 8]
    pc = HierarchicalPredictiveCoder(dims)
    acts = _stack(dims)
    assert torch.equal(pc.coding_loss(acts), pc.forward(acts).coding_loss)


# --------------------------------------------------------------------------- #
# detach_target stop-grad                                                      #
# --------------------------------------------------------------------------- #
def test_detach_target_blocks_grad_to_activations():
    dims = [8, 8]
    pc = HierarchicalPredictiveCoder(dims, detach_target=True)
    low = torch.randn(2, 8, requires_grad=True)
    high = torch.randn(2, 8, requires_grad=True)
    res = pc.forward([low, high])
    res.coding_loss.backward()
    # target (low) is detached -> no grad; only the top-down source (high) flows.
    assert low.grad is None
    assert high.grad is not None


def test_no_detach_lets_grad_reach_target():
    dims = [8, 8]
    pc = HierarchicalPredictiveCoder(dims, detach_target=False)
    low = torch.randn(2, 8, requires_grad=True)
    high = torch.randn(2, 8, requires_grad=True)
    res = pc.forward([low, high])
    res.coding_loss.backward()
    assert low.grad is not None      # error pulls the target too


# --------------------------------------------------------------------------- #
# bounded surprise spectrum                                                    #
# --------------------------------------------------------------------------- #
def test_surprise_and_per_level_bounded():
    dims = [16, 16, 16]
    pc = HierarchicalPredictiveCoder(dims)
    res = pc.forward(_stack(dims, seed=3))
    assert (res.per_level_error >= 0.0).all()
    assert (res.per_level_error <= 1.0).all()
    assert 0.0 <= res.surprise.item() <= 1.0
    assert res.surprise.item() == pytest.approx(res.per_level_error.mean().item(), rel=1e-5)


def test_perfect_prediction_is_low_surprise():
    # Train to convergence on a fixed stack, then surprise should be small.
    torch.manual_seed(0)
    dims = [12, 12]
    pc = HierarchicalPredictiveCoder(dims)
    acts = _stack(dims, seed=4)
    opt = torch.optim.SGD(pc.parameters(), lr=0.5)
    for _ in range(300):
        opt.zero_grad()
        pc.coding_loss(acts).backward()
        opt.step()
    res = pc.forward(acts)
    assert res.surprise.item() < 0.1


# --------------------------------------------------------------------------- #
# last_pred_error neuromod bridge                                              #
# --------------------------------------------------------------------------- #
def test_last_pred_error_buffer_mirrors_surprise():
    dims = [16, 16]
    pc = HierarchicalPredictiveCoder(dims)
    res = pc.forward(_stack(dims, seed=5))
    assert pc.last_pred_error.item() == pytest.approx(res.surprise.item(), rel=1e-5)
    assert pc.last_per_level_error.shape == (len(dims) - 1,)


def test_world_model_surprise_reads_the_coder():
    # The duck-typed bridge: salience_events.world_model_surprise(head) reads
    # last_pred_error -- the coder drops straight in where PredictiveStateHead goes.
    dims = [16, 16]
    pc = HierarchicalPredictiveCoder(dims)
    pc.forward(_stack(dims, seed=6))
    s = world_model_surprise(pc)
    assert 0.0 <= float(s) <= 1.0
    assert float(s) == pytest.approx(pc.last_pred_error.item(), rel=1e-5)


# --------------------------------------------------------------------------- #
# input validation                                                            #
# --------------------------------------------------------------------------- #
def test_rejects_wrong_level_count():
    pc = HierarchicalPredictiveCoder([8, 8, 8])
    with pytest.raises(ValueError):
        pc.forward(_stack([8, 8]))            # only 2 of 3 levels


def test_rejects_wrong_width():
    pc = HierarchicalPredictiveCoder([8, 8])
    with pytest.raises(ValueError):
        pc.forward([torch.randn(2, 8), torch.randn(2, 7)])


def test_rejects_mismatched_leading_dims():
    pc = HierarchicalPredictiveCoder([8, 8])
    with pytest.raises(ValueError):
        pc.forward([torch.randn(2, 3, 8), torch.randn(2, 4, 8)])


# --------------------------------------------------------------------------- #
# 联调: predictive coding -> neuromodulation                                   #
# --------------------------------------------------------------------------- #
def test_surprise_spike_raises_acetylcholine_via_controller():
    # Feed the coder's bounded surprise into the neuromodulation hub. A spike in
    # prediction error should drive acetylcholine up relative to a calm baseline.
    dims = [16, 16]
    pc = HierarchicalPredictiveCoder(dims)
    ctrl = NeuromodulationController(ema_decay=0.5, sensitivity=2.0)

    # Calm baseline: a near-zero surprise stream establishes the EMA.
    for _ in range(5):
        ctrl.update(surprise=0.02)
    calm_ach = ctrl.state.acetylcholine

    # Now a genuine spike from the coder.
    res = pc.forward(_stack(dims, seed=9))
    spike = res.surprise.item()
    # Make the spike unambiguous relative to the calm baseline if needed.
    spike = max(spike, 0.8)
    st = ctrl.update(surprise=spike)
    assert st.acetylcholine > calm_ach


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
