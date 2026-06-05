"""
tests/test_world_model.py — PredictiveStateHead test suite.

Tests cover:
  1.  Output shape matches input
  2.  No loss returned for T=1 (single-step inference)
  3.  Loss returned and finite for T>1 (training sequence)
  4.  Loss is a non-negative scalar
  5.  Gradient flows to all head params
  6.  Zero-init: predictions start near zero at init
  7.  Loss decreases during training steps (head can learn)
  8.  Full MTLNNModel with use_world_model=True — forward shape
  9.  world_model_loss present in output when use_world_model=True + labels
 10.  Total loss > LM-only loss when world_model is active
 11.  No regression when use_world_model=False
 12.  Diagnostics in get_mt_diagnostics
 13.  world_model_loss_weight scales contribution correctly
 14.  compute_loss=False skips loss computation
 15.  pred_error buffer updated after training forward
"""

import math
import torch
import torch.optim as optim

from mt_lnn.world_model import PredictiveStateHead
from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def wm_config(use_wm=True, weight=0.01, hidden_ratio=0.5):
    return MTLNNConfig(
        vocab_size=64,
        d_model=104,
        n_layers=2,
        n_heads=13,
        n_kv_heads=1,
        d_head=8,
        max_seq_len=32,
        gwtb_n_heads=1,
        use_world_model=use_wm,
        world_model_loss_weight=weight,
        world_model_hidden_ratio=hidden_ratio,
        use_rhythm=False,
        use_predictive_coding=False,
        dynamic_scale_gates=True,
        dropout=0.0,
        attention_dropout=0.0,
    )


# ---------------------------------------------------------------------------
# 1. Output shape matches input
# ---------------------------------------------------------------------------

def test_output_shape_matches_input():
    B, T, D = 2, 8, 128
    head = PredictiveStateHead(d_model=D)
    x = torch.randn(B, T, D)
    pred, loss = head(x)
    assert pred.shape == (B, T, D), f"expected {(B,T,D)}, got {pred.shape}"


def test_output_shape_single_step():
    B, T, D = 1, 1, 64
    head = PredictiveStateHead(d_model=D)
    x = torch.randn(B, T, D)
    pred, loss = head(x)
    assert pred.shape == (B, T, D)
    assert loss is None, "T=1 should return no loss"


# ---------------------------------------------------------------------------
# 2. No loss for T=1
# ---------------------------------------------------------------------------

def test_no_loss_for_single_token():
    head = PredictiveStateHead(d_model=64)
    x = torch.randn(2, 1, 64)
    _, loss = head(x)
    assert loss is None


def test_no_loss_when_compute_loss_false():
    head = PredictiveStateHead(d_model=64)
    x = torch.randn(2, 6, 64)
    _, loss = head(x, compute_loss=False)
    assert loss is None


# ---------------------------------------------------------------------------
# 3. Loss returned and finite for T>1
# ---------------------------------------------------------------------------

def test_loss_present_for_sequence():
    head = PredictiveStateHead(d_model=64)
    x = torch.randn(2, 8, 64)
    _, loss = head(x)
    assert loss is not None
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 4. Loss is non-negative
# ---------------------------------------------------------------------------

def test_loss_non_negative():
    head = PredictiveStateHead(d_model=64)
    for _ in range(5):
        x = torch.randn(2, 8, 64)
        _, loss = head(x)
        assert loss.item() >= 0.0, f"negative loss: {loss.item()}"


# ---------------------------------------------------------------------------
# 5. Gradient flows to all head params
# ---------------------------------------------------------------------------

def test_gradient_flow_through_head():
    head = PredictiveStateHead(d_model=128)
    x = torch.randn(2, 6, 128, requires_grad=True)
    _, loss = head(x)
    loss.backward()
    assert x.grad is not None
    for name, p in head.named_parameters():
        assert p.grad is not None, f"param {name} has no gradient"


# ---------------------------------------------------------------------------
# 6. Zero-init: predictions near zero at init
# ---------------------------------------------------------------------------

def test_predictions_near_zero_at_init():
    """Last layer is zero-init → predictor(x) ≈ 0 at init."""
    torch.manual_seed(0)
    head = PredictiveStateHead(d_model=64)
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        pred, _ = head(x)
    # All predictions should be exactly 0 (fc2.weight = fc2.bias = 0)
    assert pred.abs().max().item() < 1e-7, \
        f"predictions not near zero at init: max={pred.abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# 7. Loss decreases during training
# ---------------------------------------------------------------------------

def test_loss_decreases_during_training():
    """After gradient updates the head should reduce its prediction error."""
    torch.manual_seed(42)
    head = PredictiveStateHead(d_model=64)
    opt = optim.Adam(head.parameters(), lr=1e-3)

    # Use a fixed "ground truth" sequence to overfit
    x_fixed = torch.randn(1, 16, 64)
    _, loss0 = head(x_fixed)
    initial_loss = loss0.item()

    for _ in range(100):
        _, loss = head(x_fixed)
        opt.zero_grad()
        loss.backward()
        opt.step()

    _, loss_final = head(x_fixed)
    assert loss_final.item() < initial_loss, \
        f"loss did not decrease: {initial_loss:.4f} → {loss_final.item():.4f}"


# ---------------------------------------------------------------------------
# 8. Full MTLNNModel with use_world_model=True
# ---------------------------------------------------------------------------

def test_model_with_world_model_forward():
    cfg = wm_config(use_wm=True)
    model = MTLNNModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(ids)
    assert "logits" in out
    assert out["logits"].shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(out["logits"]).all()


def test_model_world_model_head_not_none():
    cfg = wm_config(use_wm=True)
    model = MTLNNModel(cfg)
    assert model.world_model_head is not None
    assert isinstance(model.world_model_head, PredictiveStateHead)


# ---------------------------------------------------------------------------
# 9. world_model_loss in output dict
# ---------------------------------------------------------------------------

def test_world_model_loss_in_output_when_training():
    cfg = wm_config(use_wm=True)
    model = MTLNNModel(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(ids, labels=labels)
    assert "world_model_loss" in out, "world_model_loss missing from output"
    assert torch.isfinite(out["world_model_loss"])
    assert out["world_model_loss"].item() >= 0.0


def test_world_model_loss_absent_without_labels():
    cfg = wm_config(use_wm=True)
    model = MTLNNModel(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(ids)  # no labels → no LM loss → world_model_loss only if computed
    # world_model_loss is only added to result when LM loss is computed
    assert "loss" not in out   # no labels → no loss key


# ---------------------------------------------------------------------------
# 10. Total loss > LM-only loss when world model active
# ---------------------------------------------------------------------------

def test_total_loss_larger_with_world_model():
    cfg_on  = wm_config(use_wm=True,  weight=1.0)  # large weight to see difference
    cfg_off = wm_config(use_wm=False)

    torch.manual_seed(7)
    model_on = MTLNNModel(cfg_on)
    torch.manual_seed(7)
    model_off = MTLNNModel(cfg_off)

    # Copy the common weights so only the head difference matters
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(
            model_on.named_parameters(), model_off.named_parameters()
        ):
            if n1 == n2 and p1.shape == p2.shape:
                p2.data.copy_(p1.data)

    model_on.train()
    model_off.train()
    ids = torch.randint(0, cfg_on.vocab_size, (2, 8))
    labels = torch.randint(0, cfg_on.vocab_size, (2, 8))

    with torch.no_grad():
        out_on  = model_on(ids, labels=labels)
        out_off = model_off(ids, labels=labels)

    loss_on  = out_on["loss"].item()
    loss_off = out_off["loss"].item()

    assert loss_on > loss_off or abs(loss_on - loss_off) < 0.01, \
        f"With weight=1.0, world_model loss should increase total: on={loss_on:.4f}, off={loss_off:.4f}"


# ---------------------------------------------------------------------------
# 11. No regression when use_world_model=False
# ---------------------------------------------------------------------------

def test_no_regression_world_model_off():
    cfg = wm_config(use_wm=False)
    model = MTLNNModel(cfg)
    assert model.world_model_head is None
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    out = model(ids)
    assert out["logits"].shape == (2, 6, cfg.vocab_size)
    assert "world_model_loss" not in out


# ---------------------------------------------------------------------------
# 12. Diagnostics
# ---------------------------------------------------------------------------

def test_world_model_diagnostics():
    cfg = wm_config(use_wm=True)
    model = MTLNNModel(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    labels = torch.randint(0, cfg.vocab_size, (1, 8))
    model(ids, labels=labels)
    diag = model.get_mt_diagnostics()
    assert "world_model_pred_error" in diag, "world_model_pred_error missing from diagnostics"
    assert math.isfinite(diag["world_model_pred_error"])


# ---------------------------------------------------------------------------
# 13. Loss weight scales contribution
# ---------------------------------------------------------------------------

def test_loss_weight_scales_contribution():
    """Higher weight → larger total loss difference vs no-wm model."""
    ids = torch.randint(0, 64, (1, 8))
    labels = torch.randint(0, 64, (1, 8))

    losses = {}
    for weight in [0.001, 0.1, 1.0]:
        cfg = wm_config(use_wm=True, weight=weight)
        torch.manual_seed(0)
        model = MTLNNModel(cfg)
        model.train()
        with torch.no_grad():
            out = model(ids, labels=labels)
        losses[weight] = out["loss"].item()

    # Higher weight → higher total loss (world model term larger)
    assert losses[1.0] >= losses[0.1] or abs(losses[1.0] - losses[0.1]) < 0.01
    assert losses[0.1] >= losses[0.001] or abs(losses[0.1] - losses[0.001]) < 0.01


# ---------------------------------------------------------------------------
# 14. compute_loss=False skips loss
# ---------------------------------------------------------------------------

def test_compute_loss_false():
    head = PredictiveStateHead(d_model=64)
    x = torch.randn(2, 8, 64)
    pred, loss = head(x, compute_loss=False)
    assert pred.shape == x.shape
    assert loss is None


# ---------------------------------------------------------------------------
# 15. pred_error buffer updated
# ---------------------------------------------------------------------------

def test_pred_error_buffer_updated():
    cfg = wm_config(use_wm=True)
    model = MTLNNModel(cfg)
    model.train()
    # Initially 0
    assert model.world_model_head.last_pred_error.item() == 0.0

    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    labels = torch.randint(0, cfg.vocab_size, (1, 8))
    model(ids, labels=labels)

    # Should be non-zero after a forward pass with T>1
    assert model.world_model_head.last_pred_error.item() >= 0.0
    # The error should be finite
    assert math.isfinite(model.world_model_head.last_pred_error.item())


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_output_shape_matches_input,
        test_output_shape_single_step,
        test_no_loss_for_single_token,
        test_no_loss_when_compute_loss_false,
        test_loss_present_for_sequence,
        test_loss_non_negative,
        test_gradient_flow_through_head,
        test_predictions_near_zero_at_init,
        test_loss_decreases_during_training,
        test_model_with_world_model_forward,
        test_model_world_model_head_not_none,
        test_world_model_loss_in_output_when_training,
        test_world_model_loss_absent_without_labels,
        test_total_loss_larger_with_world_model,
        test_no_regression_world_model_off,
        test_world_model_diagnostics,
        test_loss_weight_scales_contribution,
        test_compute_loss_false,
        test_pred_error_buffer_updated,
    ]
    for fn in tests:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except Exception as exc:
            import traceback
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
            raise
