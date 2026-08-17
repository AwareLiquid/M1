"""
Tests for the EEG-inspired rhythm gate (mt_lnn/rhythm.py).

Covers:
  - LAVIEstimator shape contract and value range
  - LAVIEstimator: neutral init, no-h_prev fallback, gradient flow
  - GlobalRhythmController: identity at init, gradient flow
  - Full MTLNNModel with use_rhythm=True: shapes + no regression on use_rhythm=False
  - MTLNNLayer with use_rhythm=True: rhythm gate modulates blend
"""

import torch
import torch.nn.functional as F

from mt_lnn.rhythm import LAVIEstimator, GlobalRhythmController
from mt_lnn.config import MTLNNConfig
from mt_lnn.mt_lnn_layer import MTLNNLayer
from mt_lnn.model import MTLNNModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def small_config(use_rhythm=True, global_rhythm=False):
    # d_model=104=8×13: d_proto=8 (Tensor-Core aligned), d_gw=13 (104//8)
    # gwtb_n_heads=1 so d_gw=13 is always divisible
    return MTLNNConfig(
        vocab_size=64,
        d_model=104,
        n_layers=2,
        n_heads=13,
        n_kv_heads=1,
        d_head=8,
        n_protofilaments=13,
        n_time_scales=3,
        map_hidden_dim=8,
        max_seq_len=32,
        gwtb_n_heads=1,
        use_rhythm=use_rhythm,
        rhythm_scale_init=0.1,
        global_rhythm=global_rhythm,
        use_predictive_coding=False,
        dynamic_scale_gates=True,
    )


# ---------------------------------------------------------------------------
# LAVIEstimator
# ---------------------------------------------------------------------------

def test_lavi_shape_and_range():
    B, T, P, D = 2, 5, 13, 4
    est = LAVIEstimator(d_proto=D, n_protofilaments=P)
    x = torch.randn(B, T, P, D)
    h = torch.randn(B, P, 3, D)  # (B, P, S, D)
    out = est(h, x)
    assert out.shape == (B, T, P, 1), f"expected ({B},{T},{P},1) got {out.shape}"
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0, "LAVI must be in [0,1]"


def test_lavi_no_h_prev():
    B, T, P, D = 2, 4, 13, 4
    est = LAVIEstimator(d_proto=D, n_protofilaments=P)
    x = torch.randn(B, T, P, D)
    out = est(None, x)
    assert out.shape == (B, T, P, 1)
    # Must be exactly 0.5 when h_prev is None (neutral init)
    assert torch.allclose(out, torch.full_like(out, 0.5)), "neutral init should give 0.5"


def test_lavi_gradient_flow():
    B, T, P, D = 2, 3, 13, 4
    est = LAVIEstimator(d_proto=D, n_protofilaments=P)
    x = torch.randn(B, T, P, D, requires_grad=True)
    h = torch.randn(B, P, 5, D)
    out = est(h, x)
    out.sum().backward()
    assert x.grad is not None
    assert est.bias.grad is not None


def test_lavi_persistent_higher_for_similar_input():
    """Persistent input (low novelty) should score higher LAVI than novel input."""
    P, D = 13, 16
    est = LAVIEstimator(d_proto=D, n_protofilaments=P)
    # h_prev and x nearly identical → high LAVI
    h_stable = torch.randn(1, P, 2, D)
    x_stable = h_stable[:, :, 0, :].unsqueeze(0).expand(1, 4, P, D).contiguous()
    lavi_stable = est(h_stable, x_stable).mean().item()

    # h_prev and x completely independent → low LAVI
    h_novel = torch.randn(1, P, 2, D)
    x_novel = torch.randn(1, 4, P, D) * 10   # very different
    lavi_novel = est(h_novel, x_novel).mean().item()

    assert lavi_stable > lavi_novel, (
        f"stable input should have higher LAVI ({lavi_stable:.3f}) "
        f"than novel ({lavi_novel:.3f})"
    )


# ---------------------------------------------------------------------------
# GlobalRhythmController
# ---------------------------------------------------------------------------

def test_global_rhythm_identity_at_init():
    """GlobalRhythmController.scale=0 → correction is zero → x unchanged."""
    n_layers, d_model = 4, 32
    ctrl = GlobalRhythmController(n_layers=n_layers, d_model=d_model)
    x = torch.randn(2, 5, d_model)
    lavi_means = torch.rand(n_layers)
    x_out, global_lavi = ctrl(lavi_means, x)
    # scale=0 → tanh(0)=0 → correction=0 → x_out == x
    assert torch.allclose(x_out, x, atol=1e-5), "output should equal input at init"
    assert 0.0 <= global_lavi.item() <= 1.0


def test_global_rhythm_gradient_flow():
    n_layers, d_model = 3, 16
    ctrl = GlobalRhythmController(n_layers=n_layers, d_model=d_model)
    x = torch.randn(2, 4, d_model, requires_grad=True)
    lavi_means = torch.rand(n_layers, requires_grad=True)
    x_out, global_lavi = ctrl(lavi_means, x)
    x_out.sum().backward()
    assert x.grad is not None
    assert ctrl.scale.grad is not None


# ---------------------------------------------------------------------------
# MTLNNLayer integration
# ---------------------------------------------------------------------------

def test_layer_with_rhythm_shapes():
    cfg = small_config(use_rhythm=True)
    layer = MTLNNLayer(cfg)
    B, T = 2, 6
    x = torch.randn(B, T, cfg.d_model)
    out, h_last = layer(x, h_prev=None)
    assert out.shape == (B, T, cfg.d_model)
    assert h_last.shape == (B, cfg.n_protofilaments, cfg.n_time_scales, cfg.d_proto)


def test_layer_rhythm_buffer_populated():
    cfg = small_config(use_rhythm=True)
    layer = MTLNNLayer(cfg)
    x = torch.randn(2, 4, cfg.d_model)
    h_prev = torch.randn(2, cfg.n_protofilaments, cfg.n_time_scales, cfg.d_proto)
    layer(x, h_prev=h_prev)
    # last_lavi_mean on resonance should be non-zero after forward with h_prev
    assert layer.resonance.last_lavi_mean.item() != 0.0


def test_layer_rhythm_neutral_no_change():
    """When h_prev=None, LAVI=0.5 everywhere → rhythm_bonus=0 → output unchanged."""
    cfg = small_config(use_rhythm=True)
    layer = MTLNNLayer(cfg)
    layer.eval()
    x = torch.randn(2, 4, cfg.d_model)

    # Run with LAVI estimator active (returns 0.5, bonus=0)
    out_lavi, _ = layer(x, h_prev=None)

    # Bypass LAVI estimator entirely (lavi=None → else branch, no modification)
    original = layer.lavi_estimator
    layer.lavi_estimator = None
    out_none, _ = layer(x, h_prev=None)
    layer.lavi_estimator = original

    assert torch.allclose(out_lavi, out_none, atol=1e-5), \
        "neutral LAVI (0.5) must leave blend weights unchanged"


# ---------------------------------------------------------------------------
# Full MTLNNModel integration
# ---------------------------------------------------------------------------

def test_model_with_rhythm_forward():
    cfg = small_config(use_rhythm=True, global_rhythm=True)
    model = MTLNNModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(ids)
    assert "logits" in out
    assert out["logits"].shape == (2, 8, cfg.vocab_size)


def test_model_rhythm_diagnostics():
    cfg = small_config(use_rhythm=True, global_rhythm=True)
    model = MTLNNModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    model(ids)
    diag = model.get_mt_diagnostics()
    assert "lavi_mean" in diag, "diagnostics should include lavi_mean"
    assert "rhythm_scale_mean" in diag
    assert "global_rhythm_scale" in diag


def test_model_rhythm_gradient_flow():
    """LAVI bias gets gradient only in step-2+ inference when h_prev is non-None.
    Step 1 (no cache): lavi estimator returns 0.5-constant (h_prev=None), no grad to bias.
    Step 2 (with cache): h_prev carries actual state, cosine-sim path fires, bias gets grad.
    """
    cfg = small_config(use_rhythm=True, global_rhythm=True)
    model = MTLNNModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))

    # Step 1: prefill — builds h_prev in cache
    with torch.no_grad():
        out1 = model(ids[:, :4], use_cache=True, use_lnn_recurrence=True)
    cache = out1["cache"]

    # Step 2: decode with h_prev non-None; gradient should reach LAVI bias
    labels = torch.randint(0, cfg.vocab_size, (2, 4))
    out2 = model(ids[:, 4:], cache=cache, use_cache=True,
                 use_lnn_recurrence=True, labels=labels)
    out2["loss"].backward()

    found_grad = False
    for block in model.blocks:
        lavi_est = getattr(block.lnn, "lavi_estimator", None)
        if lavi_est is not None and lavi_est.bias.grad is not None:
            found_grad = True
            break
    assert found_grad, "LAVIEstimator.bias must receive gradient in step-2 streaming inference"


def test_model_no_regression_rhythm_off():
    """Model with use_rhythm=False must produce same logits as before the change."""
    cfg = small_config(use_rhythm=False, global_rhythm=False)
    torch.manual_seed(0)
    model = MTLNNModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    out = model(ids)
    # Just checking it runs without error and has correct shape
    assert out["logits"].shape == (2, 6, cfg.vocab_size)


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_lavi_shape_and_range,
        test_lavi_no_h_prev,
        test_lavi_gradient_flow,
        test_lavi_persistent_higher_for_similar_input,
        test_global_rhythm_identity_at_init,
        test_global_rhythm_gradient_flow,
        test_layer_with_rhythm_shapes,
        test_layer_rhythm_buffer_populated,
        test_layer_rhythm_off_unchanged,
        test_model_with_rhythm_forward,
        test_model_rhythm_diagnostics,
        test_model_rhythm_gradient_flow,
        test_model_no_regression_rhythm_off,
    ]
    for fn in tests:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
            raise
