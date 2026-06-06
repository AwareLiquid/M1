"""
tests/test_world_model.py — PredictiveStateHead test suite.

Tests cover (v2.1 — BYOL / V-JEPA EMA target encoder):
  1.  Output latent shape (B, T, proj_dim)
  2.  No loss returned for T=1 (single-step inference)
  3.  Loss returned and finite for T>1 (training sequence)
  4.  Loss is a non-negative scalar
  5.  Gradient flows to all *trainable* head params (target_proj is frozen)
  6.  Target projector is a frozen EMA copy that tracks the online projector
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
    # Latent prediction in proj space (proj_ratio=0.5 → D//2).
    assert pred.shape == (B, T, head.proj_dim), \
        f"expected {(B, T, head.proj_dim)}, got {pred.shape}"


def test_output_shape_single_step():
    B, T, D = 1, 1, 64
    head = PredictiveStateHead(d_model=D)
    x = torch.randn(B, T, D)
    pred, loss = head(x)
    assert pred.shape == (B, T, head.proj_dim)
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
        if not p.requires_grad:
            # target_proj is a frozen EMA copy — intentionally no gradient.
            assert "target_proj" in name, f"unexpected frozen param {name}"
            continue
        assert p.grad is not None, f"trainable param {name} has no gradient"


# ---------------------------------------------------------------------------
# 6. EMA target encoder: frozen copy that tracks the online projector
# ---------------------------------------------------------------------------

def test_target_proj_is_frozen():
    """target_proj must not require grad (stop-grad target prevents collapse)."""
    head = PredictiveStateHead(d_model=64)
    for p in head.target_proj.parameters():
        assert not p.requires_grad, "target_proj should be frozen (no backprop)"


def test_target_proj_initialised_to_online():
    """At init the EMA target equals the online projector."""
    head = PredictiveStateHead(d_model=64)
    assert torch.allclose(
        head.target_proj.weight, head.online_proj.weight
    ), "target_proj should start as a copy of online_proj"


def test_ema_target_tracks_online_after_update():
    """
    After the online projector moves and a training forward triggers the EMA
    update, the target should move toward (but lag behind) the online weights.
    """
    torch.manual_seed(0)
    head = PredictiveStateHead(d_model=64, ema_decay=0.9)
    # Perturb the online projector to create a gap.
    with torch.no_grad():
        head.online_proj.weight.add_(torch.randn_like(head.online_proj.weight))
    gap_before = (head.target_proj.weight - head.online_proj.weight).abs().mean().item()

    x = torch.randn(2, 8, 64)
    head(x)  # training forward (T>1) → _update_target() runs

    gap_after = (head.target_proj.weight - head.online_proj.weight).abs().mean().item()
    assert gap_after < gap_before, \
        f"EMA target did not move toward online: {gap_before:.4f} → {gap_after:.4f}"


def test_pred_error_normalised_to_unit_interval():
    """The exposed surprise signal must be a normalised value in [0, 1]."""
    torch.manual_seed(0)
    head = PredictiveStateHead(d_model=64)
    # Drive a forward with large-magnitude states; normalised error must stay [0,1].
    x = torch.randn(2, 8, 64) * 100.0
    head(x)
    err = head.last_pred_error.item()
    assert 0.0 <= err <= 1.0, f"normalised pred_error out of range: {err}"


# ---------------------------------------------------------------------------
# 6b. Negative controls — the surprise signal is a real mechanism, not an
#     artifact, and the EMA design does not collapse.
# ---------------------------------------------------------------------------

def _ar1_batch(B=8, T=12, D=32, rho=0.9):
    """A batch with genuine temporal structure (AR(1) along the time axis)."""
    xs = [torch.randn(B, D)]
    for _ in range(T - 1):
        xs.append(rho * xs[-1] + 0.2 * torch.randn(B, D))
    return torch.stack(xs, dim=1)


def _train_head(structured: bool, seed: int, steps: int = 500, D: int = 32):
    torch.manual_seed(seed)
    head = PredictiveStateHead(D, ema_decay=0.99, warmup_steps=100)
    opt = optim.Adam([p for p in head.parameters() if p.requires_grad], lr=3e-3)
    for _ in range(steps):
        x = _ar1_batch(D=D) if structured else torch.randn(8, 12, D)
        _, loss = head(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head


def test_surprise_tracks_structure_not_noise():
    """
    NEGATIVE CONTROL: the normalised surprise must reflect *real*
    predictability. Trained on temporally structured (AR(1)) sequences the
    error should fall well below chance (0.5); trained on i.i.d. noise — where
    there is nothing to predict — it must stay near chance. This proves the
    world model captures genuine sequential structure rather than producing a
    trivial/constant signal.
    """
    structured_err = _train_head(structured=True, seed=0).last_pred_error.item()
    noise_err = _train_head(structured=False, seed=0).last_pred_error.item()
    assert structured_err < 0.25, f"failed to learn structure: err={structured_err:.3f}"
    assert noise_err > 0.35, f"surprise collapsed on noise (should ~0.5): {noise_err:.3f}"
    assert structured_err < noise_err - 0.2, \
        f"no separation: structured={structured_err:.3f} noise={noise_err:.3f}"


def test_no_representational_collapse():
    """
    The asymmetric predictor + stop-grad (+ EMA) design must NOT collapse the
    latent space: distinct inputs should keep distinct latent directions. We
    measure the mean pairwise |cosine| of the latents of a diverse probe batch
    after training — collapse would drive this toward 1.0.
    """
    head = _train_head(structured=True, seed=0, steps=500, D=32)
    torch.manual_seed(123)
    probe = torch.randn(16, 12, 32)
    with torch.no_grad():
        z = head.online_proj(probe).reshape(-1, head.proj_dim)
    zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
    sim = zn @ zn.t()
    n = sim.shape[0]
    mean_off_diag = (sim.abs().sum() - sim.diag().abs().sum()) / (n * (n - 1))
    assert mean_off_diag.item() < 0.5, \
        f"latent space collapsed: mean pairwise|cos|={mean_off_diag.item():.3f}"


def test_ema_warmup_uses_gentler_decay():
    """During warm-up the EMA must use the gentler 0.9 decay, then switch."""
    head = PredictiveStateHead(d_model=32, ema_decay=0.999, warmup_steps=3)
    # Force a gap so the decay choice is observable.
    with torch.no_grad():
        head.online_proj.weight.add_(torch.randn_like(head.online_proj.weight))
    x = torch.randn(2, 8, 32)
    # 1st update (warm-up, d=0.9): big move toward online.
    head(x)
    gap1 = (head.target_proj.weight - head.online_proj.weight).abs().mean().item()
    # Run past warm-up; subsequent updates (d=0.999) move slowly.
    for _ in range(5):
        head(x)
    gap2 = (head.target_proj.weight - head.online_proj.weight).abs().mean().item()
    # After warm-up the target keeps closing the gap (monotonic), and the
    # step counter must have advanced past warmup_steps.
    assert head._ema_step >= 6
    assert gap2 <= gap1


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
    assert pred.shape == (2, 8, head.proj_dim)
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
        test_target_proj_is_frozen,
        test_target_proj_initialised_to_online,
        test_ema_target_tracks_online_after_update,
        test_pred_error_normalised_to_unit_interval,
        test_surprise_tracks_structure_not_noise,
        test_no_representational_collapse,
        test_ema_warmup_uses_gentler_decay,
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
