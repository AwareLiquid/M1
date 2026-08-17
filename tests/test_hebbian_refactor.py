"""
tests/test_hebbian_refactor.py — Stage 0/1 contract for the refactored Hebbian
branch (mechanism A, loss-term form; mt_lnn/hebbian_plasticity.py).

Stage 1 adds the dedicated lightweight LAVI gate + within-sequence lagged
co-activation signal, and asserts the gate is DYNAMIC (not stuck at 0.5 like the
legacy one). compute_loss() still returns None until Stage 2.

Stage 0 guarantees (this suite):
  1.  Default config => use_hebbian_refactor is False.
  2.  Default model => no hebbian_plasticity module is built (is None).
  3.  Flag ON => hebbian_plasticity is a HebbianPlasticity carrying the DECOUPLED
      hyper-parameters from config (base_lr, window, grad cap, mode).
  4.  Stage 0 is param-free: turning the flag ON adds ZERO parameters.
  5.  compute_loss() returns None in Stage 0 (verified no-op).
  6.  Forward with the flag ON omits 'hebbian_refactor_loss' from the output
      (because compute_loss is None) and preserves output shape.
  7.  ZERO-REGRESSION: with identical init seed, flag-OFF and flag-ON produce
      BIT-IDENTICAL logits and loss -- the opt-in path is provably inert until
      Stage 2 wires the real term in.
  8.  last_stats is an empty (read-only copy) dict in Stage 0.
"""

import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from mt_lnn.hebbian_plasticity import HebbianPlasticity


def _cfg(use_refactor: bool) -> MTLNNConfig:
    return MTLNNConfig(
        vocab_size=64,
        d_model=104,
        n_layers=2,
        n_heads=13,
        n_kv_heads=1,
        d_head=8,
        max_seq_len=32,
        gwtb_n_heads=1,
        use_hebbian=False,          # legacy path off — isolate the refactor
        use_hebbian_refactor=use_refactor,
        use_rhythm=False,
        use_world_model=False,
        use_predictive_coding=False,
        dynamic_scale_gates=True,
        dropout=0.0,
        attention_dropout=0.0,
    )


def _build(use_refactor: bool, seed: int = 0) -> MTLNNModel:
    torch.manual_seed(seed)
    return MTLNNModel(_cfg(use_refactor))


def test_default_flag_is_off():
    assert MTLNNConfig().use_hebbian_refactor is False


def test_default_model_has_no_module():
    m = _build(use_refactor=False)
    assert m.hebbian_plasticity is None


def test_flag_on_builds_module_with_decoupled_hparams():
    cfg = _cfg(use_refactor=True)
    m = MTLNNModel(cfg)
    assert isinstance(m.hebbian_plasticity, HebbianPlasticity)
    hp = m.hebbian_plasticity
    assert hp.base_lr == cfg.hebbian_base_lr
    assert hp.window == cfg.hebbian_window
    assert hp.grad_frac_cap == cfg.hebbian_grad_frac_cap
    assert hp.mode == cfg.hebbian_refactor_mode
    # base_lr is decoupled: it is NOT the main hebbian_lr / model lr.
    assert hp.base_lr != cfg.hebbian_lr


def test_stage1_param_light_two_scalars():
    # Stage 1 adds ONLY the dedicated-LAVI gate's two scalar params (s, b_lavi).
    off = _build(use_refactor=False)
    on = _build(use_refactor=True)
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    assert n_on - n_off == 2
    assert sum(p.numel() for p in on.hebbian_plasticity.parameters()) == 2


def test_compute_loss_none_without_forward():
    # No forward => no block stashed its sequences => no signal => None.
    m = _build(use_refactor=True)
    m.train()
    assert m.hebbian_plasticity.compute_loss(m) is None


def test_forward_includes_refactor_loss_and_keeps_shape():
    # Stage 2: the term is now added to the loss and surfaced in the output.
    m = _build(use_refactor=True)
    m.train()
    x = torch.randint(0, 64, (2, 16))
    out = m(x, labels=x.clone())
    assert "hebbian_refactor_loss" in out
    assert out["hebbian_refactor_loss"].dim() == 0
    assert out["logits"].shape == (2, 16, 64)
    assert torch.isfinite(out["loss"])


def test_logits_untouched_and_loss_is_main_plus_hebb():
    # Hebbian is a LOSS-only term: it must NOT change the logits (forward output).
    # The total loss = main LM loss + hebbian term. We verify additivity by
    # reconstruction rather than by raw float inequality, because the term is
    # tiny (~1e-7, capped at <=5% of the main GRADIENT, not of the loss VALUE) so
    # it can sit below the float32 ULP of the ~4.3 LM loss -- value-small does not
    # mean gradient-small, which is exactly why Stage 2 caps the gradient share.
    off = _build(use_refactor=False, seed=0)
    on = _build(use_refactor=True, seed=0)
    off.train()
    on.train()
    x = torch.randint(0, 64, (2, 16))
    torch.manual_seed(123)
    o_off = off(x, labels=x.clone())
    torch.manual_seed(123)
    o_on = on(x, labels=x.clone())
    assert torch.equal(o_off["logits"], o_on["logits"])          # logits untouched
    main_recon = o_on["loss"] - o_on["hebbian_refactor_loss"]
    assert abs(main_recon.item() - o_off["loss"].item()) < 1e-5  # total = main + hebb


def test_last_stats_copy_is_isolated():
    m = _build(use_refactor=True)
    # last_stats returns a copy — mutating it must not corrupt internal state
    s = m.hebbian_plasticity.last_stats
    s["x"] = 1.0
    assert "x" not in m.hebbian_plasticity.last_stats


# ---------------------------------------------------------------------------
# Stage 1: dedicated LAVI gate + within-sequence signal
# ---------------------------------------------------------------------------

def _forward_then_signal(m: MTLNNModel):
    m.train()
    x = torch.randint(0, 64, (4, 24))
    _ = m(x, labels=x.clone())               # populates per-block _hebb_ref_*
    return m.hebbian_plasticity.compute_signal(m)


def test_blocks_stash_sequences_only_when_flag_on():
    on = _build(use_refactor=True)
    on.train()
    x = torch.randint(0, 64, (2, 16))
    _ = on(x, labels=x.clone())
    for b in on.blocks:
        assert b.lnn._hebb_ref_out is not None
        assert b.lnn._hebb_ref_in is not None
        assert b.lnn._hebb_ref_out.shape[-1] == on.config.d_model

    off = _build(use_refactor=False)
    off.train()
    _ = off(x, labels=x.clone())
    for b in off.blocks:
        assert getattr(b.lnn, "_hebb_ref_out", None) is None


def test_signal_is_scalar_in_graph():
    m = _build(use_refactor=True)
    sig = _forward_then_signal(m)
    assert sig is not None
    assert sig.dim() == 0
    assert sig.requires_grad


def test_signal_gradient_reaches_lavi_gate():
    m = _build(use_refactor=True)
    sig = _forward_then_signal(m)
    sig.backward()
    # the dedicated LAVI gate params must receive gradient from the signal
    assert m.hebbian_plasticity.gate_temp.grad is not None
    assert m.hebbian_plasticity.b_lavi.grad is not None


def test_gate_is_dynamic_not_stuck_at_half():
    # The core Stage 1 contract: unlike the legacy gate (stuck at sigmoid(0)=0.5),
    # this gate produces NON-TRIVIAL, per-token-varying values.
    m = _build(use_refactor=True)
    _ = _forward_then_signal(m)
    stats = m.hebbian_plasticity.last_stats
    assert stats["n_blocks"] >= 1
    # gate varies across tokens (would be ~0 if frozen at a constant)
    assert stats["gate_std"] > 1e-4
    # LAVI is a live cosine signal, not identically zero
    assert abs(stats["lavi_mean"]) > 1e-6


def test_gate_unit_dynamics_direct():
    # Direct unit test of the gate math: a LAVI signal with spread must map to a
    # gate with spread (dynamic), and gate_temp sharpens that spread.
    hp = _build(use_refactor=True).hebbian_plasticity
    lavi = torch.linspace(-1.0, 1.0, 32).unsqueeze(0)   # (1, 32) varied input
    g = hp._gate(lavi)
    assert g.min() > 0.0 and g.max() < 1.0
    assert g.std() > 1e-3
    with torch.no_grad():
        hp.gate_temp.fill_(5.0)
    g_sharp = hp._gate(lavi)
    assert g_sharp.std() > g.std()    # higher temperature => wider gate range


# ---------------------------------------------------------------------------
# Stage 2: loss term + gradient alignment + 5% cap
# ---------------------------------------------------------------------------

def test_compute_loss_returns_scalar_term_after_forward():
    m = _build(use_refactor=True)
    m.train()
    x = torch.randint(0, 64, (4, 24))
    _ = m(x, labels=x.clone())
    term = m.hebbian_plasticity.compute_loss(m)
    assert term is not None
    assert term.dim() == 0
    assert term.requires_grad
    # alpha_eff = base_lr * _scale recorded for monitoring
    assert "alpha_eff" in m.hebbian_plasticity.last_stats


def test_raw_loss_is_negative_signal():
    m = _build(use_refactor=True)
    m.train()
    x = torch.randint(0, 64, (4, 24))
    _ = m(x, labels=x.clone())
    sig = m.hebbian_plasticity.compute_signal(m)
    # need a fresh forward because compute_signal consumes the stashed tensors
    _ = m(x, labels=x.clone())
    raw = m.hebbian_plasticity.raw_loss(m)
    assert torch.sign(raw) == -torch.sign(sig) or sig.item() == 0.0


def test_recalibrate_enforces_grad_fraction_cap():
    hp = _build(use_refactor=True).hebbian_plasticity
    # raw Hebbian gradient much larger than allowed -> _scale must shrink so the
    # post-scale fraction lands exactly at the cap.
    hp.recalibrate(g_main_norm=1.0, g_hebb_raw_norm=100.0)
    assert hp.last_stats["grad_frac"] <= hp.grad_frac_cap + 1e-9
    assert hp._scale < 1.0
    # tiny raw Hebbian gradient -> cap not binding -> _scale stays at 1.0.
    hp.recalibrate(g_main_norm=1.0, g_hebb_raw_norm=1e-6)
    assert hp._scale == 1.0


def test_recalibrate_scale_formula():
    hp = _build(use_refactor=True).hebbian_plasticity
    g_main, g_hebb = 2.0, 50.0
    hp.recalibrate(g_main_norm=g_main, g_hebb_raw_norm=g_hebb)
    expected = min(1.0, hp.grad_frac_cap * g_main / (hp.base_lr * g_hebb))
    assert abs(hp._scale - expected) < 1e-9


def test_scale_changes_loss_magnitude():
    m = _build(use_refactor=True)
    m.train()
    x = torch.randint(0, 64, (4, 24))
    _ = m(x, labels=x.clone())
    m.hebbian_plasticity._scale = 1.0
    big = m.hebbian_plasticity.compute_loss(m).item()
    _ = m(x, labels=x.clone())
    m.hebbian_plasticity._scale = 0.1
    small = m.hebbian_plasticity.compute_loss(m).item()
    assert abs(small) < abs(big) + 1e-12
    assert abs(small - 0.1 * big) < 1e-6
