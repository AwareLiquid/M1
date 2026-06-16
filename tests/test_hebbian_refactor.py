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


def test_compute_loss_returns_none_in_stage0():
    m = _build(use_refactor=True)
    m.train()
    assert m.hebbian_plasticity.compute_loss(m) is None


def test_forward_omits_refactor_loss_and_keeps_shape():
    m = _build(use_refactor=True)
    m.train()
    x = torch.randint(0, 64, (2, 16))
    out = m(x, labels=x.clone())
    assert "hebbian_refactor_loss" not in out
    assert out["logits"].shape == (2, 16, 64)
    assert torch.isfinite(out["loss"])


def test_zero_regression_bit_identical_off_vs_on():
    off = _build(use_refactor=False, seed=0)
    on = _build(use_refactor=True, seed=0)
    off.train()
    on.train()
    x = torch.randint(0, 64, (2, 16))
    torch.manual_seed(123)
    o_off = off(x, labels=x.clone())
    torch.manual_seed(123)
    o_on = on(x, labels=x.clone())
    assert torch.equal(o_off["logits"], o_on["logits"])
    assert torch.equal(o_off["loss"], o_on["loss"])


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


def test_compute_loss_still_none_in_stage1():
    # Stage 1 builds the signal but does NOT add it to the loss yet.
    m = _build(use_refactor=True)
    _ = _forward_then_signal(m)
    assert m.hebbian_plasticity.compute_loss(m) is None
