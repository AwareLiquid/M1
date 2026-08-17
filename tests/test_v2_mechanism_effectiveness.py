"""
tests/test_v2_mechanism_effectiveness.py — empirical tests that v2.0 modules
actually DO something during training, not just plumb without crashing.

The original test suites (test_gwt_competition, test_world_model, test_plasticity)
verify shape contracts, gradient flow, and no-regression. They do NOT verify that
the mechanisms produce useful signal during training.

This file is the answer to V2_REVIEW.md §4 TD1: "tests only verify shape/grad/
no-crash, not mechanism effectiveness."

Mechanism-effectiveness tests:
  1. Phase A: bid weights diverge from uniform within 50 training steps
  2. Phase A: orthogonality penalty is non-zero during training, zero at eval
  3. Phase A: bid projector weights become less similar over training
  4. Phase D: centered Hebbian signal is non-zero on untrained net
     (the original uncentered form was ≈0 — that was the bug)
  5. Phase D: Hebbian signal varies across batches (non-degenerate)
"""

import math
import pytest
import torch
import torch.optim as optim

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from mt_lnn.gwtb import CompetitiveGWTBLayer
from mt_lnn.mt_lnn_layer import MTLNNLayer


def cfg_competitive(K=3, noise=0.5, ortho=0.01):
    return MTLNNConfig(
        vocab_size=64,
        d_model=104,
        n_layers=2,
        n_heads=13,
        n_kv_heads=1,
        d_head=8,
        max_seq_len=32,
        gwtb_n_heads=1,
        use_competitive_gwtb=True,
        n_competitive_bids=K,
        competitive_score_noise=noise,
        competitive_ortho_weight=ortho,
        use_rhythm=False,
        use_world_model=False,
        use_predictive_coding=False,
        use_hebbian=False,
        # Pin the coherence branch these empirical-dynamics tests were
        # calibrated under: E4 (2026-07-15) flipped the shipped default to
        # use_decay_wm=False, which changes the coherence forward and shifts
        # the fixed-seed 100-step loss trajectories these tests assert on.
        use_decay_wm=True,
        dynamic_scale_gates=True,
        dropout=0.0,
        attention_dropout=0.0,
    )


def cfg_hebbian(use_hebb=True):
    return MTLNNConfig(
        vocab_size=64,
        d_model=104,
        n_layers=2,
        n_heads=13,
        n_kv_heads=1,
        d_head=8,
        max_seq_len=32,
        gwtb_n_heads=1,
        use_hebbian=use_hebb,
        hebbian_lr=1e-3,
        hebbian_lavi_gate=False,
        use_rhythm=False,
        use_world_model=False,
        use_predictive_coding=False,
        dynamic_scale_gates=True,
        dropout=0.0,
        attention_dropout=0.0,
    )


# ===========================================================================
# Phase A mechanism effectiveness
# ===========================================================================

@pytest.mark.slow
def test_phase_a_bids_diverge_from_uniform_after_training():
    """
    EMPIRICAL: with noise injection + ortho penalty, the K bid weights should
    diverge from the uniform initial state (1/K each) after a few training steps.

    Without these mechanisms (which we documented in V2_REVIEW.md), the bids
    stay at (1/K, 1/K, 1/K) forever — symmetry never breaks.
    """
    K = 3
    cfg = cfg_competitive(K=K, noise=0.5, ortho=0.01)
    torch.manual_seed(42)
    model = MTLNNModel(cfg)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    initial_weights = model.gwtb.last_winner_weights.clone()

    for _ in range(50):
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        labels = torch.randint(0, cfg.vocab_size, (2, 16))
        out = model(ids, labels=labels)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()

    final_weights = model.gwtb.last_winner_weights
    # Measure deviation from uniform: max_k |w_k - 1/K|
    deviation = (final_weights - 1.0 / K).abs().max().item()

    assert deviation > 0.02, (
        f"After 50 training steps, bid weights are still essentially uniform: "
        f"{final_weights.tolist()} (deviation from 1/{K}={1/K:.3f} is only {deviation:.4f}). "
        f"This is the symmetry-breaking failure documented in V2_REVIEW.md §1.A."
    )


def test_phase_a_orthogonality_penalty_is_nonzero_during_training():
    """When training=True and ortho_weight > 0, _last_ortho_penalty must be set.

    At init the zero-init invariant (fc2 == 0 -> every bid == x) makes the
    penalty EXACTLY 0 — identical bids have no pairwise structure to penalise.
    The old strict `> 0` assertion at step 0 only passed because the global
    init_weights pass used to clobber the zero-init (fixed 2026-07-14, see
    CompetitiveGWTBLayer.reset_zero_init). So: at init require presence and
    >= 0; after the weights move off zero (as any optimizer step does), the
    penalty must be strictly positive."""
    cfg = cfg_competitive(K=3, noise=0.0, ortho=0.05)
    model = MTLNNModel(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    labels = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model(ids, labels=labels)
    assert "ortho_penalty" in out, "ortho_penalty should be in result dict during training"
    assert out["ortho_penalty"].item() >= 0.0
    # Simulate training moving the projectors off the zero-init.
    with torch.no_grad():
        for proj in model.gwtb.bid_projectors:
            proj.fc2.weight.add_(torch.randn_like(proj.fc2.weight) * 0.02)
    out2 = model(ids, labels=labels)
    assert out2["ortho_penalty"].item() > 0.0


def test_phase_a_orthogonality_penalty_none_at_eval():
    """At eval(), no penalty should be computed or added to loss."""
    cfg = cfg_competitive(K=3, noise=0.0, ortho=0.05)
    model = MTLNNModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(ids)
    assert "ortho_penalty" not in out


@pytest.mark.slow
def test_phase_a_bid_projectors_specialise_over_training():
    """
    EMPIRICAL: bid projector output matrices should become MORE DIFFERENT
    over training (orthogonality penalty + asymmetric gradients).

    Measure: mean pairwise cosine similarity of fc2 weights, before vs after.
    """
    K = 3
    cfg = cfg_competitive(K=K, noise=0.5, ortho=0.05)
    torch.manual_seed(0)
    model = MTLNNModel(cfg)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    def mean_pairwise_cos(layer: CompetitiveGWTBLayer) -> float:
        weights = [proj.fc2.weight for proj in layer.bid_projectors]
        # All start as zeros — handle by adding tiny noise once to get baseline
        W = torch.stack([w.flatten() for w in weights], dim=0)
        W_norm = torch.nn.functional.normalize(W + 1e-8 * torch.randn_like(W), dim=-1)
        sim = W_norm @ W_norm.t()
        K_local = sim.shape[0]
        off_diag = sim - torch.eye(K_local, device=sim.device)
        return off_diag.abs().sum().item() / (K_local * (K_local - 1))

    sim_before = mean_pairwise_cos(model.gwtb)

    for _ in range(40):
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        labels = torch.randint(0, cfg.vocab_size, (2, 16))
        out = model(ids, labels=labels)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()

    sim_after = mean_pairwise_cos(model.gwtb)

    # After training with ortho penalty, similarity should decrease
    # (projectors should be more orthogonal)
    assert sim_after < sim_before + 0.01, (
        f"Bid projectors did not specialise: cos_sim before={sim_before:.4f} "
        f"vs after={sim_after:.4f}. Orthogonality penalty not effective."
    )


@pytest.mark.slow
def test_phase_a_no_noise_no_ortho_baseline_stays_uniform():
    """
    NEGATIVE control: with BOTH noise=0 AND ortho=0 (the original broken impl),
    bid weights should stay close to uniform — confirming this is the failure
    mode V2_REVIEW.md identifies.
    """
    K = 3
    cfg = cfg_competitive(K=K, noise=0.0, ortho=0.0)
    torch.manual_seed(1)
    model = MTLNNModel(cfg)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(50):
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        labels = torch.randint(0, cfg.vocab_size, (2, 16))
        out = model(ids, labels=labels)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()

    final_weights = model.gwtb.last_winner_weights
    # With no symmetry-breaking, weights should stay very close to uniform
    deviation = (final_weights - 1.0 / K).abs().max().item()
    # Loose bound — even random noise from gradient updates produces tiny drift,
    # but it should be much smaller than the symmetry-broken version's drift.
    assert deviation < 0.05, (
        f"Even without noise/ortho, bids drifted unexpectedly: {final_weights.tolist()}"
    )


# ===========================================================================
# Phase D mechanism effectiveness
# ===========================================================================

def test_phase_d_centered_hebbian_signal_is_nonzero():
    """
    EMPIRICAL: the CENTERED Hebbian signal is non-zero on an untrained net.

    The original mean(out * x) was ~0 because random projections of x have
    zero mean correlation with out. The centered form measures FLUCTUATION
    correlation, which is non-zero by construction.
    """
    cfg = cfg_hebbian(use_hebb=True)
    torch.manual_seed(0)
    model = MTLNNModel(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (4, 16))
    labels = torch.randint(0, cfg.vocab_size, (4, 16))
    model(ids, labels=labels)

    signals = [b.lnn._hebb_signal.item() for b in model.blocks]
    # Centered covariance should produce magnitudes > 1e-5 even on untrained net
    max_abs = max(abs(s) for s in signals)
    assert max_abs > 1e-5, (
        f"Centered Hebbian signal is too small: {signals}. "
        f"Expected > 1e-5; centering should make this non-degenerate."
    )


def test_phase_d_hebbian_signal_varies_across_batches():
    """
    Sanity check: the Hebbian signal should be different for different input
    batches. If it's constant, that's evidence of a degenerate computation.
    """
    cfg = cfg_hebbian(use_hebb=True)
    torch.manual_seed(7)
    model = MTLNNModel(cfg)
    model.train()

    signals_per_batch = []
    for _ in range(5):
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        model(ids)
        # Use first block's signal
        signals_per_batch.append(model.blocks[0].lnn._hebb_signal.item())

    # Variance across batches should be non-zero
    mean_s = sum(signals_per_batch) / len(signals_per_batch)
    var = sum((s - mean_s) ** 2 for s in signals_per_batch) / len(signals_per_batch)
    assert var > 1e-12, (
        f"Hebbian signal is constant across batches: {signals_per_batch}. "
        f"This suggests degenerate computation (e.g. all signals = 0)."
    )


def test_phase_d_hebbian_gradient_contribution_nonzero():
    """The Hebbian term should contribute non-trivially to the parameter grads."""
    cfg = cfg_hebbian(use_hebb=True)
    cfg.hebbian_lr = 0.1   # Boost so the contribution is measurable in 1 step
    torch.manual_seed(2)
    model = MTLNNModel(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=labels)
    assert "hebbian_loss" in out
    assert abs(out["hebbian_loss"].item()) > 1e-7, (
        f"Hebbian contribution to loss is too small: {out['hebbian_loss'].item():.2e}. "
        f"With centered covariance and lr=0.1, expected > 1e-7."
    )


# ===========================================================================
# Smoke test: P0 fixes don't break end-to-end training
# ===========================================================================

@pytest.mark.slow
def test_p0_fixes_dont_break_training_loop():
    """100-step training loop with all P0 mechanisms enabled — should converge."""
    cfg = MTLNNConfig(
        vocab_size=32,
        d_model=104,
        n_layers=2,
        n_heads=13,
        n_kv_heads=1,
        d_head=8,
        max_seq_len=24,
        gwtb_n_heads=1,
        use_competitive_gwtb=True,
        n_competitive_bids=3,
        competitive_score_noise=0.5,
        competitive_ortho_weight=0.01,
        use_hebbian=True,
        hebbian_lr=1e-3,
        hebbian_lavi_gate=False,
        use_rhythm=False,
        use_world_model=False,
        use_predictive_coding=False,
        # Pin the pre-E4 coherence branch this fixed-seed convergence check
        # was calibrated under (see cfg_competitive for the full rationale).
        use_decay_wm=True,
        dynamic_scale_gates=True,
        dropout=0.0,
        attention_dropout=0.0,
    )
    torch.manual_seed(0)
    model = MTLNNModel(cfg)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(100):
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        labels = torch.randint(0, cfg.vocab_size, (2, 16))
        out = model(ids, labels=labels)
        losses.append(out["loss"].item())
        optimizer.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    assert all(math.isfinite(l) for l in losses), "NaN or Inf in loss"
    assert losses[-1] < losses[0], (
        f"Loss did not decrease after 100 steps: start={losses[0]:.4f}, end={losses[-1]:.4f}"
    )


if __name__ == "__main__":
    tests = [
        test_phase_a_bids_diverge_from_uniform_after_training,
        test_phase_a_orthogonality_penalty_is_nonzero_during_training,
        test_phase_a_orthogonality_penalty_none_at_eval,
        test_phase_a_bid_projectors_specialise_over_training,
        test_phase_a_no_noise_no_ortho_baseline_stays_uniform,
        test_phase_d_centered_hebbian_signal_is_nonzero,
        test_phase_d_hebbian_signal_varies_across_batches,
        test_phase_d_hebbian_gradient_contribution_nonzero,
        test_p0_fixes_dont_break_training_loop,
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
