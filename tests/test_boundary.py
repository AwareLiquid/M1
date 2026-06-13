"""
tests/test_boundary.py — P2.2 boundary-case / long-run stability tests.

These guard the stateful v2.0 modules against pathologies that only surface
after a long pre-training run (hours/days), which the per-feature unit tests
(short, single-event) do not exercise:

  A. CausalConsistencyChecker under *sustained, repeated* topic switches —
     it must keep firing on each break and recovering between, never latching
     into a degenerate stuck state, and never emit NaN / out-of-range scores.
  B. CausalConsistencyChecker numerical robustness over hundreds of updates
     with degenerate inputs (zeros, tiny, huge), both methods.
  C. PredictiveStateHead over a long training run — surprise stays bounded in
     [0,1], does NOT saturate to 0 (collapse) or 1 (no learning), latent space
     does not collapse, and no buffer goes NaN.
  D. Full MTLNNModel with all four v2.0 modules ON, driven for many varied
     forward passes — every v2 diagnostic stays finite & bounded and GWT
     competition never collapses to a monopoly (entropy stays > 0).

All run on tiny dims / modest step counts to stay fast under pytest.
"""

import warnings

import pytest
import torch
import torch.optim as optim

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn.causality import CausalConsistencyChecker
from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from mt_lnn.world_model import PredictiveStateHead


# ===========================================================================
# A. Sustained, repeated topic switches
# ===========================================================================

def _regime(D, axis, scale=0.1):
    """Anisotropic vector: dominant shared component + small perturbation on `axis`."""
    v = torch.zeros(D)
    v[0] = 10.0
    v[axis] = 1.0 + torch.randn(1).item() * scale
    return v


def test_sustained_switches_fire_and_recover_subspace():
    """
    Over many consecutive topic switches, the subspace checker must (1) dip on
    each switch and (2) recover while a regime is held — i.e. it never latches
    into a permanently-broken or permanently-consistent state.
    """
    torch.manual_seed(0)
    D = 32
    c = CausalConsistencyChecker(window=6, ema_alpha=0.6, threshold=0.35,
                                 method="subspace")
    axes = [1, 5, 9, 13, 17]   # five distinct "topics"
    dips, recoveries = [], []

    # Establish the first regime.
    for _ in range(8):
        c.update(_regime(D, axes[0]))

    for r in range(1, len(axes)):
        # Switch: record the minimum score during the transition.
        mn = 1.0
        for _ in range(4):
            c.update(_regime(D, axes[r]))
            mn = min(mn, c.consistency_score())
        dips.append(mn)
        # Hold the new regime: score should climb back up.
        for _ in range(8):
            c.update(_regime(D, axes[r]))
        recoveries.append(c.consistency_score())

    # Every switch produced a real dip ...
    assert all(d < 0.6 for d in dips), f"a switch failed to register: dips={dips}"
    # ... and every hold recovered above the dip (not stuck low).
    assert all(rec > mn for rec, mn in zip(recoveries, dips)), \
        f"checker latched low: dips={dips} recoveries={recoveries}"
    # Final state is healthy, not degenerate.
    assert recoveries[-1] > 0.6, f"did not recover after last switch: {recoveries[-1]:.3f}"


def test_sustained_updates_scores_stay_in_unit_interval():
    """Hundreds of mixed updates: scores never escape [0,1] and never NaN."""
    for method in ("cosine", "subspace"):
        torch.manual_seed(1)
        c = CausalConsistencyChecker(window=8, ema_alpha=0.4, threshold=0.3,
                                     method=method)
        for i in range(400):
            v = torch.randn(48) * (1.0 + (i % 7))   # varied magnitudes
            s = c.update(v)
            assert s == s, f"{method}: NaN score at step {i}"
            assert 0.0 <= s <= 1.0, f"{method}: score {s} out of range at step {i}"
        # Window invariant survives the long run.
        assert len(c._history) <= c.window


# ===========================================================================
# B. Degenerate-input robustness
# ===========================================================================

def test_checker_handles_degenerate_inputs_no_nan():
    """Zeros, tiny, and huge vectors must not crash or produce NaN."""
    for method in ("cosine", "subspace"):
        c = CausalConsistencyChecker(window=5, method=method)
        feed = [
            torch.zeros(16),
            torch.ones(16) * 1e-9,
            torch.ones(16) * 1e6,
            torch.zeros(16),
            torch.randn(16),
            torch.zeros(16),
        ]
        for v in feed:
            s = c.update(v)
            assert s == s and 0.0 <= s <= 1.0, f"{method}: bad score {s}"
        # effective_rank is always a finite, sane number.
        er = c.effective_rank
        assert er == er and er >= 0.0


def test_subspace_constant_history_is_stable():
    """An all-identical (rank-deficient) window must not break the SVD path."""
    c = CausalConsistencyChecker(window=6, method="subspace")
    h = torch.ones(24)
    for _ in range(12):
        s = c.update(h)
        assert s == s and 0.0 <= s <= 1.0
    # Colinear window → effective rank near 1, score high (perfectly consistent).
    assert c.effective_rank < 1.5
    assert c.consistency_score() > 0.8


# ===========================================================================
# C. World model long-run stability
# ===========================================================================

def _ar1_batch(B=8, T=12, D=24, rho=0.9):
    xs = [torch.randn(B, D)]
    for _ in range(T - 1):
        xs.append(rho * xs[-1] + 0.2 * torch.randn(B, D))
    return torch.stack(xs, dim=1)


@pytest.mark.slow
def test_world_model_long_run_surprise_bounded_no_collapse():
    """
    Train the predictive head for a long run on structured data and assert:
      - surprise stays strictly inside [0,1] throughout (never NaN),
      - it ends well below chance (learned) but not pinned at 0 (collapse),
      - the latent space does not collapse (pairwise |cos| stays bounded).
    """
    torch.manual_seed(0)
    D = 24
    head = PredictiveStateHead(D, ema_decay=0.99, warmup_steps=100)
    opt = optim.Adam([p for p in head.parameters() if p.requires_grad], lr=3e-3)

    for step in range(700):
        x = _ar1_batch(D=D)
        _, loss = head(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        err = head.last_pred_error.item()
        assert err == err, f"surprise went NaN at step {step}"
        assert 0.0 <= err <= 1.0, f"surprise {err} out of range at step {step}"

    final_err = head.last_pred_error.item()
    assert final_err < 0.3, f"head failed to learn over long run: {final_err:.3f}"
    assert final_err > 1e-4, f"surprise collapsed to ~0: {final_err:.5f}"
    assert head.last_pred_error_raw.item() == head.last_pred_error_raw.item()

    # No representational collapse after the long run.
    torch.manual_seed(7)
    probe = torch.randn(16, 12, D)
    with torch.no_grad():
        z = head.online_proj(probe).reshape(-1, head.proj_dim)
    zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
    sim = zn @ zn.t()
    n = sim.shape[0]
    mean_off = (sim.abs().sum() - sim.diag().abs().sum()) / (n * (n - 1))
    assert mean_off.item() < 0.5, f"latent collapsed over long run: {mean_off.item():.3f}"


def test_world_model_constant_input_no_nan():
    """A degenerate constant sequence must not NaN the normalised surprise."""
    head = PredictiveStateHead(32)
    x = torch.ones(4, 10, 32)
    _, loss = head(x)
    assert loss is None or torch.isfinite(loss)
    err = head.last_pred_error.item()
    assert err == err and 0.0 <= err <= 1.0


# ===========================================================================
# D. Full model long-run diagnostics
# ===========================================================================

def _full_v2_model():
    cfg = MTLNNConfig(
        vocab_size=128, d_model=104, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=64, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
        use_competitive_gwtb=True, n_competitive_bids=3,
        use_world_model=True, use_hebbian=True, global_rhythm=True,
    )
    return MTLNNModel(cfg)


def test_full_model_long_run_diagnostics_bounded_and_no_collapse():
    """
    Drive a full v2 model through many varied forward passes and verify every
    v2 diagnostic stays finite & in-range, and GWT competition never collapses
    to a monopoly (entropy stays clearly > 0) across the whole run.
    """
    import math

    torch.manual_seed(0)
    model = _full_v2_model().eval()
    entropies = []
    surprises = []

    with torch.no_grad():
        for i in range(40):
            T = 8 + (i % 24)                      # varied sequence lengths
            ids = torch.randint(0, 128, (2, T))
            model(ids, labels=ids)
            diag = model.get_mt_diagnostics()

            # All v2 scalars finite.
            for k, v in diag.items():
                if isinstance(v, float):
                    assert math.isfinite(v), f"diag[{k}]={v} not finite at step {i}"

            ent = diag.get("gwtb_competition_entropy")
            if ent is not None:
                entropies.append(ent)
            surp = diag.get("world_model_pred_error")
            if surp is not None:
                assert 0.0 <= surp <= 1.0, f"surprise {surp} out of range"
                surprises.append(surp)

    # Competition never collapsed to a single winner (entropy floor > 0).
    assert entropies, "no competition entropy recorded"
    assert min(entropies) > 0.05, \
        f"GWT competition collapsed toward monopoly: min entropy={min(entropies):.4f}"
    # And it is below the uniform ceiling log(K) — it's a real distribution.
    assert max(entropies) <= math.log(3) + 1e-3


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

# ===========================================================================
# E. Decoding past max_seq_len (RoPE / attn-bias / GWTB causal tables must
#    extend on demand instead of silently truncating the position window).
# ===========================================================================

def test_incremental_decode_past_max_seq_len_stays_finite():
    """KV-cache decoding beyond the configured max_seq_len must keep producing
    finite logits. Regression: RoPE sin/cos, the attention _delta/_causal
    buffers, and the GWTB causal mask were all capped at max_seq_len and
    silently returned short slices → shape mismatch / wrong masking."""
    torch.manual_seed(0)
    max_len = 16
    cfg = MTLNNConfig(
        vocab_size=64, d_model=104, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=max_len, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
    )
    model = MTLNNModel(cfg).eval()
    ids = torch.randint(0, 64, (1, 4))
    with torch.no_grad():
        out = model(ids, use_cache=True)
        cache, off = out["cache"], 4
        cur = out["logits"][:, -1:].argmax(-1)
        # Decode well past max_seq_len (to ~4x the window).
        for _ in range(max_len * 4):
            o = model(cur, cache=cache, use_cache=True, position_offset=off)
            assert torch.isfinite(o["logits"]).all()
            cache, off = o["cache"], off + 1
            cur = o["logits"][:, -1:].argmax(-1)
    assert off > max_len, "test did not actually exceed max_seq_len"


def test_single_forward_longer_than_max_seq_len():
    """A one-shot forward on a sequence longer than max_seq_len must also work
    (full-attention path, no cache)."""
    torch.manual_seed(1)
    cfg = MTLNNConfig(
        vocab_size=64, d_model=104, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=16, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
    )
    model = MTLNNModel(cfg).eval()
    ids = torch.randint(0, 64, (1, 40))   # 40 > 16
    with torch.no_grad():
        out = model(ids, labels=ids)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["loss"])


if __name__ == "__main__":
    tests = [
        test_sustained_switches_fire_and_recover_subspace,
        test_sustained_updates_scores_stay_in_unit_interval,
        test_checker_handles_degenerate_inputs_no_nan,
        test_subspace_constant_history_is_stable,
        test_world_model_long_run_surprise_bounded_no_collapse,
        test_world_model_constant_input_no_nan,
        test_full_model_long_run_diagnostics_bounded_and_no_collapse,
        test_incremental_decode_past_max_seq_len_stays_finite,
        test_single_forward_longer_than_max_seq_len,
    ]
    for fn in tests:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except Exception:
            import traceback
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
            raise
