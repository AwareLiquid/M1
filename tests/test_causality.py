"""
tests/test_causality.py — CausalConsistencyChecker + deliberation integration.

Tests cover:
  1.  Neutral score at start (1.0 before first update)
  2.  Stable trajectory keeps high score
  3.  Anti-correlated jump drops score significantly
  4.  Oscillating trajectory stays low after threshold
  5.  Score recovers after trajectory re-stabilises
  6.  reset() restores state cleanly
  7.  Various h_prev shapes are accepted
  8.  Window size limits history correctly
  9.  RouteDecision has causal_consistency field
 10.  RouterThresholds has consistency_floor field
 11.  decide() with consistency_signal=None: backward-compat (no change)
 12.  decide() with low consistency → SELF_CRITIQUE (causal_break)
 13.  decide() with high consistency + low entropy → LOCAL (no override)
 14.  decide() causal_consistency field populated in decision
 15.  Full simulated inference loop: checker + router
"""

import math
import torch
import torch.nn.functional as F

from mt_lnn.causality import CausalConsistencyChecker
from mt_lnn.deliberation import (
    DeliberationRouter,
    RouterThresholds,
    RouteDecision,
    Route,
    token_entropy,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_checker(window=8, alpha=0.4, threshold=0.3):
    return CausalConsistencyChecker(window=window, ema_alpha=alpha, threshold=threshold)


def fill_stable(checker: CausalConsistencyChecker, vec: torch.Tensor, n: int):
    """Update checker n times with the same vector to build stable history."""
    for _ in range(n):
        checker.update(vec)


# ---------------------------------------------------------------------------
# 1. Neutral score at start
# ---------------------------------------------------------------------------

def test_initial_score_is_one():
    checker = make_checker()
    assert checker.consistency_score() == 1.0


def test_is_consistent_before_first_update():
    checker = make_checker()
    assert checker.is_consistent


def test_steps_seen_zero_at_start():
    checker = make_checker()
    assert checker.steps_seen == 0


# ---------------------------------------------------------------------------
# 2. Stable trajectory keeps score high
# ---------------------------------------------------------------------------

def test_stable_trajectory_score_stays_high():
    checker = make_checker(window=8, alpha=0.4)
    h = torch.randn(32)
    for _ in range(16):
        checker.update(h)
    assert checker.consistency_score() > 0.7, (
        f"stable trajectory score={checker.consistency_score():.3f} unexpectedly low"
    )


def test_stable_trajectory_is_consistent():
    checker = make_checker(threshold=0.5)
    h = torch.ones(16)
    for _ in range(12):
        checker.update(h)
    assert checker.is_consistent


# ---------------------------------------------------------------------------
# 3. Anti-correlated jump drops score
# ---------------------------------------------------------------------------

def test_anticorrelated_jump_drops_score():
    """
    Build a stable history then inject anti-correlated vectors.
    Score should drop significantly below the initial stable value.
    """
    checker = make_checker(window=6, alpha=0.6, threshold=0.3)
    D = 64
    h_stable = torch.ones(D)
    fill_stable(checker, h_stable, 10)
    score_stable = checker.consistency_score()

    # Anti-correlated: cos_sim = -1 → mapped to 0
    h_jump = -torch.ones(D)
    for _ in range(5):
        checker.update(h_jump)
    score_after = checker.consistency_score()

    assert score_after < score_stable - 0.2, (
        f"score did not drop: stable={score_stable:.3f}, after={score_after:.3f}"
    )


def test_first_anticorrelated_jump_below_threshold():
    """
    A single anti-correlated step from a stable trajectory should drop the
    score below threshold immediately.

    The checker detects the MOMENT of the break (not sustained wrong state).
    After the window refills with new-regime vectors, the score adapts —
    this is correct behaviour: the alarm fires at the transition boundary.
    """
    checker = make_checker(window=4, alpha=0.7, threshold=0.5)
    D = 32
    h_stable = torch.ones(D)
    fill_stable(checker, h_stable, 10)
    score_stable = checker.consistency_score()
    assert score_stable > 0.9, f"stable plateau too low: {score_stable:.3f}"

    # Single anti-correlated step: cos_sim = -1 → mapped to 0
    # EMA: 0.7 * 0 + 0.3 * 1.0 = 0.3 < threshold=0.5
    score_first_jump = checker.update(-torch.ones(D))
    assert score_first_jump < checker.threshold, (
        f"first jump score={score_first_jump:.3f} not below threshold={checker.threshold}"
    )
    assert not checker.is_consistent


# ---------------------------------------------------------------------------
# 4. Oscillating trajectory stays low
# ---------------------------------------------------------------------------

def test_oscillating_trajectory_stays_low():
    """Alternating +h / -h should keep score below stable baseline."""
    checker = make_checker(window=4, alpha=0.5)
    D = 32
    h_pos = torch.ones(D)
    h_neg = -torch.ones(D)
    fill_stable(checker, h_pos, 6)
    score_stable = checker.consistency_score()

    # Oscillate
    for i in range(10):
        checker.update(h_pos if i % 2 == 0 else h_neg)

    score_osc = checker.consistency_score()
    # Oscillating score should be noticeably below the stable plateau
    assert score_osc < score_stable - 0.1, (
        f"oscillating score={score_osc:.3f} not clearly below stable={score_stable:.3f}"
    )


# ---------------------------------------------------------------------------
# 5. Score recovers after re-stabilisation
# ---------------------------------------------------------------------------

def test_score_recovers_after_restabilisation():
    checker = make_checker(window=4, alpha=0.5)
    D = 32
    h_a = torch.ones(D)
    h_b = -torch.ones(D)

    fill_stable(checker, h_a, 8)
    for _ in range(4):
        checker.update(h_b)
    score_after_jump = checker.consistency_score()

    # Re-stabilise with h_b (new regime)
    fill_stable(checker, h_b, 12)
    score_recovered = checker.consistency_score()

    assert score_recovered > score_after_jump + 0.1, (
        f"score did not recover: jump={score_after_jump:.3f}, "
        f"recovered={score_recovered:.3f}"
    )


# ---------------------------------------------------------------------------
# 6. reset() clears state
# ---------------------------------------------------------------------------

def test_reset_restores_initial_state():
    checker = make_checker()
    h = torch.randn(16)
    for _ in range(10):
        checker.update(h)
    assert checker.steps_seen == 10

    checker.reset()
    assert checker.consistency_score() == 1.0
    assert checker.steps_seen == 0
    assert checker.is_consistent


def test_after_reset_behaves_like_fresh():
    checker = make_checker(window=4, alpha=0.7)
    h = -torch.ones(32)
    for _ in range(8):
        checker.update(h)
    checker.reset()

    # Should behave as if fresh — no history to compare against
    score_after_reset = checker.update(h)
    assert score_after_reset == 1.0, "first update after reset should be 1.0"


# ---------------------------------------------------------------------------
# 7. Various h_prev shapes are accepted
# ---------------------------------------------------------------------------

def test_accepts_4d_h_prev():
    checker = make_checker()
    h = torch.randn(2, 13, 5, 8)    # (B, P, S, D)
    score = checker.update(h)
    assert 0.0 <= score <= 1.0


def test_accepts_3d_h_prev():
    checker = make_checker()
    h = torch.randn(2, 13, 64)      # (B, P, D)
    score = checker.update(h)
    assert 0.0 <= score <= 1.0


def test_accepts_2d_h_prev():
    checker = make_checker()
    h = torch.randn(2, 832)         # (B, d_model)
    score = checker.update(h)
    assert 0.0 <= score <= 1.0


def test_accepts_1d_h_prev():
    checker = make_checker()
    h = torch.randn(64)
    score = checker.update(h)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 8. Window size limits history
# ---------------------------------------------------------------------------

def test_window_limits_history_buffer():
    W = 4
    checker = make_checker(window=W)
    h = torch.randn(16)
    for _ in range(W + 10):
        checker.update(h)
    # Internal deque should not exceed window size
    assert len(checker._history) <= W


# ---------------------------------------------------------------------------
# 9. RouteDecision has causal_consistency field
# ---------------------------------------------------------------------------

def test_route_decision_has_causal_field():
    d = RouteDecision(route=Route.LOCAL, reason="test", entropy=1.0)
    assert hasattr(d, "causal_consistency")
    assert d.causal_consistency is None


def test_route_decision_causal_field_assignable():
    d = RouteDecision(
        route=Route.SELF_CRITIQUE,
        reason="causal_break",
        entropy=2.5,
        causal_consistency=0.15,
    )
    assert d.causal_consistency == 0.15


# ---------------------------------------------------------------------------
# 10. RouterThresholds has consistency_floor
# ---------------------------------------------------------------------------

def test_router_thresholds_has_consistency_floor():
    t = RouterThresholds()
    assert hasattr(t, "consistency_floor")
    assert isinstance(t.consistency_floor, float)
    assert 0.0 < t.consistency_floor < 1.0


def test_router_thresholds_custom_floor():
    t = RouterThresholds(consistency_floor=0.5)
    assert t.consistency_floor == 0.5


# ---------------------------------------------------------------------------
# 11. decide() with consistency_signal=None: backward-compat
# ---------------------------------------------------------------------------

def test_decide_no_consistency_signal_low_entropy():
    """Low entropy + no consistency signal → LOCAL (unchanged from before)."""
    router = DeliberationRouter()
    # Logits heavily peaked on one token → low entropy
    logits = torch.zeros(100)
    logits[0] = 20.0
    d = router.decide(logits, query="test", evidence_log=[])
    assert d.route == Route.LOCAL
    assert d.causal_consistency is None


def test_decide_no_consistency_signal_high_entropy():
    """High entropy + fact gap + no consistency → CLOUD.

    100 uniform logits: H = log(100) ≈ 4.6.  Set high=4.0 so 4.6 > high,
    putting this squarely in the high-entropy branch that checks fact_gap.
    """
    router = DeliberationRouter(thresholds=RouterThresholds(low=2.0, high=4.0))
    logits = torch.zeros(100)  # uniform → H ≈ 4.6 > 4.0
    d = router.decide(logits, query="capital of France", evidence_log=[])
    assert d.route == Route.CLOUD


# ---------------------------------------------------------------------------
# 12. decide() with low consistency → SELF_CRITIQUE (causal_break)
# ---------------------------------------------------------------------------

def test_low_consistency_forces_self_critique():
    """Even with low-entropy (confident) logits, broken trajectory → SELF_CRITIQUE."""
    router = DeliberationRouter(thresholds=RouterThresholds(consistency_floor=0.4))
    # Very low entropy logits (model is "confident")
    logits = torch.zeros(100)
    logits[0] = 30.0

    decision = router.decide(
        logits,
        query="test",
        evidence_log=[],
        consistency_signal=0.2,   # below floor=0.4
    )
    assert decision.route == Route.SELF_CRITIQUE
    assert decision.reason == "causal_break"
    assert decision.causal_consistency == 0.2


def test_low_consistency_on_various_entropy_levels():
    """Causal break overrides routing at ANY entropy level."""
    router = DeliberationRouter(thresholds=RouterThresholds(
        low=3.0, high=5.0, consistency_floor=0.35
    ))
    for entropy_logit_scale in [30.0, 2.0, 0.0]:  # low / medium / high entropy
        logits = torch.zeros(100)
        if entropy_logit_scale > 0:
            logits[0] = entropy_logit_scale  # more peaked = lower entropy
        decision = router.decide(
            logits,
            query="query",
            evidence_log=[],
            consistency_signal=0.1,   # clear break
        )
        assert decision.route == Route.SELF_CRITIQUE, (
            f"expected SELF_CRITIQUE for scale={entropy_logit_scale}, "
            f"got {decision.route}"
        )
        assert decision.reason == "causal_break"


# ---------------------------------------------------------------------------
# 13. decide() with high consistency + low entropy → LOCAL (no override)
# ---------------------------------------------------------------------------

def test_high_consistency_does_not_override_local():
    """High consistency + low entropy → router should return LOCAL."""
    router = DeliberationRouter()
    logits = torch.zeros(100)
    logits[0] = 20.0  # low entropy
    decision = router.decide(
        logits,
        query="test",
        evidence_log=[],
        consistency_signal=0.9,   # clearly above floor=0.3
    )
    assert decision.route == Route.LOCAL


def test_above_floor_no_causal_override():
    """Consistency just above floor should not trigger override."""
    router = DeliberationRouter(thresholds=RouterThresholds(consistency_floor=0.3))
    logits = torch.zeros(100)
    logits[0] = 20.0
    decision = router.decide(
        logits,
        query="test",
        evidence_log=[],
        consistency_signal=0.31,  # just above floor
    )
    # Should follow normal entropy routing, not causal override
    assert decision.reason != "causal_break"


# ---------------------------------------------------------------------------
# 14. causal_consistency field is populated in decision
# ---------------------------------------------------------------------------

def test_causal_consistency_populated_when_break():
    router = DeliberationRouter()
    logits = torch.zeros(50)
    d = router.decide(logits, query="q", evidence_log=[], consistency_signal=0.1)
    assert d.causal_consistency == 0.1


def test_causal_consistency_none_when_no_signal():
    router = DeliberationRouter()
    logits = torch.zeros(50)
    logits[0] = 20.0
    d = router.decide(logits, query="q", evidence_log=[])
    assert d.causal_consistency is None


# ---------------------------------------------------------------------------
# 15. Full simulated inference loop
# ---------------------------------------------------------------------------

def test_full_inference_loop_no_break():
    """
    Simulate a smooth inference session.
    Router should stay LOCAL throughout (no causal override).
    """
    checker = make_checker(window=6, alpha=0.4, threshold=0.3)
    router = DeliberationRouter(thresholds=RouterThresholds(
        low=3.0, high=5.0, consistency_floor=0.3
    ))

    D = 64
    h_base = torch.randn(D)
    logits_confident = torch.zeros(100)
    logits_confident[0] = 15.0

    cloud_count = 0
    self_critique_due_to_causal = 0

    for step in range(20):
        # Smooth trajectory: slightly perturbed h_base
        h_t = h_base + 0.01 * torch.randn(D)
        score = checker.update(h_t)
        d = router.decide(
            logits_confident,
            query="steady query",
            evidence_log=[],
            consistency_signal=score,
        )
        if d.route == Route.CLOUD:
            cloud_count += 1
        if d.route == Route.SELF_CRITIQUE and d.reason == "causal_break":
            self_critique_due_to_causal += 1

    assert self_critique_due_to_causal == 0, \
        f"{self_critique_due_to_causal} causal breaks on smooth trajectory"


def test_full_inference_loop_with_break():
    """
    Simulate an inference session that hits a causal break mid-way.
    At least one step should trigger causal override.
    """
    checker = make_checker(window=4, alpha=0.6, threshold=0.4)
    router = DeliberationRouter(thresholds=RouterThresholds(
        low=3.0, high=5.0, consistency_floor=0.4
    ))

    D = 32
    h_stable = torch.ones(D)
    h_jump = -torch.ones(D)  # anti-correlated
    logits_confident = torch.zeros(100)
    logits_confident[0] = 20.0  # low entropy — model "confident"

    # Phase 1: stable
    for _ in range(10):
        checker.update(h_stable)

    # Phase 2: sudden jump (model hallucinates?)
    causal_overrides = 0
    for _ in range(8):
        score = checker.update(h_jump)
        d = router.decide(
            logits_confident,
            query="question",
            evidence_log=[],
            consistency_signal=score,
        )
        if d.route == Route.SELF_CRITIQUE and d.reason == "causal_break":
            causal_overrides += 1

    assert causal_overrides > 0, \
        "expected at least one causal override during trajectory break"


# ---------------------------------------------------------------------------
# Phase B v2.1: subspace method (anisotropy-robust break detection)
# ---------------------------------------------------------------------------

def _aniso_stable(D=32, scale=0.1):
    """Large shared component + small perturbation in subspace span{e1,e2}."""
    base = torch.zeros(D); base[0] = 10.0
    p = torch.zeros(D)
    p[1] = torch.randn(1).item() * scale
    p[2] = torch.randn(1).item() * scale
    return base + p


def _aniso_switched(D=32, scale=0.1):
    """Same large shared component, perturbation switched to span{e3,e4}."""
    base = torch.zeros(D); base[0] = 10.0
    p = torch.zeros(D)
    p[3] = torch.randn(1).item() * scale
    p[4] = torch.randn(1).item() * scale
    return base + p


def test_subspace_detects_break_cosine_blind():
    """
    The headline anisotropy case: a dominant shared direction masks a real
    topic switch from the cosine detector, but the subspace detector catches
    it. subspace must dip below 0.5 (here below the 0.35 self-critique floor),
    while cosine moves < 0.1.
    """
    torch.manual_seed(0)
    results = {}
    for method in ("cosine", "subspace"):
        torch.manual_seed(0)  # identical vector stream for a fair comparison
        c = CausalConsistencyChecker(window=6, ema_alpha=0.6,
                                     threshold=0.35, method=method)
        for _ in range(8):
            c.update(_aniso_stable())
        pre = c.consistency_score()
        min_during = pre
        for _ in range(8):
            c.update(_aniso_switched())
            min_during = min(min_during, c.consistency_score())
        results[method] = (pre, min_during)

    cos_pre, cos_min = results["cosine"]
    sub_pre, sub_min = results["subspace"]

    # Cosine is essentially blind: barely moves through the switch.
    assert abs(cos_pre - cos_min) < 0.1, \
        f"cosine unexpectedly reacted: {cos_pre:.3f} -> {cos_min:.3f}"
    # Subspace fires hard enough to cross the self-critique floor.
    assert sub_min < 0.5, f"subspace did not fire on the break: min={sub_min:.3f}"
    assert sub_pre - sub_min > 0.3, \
        f"subspace drop too small: {sub_pre:.3f} -> {sub_min:.3f}"


def test_cosine_method_is_backward_compatible():
    """
    method='cosine' (the default) must reproduce the exact pre-v2.1 mapping:
    EMA of (cos(h, window_mean)+1)/2.  Verified against a manual computation.
    """
    torch.manual_seed(1)
    D = 16
    vecs = [torch.randn(D) for _ in range(5)]

    c = CausalConsistencyChecker(window=8, ema_alpha=0.4, threshold=0.3,
                                 method="cosine")
    # Manual reference replicating the documented algorithm.
    import torch.nn.functional as F
    hist = []
    ref = 1.0
    for v in vecs:
        score = c.update(v)
        if hist:
            mean_ref = torch.stack(hist, dim=0).mean(dim=0)
            hn = F.normalize(v.unsqueeze(0), dim=-1, eps=1e-8).squeeze(0)
            mn = F.normalize(mean_ref.unsqueeze(0), dim=-1, eps=1e-8).squeeze(0)
            sim01 = max(0.0, min(1.0, (torch.dot(hn, mn).item() + 1.0) / 2.0))
            ref = 0.4 * sim01 + 0.6 * ref
        hist.append(v.clone())
        assert abs(score - ref) < 1e-5, f"cosine drift: {score:.6f} vs {ref:.6f}"


def test_effective_rank_diagnostic_tracks_complexity():
    """
    effective_rank ≈ 1 for a near-colinear window, and rises when the window
    spans several independent directions.
    """
    torch.manual_seed(2)
    D = 32
    base = torch.randn(D)

    # Colinear window (scaled copies of one direction) → eff_rank ≈ 1.
    c1 = CausalConsistencyChecker(window=6, method="subspace")
    for k in range(6):
        c1.update(base * (1.0 + 0.01 * k))
    assert c1.effective_rank < 1.5, f"colinear eff_rank too high: {c1.effective_rank:.2f}"

    # Diverse window (independent directions) → eff_rank well above 1.
    c2 = CausalConsistencyChecker(window=6, method="subspace")
    for _ in range(6):
        c2.update(torch.randn(D))
    assert c2.effective_rank > 2.0, f"diverse eff_rank too low: {c2.effective_rank:.2f}"


def test_principal_subspace_exclude_last_isolates_the_break():
    """exclude_last must drop the just-ingested suspect state from the basis.

    The canonical ``update(h); steer(h)`` order appends ``h`` before the subspace
    is built. If ``h`` is a break, *including* it lets its own direction become a
    principal axis — its off-subspace residual collapses to ~0 and a corrective
    projection removes nothing. Excluding it keeps the residual ≈ the full break
    magnitude, which is what makes steering actually act. Pin that gap.
    """
    torch.manual_seed(0)
    chk = CausalConsistencyChecker(window=8, method="subspace", energy_keep=0.9)
    base = torch.randn(1, 13, 5, 8)
    for _ in range(6):
        chk.update(base + 0.02 * torch.randn(1, 13, 5, 8))   # smooth legal past
    hbad = base + 4.0 * torch.randn(1, 13, 5, 8)             # an off-subspace jump
    chk.update(hbad)                                          # appends the suspect

    def off_residual(basis):
        Vk, mean = basis
        hf = hbad.reshape(-1).float() - mean
        return float((hf - Vk.t() @ (Vk @ hf)).norm())

    incl = chk.principal_subspace()                  # includes hbad (polluted)
    excl = chk.principal_subspace(exclude_last=True)  # legal past only
    assert incl is not None and excl is not None
    r_incl, r_excl = off_residual(incl), off_residual(excl)
    # Including the suspect all but erases its own residual; excluding keeps it.
    assert r_incl < 1.0, f"included-self residual unexpectedly large: {r_incl:.3f}"
    assert r_excl > 10.0 * max(r_incl, 1e-6), (
        f"exclude_last failed to isolate the break: incl={r_incl:.4f} excl={r_excl:.4f}"
    )


def test_principal_subspace_exclude_last_needs_two_remaining():
    """With only 2 states, excluding the last leaves 1 → no subspace (None)."""
    chk = CausalConsistencyChecker(window=8, method="subspace")
    chk.update(torch.randn(16))
    chk.update(torch.randn(16))
    assert chk.principal_subspace() is not None          # 2 states → ok
    assert chk.principal_subspace(exclude_last=True) is None  # 1 left → None


def test_invalid_method_raises():
    try:
        CausalConsistencyChecker(method="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown method")


def test_from_config_builds_checker():
    """CausalConsistencyChecker.from_config reads causal_check_* fields."""
    from mt_lnn.config import MTLNNConfig
    cfg = MTLNNConfig(
        vocab_size=64, d_model=104, n_layers=1, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=32, gwtb_n_heads=1,
        causal_check_method="subspace",
        causal_check_window=7,
        causal_check_threshold=0.42,
    )
    c = CausalConsistencyChecker.from_config(cfg)
    assert c.method == "subspace"
    assert c.window == 7
    assert abs(c.threshold - 0.42) < 1e-9
    # Overrides win.
    c2 = CausalConsistencyChecker.from_config(cfg, window=3)
    assert c2.window == 3 and c2.method == "subspace"


def test_subspace_break_triggers_self_critique():
    """
    End-to-end: an anisotropic break drives the subspace score below the
    router's consistency_floor, forcing SELF_CRITIQUE even on low-entropy
    (confident-looking) logits.
    """
    torch.manual_seed(0)
    c = CausalConsistencyChecker(window=6, ema_alpha=0.6,
                                 threshold=0.35, method="subspace")
    router = DeliberationRouter(thresholds=RouterThresholds(
        low=3.0, high=5.0, consistency_floor=0.35))

    for _ in range(8):
        c.update(_aniso_stable())

    # Low-entropy logits (confident) — without the causal check this routes LOCAL.
    confident_logits = torch.zeros(64); confident_logits[3] = 20.0

    fired = False
    for _ in range(8):
        score = c.update(_aniso_switched())
        d = router.decide(confident_logits, query="q", evidence_log=[],
                          consistency_signal=score)
        if d.route == Route.SELF_CRITIQUE and d.reason == "causal_break":
            fired = True
            break
    assert fired, "subspace break failed to trigger causal self-critique override"


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_initial_score_is_one,
        test_is_consistent_before_first_update,
        test_steps_seen_zero_at_start,
        test_stable_trajectory_score_stays_high,
        test_stable_trajectory_is_consistent,
        test_anticorrelated_jump_drops_score,
        test_anticorrelated_jump_below_threshold,
        test_oscillating_trajectory_stays_low,
        test_score_recovers_after_restabilisation,
        test_reset_restores_initial_state,
        test_after_reset_behaves_like_fresh,
        test_accepts_4d_h_prev,
        test_accepts_3d_h_prev,
        test_accepts_2d_h_prev,
        test_accepts_1d_h_prev,
        test_window_limits_history_buffer,
        test_route_decision_has_causal_field,
        test_route_decision_causal_field_assignable,
        test_router_thresholds_has_consistency_floor,
        test_router_thresholds_custom_floor,
        test_decide_no_consistency_signal_low_entropy,
        test_decide_no_consistency_signal_high_entropy,
        test_low_consistency_forces_self_critique,
        test_low_consistency_on_various_entropy_levels,
        test_high_consistency_does_not_override_local,
        test_above_floor_no_causal_override,
        test_causal_consistency_populated_when_break,
        test_causal_consistency_none_when_no_signal,
        test_full_inference_loop_no_break,
        test_full_inference_loop_with_break,
        test_subspace_detects_break_cosine_blind,
        test_cosine_method_is_backward_compatible,
        test_effective_rank_diagnostic_tracks_complexity,
        test_principal_subspace_exclude_last_isolates_the_break,
        test_principal_subspace_exclude_last_needs_two_remaining,
        test_invalid_method_raises,
        test_from_config_builds_checker,
        test_subspace_break_triggers_self_critique,
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
