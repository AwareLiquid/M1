"""
tests/test_active_inference.py — behavioural contract for Expected-Free-Energy
autonomous goal selection (mt_lnn/active_inference.py, 2026-06-15).

ActiveInferencePlanner scores a bank of candidate goal latents against an
imagined future (from LatentImagination) and selects the goal that minimises
Expected Free Energy (Friston 2015/2017; Da Costa 2020):

    G(goal) = -(w_p * pragmatic(goal) + w_e * epistemic(goal))

  * pragmatic (exploit): confidence-weighted alignment of the imagined
    trajectory with the goal -- goals the world model expects to reach;
  * epistemic (explore): novelty of the goal vs the present latent -- goals far
    from "now" promise more information.

We pin:
  * shapes/types and that the selected index is argmin(EFE);
  * pure pragmatic (w_e=0) over a MOVING imagined trajectory picks the goal the
    trajectory actually heads toward (not the present);
  * pure epistemic (w_p=0) picks the most novel (farthest-from-present) goal;
  * the explore/exploit knob: raising info_gain_weight flips the choice from a
    near reachable goal to a novel far goal;
  * zero trainable params; duck-typed on imagination (no model.py coupling);
  * determinism, batch handling, and input validation.

Run:  python -m pytest tests/test_active_inference.py -v
"""
import math
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from mt_lnn import (                                                  # noqa: E402
    ActiveInferencePlanner,
    EFEPlan,
    LatentImagination,
)


# --------------------------------------------------------------------------- #
# a controllable stub head: online_proj = identity, predictor = configurable   #
# --------------------------------------------------------------------------- #
class _StubHead(nn.Module):
    """Minimal PredictiveStateHead-like object with a known latent dynamic.

    online_proj is identity (P == d_model), so the present latent is just the
    (normalised) hidden state; predictor is a caller-supplied callable so the
    imagined trajectory is exactly predictable.
    """

    def __init__(self, P, predictor=None, last_pred_error=0.0):
        super().__init__()
        self.proj_dim = P
        self.online_proj = nn.Identity()
        self.predictor = predictor if predictor is not None else nn.Identity()
        self.register_buffer("last_pred_error", torch.tensor(float(last_pred_error)))


def _basis(d, i):
    v = torch.zeros(d)
    v[i] = 1.0
    return v


def _rotation(d, theta):
    """Block 2D rotation in the (0,1) plane; orthogonal (norm-preserving)."""
    A = torch.eye(d)
    c, s = math.cos(theta), math.sin(theta)
    A[0, 0] = c; A[0, 1] = -s
    A[1, 0] = s; A[1, 1] = c
    return A


def _static_planner(P=8, **kw):
    """Planner over an identity-predictor head: the trajectory sits at present."""
    head = _StubHead(P)
    imag = LatentImagination(head, horizon=3)
    return ActiveInferencePlanner(imag, **kw)


# --------------------------------------------------------------------------- #
# shapes / selection                                                          #
# --------------------------------------------------------------------------- #
def test_score_goals_shapes_and_argmin():
    P = 8
    planner = _static_planner(P)
    hidden = torch.randn(2, P)
    goals = torch.randn(5, P)
    plan = planner.score_goals(hidden, goals)

    assert isinstance(plan, EFEPlan)
    assert plan.efe.shape == (2, 5)
    assert plan.pragmatic.shape == (2, 5)
    assert plan.epistemic.shape == (2, 5)
    assert plan.selected.shape == (2,)
    # selected is exactly the argmin of EFE.
    assert torch.equal(plan.selected, torch.argmin(plan.efe, dim=-1))
    # values bounded in [0, 1].
    for t in (plan.pragmatic, plan.epistemic):
        assert (t >= 0.0).all() and (t <= 1.0).all()
    # chosen_goals returns the selected normalised goals.
    cg = plan.chosen_goals()
    assert cg.shape == (2, P)


def test_single_goal_vector_is_accepted():
    P = 6
    planner = _static_planner(P)
    hidden = torch.randn(1, P)
    plan = planner.score_goals(hidden, _basis(P, 0))     # (P,) -> (1 goal)
    assert plan.efe.shape == (1, 1)
    assert plan.selected.tolist() == [0]


# --------------------------------------------------------------------------- #
# pragmatic value tracks the imagined FUTURE, not the present                  #
# --------------------------------------------------------------------------- #
def test_pure_pragmatic_picks_goal_the_trajectory_heads_toward():
    # Moving trajectory: predictor = 90-degree rotation, so from z0=e0 the
    # imagined latents are e1 (step 1) then -e0 (step 2). A goal at e1 lies on
    # the predicted path; a goal at e0 (the present) does not.
    P = 8
    head = _StubHead(P, predictor=lambda x, R=_rotation(P, math.pi / 2): x @ R.T)
    planner = ActiveInferencePlanner(
        LatentImagination(head, horizon=2),
        pragmatic_weight=1.0,
        info_gain_weight=0.0,                            # pure exploit
    )
    hidden = _basis(P, 0).unsqueeze(0)                   # present = e0
    goals = torch.stack([_basis(P, 1), _basis(P, 0)])    # [on-path e1, present e0]
    plan = planner.score_goals(hidden, goals)
    # The trajectory heads toward e1, so the reachable goal (index 0) wins.
    assert plan.selected.tolist() == [0]
    assert plan.pragmatic[0, 0] > plan.pragmatic[0, 1]


# --------------------------------------------------------------------------- #
# epistemic value rewards novelty                                             #
# --------------------------------------------------------------------------- #
def test_pure_epistemic_picks_most_novel_goal():
    # Identity predictor -> trajectory stays at present e0. With pure epistemic
    # weighting the agent should pick the goal FARTHEST from the present.
    P = 8
    planner = _static_planner(P, pragmatic_weight=0.0, info_gain_weight=1.0)
    hidden = _basis(P, 0).unsqueeze(0)                   # present = e0
    goals = torch.stack([
        _basis(P, 0),                                    # identical to present (novelty 0)
        _basis(P, 1),                                    # orthogonal (novelty 0.5)
        -_basis(P, 0),                                   # opposite (novelty 1.0)
    ])
    plan = planner.score_goals(hidden, goals)
    assert plan.selected.tolist() == [2]                 # the opposite goal is most novel
    # epistemic is monotone in novelty.
    assert plan.epistemic[0, 0] < plan.epistemic[0, 1] < plan.epistemic[0, 2]


# --------------------------------------------------------------------------- #
# the explore/exploit knob                                                     #
# --------------------------------------------------------------------------- #
def test_info_gain_weight_flips_exploit_to_explore():
    # Identity predictor -> pragmatic(goal) = 1 - epistemic(goal). A small w_e
    # keeps the near (reachable) goal; a large w_e flips to the novel far goal.
    P = 8
    hidden = _basis(P, 0).unsqueeze(0)
    near = _basis(P, 0)                                  # at present: high pragmatic
    far = -_basis(P, 0)                                  # opposite: high epistemic
    goals = torch.stack([near, far])

    exploit = _static_planner(P, pragmatic_weight=1.0, info_gain_weight=0.1)
    explore = _static_planner(P, pragmatic_weight=1.0, info_gain_weight=5.0)

    assert exploit.score_goals(hidden, goals).selected.tolist() == [0]   # near
    assert explore.score_goals(hidden, goals).selected.tolist() == [1]   # far


def test_efe_equals_negative_weighted_sum():
    P = 6
    planner = _static_planner(P, pragmatic_weight=1.3, info_gain_weight=0.7)
    hidden = torch.randn(2, P)
    goals = torch.randn(4, P)
    plan = planner.score_goals(hidden, goals)
    expected = -(1.3 * plan.pragmatic + 0.7 * plan.epistemic)
    assert torch.allclose(plan.efe, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# housekeeping: zero params, determinism, validation                          #
# --------------------------------------------------------------------------- #
def test_zero_parameters_and_not_an_nn_module():
    planner = _static_planner(8)
    assert planner.n_parameters == 0
    assert not isinstance(planner, nn.Module)


def test_deterministic_for_fixed_input():
    P = 8
    planner = _static_planner(P)
    hidden = torch.randn(3, P)
    goals = torch.randn(5, P)
    a = planner.score_goals(hidden, goals)
    b = planner.score_goals(hidden, goals)
    assert torch.equal(a.efe, b.efe)
    assert torch.equal(a.selected, b.selected)


def test_select_goal_matches_plan():
    P = 8
    planner = _static_planner(P)
    hidden = torch.randn(2, P)
    goals = torch.randn(4, P)
    idx = planner.select_goal(hidden, goals)
    assert torch.equal(idx, planner.score_goals(hidden, goals).selected)


def test_construction_and_input_validation():
    head = _StubHead(8)
    imag = LatentImagination(head, horizon=2)
    with pytest.raises(ValueError):
        ActiveInferencePlanner(imag, pragmatic_weight=-1.0)
    with pytest.raises(ValueError):
        ActiveInferencePlanner(imag, info_gain_weight=-0.1)
    with pytest.raises(TypeError):
        ActiveInferencePlanner(object())                 # no .imagine / .head

    planner = ActiveInferencePlanner(imag)
    with pytest.raises(ValueError):
        planner.score_goals(torch.randn(2, 8), torch.randn(3, 7))   # wrong goal width


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
