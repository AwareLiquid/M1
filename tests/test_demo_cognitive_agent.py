"""
tests/test_demo_cognitive_agent.py -- autonomous cognitive agent demo contract.

Pins ``examples/demo_cognitive_agent.py``: the full perceive -> remember -> attend
-> imagine+validate -> survive -> act-safe loop. Each stage genuinely exercises a
real module (grid cells, SpatialMemory, the top-down pathway, LatentImagination,
physics_ops collision/kinematics, BlindRolloutGuard, CircuitBreaker), so the demo
is an integration test of the system, not a mock.

Honest, default-off discipline checked here:
  * top-down with the gate CLOSED is a STRICT no-op (bit-identical hidden state);
    OPEN it steers, and a larger gate steers at least as much.
  * the imagination driver adds ZERO parameters and its confidence decays.
  * the straight path HITS an obstacle while a physics-validated detour is chosen.
  * the guard coasts on the world model then goes DARK ("lost perception").
  * every safety-clamped command stays finite and inside the actuator envelope.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_cognitive_agent import (   # noqa: E402
    run_demo, print_report, build_scene, perceive, remember, attend,
    imagine_and_plan, survive, act_safe,
)


def _args(seed=0, gate=1.0, horizon=5, floor=0.35, max_blind=4):
    return type("A", (), {
        "seed": seed, "gate": gate, "horizon": horizon,
        "floor": floor, "max_blind": max_blind,
    })()


# --- stage 1: perception ---------------------------------------------------

def test_grid_codes_separate_distinct_places():
    p = perceive(build_scene())
    assert p["self_cos"] > 0.99            # a place is identical to itself
    assert p["mean_offdiag_cos"] < 0.9     # distinct objects get distinct codes


# --- stage 2: memory -------------------------------------------------------

def test_cognitive_map_recalls_by_place():
    m = remember(build_scene(), seed=0)
    assert m["mem_params"] == 0            # zero trainable params
    assert m["clean_recall"] > 0.9         # pattern completion at the true place
    assert m["fuzzy_recall"] > 0.5         # graceful under a noisy positional cue


# --- stage 3: top-down attention ------------------------------------------

def test_closed_gate_is_strict_no_op_open_gate_steers():
    info, _model, hidden = attend(build_scene(), seed=0, gate=1.0)
    assert info["closed_diff"] == 0.0      # bit-exact: goal == no goal at init
    assert info["open_diff"] > 1e-4        # opening the gate steers the hidden state
    assert info["hidden_finite"]
    assert hidden.shape[-1] == 104


def test_larger_gate_steers_at_least_as_much():
    weak, _, _ = attend(build_scene(), seed=0, gate=0.5)
    strong, _, _ = attend(build_scene(), seed=0, gate=2.0)
    assert strong["open_diff"] >= weak["open_diff"]


# --- stage 4: imagination + physics-validated planning --------------------

def test_imagination_is_zero_param_and_confidence_decays():
    scene = build_scene()
    info, model, hidden = attend(scene, seed=0, gate=1.0)
    plan = imagine_and_plan(scene, model, hidden, horizon=5,
                            dt=0.1, v_max=30.0, a_max=600.0)
    assert plan["added_params"] == 0
    assert plan["conf_decays"]


def test_straight_path_blocked_and_a_safe_detour_is_committed():
    scene = build_scene()
    _info, model, hidden = attend(scene, seed=0, gate=1.0)
    plan = imagine_and_plan(scene, model, hidden, horizon=5,
                            dt=0.1, v_max=30.0, a_max=600.0)
    assert plan["straight_blocked"]                       # straight line hits a wall
    assert not plan["candidates"]["straight"]["collision_free"]
    assert plan["chosen"] is not None                     # a feasible plan exists
    chosen = plan["candidates"][plan["chosen"]]
    assert chosen["collision_free"] and chosen["kinematically_ok"]


# --- stage 5: blind survival ----------------------------------------------

def test_guard_coasts_then_goes_dark():
    scene = build_scene()
    _info, model, hidden = attend(scene, seed=0, gate=1.0)
    sv = survive(model, hidden, horizon=5, floor=0.35, max_blind=4)
    assert sv["n_coast"] > 0               # coasts on the world model first
    assert sv["went_dark"]                 # then honestly stops when trust runs out
    assert sv["sources"][0] == "live"      # live while the feed is present


# --- stage 6: safe actuation ----------------------------------------------

def test_breaker_keeps_every_command_in_bounds():
    scene = build_scene()
    _info, model, hidden = attend(scene, seed=0, gate=1.0)
    plan = imagine_and_plan(scene, model, hidden, horizon=5,
                            dt=0.1, v_max=30.0, a_max=600.0)
    import math
    safe = act_safe(scene, plan["chosen_path"], dt=0.1, max_turn=math.radians(45.0))
    assert safe["all_in_bounds"]           # finite + within +/- envelope, always
    assert safe["n_tripped"] > 0           # the spliced garbage trips the breaker


# --- end to end ------------------------------------------------------------

def test_full_loop_verdict_ok():
    s = run_demo(_args())
    assert s["ok"]


def test_report_prints_ok_verdict(capsys):
    print_report(run_demo(_args()))
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "AUTONOMOUS COGNITIVE AGENT" in out
