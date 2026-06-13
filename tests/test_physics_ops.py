"""
tests/test_physics_ops.py -- composable Newtonian dynamics operators (L4 step 3).

Each operator is pinned against an ANALYTIC invariant (momentum conservation,
exact velocity under constant acceleration, swapped velocities in an elastic
head-on hit, a near-constant two-body orbit radius), and a final composition
test rolls gravity + a bouncing floor + a wall forward to answer a real "will
the ball clear the wall?" query -- physical reasoning that is *computed*, not
memorised.
"""
import math
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.physics_ops import (  # noqa: E402
    PhysicsRollout,
    integrate,
    integrate_verlet,
    uniform_gravity,
    pairwise_gravity,
    kinetic_energy,
    momentum,
    overlapping_pairs,
    resolve_sphere_collisions,
    reflect_in_box,
    rollout,
)


# --- integration ----------------------------------------------------------

def test_integrate_is_semi_implicit_euler():
    pos = torch.tensor([[0.0, 0.0]])
    vel = torch.tensor([[1.0, 0.0]])
    acc = torch.tensor([[0.0, -2.0]])
    p1, v1 = integrate(pos, vel, acc, dt=0.5)
    # v' = v + a*dt = (1, -1);  x' = x + v'*dt = (0.5, -0.5)
    assert torch.allclose(v1, torch.tensor([[1.0, -1.0]]), atol=1e-6)
    assert torch.allclose(p1, torch.tensor([[0.5, -0.5]]), atol=1e-6)


def test_integrate_constant_accel_velocity_is_exact():
    # under constant accel, symplectic Euler integrates velocity exactly
    pos = torch.zeros(3, 2)
    vel = torch.zeros(3, 2)
    g = torch.tensor([0.0, -9.81]).expand(3, 2)
    dt, n = 0.01, 100
    p, v = pos, vel
    for _ in range(n):
        p, v = integrate(p, v, g, dt)
    assert torch.allclose(v[:, 1], torch.full((3,), -9.81 * n * dt), atol=1e-4)


def test_integrate_is_differentiable():
    pos = torch.zeros(1, 2, 2, requires_grad=True)
    vel = torch.ones(1, 2, 2, requires_grad=True)
    acc = torch.zeros(1, 2, 2)
    p1, _ = integrate(pos, vel, acc, dt=0.1)
    p1.sum().backward()
    assert pos.grad is not None and torch.isfinite(vel.grad).all()


def test_integrate_2d_input_squeezes_batch_via_rollout():
    s = rollout(torch.zeros(2, 2), torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                steps=3, dt=0.1)
    assert s.positions.dim() == 3                          # (T+1, N, D), batch squeezed
    assert s.positions.shape == (4, 2, 2)


# --- velocity Verlet (2nd-order symplectic) -------------------------------

_f64 = torch.float64


def test_integrate_verlet_is_kick_drift_kick():
    pos = torch.tensor([[0.0, 0.0]], dtype=_f64)
    vel = torch.tensor([[1.0, 0.0]], dtype=_f64)

    def accel(x):
        return torch.tensor([[0.0, -2.0]], dtype=_f64).expand_as(x)

    x1, v1, a1 = integrate_verlet(pos, vel, accel, dt=0.5)
    # v_half = v + 0.5 a dt = (1, -0.5); x' = x + v_half dt = (0.5, -0.25)
    # v'     = v_half + 0.5 a dt = (1, -1)
    assert torch.allclose(x1, torch.tensor([[0.5, -0.25]], dtype=_f64), atol=1e-12)
    assert torch.allclose(v1, torch.tensor([[1.0, -1.0]], dtype=_f64), atol=1e-12)
    assert torch.allclose(a1, torch.tensor([[0.0, -2.0]], dtype=_f64), atol=1e-12)


def test_verlet_is_exact_under_constant_acceleration():
    # velocity Verlet integrates a constant-acceleration (projectile) path EXACTLY
    g = torch.tensor([[0.0, -9.81]], dtype=_f64)

    def accel(x):
        return g.expand_as(x)

    x = torch.tensor([[0.0, 0.0]], dtype=_f64)
    v = torch.tensor([[2.0, 5.0]], dtype=_f64)
    x0, v0 = x.clone(), v.clone()
    dt, n = 0.01, 200
    a = None
    for _ in range(n):
        x, v, a = integrate_verlet(x, v, accel, dt, accel=a)
    t = n * dt
    expect_x = x0 + v0 * t + 0.5 * g * t * t
    expect_v = v0 + g * t
    assert torch.allclose(x, expect_x, atol=1e-10)
    assert torch.allclose(v, expect_v, atol=1e-10)


def _sho_energy_drift(integrator: str, dt: float, steps: int) -> float:
    # harmonic oscillator a = -x; total energy 0.5(x^2 + v^2) is conserved exactly
    def spring(pos, vel):
        return -pos

    x0 = torch.tensor([[1.0, 0.0]], dtype=_f64)
    v0 = torch.zeros(1, 2, dtype=_f64)
    r = rollout(x0, v0, steps=steps, dt=dt, accel_fn=spring, integrator=integrator)
    pe = 0.5 * (r.positions ** 2).sum(-1).reshape(r.kinetic_energy.shape)
    E = r.kinetic_energy + pe
    return float(E.max() - E.min())


def test_verlet_energy_drift_is_second_order():
    # Euler drift ~ O(dt) (halving dt -> ~2x less); Verlet ~ O(dt^2) (-> ~4x less)
    e1 = _sho_energy_drift("symplectic_euler", 0.1, 2000)
    e2 = _sho_energy_drift("symplectic_euler", 0.05, 4000)
    v1 = _sho_energy_drift("verlet", 0.1, 2000)
    v2 = _sho_energy_drift("verlet", 0.05, 4000)
    assert abs(e1 / e2 - 2.0) < 0.2          # first order
    assert abs(v1 / v2 - 4.0) < 0.3          # second order


def test_verlet_conserves_energy_far_better_than_euler():
    euler = _sho_energy_drift("symplectic_euler", 0.1, 2000)
    verlet = _sho_energy_drift("verlet", 0.1, 2000)
    assert verlet < euler / 10.0             # ~40x in practice


def test_verlet_rollout_is_time_reversible():
    def spring(pos, vel):
        return -pos

    x0 = torch.tensor([[1.0, 0.0]], dtype=_f64)
    v0 = torch.tensor([[0.0, 0.3]], dtype=_f64)
    fwd = rollout(x0, v0, steps=300, dt=0.1, accel_fn=spring, integrator="verlet")
    back = rollout(fwd.final_positions, -fwd.final_velocities, steps=300, dt=0.1,
                   accel_fn=spring, integrator="verlet")
    assert torch.allclose(back.final_positions, x0, atol=1e-9)
    assert torch.allclose(back.final_velocities, -v0, atol=1e-9)


def test_rollout_rejects_unknown_integrator():
    with pytest.raises(ValueError):
        rollout(torch.zeros(1, 2), torch.zeros(1, 2), steps=2, dt=0.1,
                integrator="midpoint")


# --- forces ---------------------------------------------------------------

def test_uniform_gravity_broadcasts_field():
    pos = torch.randn(4, 2)
    a = uniform_gravity(pos, [0.0, -9.81])
    assert a.shape == (4, 2)
    assert torch.allclose(a[:, 1], torch.full((4,), -9.81))
    assert torch.allclose(a[:, 0], torch.zeros(4))


def test_pairwise_gravity_two_body_reaction_and_magnitude():
    # two equal unit masses 2 apart on the x-axis, no softening
    pos = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    a = pairwise_gravity(pos, mass=1.0, G=1.0, softening=0.0)
    # a_0 points toward +x (toward body 1), a_1 toward -x; equal & opposite
    assert a[0, 0].item() == pytest.approx(1.0 / 4.0, abs=1e-6)   # G m / r^2 = 1/4
    assert a[1, 0].item() == pytest.approx(-1.0 / 4.0, abs=1e-6)
    assert torch.allclose(a[0] + a[1], torch.zeros(2), atol=1e-6)  # momentum-conserving


def test_pairwise_gravity_conserves_momentum_random():
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(5, 3, generator=g)
    mass = torch.rand(5, generator=g) + 0.5
    a = pairwise_gravity(pos, mass=mass, softening=1e-2)
    net = (mass.view(5, 1) * a).sum(0)                     # sum m_i a_i
    assert torch.allclose(net, torch.zeros(3), atol=1e-5)


def test_pairwise_gravity_is_differentiable():
    pos = torch.randn(1, 4, 2, requires_grad=True)
    pairwise_gravity(pos, mass=1.0).pow(2).sum().backward()
    assert pos.grad is not None and torch.isfinite(pos.grad).all()


# --- diagnostics ----------------------------------------------------------

def test_kinetic_energy_matches_formula():
    vel = torch.tensor([[3.0, 4.0], [0.0, 2.0]])           # speeds 5 and 2
    ke = kinetic_energy(vel, mass=2.0)
    assert ke.item() == pytest.approx(0.5 * 2 * (25.0 + 4.0))


def test_momentum_matches_formula():
    vel = torch.tensor([[1.0, 0.0], [-1.0, 2.0]])
    p = momentum(vel, mass=torch.tensor([2.0, 1.0]))
    assert torch.allclose(p, torch.tensor([1.0, 2.0]), atol=1e-6)   # (2-1, 0+2)


# --- contacts -------------------------------------------------------------

def test_overlapping_pairs_thresholds_on_radius_sum():
    pos = torch.tensor([[0.0, 0.0], [0.9, 0.0], [5.0, 0.0]])
    over = overlapping_pairs(pos, radius=0.5)              # sum radii = 1.0
    assert over[0, 1] and over[1, 0]                       # dist 0.9 < 1.0
    assert not over[0, 2] and not over[1, 2]               # far apart
    assert over.diagonal().sum() == 0                      # no self overlap


def test_elastic_head_on_equal_mass_swaps_velocities():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    vel = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])          # approaching
    v2 = resolve_sphere_collisions(pos, vel, radius=0.6, mass=1.0, restitution=1.0)
    assert torch.allclose(v2[0], torch.tensor([-1.0, 0.0]), atol=1e-6)
    assert torch.allclose(v2[1], torch.tensor([1.0, 0.0]), atol=1e-6)


def test_unequal_mass_elastic_matches_1d_formula():
    # 1D elastic: v1' = ((m1-m2)v1 + 2 m2 v2)/(m1+m2), v2' similarly
    m1, m2, u1, u2 = 3.0, 1.0, 2.0, -1.0
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    vel = torch.tensor([[u1, 0.0], [u2, 0.0]])
    v2 = resolve_sphere_collisions(pos, vel, radius=0.6,
                                   mass=torch.tensor([m1, m2]), restitution=1.0)
    v1p = ((m1 - m2) * u1 + 2 * m2 * u2) / (m1 + m2)
    v2p = ((m2 - m1) * u2 + 2 * m1 * u1) / (m1 + m2)
    assert v2[0, 0].item() == pytest.approx(v1p, abs=1e-6)
    assert v2[1, 0].item() == pytest.approx(v2p, abs=1e-6)


def test_collision_conserves_momentum_and_energy_when_elastic():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    vel = torch.tensor([[2.0, 0.5], [-1.0, 0.0]])
    mass = torch.tensor([2.0, 3.0])
    v2 = resolve_sphere_collisions(pos, vel, radius=0.6, mass=mass, restitution=1.0)
    assert torch.allclose(momentum(vel, mass), momentum(v2, mass), atol=1e-5)
    assert kinetic_energy(vel, mass).item() == pytest.approx(
        kinetic_energy(v2, mass).item(), abs=1e-5)


def test_inelastic_collision_dissipates_energy_but_keeps_momentum():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    vel = torch.tensor([[2.0, 0.0], [-1.0, 0.0]])
    mass = torch.tensor([1.0, 1.0])
    v2 = resolve_sphere_collisions(pos, vel, radius=0.6, mass=mass, restitution=0.5)
    assert torch.allclose(momentum(vel, mass), momentum(v2, mass), atol=1e-5)
    assert kinetic_energy(v2, mass).item() < kinetic_energy(vel, mass).item()


def test_separating_pair_is_untouched():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    vel = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])          # moving apart
    v2 = resolve_sphere_collisions(pos, vel, radius=0.6, restitution=1.0)
    assert torch.allclose(v2, vel, atol=1e-7)              # no spurious impulse


def test_reflect_in_box_floor_bounce():
    pos = torch.tensor([[1.0, -0.2]])                      # dipped below floor y=0
    vel = torch.tensor([[1.0, -3.0]])
    inf = float("inf")
    p, v = reflect_in_box(pos, vel, lo=[-inf, 0.0], hi=[inf, inf], restitution=1.0)
    assert p[0, 1].item() == pytest.approx(0.0)            # clamped to floor
    assert v[0, 1].item() == pytest.approx(3.0)            # vy flipped, speed kept
    assert v[0, 0].item() == pytest.approx(1.0)            # vx untouched


def test_reflect_in_box_restitution_damps():
    pos = torch.tensor([[0.0, -0.1]])
    vel = torch.tensor([[0.0, -4.0]])
    inf = float("inf")
    _, v = reflect_in_box(pos, vel, lo=[-inf, 0.0], hi=[inf, inf], restitution=0.5)
    assert v[0, 1].item() == pytest.approx(2.0)            # 0.5 * 4


# --- rollout & conservation -----------------------------------------------

def test_rollout_free_fall_horizontal_velocity_constant():
    pos = torch.tensor([[0.0, 10.0]])
    vel = torch.tensor([[2.0, 0.0]])
    s = rollout(pos, vel, steps=50, dt=0.01, gravity=[0.0, -9.81])
    assert s.positions.shape == (51, 1, 2)
    # vx is exactly constant; vy = -g * t exactly (constant accel)
    assert torch.allclose(s.velocities[:, 0, 0], torch.full((51,), 2.0), atol=1e-6)
    assert s.velocities[-1, 0, 1].item() == pytest.approx(-9.81 * 50 * 0.01, abs=1e-4)


def test_rollout_elastic_floor_returns_to_launch_height():
    # drop with upward kick, perfectly elastic floor, no air drag -> energy conserved
    pos = torch.tensor([[0.0, 1.0]])
    vel = torch.tensor([[0.0, 0.0]])
    s = rollout(pos, vel, steps=400, dt=0.005, gravity=[0.0, -9.81],
                bounds=([float("-inf"), 0.0], [float("inf"), float("inf")]),
                restitution=1.0)
    # KE+PE conserved => the apex height after bouncing stays near the drop height
    max_h = s.positions[:, 0, 1].max().item()
    assert max_h == pytest.approx(1.0, abs=0.05)
    assert s.positions[:, 0, 1].min().item() >= -1e-6     # never tunnels through floor


def test_rollout_two_body_orbit_radius_is_near_constant():
    # two equal masses on a circular orbit about their COM; separation should hold
    G, m, r = 1.0, 1.0, 2.0
    v_rel = math.sqrt(G * (m + m) / r)                     # relative orbital speed
    pos = torch.tensor([[-r / 2, 0.0], [r / 2, 0.0]])
    vel = torch.tensor([[0.0, -v_rel / 2], [0.0, v_rel / 2]])

    def grav(p, v):
        return pairwise_gravity(p, mass=m, G=G, softening=0.0)

    s = rollout(pos, vel, steps=2000, dt=0.001, accel_fn=grav)
    sep = (s.positions[:, 1, :] - s.positions[:, 0, :]).norm(dim=-1)
    assert sep.max().item() == pytest.approx(r, abs=0.1)
    assert sep.min().item() == pytest.approx(r, abs=0.1)
    # closed system: COM momentum stays ~zero throughout
    pmom = (m * s.velocities[:, 0] + m * s.velocities[:, 1])
    assert pmom.norm(dim=-1).max().item() < 1e-3


def test_rollout_isolated_collision_conserves_momentum_over_time():
    pos = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    vel = torch.tensor([[1.5, 0.0], [-0.5, 0.0]])          # closing
    s = rollout(pos, vel, steps=200, dt=0.01, radius=0.3, mass=1.0, restitution=1.0)
    p0 = momentum(s.velocities[0], 1.0)
    pT = momentum(s.velocities[-1], 1.0)
    assert torch.allclose(p0, pT, atol=1e-4)
    # an elastic hit happened: closing pair ends up separating
    assert s.velocities[-1, 0, 0].item() < s.velocities[-1, 1, 0].item()


def test_rollout_rejects_zero_steps():
    with pytest.raises(ValueError):
        rollout(torch.zeros(2, 2), torch.zeros(2, 2), steps=0, dt=0.1)


# --- composition: the flagship physics-reasoning query --------------------

def _ball_clears_wall(restitution: float, *, v0=(3.0, 4.0), wall_x=2.4,
                      wall_h=0.6, steps=800, dt=0.005) -> bool:
    """Roll a ball under gravity bouncing on the floor; does its path clear a
    wall of height ``wall_h`` standing at ``wall_x``? Computed, not stored."""
    pos = torch.tensor([[0.0, 0.0]])
    vel = torch.tensor([list(v0)])
    s = rollout(pos, vel, steps=steps, dt=dt, gravity=[0.0, -9.81],
                bounds=([float("-inf"), 0.0], [float("inf"), float("inf")]),
                restitution=restitution)
    xs = s.positions[:, 0, 0]
    ys = s.positions[:, 0, 1]
    # find the first snapshot at/after the wall's x; clears iff it is above wall_h
    past = (xs >= wall_x).nonzero()
    if past.numel() == 0:
        return False                                       # never reaches the wall
    k = int(past[0])
    return bool(ys[k].item() >= wall_h)


def test_composition_restitution_decides_whether_ball_clears_wall():
    """The same code, same geometry, different elasticity -> different outcome.
    Whether the ball clears the wall is COMPUTED by rolling the dynamics, not
    looked up: a lossy bounce (low e) saps the second arc so the ball is too low
    by the time it reaches the wall, while an elastic bounce (high e) keeps
    enough height to clear it."""
    # tuned so the decisive crossing is on the post-bounce arc
    assert _ball_clears_wall(restitution=1.0, wall_x=4.2, wall_h=0.45) is True
    assert _ball_clears_wall(restitution=0.4, wall_x=4.2, wall_h=0.45) is False


def test_rollout_summary_accessors_are_correct_unbatched_and_batched():
    """``steps`` / ``final_positions`` / ``final_velocities`` must index the time
    axis correctly for BOTH the unbatched (T+1, N, D) and batched
    (B, T+1, N, D) layouts -- a regression guard for the accessors that
    previously assumed a batch dim and silently returned garbage when the
    rollout was called (as every caller does) without one."""
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])           # (N=2, D=2)
    vel = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    k = 5

    # unbatched: positions (T+1, N, D)
    s = rollout(pos, vel, steps=k, dt=0.1)
    assert s.positions.shape == (k + 1, 2, 2)
    assert s.steps == k
    assert s.final_positions.shape == (2, 2)
    assert torch.equal(s.final_positions, s.positions[-1])
    assert torch.equal(s.final_velocities, s.velocities[-1])

    # batched: positions (B, T+1, N, D)
    sb = rollout(pos.unsqueeze(0), vel.unsqueeze(0), steps=k, dt=0.1)
    assert sb.positions.shape == (1, k + 1, 2, 2)
    assert sb.steps == k
    assert sb.final_positions.shape == (1, 2, 2)
    assert torch.equal(sb.final_positions, sb.positions[:, -1])
    assert torch.equal(sb.final_velocities, sb.velocities[:, -1])


def test_physics_ops_module_does_not_import_model():
    import mt_lnn.physics_ops as po
    src = open(po.__file__, encoding="utf-8").read()
    assert "import model" not in src and "from .model" not in src


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except Exception:
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
            raise
