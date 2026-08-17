"""
examples/demo_symplectic_integrator.py -- why the imagination rollout should
integrate with velocity Verlet, not first-order Euler (P1 engineering).

The story
---------
An imagination / world-model rollout coasts a state forward under forces for
many steps. If the integrator leaks energy, a long rollout slowly *invents*
energy it never had -- a planet spirals out, a pendulum winds up -- and the
"what happens next" prediction drifts away from physics. The fix is not a
smaller step (expensive); it is a *symplectic* stepper whose energy error stays
bounded for all time.

This demo flies one body on a circular Kepler orbit (central inverse-square
force, GM = 1, r0 = 1, v0 = 1) for many orbits and compares the two integrators
that ``physics_ops.rollout`` now offers:

    semi-implicit Euler  -- 1st order, energy error O(dt) and *accumulating*;
    velocity Verlet      -- 2nd order, time-reversible, energy error O(dt^2)
                            and *bounded* (no secular drift).

We report, over the whole flight, the worst energy drift and the worst orbit-
radius drift for each. The closed truth: a perfect circular orbit keeps both E
and r constant, so any drift is pure integrator error -- and Verlet's is smaller
by orders of magnitude. A second panel halves dt and shows the error shrinking
~2x for Euler but ~4x for Verlet, the signature of 1st- vs 2nd-order accuracy.

Honest scope
------------
Pure physics operators, no model, zero trainable parameters. A central-force
orbit is a hand-readable stand-in for a real latent rollout; wiring Verlet into
the live imagination loop is a separate integration step (the ``integrator=``
switch makes it a one-line change). Deterministic, CPU, sub-second, ASCII-only.

Run::

    python examples/demo_symplectic_integrator.py
    python examples/demo_symplectic_integrator.py --orbits 64 --dt 0.02
"""

from __future__ import annotations

import argparse
import math

import torch

from mt_lnn.physics_ops import rollout

_f64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=_f64)


def _central_accel(GM: float):
    """Inverse-square pull toward the origin: a = -GM * x / |x|^3 (position-only)."""
    def accel_fn(pos, vel):                       # rollout passes (pos, vel)
        r = pos.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return -GM * pos / r.pow(3)
    return accel_fn


def _fly(integrator: str, *, GM: float, steps: int, dt: float):
    """Fly the unit circular orbit (r0=1, omega=sqrt(GM)) for ``steps`` steps."""
    x0 = _t([[1.0, 0.0]])                          # (N=1, D=2), r0 = 1
    v0 = _t([[0.0, math.sqrt(GM)]])                # circular speed sqrt(GM/r0)
    return rollout(x0, v0, steps=steps, dt=dt,
                   accel_fn=_central_accel(GM), integrator=integrator)


def _orbit_drifts(integrator: str, *, GM: float, steps: int, dt: float) -> dict:
    """Fly the circular orbit and return worst energy / radius drift (fractional)."""
    roll = _fly(integrator, GM=GM, steps=steps, dt=dt)
    pos = roll.positions                           # (T+1, N, D)
    vel = roll.velocities
    r = pos.norm(dim=-1).squeeze(-1)               # (T+1,)
    speed2 = vel.pow(2).sum(-1).squeeze(-1)        # (T+1,)
    energy = 0.5 * speed2 - GM / r.clamp_min(1e-12)  # specific orbital energy
    e0 = float(energy[0])
    energy_drift = float((energy - e0).abs().max() / abs(e0))
    radius_drift = float((r - 1.0).abs().max())    # r0 == 1
    return {"energy_drift": energy_drift, "radius_drift": radius_drift,
            "radii": r}


def _position_error(integrator: str, *, GM: float, steps: int, dt: float) -> float:
    """Worst position error vs the EXACT circular orbit x(t)=(cos wt, sin wt).

    With r0=1 and v0=sqrt(GM) the orbit is the unit circle traced at angular
    speed w=sqrt(GM); the closed-form path is the unambiguous ground truth, so
    this is a clean integrator-ORDER probe (Euler O(dt), Verlet O(dt^2))."""
    roll = _fly(integrator, GM=GM, steps=steps, dt=dt)
    w = math.sqrt(GM)
    t = roll.times                                 # (T+1,)
    exact = torch.stack([torch.cos(w * t), torch.sin(w * t)], dim=-1)  # (T+1, D)
    pos = roll.positions.squeeze(-2)               # (T+1, D)  (single body)
    return float((pos - exact).norm(dim=-1).max())


def _ascii_radius(radii: torch.Tensor, width: int = 56, rows: int = 7) -> str:
    """A staircase of orbit radius vs step -- a flat line is a faithful orbit."""
    n = radii.shape[0]
    cols = min(width, n)
    idx = torch.linspace(0, n - 1, cols).round().long()
    sample = radii[idx]
    hi = max(float(sample.max()), 1.0 + 1e-6)
    lo = min(float(sample.min()), 1.0 - 1e-6)
    span = max(hi - lo, 1e-9)
    out = []
    for rr in range(rows):
        level = hi - (rr + 0.5) * span / rows
        line = "".join("*" if float(sample[i]) >= level else " " for i in range(cols))
        out.append(f"   r={level:6.4f} |{line}")
    out.append("            +" + "-" * cols)
    out.append(f"             step 0 .. {n - 1}   (orbit radius; truth r = 1.0000)")
    return "\n".join(out)


def run_demo(args) -> dict:
    GM = 1.0
    period = 2.0 * math.pi * math.sqrt(1.0 / GM)   # circular period at r0 = 1
    steps = int(round(args.orbits * period / args.dt))
    euler = _orbit_drifts("symplectic_euler", GM=GM, steps=steps, dt=args.dt)
    verlet = _orbit_drifts("verlet", GM=GM, steps=steps, dt=args.dt)

    # convergence panel: fly a fixed TIME, halve dt, and compare the global
    # position error against the exact circular orbit -- the unambiguous order
    # probe (Euler O(dt) -> 2x drop, Verlet O(dt^2) -> 4x drop per halving).
    short = max(int(round(2.0 * period / args.dt)), 8)
    eu_h = _position_error("symplectic_euler", GM=GM, steps=short, dt=args.dt)
    eu_hh = _position_error("symplectic_euler", GM=GM, steps=short * 2, dt=args.dt / 2)
    ve_h = _position_error("verlet", GM=GM, steps=short, dt=args.dt)
    ve_hh = _position_error("verlet", GM=GM, steps=short * 2, dt=args.dt / 2)
    eps = 1e-18
    euler_order = math.log2(max(eu_h, eps) / max(eu_hh, eps))
    verlet_order = math.log2(max(ve_h, eps) / max(ve_hh, eps))
    return {
        "orbits": args.orbits, "dt": args.dt, "steps": steps,
        "euler_energy": euler["energy_drift"], "euler_radius": euler["radius_drift"],
        "verlet_energy": verlet["energy_drift"], "verlet_radius": verlet["radius_drift"],
        "euler_radii": euler["radii"], "verlet_radii": verlet["radii"],
        "euler_order": euler_order, "verlet_order": verlet_order,
    }


def print_report(s: dict) -> None:
    line = "=" * 64
    print(line)
    print("SYMPLECTIC INTEGRATOR -- bounded energy over a long imagination rollout")
    print(line)
    print(f"\n  circular Kepler orbit (GM=1, r0=1): {s['orbits']} orbits, "
          f"dt={s['dt']}, {s['steps']} steps")
    ratio_e = s["euler_energy"] / max(s["verlet_energy"], 1e-18)
    ratio_r = s["euler_radius"] / max(s["verlet_radius"], 1e-18)
    print(f"\n  {'integrator':<20}{'energy drift':>16}{'radius drift':>16}")
    print(f"  {'-' * 20}{'-' * 16:>16}{'-' * 16:>16}")
    print(f"  {'semi-implicit Euler':<20}{s['euler_energy']:>16.2e}"
          f"{s['euler_radius']:>16.2e}")
    print(f"  {'velocity Verlet':<20}{s['verlet_energy']:>16.2e}"
          f"{s['verlet_radius']:>16.2e}")
    print(f"\n  Verlet is better by    : {ratio_e:.0f}x (energy), "
          f"{ratio_r:.0f}x (radius)")

    print("\n  Euler orbit radius (spirals -- energy leaks in):")
    print(_ascii_radius(s["euler_radii"]))
    print("\n  Verlet orbit radius (holds -- energy bounded):")
    print(_ascii_radius(s["verlet_radii"]))

    print(f"\n  -- accuracy order (halve dt, position-error drop vs exact orbit) --")
    print(f"  Euler  drop factor    : {2 ** s['euler_order']:.2f}x  "
          f"(~2x  -> 1st order, O(dt))")
    print(f"  Verlet drop factor    : {2 ** s['verlet_order']:.2f}x  "
          f"(~4x  -> 2nd order, O(dt^2))")

    ok = (
        s["verlet_energy"] < s["euler_energy"] / 10.0
        and s["verlet_radius"] < s["euler_radius"] / 10.0
        and 0.6 < s["euler_order"] < 1.5       # Euler ~1st order (2x drop)
        and s["verlet_order"] > 1.6             # Verlet >=2nd order (>=4x drop)
    )
    print("\n" + line)
    if ok:
        print("VERDICT [OK]: velocity Verlet conserves orbital energy/radius orders")
        print("        of magnitude better than Euler over a long rollout, and its")
        print("        error falls ~4x per dt-halving (2nd order) vs Euler's ~2x.")
        print("        The right stepper for long imagination rollouts.")
    else:
        print("VERDICT [FAIL]: integrator comparison not as expected -- see above.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="symplectic integrator (Verlet vs Euler) demo")
    ap.add_argument("--orbits", type=float, default=32.0,
                    help="number of circular orbits to fly (default 32).")
    ap.add_argument("--dt", type=float, default=0.05,
                    help="integration timestep (default 0.05).")
    args = ap.parse_args()
    print_report(run_demo(args))


if __name__ == "__main__":
    main()
