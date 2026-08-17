"""
examples/demo_attractor_ops.py -- composable attractor / self-stabilization
diagnostics (P0 learning): where settling dynamics go, how fast, and how robust.

The story
---------
The continuous-time core relaxes toward equilibria -- memory recall settling on a
stored pattern, an imagination rollout coasting to a fixed point. This demo
instruments that settling with model-free dynamical-systems operators on two
hand-readable systems:

    (1) a STABLE linear map  x -> A x + b  (rho < 1): it has a fixed point, a
        spectral radius, an analytic convergence rate, and a settling time -- and
        the measured rate from the relaxation trajectory matches the closed form;

    (2) a DIVERGENT linear map (rho > 1): no stable attractor, negative rate, the
        Lyapunov energy grows every step -- the diagnostic flags it.

Then a NONLINEAR system, ``xdot = -x + x^3`` (stable at 0, unstable at +-1): the
basin probe pushes outward from 0 and recovers the basin half-width = 1, the
edge of stability.

SRTD-style, this is exactly the instrumentation a *self-stabilization* check
wants: is the representation settling, how fast, and how big a shock can it
absorb before tipping into another basin.

Honest scope
------------
Pure dynamical-systems operators, no model, zero trainable parameters. The maps
here are synthetic stand-ins for the real settling core; wiring this to watch
live LTC trajectories is a separate integration step. Deterministic, CPU,
sub-second, ASCII-only.

Run::

    python examples/demo_attractor_ops.py
    python examples/demo_attractor_ops.py --rho 0.7
"""

from __future__ import annotations

import argparse

import torch

from mt_lnn.attractor_ops import (
    analyze_linear_attractor,
    asymptotic_rate,
    basin_radius,
    fixed_point,
    is_contraction,
    linear_step,
    lyapunov_descent,
    relax,
    spectral_radius,
)

_f64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=_f64)


def _ascii_decay(distances: torch.Tensor, width: int = 48, rows: int = 8) -> str:
    """A log-scale descent plot of distance-to-fixed-point vs step."""
    d = distances.clamp_min(1e-12)
    ld = torch.log10(d)
    hi, lo = float(ld[0]), float(ld.min())
    span = max(hi - lo, 1e-9)
    n = d.shape[0]
    cols = min(width, n)
    idx = torch.linspace(0, n - 1, cols).round().long()
    out = []
    for r in range(rows):
        level = hi - (r + 0.5) * span / rows
        line = "".join("*" if float(ld[i]) >= level else " " for i in idx)
        out.append(f"   1e{level:+05.1f} |{line}")
    out.append("          +" + "-" * cols)
    out.append(f"           step 0 .. {n - 1}   (distance to x*, log scale)")
    return "\n".join(out)


def run_demo(args) -> dict:
    rho = args.rho
    # (1) stable map: diag(rho, rho/2) -> spectral radius == rho < 1
    A = torch.diag(_t([rho, rho / 2.0]))
    b = _t([1.0, 2.0])
    rep = analyze_linear_attractor(A, b, tol=1e-3, max_steps=4000)
    dist_full = (rep.trajectory - rep.fixed_point).norm(dim=-1)
    # show the trajectory up to ~2x settling time so the decay is legible
    show = min(dist_full.shape[0], 2 * rep.settling_time + 10)
    dist = dist_full[:show]
    lyap = lyapunov_descent(rep.trajectory, target=rep.fixed_point)
    nz = lyap[lyap > 0]

    # global contraction => basin unbounded (returns the cap)
    lin_basin = float(basin_radius(linear_step(A, b), rep.fixed_point,
                                   _t([1.0, 0.0]), max_radius=5.0, steps=800))

    # (2) divergent map: rho > 1
    Ad = torch.diag(_t([1.0 / rho if rho < 1 else 1.3, 1.1]))
    div_contracts = is_contraction(Ad)
    div_rate = float(asymptotic_rate(Ad))
    div_traj = relax(linear_step(Ad, _t([0.0, 0.0])), _t([1.0, 1.0]), 8)
    div_lyap = lyapunov_descent(div_traj, target=_t([0.0, 0.0]))

    # (3) nonlinear: xdot = -x + x^3  ->  basin of 0 is (-1, 1)
    dt = 0.01

    def nl_step(x):
        return x + dt * (-x + x ** 3)

    nl_basin = float(basin_radius(nl_step, _t([0.0]), _t([1.0]),
                                  max_radius=2.0, steps=4000))

    return {
        "rho": rho,
        "fixed_point": rep.fixed_point,
        "spectral_radius": float(rep.spectral_radius),
        "is_contraction": rep.is_contraction,
        "asymptotic_rate": float(rep.asymptotic_rate),
        "empirical_rate": float(rep.empirical_rate),
        "settling_time": rep.settling_time,
        "distances": dist,
        "max_lyap_ratio": float(nz.max()) if nz.numel() else 0.0,
        "lin_basin": lin_basin, "basin_cap": 5.0,
        "div_contracts": div_contracts,
        "div_rate": div_rate,
        "div_min_lyap": float(div_lyap.min()),
        "nl_basin": nl_basin,
    }


def print_report(s: dict) -> None:
    line = "=" * 64
    print(line)
    print("ATTRACTOR / SELF-STABILIZATION -- does it settle, how fast, how robust")
    print(line)

    print(f"\n  -- (1) STABLE linear map  x -> A x + b  (rho = {s['rho']:.2f}) --")
    print(f"  fixed point x*        : "
          f"[{s['fixed_point'][0]:.3f}, {s['fixed_point'][1]:.3f}]")
    print(f"  spectral radius rho   : {s['spectral_radius']:.4f}   "
          f"(< 1 -> contraction: {s['is_contraction']})")
    print(f"  rate -log(rho)        : analytic {s['asymptotic_rate']:.4f} "
          f"vs measured {s['empirical_rate']:.4f}")
    print(f"  settling time (1e-3)  : {s['settling_time']} steps")
    print(f"  max Lyapunov ratio    : {s['max_lyap_ratio']:.4f}   "
          f"(< 1 -> energy descends every step)")
    print("\n  distance to x* over the relaxation:")
    print(_ascii_decay(s["distances"]))
    print(f"\n  basin half-width (linear, global contraction) : "
          f"{s['lin_basin']:.1f}  (= cap {s['basin_cap']:.1f}: unbounded)")

    print(f"\n  -- (2) DIVERGENT linear map (rho > 1) --")
    print(f"  is_contraction        : {s['div_contracts']}   (no stable attractor)")
    print(f"  rate -log(rho)        : {s['div_rate']:+.4f}   (negative -> blows up)")
    print(f"  min Lyapunov ratio    : {s['div_min_lyap']:.4f}   "
          f"(> 1 -> energy grows every step)")

    print(f"\n  -- (3) NONLINEAR  xdot = -x + x^3  (stable 0, unstable +-1) --")
    print(f"  basin half-width      : {s['nl_basin']:.4f}   "
          f"(recovers the unstable boundary at 1.0)")

    ok = (
        s["is_contraction"]
        and abs(s["empirical_rate"] - s["asymptotic_rate"]) < 1e-2
        and s["max_lyap_ratio"] < 1.0
        and s["lin_basin"] == s["basin_cap"]
        and not s["div_contracts"]
        and s["div_rate"] < 0.0
        and s["div_min_lyap"] > 1.0
        and abs(s["nl_basin"] - 1.0) < 1e-2
    )
    print("\n" + line)
    if ok:
        print("VERDICT [OK]: the diagnostic MEASURES stability -- a contraction's")
        print("        rate/settling match the closed form, a divergent map is")
        print("        flagged, and the basin probe recovers the stability edge.")
        print("        Model-free self-stabilization instrumentation.")
    else:
        print("VERDICT [FAIL]: attractor diagnostics not as expected -- see above.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="composable attractor self-stabilization demo")
    ap.add_argument("--rho", type=float, default=0.9,
                    help="spectral radius of the stable map (0<rho<1, default 0.9).")
    args = ap.parse_args()
    print_report(run_demo(args))


if __name__ == "__main__":
    main()
