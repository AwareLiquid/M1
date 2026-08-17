"""
examples/demo_physics_ops.py -- composable physics operators (L4 step 3).

The story
---------
A ball is launched across a courtyard. A wall stands in its way. Standing
between the launch point and the wall is a perfectly hard floor it can bounce
off. Question: *does the ball clear the wall?* The answer is not stored anywhere
-- it has to be COMPUTED by simulating the dynamics from the launch velocity,
gravity, and how bouncy the floor is. We answer it by composing the operators in
:mod:`mt_lnn.physics_ops`:

    uniform_gravity  ->  integrate (symplectic)  ->  reflect_in_box (floor)

rolled forward into a trajectory, then rendered as an ASCII side-view. Then we
sweep the floor's restitution and watch the verdict flip: a lossy bounce saps
the second arc and the ball smacks into the wall; an elastic bounce keeps enough
height to sail over -- emergent, computed physics, not a lookup.

Honest scope
------------
Pure deterministic mechanics, no model. This exercises the *operator* layer (the
"compute physical dynamics" piece next to ``spatial_ops``'s "compute geometry").
First-order symplectic Euler: energy-stable, not exact. ASCII-only, CPU,
sub-second.

Run::

    python examples/demo_physics_ops.py
    python examples/demo_physics_ops.py --restitution 0.4
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import torch

from mt_lnn.physics_ops import rollout


WALL_X = 4.2
WALL_H = 0.45
FLOOR = ([float("-inf"), 0.0], [float("inf"), float("inf")])


def run_demo(args) -> dict:
    pos = torch.tensor([[0.0, 0.0]])
    vel = torch.tensor([[args.vx, args.vy]])
    traj = rollout(pos, vel, steps=args.steps, dt=args.dt,
                   gravity=[0.0, -9.81], bounds=FLOOR,
                   restitution=args.restitution)

    xs = traj.positions[:, 0, 0]
    ys = traj.positions[:, 0, 1]

    # decisive crossing: first snapshot at/after the wall's x
    past = (xs >= WALL_X).nonzero()
    if past.numel() == 0:
        clears = False
        cross_h = None
    else:
        k = int(past[0])
        cross_h = ys[k].item()
        clears = cross_h >= WALL_H

    n_bounces = int(((ys[:-1] <= 1e-4) & (ys[1:] > 1e-4)).sum())

    # sweep restitution to find where the verdict flips
    flip = None
    for e in [round(0.1 * i, 2) for i in range(1, 11)]:
        if _clears(args, e):
            flip = e
            break

    return {
        "xs": xs, "ys": ys, "restitution": args.restitution,
        "clears": clears, "cross_h": cross_h, "n_bounces": n_bounces,
        "flip_restitution": flip, "vx": args.vx, "vy": args.vy,
        "max_height": ys.max().item(), "range_x": xs.max().item(),
    }


def _clears(args, restitution: float) -> bool:
    pos = torch.tensor([[0.0, 0.0]])
    vel = torch.tensor([[args.vx, args.vy]])
    traj = rollout(pos, vel, steps=args.steps, dt=args.dt,
                   gravity=[0.0, -9.81], bounds=FLOOR, restitution=restitution)
    xs = traj.positions[:, 0, 0]
    ys = traj.positions[:, 0, 1]
    past = (xs >= WALL_X).nonzero()
    if past.numel() == 0:
        return False
    return bool(ys[int(past[0])].item() >= WALL_H)


def render(xs: torch.Tensor, ys: torch.Tensor, *, width: int = 56, height: int = 14) -> str:
    """ASCII side-view: 'o' = ball path, '|' = wall, '_' = floor."""
    x_max = max(float(xs.max()), WALL_X) * 1.05 + 1e-6
    y_max = max(float(ys.max()), WALL_H) * 1.15 + 1e-6
    grid = [[" " for _ in range(width)] for _ in range(height)]

    def col(x):
        return min(width - 1, max(0, int(x / x_max * (width - 1))))

    def row(y):
        r = int(y / y_max * (height - 1))
        return min(height - 1, max(0, (height - 1) - r))      # invert: up = smaller row

    # trajectory
    for i in range(xs.shape[0]):
        grid[row(float(ys[i]))][col(float(xs[i]))] = "o"
    # wall column up to its height
    wc = col(WALL_X)
    for y in [WALL_H * t / 8 for t in range(9)]:
        grid[row(y)][wc] = "|"
    # floor line
    floor_row = row(0.0)
    for c in range(width):
        if grid[floor_row][c] == " ":
            grid[floor_row][c] = "_"
    return "\n".join("  " + "".join(r) for r in grid)


def print_report(s: dict) -> None:
    line = "=" * 60
    print(line)
    print("COMPOSABLE PHYSICS OPERATORS -- does the ball clear the wall?")
    print(line)
    print(f"\n  launch v=({s['vx']:.1f}, {s['vy']:.1f})  gravity=-9.81  "
          f"floor restitution={s['restitution']:.2f}")
    print(f"  wall at x={WALL_X:.1f} height={WALL_H:.2f}  bounces={s['n_bounces']}")
    print("  composed: uniform_gravity -> integrate -> reflect_in_box(floor)\n")
    print(render(s["xs"], s["ys"]))
    print(f"\n  trajectory range x : {s['range_x']:.2f}   max height : {s['max_height']:.2f}")
    if s["cross_h"] is None:
        print(f"  ball never reaches the wall at x={WALL_X:.1f}")
    else:
        verb = "CLEARS" if s["clears"] else "HITS"
        print(f"  at the wall the ball is at height {s['cross_h']:.2f} -> {verb} the wall")
    if s["flip_restitution"] is not None:
        print(f"  -> the ball only clears once floor restitution >= {s['flip_restitution']:.2f}")
    else:
        print("  -> the ball does not clear the wall at any tested restitution")
    print("\n" + line)
    print("VERDICT [OK]: the outcome is COMPUTED by rolling the dynamics forward")
    print("        (composed from zero-parameter operators), not memorised.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="composable physics operators demo")
    ap.add_argument("--restitution", type=float, default=1.0,
                    help="floor bounciness. Default 1.0 (elastic) clears the wall.")
    ap.add_argument("--vx", type=float, default=3.0, help="launch x-velocity")
    ap.add_argument("--vy", type=float, default=4.0, help="launch y-velocity")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--dt", type=float, default=0.005)
    args = ap.parse_args()
    print_report(run_demo(args))


if __name__ == "__main__":
    main()
