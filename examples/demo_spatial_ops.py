"""
examples/demo_spatial_ops.py -- composable geometric operators (L4 step 2).

The story
---------
An agent stands on a field of stepping-stones split by a gap. It can stride at
most ``--reach`` metres at a time. Question: *can it reach the goal?* This is a
spatial-reasoning query that must be COMPUTED from the geometry + the step
length, not looked up. We answer it by composing three zero-parameter operators
from :mod:`mt_lnn.spatial_ops`:

    pairwise_distance  ->  radius_graph(reach)  ->  reachable_from(start)

and render the result as an ASCII map. Then we sweep the stride and show the gap
"unlocks" the far side once the stride is long enough -- emergent, computed
reachability, not a stored answer.

Honest scope
------------
Pure geometry, no model. This demonstrates the *operator* layer (the missing
"compute over space" piece next to ``spatial.py``'s perception encoders). It is
deterministic, CPU, sub-second, ASCII-only.

Run::

    python examples/demo_spatial_ops.py
    python examples/demo_spatial_ops.py --reach 1.6
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch

from mt_lnn.spatial_ops import (
    pairwise_distance,
    radius_graph,
    reachable_from,
    hop_distance,
)


def stepping_stones() -> Tuple[torch.Tensor, int, int, float]:
    """A 5x3 lattice of stones with a vertical gap (column 2 removed).

    Returns (coords (N,2), start_index, goal_index, gap_width).
    """
    pts: List[Tuple[float, float]] = []
    meta = []
    for x in range(5):
        if x == 2:
            continue                                   # the gap column
        for y in range(3):
            pts.append((float(x), float(y)))
            meta.append((x, y))
    coords = torch.tensor(pts)
    start = meta.index((0, 1))                          # left-middle
    goal = meta.index((4, 1))                           # right-middle
    return coords, start, goal, 2.0                     # gap spans x=1..x=3 => width 2


def render(coords: torch.Tensor, reach: torch.Tensor, start: int, goal: int) -> str:
    """ASCII map: S=start, G=goal(reached) / g=goal(unreached), #=reachable,
    .=unreachable stone, space=gap."""
    xs = coords[:, 0].long()
    ys = coords[:, 1].long()
    W, H = int(xs.max()) + 1, int(ys.max()) + 1
    grid = [[" " for _ in range(W)] for _ in range(H)]
    for i in range(coords.shape[0]):
        x, y = int(xs[i]), int(ys[i])
        if i == start:
            ch = "S"
        elif i == goal:
            ch = "G" if reach[i] > 0 else "g"
        else:
            ch = "#" if reach[i] > 0 else "."
        grid[y][x] = ch
    # print top row last so y increases upward
    return "\n".join("  " + " ".join(row) for row in reversed(grid))


def run_demo(args) -> dict:
    coords, start, goal, gap = stepping_stones()
    src = torch.zeros(coords.shape[0])
    src[start] = 1.0

    adj = radius_graph(coords, radius=args.reach)
    reach = reachable_from(adj, src)
    hops = hop_distance(adj, src)

    goal_reached = bool(reach[goal] > 0)
    goal_hops = hops[goal].item()

    # sweep to find the stride at which the far side unlocks
    strides = [round(0.2 * i, 2) for i in range(5, 16)]   # 1.0 .. 3.0
    unlock = None
    for s in strides:
        r = reachable_from(radius_graph(coords, radius=s), src)
        if r[goal] > 0:
            unlock = s
            break

    return {
        "coords": coords, "start": start, "goal": goal, "gap": gap,
        "reach": reach, "stride": args.reach,
        "goal_reached": goal_reached, "goal_hops": goal_hops,
        "unlock_stride": unlock,
        "n_reachable": int((reach > 0).sum()),
        "n_stones": coords.shape[0],
    }


def print_report(s: dict) -> None:
    line = "=" * 60
    print(line)
    print("COMPOSABLE SPATIAL OPERATORS -- reachability across a gap")
    print(line)
    print(f"\n  stones={s['n_stones']}  gap width={s['gap']:.1f}  agent stride={s['stride']:.2f}")
    print(f"  composed query: distance -> radius_graph({s['stride']:.2f}) -> reachable_from(start)\n")
    print(render(s["coords"], s["reach"], s["start"], s["goal"]))
    print(f"\n  reachable stones : {s['n_reachable']}/{s['n_stones']}")
    if s["goal_reached"]:
        print(f"  GOAL reachable in {int(s['goal_hops'])} hops with stride {s['stride']:.2f}")
    else:
        print(f"  GOAL NOT reachable with stride {s['stride']:.2f} (gap {s['gap']:.1f} too wide)")
    if s["unlock_stride"] is not None:
        print(f"  -> the far side unlocks once stride >= {s['unlock_stride']:.2f}")
    print("\n" + line)
    print("VERDICT [OK]: reachability is COMPUTED from geometry + step length")
    print("        (composed from zero-parameter operators), not memorised.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="composable spatial operators demo")
    ap.add_argument("--reach", type=float, default=1.1,
                    help="agent max stride (metres). Default 1.1 cannot cross the gap.")
    args = ap.parse_args()
    print_report(run_demo(args))


if __name__ == "__main__":
    main()
