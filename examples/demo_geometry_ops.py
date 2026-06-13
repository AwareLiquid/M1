"""
examples/demo_geometry_ops.py -- composable Fisher-Rao information geometry
(P0#1: the simplex is the space the L2 place-cell codes actually live on).

The story
---------
``PlaceCellCode.forward`` ends in ``torch.softmax(...)``: every place code is a
*probability distribution* -- a point on the simplex Delta^{n-1}, not a free
vector in R^n. Yet L2 spatial memory reads and writes those codes with plain
Euclidean algebra (``key @ M``). Euclidean straight lines cut THROUGH the
interior of the simplex and measure distance with the wrong ruler; the correct,
coordinate-free ruler is the Fisher-Rao metric, under which the simplex is a
curved manifold (a piece of a sphere of radius 2).

This demo takes two real-shaped place codes and walks from one to the other two
ways:

    Euclidean chord :  m_euc(t) = (1-t) p + t q          (the metric L2 assumes)
    Fisher-Rao path :  geodesic_interpolate(p, q, t)     (the metric it should)

and shows they DISAGREE -- the Euclidean midpoint is not the geometric midpoint,
it is closer/blurrier by the natural (Fisher-Rao) ruler. We compose:

    fisher_rao_distance  /  geodesic_interpolate  /  log_map / exp_map /
    fisher_norm  /  geodesic_barycenter

all zero-parameter pure operators from :mod:`mt_lnn.geometry_ops`, and verify
the laws that make the geodesic the *right* interpolation: it is constant-speed
and its half-length equals half the endpoint distance (the Euclidean chord is
neither).

Honest scope
------------
Pure geometry, no model, no training. This demonstrates the *operator* layer --
the "compute on the simplex with the correct metric" piece that L2 memory is
currently missing. Deterministic, CPU, sub-second, ASCII-only. It does not (yet)
re-wire spatial_memory.py; it makes the metric-inconsistency concrete and
quantifiable so that rewrite is grounded, not 皮毛.

Run::

    python examples/demo_geometry_ops.py
    python examples/demo_geometry_ops.py --steps 8
"""

from __future__ import annotations

import argparse
from typing import List

import torch

from mt_lnn.geometry_ops import (
    bhattacharyya_coefficient,
    fisher_rao_distance,
    geodesic_barycenter,
    geodesic_interpolate,
    is_on_simplex,
)

_f64 = torch.float64


def _place_codes() -> tuple[torch.Tensor, torch.Tensor]:
    """Two place-cell-shaped codes: softmax of fixed logits over 6 cells.

    Peaked on different cells (an agent at two different locations), exactly the
    simplex-valued objects ``PlaceCellCode.forward`` emits.
    """
    logits_p = torch.tensor([3.0, 1.5, 0.2, -1.0, -0.5, 0.1], dtype=_f64)
    logits_q = torch.tensor([-1.0, -0.5, 0.3, 2.8, 1.2, -0.2], dtype=_f64)
    p = torch.softmax(logits_p, dim=-1)
    q = torch.softmax(logits_q, dim=-1)
    return p, q


def _bar(prob: torch.Tensor, width: int = 24) -> str:
    """A horizontal ASCII bar chart of a distribution (peak normalised)."""
    peak = float(prob.max())
    rows = []
    for i, v in enumerate(prob.tolist()):
        n = int(round(width * v / peak)) if peak > 0 else 0
        rows.append(f"    cell {i}: {'#' * n}{'.' * (width - n)} {v:5.3f}")
    return "\n".join(rows)


def run_demo(args) -> dict:
    p, q = _place_codes()
    d_pq = float(fisher_rao_distance(p, q))

    # the two midpoints
    geo_mid = geodesic_interpolate(p, q, 0.5)
    euc_mid = 0.5 * (p + q)

    # how far is each midpoint, by the correct (Fisher-Rao) ruler?
    d_geo = float(fisher_rao_distance(p, geo_mid))           # == d_pq / 2 exactly
    d_euc = float(fisher_rao_distance(p, euc_mid))
    # the geometric error the Euclidean chord makes at the midpoint
    chord_gap = float((geo_mid - euc_mid).abs().max())

    # constant-speed check along the geodesic: d(p, gamma(t)) should be t * d_pq
    ts = [i / args.steps for i in range(args.steps + 1)]
    speed_rows: List[tuple] = []
    max_speed_err = 0.0
    for t in ts:
        g = geodesic_interpolate(p, q, t)
        dt = float(fisher_rao_distance(p, g))
        err = abs(dt - t * d_pq)
        max_speed_err = max(max_speed_err, err)
        speed_rows.append((t, dt, t * d_pq, bool(is_on_simplex(g))))

    # the geodesic midpoint is also the equal-weight Karcher barycenter
    bary = geodesic_barycenter(torch.stack([p, q]),
                               torch.tensor([0.5, 0.5], dtype=_f64))
    bary_gap = float((bary - geo_mid).abs().max())

    return {
        "p": p, "q": q, "geo_mid": geo_mid, "euc_mid": euc_mid,
        "d_pq": d_pq, "d_geo": d_geo, "d_euc": d_euc,
        "bc": float(bhattacharyya_coefficient(p, q)),
        "chord_gap": chord_gap, "max_speed_err": max_speed_err,
        "bary_gap": bary_gap, "speed_rows": speed_rows,
        "euc_on_simplex": bool(is_on_simplex(euc_mid)),
    }


def print_report(s: dict) -> None:
    line = "=" * 64
    print(line)
    print("FISHER-RAO INFORMATION GEOMETRY -- place codes live on the simplex")
    print(line)

    print("\n  p = softmax(logits_p)   (place code A)")
    print(_bar(s["p"]))
    print("\n  q = softmax(logits_q)   (place code B)")
    print(_bar(s["q"]))

    print(f"\n  Bhattacharyya overlap   BC(p,q) = {s['bc']:.4f}")
    print(f"  Fisher-Rao distance     d(p,q)  = {s['d_pq']:.4f}  (in [0, pi])")

    print("\n  -- two ways to stand 'halfway' between p and q --")
    print("\n  Fisher-Rao geodesic midpoint  gamma(0.5):")
    print(_bar(s["geo_mid"]))
    print("\n  Euclidean chord midpoint      0.5*(p+q):")
    print(_bar(s["euc_mid"]))

    print(f"\n  max per-cell gap between them          : {s['chord_gap']:.4f}")
    print(f"  d(p, geodesic midpoint) [exact half]   : {s['d_geo']:.4f}"
          f"   (= d/2 = {s['d_pq'] / 2:.4f})")
    print(f"  d(p, Euclidean  midpoint)              : {s['d_euc']:.4f}"
          f"   (NOT d/2 -- wrong ruler)")

    print("\n  -- constant-speed law: d(p, gamma(t)) should equal t * d(p,q) --")
    print("     t     measured   t*d(p,q)   on-simplex")
    for t, dt, td, on in s["speed_rows"]:
        print(f"   {t:4.2f}    {dt:7.4f}    {td:7.4f}      {'yes' if on else 'NO'}")
    print(f"\n  max constant-speed error               : {s['max_speed_err']:.2e}")
    print(f"  |Karcher barycenter - geodesic midpoint|: {s['bary_gap']:.2e}")

    ok = (
        s["max_speed_err"] < 1e-6
        and s["bary_gap"] < 1e-5
        and abs(s["d_geo"] - s["d_pq"] / 2) < 1e-6
        and s["chord_gap"] > 1e-3            # the chord genuinely disagrees
    )
    print("\n" + line)
    if ok:
        print("VERDICT [OK]: the Fisher-Rao geodesic is the constant-speed midpoint")
        print("        on the simplex; the Euclidean chord L2 currently assumes is")
        print("        a different, metric-inconsistent path. Distance is COMPUTED")
        print("        from the correct curved geometry, not Euclidean shortcut.")
    else:
        print("VERDICT [FAIL]: geometric invariants not satisfied -- see numbers above.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="composable Fisher-Rao geometry demo")
    ap.add_argument("--steps", type=int, default=10,
                    help="number of geodesic samples for the constant-speed table.")
    args = ap.parse_args()
    print_report(run_demo(args))


if __name__ == "__main__":
    main()
