"""
examples/demo_topology_ops.py -- composable TDA on a state population (P0#2):
counting clusters/attractors with Betti-0 and catching topology drift with SRTD.

The story
---------
A population of hidden states / place codes is a point cloud, and its *shape*
carries information no single state does: how many well-separated clusters (read:
attractor basins / distinct place fields) it has, and how strongly separated they
are. That count is the 0-th Betti number; the separation strengths are the H0
persistence barcode. We build a population with THREE clean clusters and show:

    betti0_curve          -- the component count falls 3 -> 1 as the merge scale
                             grows (the discrete fingerprint of three clusters);
    persistent_homology_h0-- the barcode has two long bars (the inter-cluster
                             gaps) and total persistence measures separation.

Then we stress it two ways and watch the topology:

    (1) small jitter      -- topology UNCHANGED: Betti-0 stays 3, SRTD ~ 0;
    (2) cluster collapse  -- topology BROKEN: the clusters fuse, Betti-0 drops,
                             and SRTD spikes.

SRTD (Symmetric Relative-Topology Divergence) is exactly the kind of zero-param
tripwire a *topological failsafe* would watch: a large SRTD between the live
population and its healthy reference means "the representation just changed
shape" -- mode collapse, a steering edit gone wrong, a regime change.

Honest scope
------------
Pure geometry, no model, H0 (connected components) only. 2-D synthetic clouds
for legible ASCII; the operators are dimension-agnostic. Deterministic, CPU,
sub-second, ASCII-only.

Run::

    python examples/demo_topology_ops.py
    python examples/demo_topology_ops.py --spread 6.0
"""

from __future__ import annotations

import argparse
from typing import List

import torch

from mt_lnn.topology_ops import (
    betti0,
    betti0_curve,
    persistent_homology_h0,
    srtd,
    total_persistence,
)

_f64 = torch.float64


def _three_clusters(spread: float, jitter: float, seed: int = 0) -> torch.Tensor:
    """Three blobs of 5 points each, centres ``spread`` apart, radius ``jitter``."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[0.0, 0.0], [spread, 0.0], [spread / 2, spread]], dtype=_f64)
    pts = []
    for c in centres:
        blob = c + jitter * torch.randn(5, 2, generator=g, dtype=_f64)
        pts.append(blob)
    return torch.cat(pts, dim=0)                       # (15, 2)


def _ascii_curve(curve: torch.Tensor, thresholds: torch.Tensor) -> str:
    """A tiny step plot of Betti-0 vs scale."""
    rows = []
    top = int(curve.max())
    for level in range(top, 0, -1):
        cells = "".join("#" if int(c) >= level else " " for c in curve)
        rows.append(f"   b0={level:2d} |{cells}")
    axis = "        +" + "-" * len(curve)
    ticks = f"         scale 0 .. {float(thresholds[-1]):.1f}"
    return "\n".join(rows) + "\n" + axis + "\n" + ticks


def _ascii_barcode(deaths: torch.Tensor, scale: float) -> str:
    """Horizontal bars for the finite H0 lifetimes (births all at 0)."""
    width = 40
    rows = []
    for d in deaths.tolist():
        n = int(round(width * d / scale)) if scale > 0 else 0
        rows.append(f"    |{'=' * n}{' ' * (width - n)}| {d:6.3f}")
    rows.append(f"    |{'=' * width}>  inf  (the one immortal component)")
    return "\n".join(rows)


def run_demo(args) -> dict:
    spread = args.spread
    ref = _three_clusters(spread, jitter=0.15, seed=0)
    jittered = _three_clusters(spread, jitter=0.15, seed=1)         # same shape, new noise
    collapsed = _three_clusters(spread=0.4, jitter=0.15, seed=0)    # clusters fused

    bc = persistent_homology_h0(ref)
    deaths = bc.finite_deaths

    # a merge scale that sits between intra-cluster (~jitter) and inter-cluster (~spread)
    probe = spread / 2.0
    thresholds = torch.linspace(0.0, spread * 1.2, 25, dtype=_f64)
    curve = betti0_curve(ref, thresholds)

    return {
        "spread": spread,
        "ref": ref, "jittered": jittered, "collapsed": collapsed,
        "deaths": deaths,
        "total_persistence": float(total_persistence(ref)),
        "probe": probe,
        "b0_ref": betti0(ref, probe),
        "b0_jittered": betti0(jittered, probe),
        "b0_collapsed": betti0(collapsed, probe),
        "srtd_jitter": float(srtd(ref, jittered)),
        "srtd_collapse": float(srtd(ref, collapsed)),
        "curve": curve, "thresholds": thresholds,
    }


def print_report(s: dict) -> None:
    line = "=" * 64
    print(line)
    print("TOPOLOGICAL DATA ANALYSIS -- counting clusters, catching drift")
    print(line)

    print(f"\n  population = 3 clusters x 5 points, centres {s['spread']:.1f} apart\n")
    print("  Betti-0 vs merge scale (step plot):")
    print(_ascii_curve(s["curve"], s["thresholds"]))

    print("\n  H0 persistence barcode (finite bars = inter-component gaps):")
    print(_ascii_barcode(s["deaths"], s["spread"] * 1.2))
    print(f"\n  total persistence (sum of finite bars) : {s['total_persistence']:.3f}")

    print(f"\n  -- topology at probe scale eps = {s['probe']:.2f} --")
    print(f"  Betti-0 reference population            : {s['b0_ref']}   (3 clusters)")
    print(f"  Betti-0 after small jitter              : {s['b0_jittered']}   "
          f"(shape preserved)")
    print(f"  Betti-0 after cluster collapse          : {s['b0_collapsed']}   "
          f"(shape BROKEN)")

    print("\n  -- SRTD: topology drift from the healthy reference --")
    print(f"  SRTD(ref, jittered)  = {s['srtd_jitter']:.4f}   (tiny: same topology)")
    print(f"  SRTD(ref, collapsed) = {s['srtd_collapse']:.4f}   (large: a tripwire fires)")

    ok = (
        s["b0_ref"] == 3
        and s["b0_jittered"] == 3
        and s["b0_collapsed"] < 3
        and s["srtd_collapse"] > 5.0 * max(s["srtd_jitter"], 1e-9)
    )
    print("\n" + line)
    if ok:
        print("VERDICT [OK]: Betti-0 COMPUTES the cluster count from the cloud, and")
        print("        SRTD flags the collapse (>>) while ignoring benign jitter (~0)")
        print("        -- a zero-parameter topological health check / failsafe signal.")
    else:
        print("VERDICT [FAIL]: topological invariants not as expected -- see above.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="composable TDA (Betti-0 / persistence / SRTD) demo")
    ap.add_argument("--spread", type=float, default=5.0,
                    help="distance between the three cluster centres (default 5.0).")
    args = ap.parse_args()
    print_report(run_demo(args))


if __name__ == "__main__":
    main()
