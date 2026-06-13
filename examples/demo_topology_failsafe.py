"""
examples/demo_topology_failsafe.py -- a zero-parameter tripwire on the *shape*
of the live representation (P1 engineering).

The story
---------
A healthy population of latent states has structure -- distinct modes, well-
separated clusters (3 concepts, say). Two failure modes erase that structure:
*mode collapse* (every state fuses into one blob -- the model has stopped
distinguishing things) and *regime change* (the cluster count jumps). Loss
curves and activation norms can look fine while this happens; what changes is
the *topology* of the state cloud.

:class:`mt_lnn.failsafe.TopologyBreaker` watches exactly that. It latches a
healthy reference cloud, then every tick measures the H0-barcode divergence
(SRTD) of the live cloud from the reference and -- optionally -- its connected-
component count (Betti-0). A debounced finite-state machine (mirroring the
classic :class:`CircuitBreaker`) trips after a few consecutive bad ticks and
closes again after the structure recovers, so a single noisy tick never
flips it.

This demo feeds the breaker a stream of clouds:

    benign jitter  -> SRTD stays ~ 0, never trips;
    mode collapse  -> clusters fuse, SRTD spikes, breaker trips after debounce;
    recovery       -> clusters separate again, breaker closes after a clean run.

Honest scope
------------
A pure external health *signal*, zero trainable parameters; it never touches the
representation -- a caller decides what to do when it trips (freeze, roll back a
steering edit, fall back to a safe policy). The synthetic clusters here stand in
for a real latent population; tapping live hidden states is a separate wiring
step. Deterministic, CPU, sub-second, ASCII-only.

Run::

    python examples/demo_topology_failsafe.py
    python examples/demo_topology_failsafe.py --threshold 0.5
"""

from __future__ import annotations

import argparse

import torch

from mt_lnn.failsafe import TopologyBreaker

_f64 = torch.float64


def _clusters(spread: float, jitter: float, seed: int) -> torch.Tensor:
    """Three blobs of 5 points around centres scaled by ``spread`` (collapse -> 0)."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]], dtype=_f64)
    pts = []
    for c in centres:
        noise = torch.randn(5, 2, generator=g, dtype=_f64) * jitter
        pts.append(c * spread + noise)
    return torch.cat(pts, dim=0)                    # (15, 2)


def run_demo(args) -> dict:
    thr = args.threshold
    # healthy reference: three well-separated clusters.
    ref = _clusters(spread=1.0, jitter=0.2, seed=0)
    brk = TopologyBreaker(srtd_threshold=thr, eps=3.0,
                          trip_after=2, reset_after=3, reference=ref)

    log = []                                        # (phase, srtd, betti0, tripped, reason)

    # phase 1: benign jitter -- same structure, fresh noise. 4 ticks.
    benign_tripped = False
    for k in range(4):
        r = brk.step(_clusters(spread=1.0, jitter=0.2, seed=10 + k))
        benign_tripped = benign_tripped or r.tripped
        log.append(("jitter", r.srtd, r.betti0, r.tripped, r.reason))

    # phase 2: mode collapse -- clusters fuse toward a single blob. 4 ticks.
    collapse_trip_tick = None
    for k in range(4):
        r = brk.step(_clusters(spread=0.05, jitter=0.2, seed=20 + k))
        if r.tripped and collapse_trip_tick is None:
            collapse_trip_tick = k
        log.append(("collapse", r.srtd, r.betti0, r.tripped, r.reason))

    # phase 3: recovery -- structure returns. 4 ticks.
    recovery_close_tick = None
    for k in range(4):
        r = brk.step(_clusters(spread=1.0, jitter=0.2, seed=30 + k))
        if not r.tripped and recovery_close_tick is None:
            recovery_close_tick = k
        log.append(("recover", r.srtd, r.betti0, r.tripped, r.reason))

    return {
        "threshold": thr,
        "n_parameters": brk.n_parameters,
        "benign_tripped": benign_tripped,
        "collapse_trip_tick": collapse_trip_tick,
        "recovery_close_tick": recovery_close_tick,
        "log": log,
    }


def print_report(s: dict) -> None:
    line = "=" * 64
    print(line)
    print("TOPOLOGY FAILSAFE -- a tripwire on the SHAPE of the representation")
    print(line)
    print(f"\n  srtd_threshold = {s['threshold']:.2f}, eps = 3.0, "
          f"trip_after = 2, reset_after = 3")
    print(f"  trainable parameters : {s['n_parameters']}  (pure external signal)")

    print(f"\n  {'phase':<10}{'srtd':>10}{'betti0':>8}{'tripped':>9}{'reason':>9}")
    print(f"  {'-' * 10}{'-' * 10:>10}{'-' * 8:>8}{'-' * 9:>9}{'-' * 9:>9}")
    for phase, srtd, b0, tripped, reason in s["log"]:
        b0s = "-" if b0 is None else str(b0)
        print(f"  {phase:<10}{srtd:>10.3f}{b0s:>8}"
              f"{('TRIP' if tripped else 'ok'):>9}{(reason or '-'):>9}")

    print(f"\n  benign jitter ever tripped : {s['benign_tripped']}  (want False)")
    print(f"  collapse tripped at tick   : {s['collapse_trip_tick']}  "
          f"(debounced: want 1, i.e. 2nd bad tick)")
    print(f"  recovery closed at tick    : {s['recovery_close_tick']}  "
          f"(debounced: want 2, i.e. 3rd clean tick)")

    ok = (
        not s["benign_tripped"]
        and s["collapse_trip_tick"] == 1
        and s["recovery_close_tick"] == 2
        and s["n_parameters"] == 0
    )
    print("\n" + line)
    if ok:
        print("VERDICT [OK]: the breaker ignores benign jitter, TRIPS on a mode")
        print("        collapse only after the debounce confirms it, and CLOSES")
        print("        again once the structure recovers -- a zero-parameter")
        print("        topological health signal, not a representation edit.")
    else:
        print("VERDICT [FAIL]: failsafe behaviour not as expected -- see above.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="topological failsafe (TopologyBreaker) demo")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="SRTD trip threshold (> 0, default 1.0).")
    args = ap.parse_args()
    print_report(run_demo(args))


if __name__ == "__main__":
    main()
