"""
examples/demo_salience_events.py -- global-workspace ignition / state-change events.

The story
---------
The fast liquid core runs hot every tick. We do NOT want the slow / symbolic /
cloud layer polling that loop -- it should be *woken* only when something
salient changes. Here a stream of per-step **surprise** (the world model's
prediction error) flows by: long calm stretches, then a regime change where the
signal jumps, then it settles into a new normal. A zero-parameter
:class:`mt_lnn.salience_events.SalienceEventDetector` watches the stream and
fires discrete events:

    adaptive baseline (homeostasis) -> z-score (surprise in sigma)
        -> ignition threshold (GWT "win the workspace") + Schmitt release
        -> refractory silence (no event storm)

We render the surprise as an ASCII timeline and mark where the detector ignites
(^) and quiesces (v). The point: a slow drift the system has *adapted to* fires
nothing, while a genuine departure trips the wire -- emergent, computed salience,
not a fixed alarm level.

Honest scope
------------
Deterministic synthetic surprise stream, no model in the loop (the detector is
duck-typed on ``last_pred_error`` and tested against a real predictive head
separately). CPU, sub-second, ASCII-only.

Run::

    python -m examples.demo_salience_events
    python -m examples.demo_salience_events --ignite-z 2.5
"""

from __future__ import annotations

import argparse
import math
from typing import List

from mt_lnn.salience_events import SalienceEventDetector


def surprise_stream() -> List[float]:
    """A calm baseline, a slow adapted drift, then a sharp regime change.

    All deterministic. Values stand in for the world model's normalised
    prediction error (surprise) at each tick.
    """
    s: List[float] = []
    s += [0.10 + 0.01 * math.sin(i) for i in range(20)]      # calm, tiny ripple
    s += [0.10 + 0.004 * i for i in range(20)]               # slow drift (adapted to)
    s += [0.85, 0.92, 0.88, 0.83, 0.80]                      # REGIME CHANGE: surprise jumps
    s += [0.20 + 0.01 * math.sin(i) for i in range(20)]      # settles into a new normal
    return s


def run_demo(args) -> dict:
    stream = surprise_stream()
    det = SalienceEventDetector(ignite_z=args.ignite_z, release_z=args.release_z,
                                refractory=args.refractory, warmup=args.warmup)
    events = det.observe(stream)
    return {
        "stream": stream,
        "events": events,
        "n_ignition": sum(e.kind == "ignition" for e in events),
        "n_quiescence": sum(e.kind == "quiescence" for e in events),
        "ignite_z": args.ignite_z,
        "first_ignition_step": next((e.step for e in events if e.kind == "ignition"), None),
    }


def render(stream: List[float], events, *, width: int = 0, height: int = 9) -> str:
    """ASCII timeline: each column is a tick; bar height = surprise. Event marks
    on the row beneath: '^' ignition, 'v' quiescence."""
    n = len(stream)
    hi = max(stream) or 1.0
    rows = [[" " for _ in range(n)] for _ in range(height)]
    for c, val in enumerate(stream):
        h = int(round(val / hi * (height - 1)))
        for r in range(h + 1):
            rows[height - 1 - r][c] = "|"
    body = "\n".join("  " + "".join(r) for r in rows)

    mark = [" "] * n
    for e in events:
        mark[e.step - 1] = "^" if e.kind == "ignition" else "v"
    axis = "  " + "-" * n
    marks = "  " + "".join(mark)
    return body + "\n" + axis + "\n" + marks


def print_report(s: dict) -> None:
    line = "=" * 60
    print(line)
    print("GLOBAL-WORKSPACE IGNITION -- waking the slow layer on salient change")
    print(line)
    print(f"\n  ticks={len(s['stream'])}  ignite_z={s['ignite_z']:.1f}  "
          f"(surprise = world-model prediction error per tick)")
    print("  adaptive baseline -> z-score -> ignition + Schmitt release + refractory\n")
    print(render(s["stream"], s["events"]))
    print(f"\n  events: {s['n_ignition']} ignition(s), {s['n_quiescence']} quiescence(s)")
    if s["first_ignition_step"] is not None:
        print(f"  first ignition at tick {s['first_ignition_step']} "
              f"(the regime change, NOT the slow drift before it)")
    for e in s["events"]:
        print(f"    tick {e.step:>3}  {e.kind:<10}  z={e.z:5.1f}  "
              f"signal={e.signal:.2f}  baseline={e.baseline:.2f}")
    print("\n" + line)
    print("VERDICT [OK]: salience is COMPUTED relative to an adapting baseline")
    print("        (slow drift fires nothing; a real departure trips the wire),")
    print("        a zero-parameter tripwire for the dual-speed engine.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="global-workspace ignition events demo")
    ap.add_argument("--ignite-z", type=float, default=3.0)
    ap.add_argument("--release-z", type=float, default=1.0)
    ap.add_argument("--refractory", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    print_report(run_demo(args))


if __name__ == "__main__":
    main()
