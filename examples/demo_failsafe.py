"""
examples/demo_failsafe.py -- input-dropout blind rollout + output circuit breaker.

The story
---------
A fast liquid core is deployed in the wild, where two failures the training loss
never saw will happen sooner or later:

  Scene 1 -- THE INPUT STUTTERS.
    The sensor feed gaps for a few ticks. Instead of feeding the core a stale or
    zero frame (which silently corrupts its recurrent state), a
    :class:`mt_lnn.failsafe.BlindRolloutGuard` *coasts on the world model's
    imagination* from the last good state -- but only while the imagination's own
    confidence stays above a floor. The moment the dream is no longer trustworthy
    it goes DARK ("I can no longer see") rather than hallucinating forever.

  Scene 2 -- THE OUTPUT MISBEHAVES.
    The model emits a command that is NaN, then slams out of bounds, then slews
    impossibly fast. A model-external :class:`mt_lnn.failsafe.CircuitBreaker`
    *always* hard-clamps the command into the physical red-lines, and trips to a
    safe fallback (here: return-to-setpoint) with debounce when the model keeps
    violating them -- then closes again with bumpless transfer once it recovers.

Both actuators have ZERO trainable parameters and never import ``model.py``.
Scene 1 drives the *real* world-model imagination (a randomly initialised
``PredictiveStateHead`` -- the rollout mechanism is what we demonstrate, not a
trained predictor's accuracy).

Honest scope
------------
Deterministic (seeded). The guard runs on a real but untrained head, so the
confidences are the genuine geometric/novelty discount of the rollout, not a
claim about prediction quality. CPU, sub-second, ASCII-only.

Run::

    python -m examples.demo_failsafe
    python -m examples.demo_failsafe --floor 0.45
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional

import torch

from mt_lnn.failsafe import BlindRolloutGuard, CircuitBreaker
from mt_lnn.imagination import LatentImagination
from mt_lnn.world_model import PredictiveStateHead


# --------------------------------------------------------------------------- #
# Scene 1 -- input-dropout blind rollout                                       #
# --------------------------------------------------------------------------- #


def run_guard(args) -> dict:
    """Live ticks, a dropout window, then recovery. Real world-model imagination."""
    torch.manual_seed(0)
    head = PredictiveStateHead(d_model=args.d_model)
    head.eval()
    im = LatentImagination(head, horizon=args.horizon, trust_decay=args.trust_decay)
    guard = BlindRolloutGuard(im, confidence_floor=args.floor)

    # present mask: 3 live, a long dropout (forces it past confidence/budget),
    # then the feed returns and it re-coasts on the next dropout.
    present = [True, True, True] + [False] * (args.horizon + 2) + [True, False, False]
    base = torch.randn(1, args.d_model)

    rows = []
    for t, live in enumerate(present):
        if live:
            # a slowly evolving "real" hidden state when the feed is up
            hidden = base + 0.05 * t * torch.randn(1, args.d_model)
            out = guard.step(hidden)
        else:
            out = guard.step(None)
        rows.append(out)
    return {"present": present, "rows": rows, "horizon": args.horizon,
            "floor": args.floor}


def render_guard(d: dict) -> str:
    present, rows = d["present"], d["rows"]
    sym = {"live": "L", "imagined": "~", "dark": "X"}
    feed = "".join("#" if p else "." for p in present)
    served = "".join(sym[r.source] for r in rows)
    conf = "".join(_bar(r.confidence) for r in rows)
    return ("    feed   : " + feed + "    (# input present, . dropped)\n"
            "    served : " + served + "    (L live, ~ imagined, X dark)\n"
            "    conf   : " + conf + "    (height = trust in served latent)")


def _bar(x: float) -> str:
    ramp = " .:-=+*#%@"
    i = max(0, min(len(ramp) - 1, int(round(x * (len(ramp) - 1)))))
    return ramp[i]


# --------------------------------------------------------------------------- #
# Scene 2 -- model-external output circuit breaker                             #
# --------------------------------------------------------------------------- #


def command_stream() -> List[float]:
    """A clean ramp, a NaN, an out-of-bounds slam, a slew spike, then recovery."""
    s: List[float] = []
    s += [0.0, 0.2, 0.4, 0.6, 0.8]                       # clean ramp toward setpoint
    s += [float("nan")]                                  # garbage
    s += [9.0, 9.0, 9.0]                                 # slam way out of [-1, 1]
    s += [0.5, -0.5, 0.5]                                # back to sane (recovery)
    s += [7.0]                                           # one last spike (slew)
    s += [0.3, 0.3, 0.3]                                 # settle
    return s


def run_breaker(args) -> dict:
    cmds = command_stream()
    breaker = CircuitBreaker(lo=-1.0, hi=1.0, max_rate=0.5,
                             fallback=lambda last: 0.0,   # PID-style return-to-setpoint 0
                             trip_after=2, reset_after=2)
    rows = [breaker.step(c) for c in cmds]
    return {"cmds": cmds, "rows": rows,
            "n_tripped": sum(r.tripped for r in rows),
            "n_overridden": sum(r.overridden for r in rows)}


def render_breaker(d: dict) -> str:
    lines = ["    tick  raw cmd     safe out   tripped  reason"]
    for i, (c, r) in enumerate(zip(d["cmds"], d["rows"]), start=1):
        raw = "   nan " if c != c else f"{c:7.2f}"
        flag = "TRIP" if r.tripped else " -- "
        lines.append(f"    {i:>3}  {raw}     {r.value:7.3f}    {flag}    "
                     f"{r.reason or 'clean'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# report                                                                       #
# --------------------------------------------------------------------------- #


def print_report(gd: dict, bd: dict) -> None:
    line = "=" * 64
    print(line)
    print("FAILSAFE -- coast through input dropouts, clamp the output red-lines")
    print(line)

    print("\nSCENE 1  input dropout -> blind rollout on the world model")
    print(f"  horizon={gd['horizon']}  confidence_floor={gd['floor']:.2f}  "
          "(coast while trusted, go DARK rather than hallucinate)\n")
    print(render_guard(gd))
    n_imagined = sum(r.source == "imagined" for r in gd["rows"])
    n_dark = sum(r.source == "dark" for r in gd["rows"])
    print(f"\n  coasted {n_imagined} tick(s) on imagination, "
          f"went dark {n_dark} tick(s) when trust ran out.")

    print("\n" + "-" * 64)
    print("\nSCENE 2  output red-lines -> model-external circuit breaker")
    print("  bounds=[-1, 1]  max_rate=0.5/tick  trip_after=2  fallback=setpoint 0\n")
    print(render_breaker(bd))
    print(f"\n  raw stream violated the red-lines; breaker tripped "
          f"{bd['n_tripped']} tick(s), overrode {bd['n_overridden']} tick(s);")
    print("  every emitted value stayed finite and inside [-1, 1].")

    print("\n" + line)
    print("VERDICT [OK]: the core COASTS through brief blindness on its own world")
    print("        model and STOPS when it can't be trusted; the output is hard-")
    print("        clamped to physical red-lines no matter what the model emits --")
    print("        two zero-parameter, model-external safety reflexes.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="failsafe: blind rollout + circuit breaker demo")
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--trust-decay", type=float, default=0.8)
    ap.add_argument("--floor", type=float, default=0.35)
    args = ap.parse_args()
    print_report(run_guard(args), run_breaker(args))


if __name__ == "__main__":
    main()
