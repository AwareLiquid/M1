"""Generate a synthetic reasoning_trace.jsonl for UI demo / screenshots.

Doesn't load any LM — just emits a plausible mix of LOCAL / SELF_CRITIQUE /
CLOUD / inject events via ReasoningTrace so trace_timeline.html has
something to render out of the box.

    python scripts/demo_trace_synth.py --out demo_trace.jsonl
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from mt_lnn.reasoning_trace import ReasoningTrace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="demo_trace.jsonl")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        out.unlink()
    rng = random.Random(args.seed)
    trace = ReasoningTrace(str(out), session_id="demo_synth", phi_every=8)

    inject_done = False
    for i in range(args.n):
        # Entropy: bursty pattern — calm, then a high-entropy stretch ≈ step 30, then calm
        base = 1.8 + 0.6 * math.sin(i / 9.0)
        if 28 <= i <= 36:
            base += 3.2  # high-entropy region triggers cloud
        elif 60 <= i <= 65:
            base += 1.6  # mid-entropy region triggers self_critique
        ent = max(0.1, base + rng.gauss(0, 0.25))

        if 28 <= i <= 36 and not inject_done:
            trace.record_route(route="cloud", reason="entropy>5 + fact_gap",
                               extras={"entropy": ent, "fact_gap": 0.78})
            trace.record_cloud_inject(source="mock_oracle",
                                      query="origin of m-theory",
                                      fact_len=212)
            inject_done = True
            route = "cloud"
        elif 60 <= i <= 65:
            trace.record_route(route="self_critique", reason="mid-entropy",
                               extras={"entropy": ent})
            route = "self_critique"
        else:
            route = "local"

        phi = None
        if i > 0 and i % 8 == 0:
            phi = 0.05 + 0.25 * rng.random() + (0.4 if 28 <= i <= 36 else 0)
        trace.record_token(token_id=rng.randint(100, 30000),
                           entropy=ent, route=route, phi=phi)

    trace.close()
    print(f"wrote {args.n} synthetic trace events to {out.resolve()}")


if __name__ == "__main__":
    main()
