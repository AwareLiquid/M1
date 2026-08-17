"""
examples/demo_sensory_frontend.py -- close the perception loop: a raw, jittered
sensor stream becomes backbone-ready tokens, with dropouts handled honestly
(P1 closed-loop, step 1).

The story
---------
A real sensor does not tick on the liquid core's fixed ``dt`` clock: samples
arrive jittered, a burst of frames drops, the clock drifts. Feeding that raw
stream straight into a fixed-step recurrent core silently violates its
discretisation. :class:`mt_lnn.sensory_frontend.SensoryFrontend` is the temporal
front that fixes it -- composing two pieces that already exist, with zero
backbone coupling:

    ingest_ops.align_stream      -- resample the jittered stream onto the fixed-dt
                                    grid AND flag which steps are real-sample-backed
                                    vs deep inside a dropout;
    multimodal.ModalityProjector -- project each aligned sample into d_model tokens.

This demo drives the frontend with a 12-channel stream whose timestamps jitter
and that drops a burst of frames mid-way. It shows: (1) the stream lands on a
clean ``dt`` grid; (2) the coverage mask flags exactly the dropout steps (the
ones a :class:`~mt_lnn.failsafe.BlindRolloutGuard` would coast on); (3) the
projected ``(B, M, d_model)`` tokens feed ``MTLNNModel.forward(inputs_embeds=...)``
and produce finite logits -- the loop is closed.

Honest scope
------------
A working, tested temporal injection path -- not a pretrained perceptual tower.
Linear resampling is exact on affine signals only; a too-wide gap is *flagged*,
not invented. Deterministic, CPU, sub-second, ASCII-only.

Run::

    python examples/demo_sensory_frontend.py
    python examples/demo_sensory_frontend.py --dt 0.05
"""

from __future__ import annotations

import argparse

import torch

from mt_lnn.model import MTLNNConfig, MTLNNModel
from mt_lnn.sensory_frontend import SensoryFrontend

_F = torch.float32
_D = 104           # d_model; divisible by n_heads=13 (protofilament count)


def _tiny_model(max_seq_len: int = 64) -> MTLNNModel:
    cfg = MTLNNConfig(
        vocab_size=64, d_model=_D, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=max_seq_len, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
    )
    return MTLNNModel(cfg).eval()


def _ascii_coverage(covered: torch.Tensor) -> str:
    """A timeline: '#' a real-sample-backed step, '.' a coasted dropout step."""
    cells = "".join("#" if bool(c) else "." for c in covered)
    return f"   |{cells}|   ('#' trust sample, '.' coast / BlindRolloutGuard)"


def run_demo(args) -> dict:
    torch.manual_seed(0)
    dt = args.dt
    F = 12                                              # raw sensor channels
    B = 2

    # a jittered clock with a burst dropout: samples up to t~0.2, a hole, then
    # samples resume near t~1.0 -- the grid steps inside the hole are untrusted.
    timestamps = torch.tensor(
        [0.0, 0.09, 0.21, 0.31, 1.00, 1.08, 1.21, 1.29], dtype=_F
    )
    values = torch.randn(B, timestamps.shape[0], F)
    n_steps = int(round(1.3 / dt)) + 1

    fe = SensoryFrontend(in_dim=F, d_model=_D, dt=dt)
    enc = fe(values, timestamps, n_steps=n_steps, t_start=0.0)

    model = _tiny_model(max_seq_len=max(64, n_steps + 4))
    with torch.no_grad():
        out = model(inputs_embeds=enc.embeds, pad_mask=enc.pad_mask, use_cache=True)
    logits = out["logits"]

    return {
        "dt": dt, "F": F, "d_model": _D, "B": B,
        "n_raw": int(timestamps.shape[0]),
        "n_steps": enc.n_steps,
        "n_gap_steps": enc.n_gap_steps,
        "coverage_ratio": enc.coverage_ratio,
        "fully_covered": enc.fully_covered,
        "covered_row": enc.covered[0],
        "embeds_shape": tuple(enc.embeds.shape),
        "logits_shape": tuple(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
    }


def print_report(s: dict) -> None:
    line = "=" * 64
    print(line)
    print("SENSORY FRONTEND -- raw jittered stream -> backbone-ready tokens")
    print(line)
    print(f"\n  raw stream            : {s['n_raw']} samples, {s['F']} channels, "
          f"jittered clock + a burst dropout")
    print(f"  aligned to grid       : dt={s['dt']}, {s['n_steps']} uniform steps")
    print(f"  dropout steps flagged : {s['n_gap_steps']}  "
          f"(coverage {s['coverage_ratio'] * 100:.0f}%, fully_covered={s['fully_covered']})")
    print("\n  coverage timeline (batch item 0):")
    print(_ascii_coverage(s["covered_row"]))

    print(f"\n  projected tokens      : {s['embeds_shape']}  "
          f"(B, M, d_model={s['d_model']}) -> inputs_embeds")
    print(f"  backbone logits       : {s['logits_shape']}  finite={s['logits_finite']}")

    ok = (
        s["n_steps"] > s["n_raw"]               # resampled onto a denser uniform grid
        and s["n_gap_steps"] > 0                # the burst dropout is flagged
        and not s["fully_covered"]
        and s["embeds_shape"][-1] == s["d_model"]
        and s["logits_shape"][:2] == (s["B"], s["n_steps"])
        and s["logits_finite"]
    )
    print("\n" + line)
    if ok:
        print("VERDICT [OK]: the raw jittered stream is aligned to the core's dt")
        print("        grid, its dropout is flagged for the BlindRolloutGuard to")
        print("        coast on, and the projected tokens drive the backbone to")
        print("        finite logits -- the perception loop is closed, zero")
        print("        model.py coupling.")
    else:
        print("VERDICT [FAIL]: frontend did not close the loop -- see above.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="raw streaming-sensor frontend demo")
    ap.add_argument("--dt", type=float, default=0.1,
                    help="the liquid core's fixed timestep to align onto (default 0.1).")
    args = ap.parse_args()
    print_report(run_demo(args))


if __name__ == "__main__":
    main()
