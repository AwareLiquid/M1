"""
examples/demo_top_down_modulation.py -- a high-level goal biases every block, with
the pathway closed (a strict no-op) until it is opened (P1 closed-loop, step 2).

The story
---------
A brain's higher areas do not just receive bottom-up sensation -- they send
top-down feedback that *biases* lower-level processing toward the current goal or
context. The backbone gains exactly that pathway here: ``MTLNNModel.forward`` now
accepts a ``top_down`` goal/context vector that each block folds in as a
ZERO-INIT-GATED residual, injected after attention and before the liquid core::

    x = x + tanh(gate) * proj(LayerNorm(top_down))

Two honest properties make this safe to add to a trained model:

  1. CLOSED GATE == STRICT NO-OP. ``gate`` starts at 0, so ``tanh(0)=0`` and the
     residual is exactly zero -- feeding any goal is bit-identical to feeding
     none. A pretrained checkpoint is untouched until training opens the gate.
  2. OPEN GATE STEERS. Once the gate opens, two different goals push the SAME
     input to measurably different next-token distributions -- the top-down
     pathway is live and the goal actually matters.

This demo drives a tiny model with a fixed token sequence and shows: (1) with the
gate closed, goal-vs-none logits are bit-identical (zero regression); (2) with the
gate open, two distinct goals diverge the output distribution. The whole feature
is gated by ``config.use_top_down`` and defaults OFF -- zero coupling for any
model that does not ask for it.

Honest scope
------------
A working, tested top-down *injection path* -- not a trained controller. The gate
starts closed (no effect); a real deployment LEARNS to open and aim it. The GWT
docking entry (``top_down_to_gwtb``) is exercised by the test-suite, not here.
Deterministic, CPU, sub-second, ASCII-only.

Run::

    python examples/demo_top_down_modulation.py
    python examples/demo_top_down_modulation.py --gate 2.0
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel

_D = 104           # d_model; divisible by n_heads=13 (protofilament count)


def _tiny_model() -> MTLNNModel:
    cfg = MTLNNConfig(
        vocab_size=64, d_model=_D, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=64, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0, use_top_down=True,
    )
    return MTLNNModel(cfg).eval()


def _ascii_bar(value: float, lo: float, hi: float, width: int = 40) -> str:
    """A simple ASCII gauge: where `value` sits in [lo, hi]."""
    span = max(hi - lo, 1e-12)
    fill = int(round((min(max(value, lo), hi) - lo) / span * width))
    return "[" + "#" * fill + "-" * (width - fill) + "]"


def run_demo(args) -> dict:
    torch.manual_seed(0)
    B, T = 1, 10
    ids = torch.randint(0, 64, (B, T))
    goal_a = torch.randn(B, _D)
    goal_b = torch.randn(B, _D)

    model = _tiny_model()

    with torch.no_grad():
        # 1. Gate CLOSED (init): a goal must be a strict no-op vs no goal at all.
        none = model(input_ids=ids, top_down=None)["logits"]
        closed = model(input_ids=ids, top_down=goal_a)["logits"]
        closed_diff = (none - closed).abs().max().item()

        # 2. Open every block's gate, then steer with two distinct goals.
        for blk in model.blocks:
            blk.top_down_gate.data.fill_(float(args.gate))
        log_a = model(input_ids=ids, top_down=goal_a)["logits"]
        log_b = model(input_ids=ids, top_down=goal_b)["logits"]

    # how far the two goals push the LAST-position next-token distribution apart
    pa = F.softmax(log_a[0, -1], dim=-1)
    pb = F.softmax(log_b[0, -1], dim=-1)
    kl_ab = float((pa * (pa / (pb + 1e-9)).log()).sum())          # KL(a || b)
    argmax_a = int(pa.argmax())
    argmax_b = int(pb.argmax())
    open_diff = (log_a - log_b).abs().max().item()

    return {
        "gate": float(args.gate), "B": B, "T": T, "d_model": _D,
        "closed_diff": closed_diff,
        "open_diff": open_diff,
        "kl_ab": kl_ab,
        "argmax_a": argmax_a, "argmax_b": argmax_b,
        "logits_finite": bool(torch.isfinite(log_a).all() and torch.isfinite(log_b).all()),
    }


def print_report(s: dict) -> None:
    line = "=" * 64
    print(line)
    print("TOP-DOWN MODULATION -- a goal biases every block, closed until opened")
    print(line)

    print(f"\n  setup                 : B={s['B']}, T={s['T']}, d_model={s['d_model']}, "
          f"2-layer tiny model (use_top_down=True)")

    print("\n  [1] gate CLOSED (init) -- a goal must change nothing:")
    print(f"        max|logits(goal) - logits(none)| = {s['closed_diff']:.2e}")
    print(f"        {_ascii_bar(s['closed_diff'], 0.0, 1e-6)}  (want ~0: strict no-op)")

    print(f"\n  [2] gate OPEN (={s['gate']}) -- two distinct goals must steer:")
    print(f"        max|logits(goalA) - logits(goalB)| = {s['open_diff']:.3f}")
    print(f"        KL(next-token A || B) at last step = {s['kl_ab']:.3f} nats")
    print(f"        next-token argmax: goalA -> {s['argmax_a']:>2d} | goalB -> {s['argmax_b']:>2d}")
    print(f"        {_ascii_bar(s['kl_ab'], 0.0, 1.0)}  (want > 0: goal actually matters)")

    ok = (
        s["closed_diff"] == 0.0          # closed gate is a STRICT no-op (bit-exact)
        and s["open_diff"] > 1e-3        # open gate genuinely steers the logits
        and s["kl_ab"] > 1e-4            # the two goals diverge the distribution
        and s["logits_finite"]
    )
    print("\n" + line)
    if ok:
        print("VERDICT [OK]: with the gate closed a top-down goal is bit-identical to")
        print("        no goal (a trained checkpoint is untouched); with the gate open")
        print("        two different goals steer the same input to different next-token")
        print("        distributions -- the top-down pathway is live, default-off, zero")
        print("        regression.")
    else:
        print("VERDICT [FAIL]: top-down pathway did not behave as specified -- see above.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="top-down modulation demo")
    ap.add_argument("--gate", type=float, default=1.0,
                    help="value to open each block's top-down gate to (default 1.0).")
    args = ap.parse_args()
    print_report(run_demo(args))


if __name__ == "__main__":
    main()
