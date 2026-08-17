"""
examples/demo_causal_decoding.py -- L3 causal steering INSIDE real decoding.

Why this demo exists (and how it differs from demo_causal_spatial_steering)
---------------------------------------------------------------------------
``demo_causal_spatial_steering.py`` proves the steering geometry on a *synthetic*
2-D belief trajectory that lives OUTSIDE the model. This demo proves the same
mechanism where it actually has to work: inside ``MTLNNModel.generate()``, acting
on the *live LNN recurrent cache* between decode steps. It is the production-path
validation of L3 -- the answer to "does the actuator do anything once it is wired
into real autoregressive decoding?"

The story
---------
An agent decodes a sequence. Its hidden "belief" is the LNN recurrent state that
the KV/recurrent cache threads from one token to the next. A *confident
hallucination* looks like an abrupt jump in that state while the output tokens
still look fluent -- exactly what :class:`CausalConsistencyChecker` detects and
:class:`CausalActivationSteerer` corrects by projecting the state back onto the
legal causal subspace (the directions the recent trajectory actually moved
along). We simulate that jump by injecting one off-subspace perturbation into the
recurrent cache mid-decode, then compare three runs:

  * **baseline**       -- no injection (the trajectory we want to stay near);
  * **injection only** -- the jump propagates through the recurrence, dragging the
    belief state away from baseline for the rest of decoding;
  * **injection + steering** -- :class:`CausalDecodeSteerer`, attached as the
    generic ``generate(step_callback=...)`` hook, detects the break and projects
    the corrupted state back, pulling the trajectory measurably back toward
    baseline.

It also checks the **safety** property that matters for production: on the clean
(un-injected) run the gate never trips, so steering is a no-op and decoding is
bit-for-bit identical to vanilla -- the correction can only help on a detected
break, never perturb a healthy trajectory.

Honest scope
------------
On an *untrained* model the output tokens are degenerate (greedy collapses), so
this demo measures the effect where it is real and observable: the *recurrent
state trajectory*. Whether steering reduces end-task hallucination in tokens is a
question for a *trained* ``serve.pt`` -- this harness is exactly the tool that
will measure it there (swap in a checkpoint; the metrics carry over).

Decoupling
----------
Uses only public APIs: ``MTLNNModel.generate(step_callback=...)`` (a generic hook;
the model imports nothing causal) and ``mt_lnn.causal_decoding.CausalDecodeSteerer``
(imports only ``causality`` + ``causal_steering``; never ``model.py``). Zero
trainable parameters are added. Deterministic given ``--seed``. CPU, ~1 second.

Run::

    python examples/demo_causal_decoding.py
    python examples/demo_causal_decoding.py --plot
    python examples/demo_causal_decoding.py --seed 3 --noise 6
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from mt_lnn.causal_decoding import CausalDecodeSteerer


# ---------------------------------------------------------------------------
# 1. A tiny decoder and a controlled "hallucination" injection
# ---------------------------------------------------------------------------

def build_model(*, seed: int = 0, d_model: int = 104, n_layers: int = 2,
                max_seq_len: int = 96) -> MTLNNModel:
    """A small MT-LNN decoder (untrained -- we probe the recurrent trajectory)."""
    torch.manual_seed(seed)
    cfg = MTLNNConfig(
        vocab_size=64, d_model=d_model, n_layers=n_layers, n_heads=13,
        n_kv_heads=1, d_head=8, max_seq_len=max_seq_len, gwtb_n_heads=1,
        dropout=0.0, attention_dropout=0.0,
    )
    return MTLNNModel(cfg).eval()


def decode_with_probe(
    model: MTLNNModel,
    prompt: torch.Tensor,
    *,
    inject_step: Optional[int],
    steer_on: bool,
    max_new_tokens: int,
    noise: float,
    seed: int,
) -> Tuple[Dict[int, torch.Tensor], Optional[CausalDecodeSteerer], torch.Tensor]:
    """Decode while probing layer-0 recurrent state each step.

    If ``inject_step`` is not None, one off-subspace perturbation is added to
    *every* layer's recurrent state at that step (a simulated belief jump). If
    ``steer_on``, a :class:`CausalDecodeSteerer` runs as the decode hook.

    Returns ``(states, steerer, tokens)`` where ``states[step]`` is the flattened
    layer-0 recurrent state observed at that step.
    """
    g = torch.Generator().manual_seed(seed)
    steerer = (CausalDecodeSteerer(strength=1.0, floor=0.5, layers="all", window=6)
               if steer_on else None)
    states: Dict[int, torch.Tensor] = {}

    def hook(cache, step: int) -> None:
        if inject_step is not None and step == inject_step:
            for i in range(len(cache.layers)):
                lc = cache.layers[i]
                h = lc[1]
                bump = noise * torch.randn(h.shape, generator=g) * h.detach().std()
                cache.layers[i] = (lc[0], h + bump, lc[2])
        if steerer is not None:
            steerer(cache, step)
        states[step] = cache.layers[0][1].reshape(-1).clone()

    tokens = model.generate(
        prompt, max_new_tokens=max_new_tokens, do_sample=False, step_callback=hook,
    )
    return states, steerer, tokens


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------

def drift_curve(states: Dict[int, torch.Tensor],
                baseline: Dict[int, torch.Tensor]) -> List[Tuple[int, float]]:
    """Per-step L2 distance of a run's recurrent state from the baseline run."""
    steps = sorted(states.keys() & baseline.keys())
    return [(s, float((states[s] - baseline[s]).norm())) for s in steps]


def mean_drift(states: Dict[int, torch.Tensor],
               baseline: Dict[int, torch.Tensor], *, frm: int) -> float:
    """Mean post-injection drift from baseline over steps >= ``frm``."""
    pairs = [(s, d) for s, d in drift_curve(states, baseline) if s >= frm]
    return sum(d for _, d in pairs) / max(1, len(pairs))


# ---------------------------------------------------------------------------
# 3. End-to-end
# ---------------------------------------------------------------------------

def run_demo(args) -> dict:
    model = build_model(seed=args.seed, d_model=args.d_model, n_layers=args.n_layers)
    g = torch.Generator().manual_seed(args.seed + 100)
    prompt = torch.randint(0, 64, (1, args.prompt_len), generator=g)

    inj_seed = args.seed + 7
    base, _, base_tok = decode_with_probe(
        model, prompt, inject_step=None, steer_on=False,
        max_new_tokens=args.steps, noise=args.noise, seed=inj_seed)
    inj, _, _ = decode_with_probe(
        model, prompt, inject_step=args.inject_step, steer_on=False,
        max_new_tokens=args.steps, noise=args.noise, seed=inj_seed)
    both, steerer, _ = decode_with_probe(
        model, prompt, inject_step=args.inject_step, steer_on=True,
        max_new_tokens=args.steps, noise=args.noise, seed=inj_seed)

    # Safety: steering a clean (un-injected) run must be a no-op.
    clean_states, clean_steer, clean_tok = decode_with_probe(
        model, prompt, inject_step=None, steer_on=True,
        max_new_tokens=args.steps, noise=args.noise, seed=inj_seed)

    drift_inj = mean_drift(inj, base, frm=args.inject_step)
    drift_both = mean_drift(both, base, frm=args.inject_step)
    reduction_pct = (100.0 * (drift_inj - drift_both) / drift_inj
                     if drift_inj > 1e-9 else 0.0)
    clean_unchanged = bool(torch.equal(clean_tok, base_tok))

    verdict = (
        (steerer.n_corrections >= 1)
        and (drift_both < drift_inj)
        and (reduction_pct > 5.0)
        and (clean_steer.n_corrections == 0)
        and clean_unchanged
    )

    summary = {
        "corrections": steerer.n_corrections,
        "total_correction_norm": steerer.total_correction_norm,
        "drift_injection": drift_inj,
        "drift_steered": drift_both,
        "drift_reduction_pct": reduction_pct,
        "clean_run_corrections": clean_steer.n_corrections,
        "clean_run_unchanged": clean_unchanged,
        "inject_curve": drift_curve(inj, base),
        "steer_curve": drift_curve(both, base),
        "inject_step": args.inject_step,
        "n_steps": args.steps,
        "verdict": bool(verdict),
    }
    print_report(summary)
    if args.plot:
        maybe_plot(summary, args.plot_path)
    return summary


def print_report(summary: dict) -> None:
    print("=" * 66)
    print("L3 causal steering INSIDE real decoding -- correct a belief jump")
    print("=" * 66)
    print(f"steps={summary['n_steps']}  inject_step={summary['inject_step']}  "
          f"(off-subspace perturbation into the live recurrent cache)")
    print("-" * 66)
    print("step | drift-from-baseline   inject-only [I]  vs  +steering [S]")
    inj = dict(summary["inject_curve"])
    steer = dict(summary["steer_curve"])
    for s in sorted(inj):
        di, ds = inj[s], steer.get(s, 0.0)
        mark = "  <-- break" if s == summary["inject_step"] else ""
        bar_i = "I" * int(round(min(di, 6.0) * 6))
        bar_s = "S" * int(round(min(ds, 6.0) * 6))
        print(f" {s:3d} | I {di:5.2f} {bar_i}")
        print(f"     | S {ds:5.2f} {bar_s}{mark}")
    print("-" * 66)
    print(f"corrections applied            : {summary['corrections']} "
          f"(total norm {summary['total_correction_norm']:.3f})")
    print(f"mean post-injection drift  I   : {summary['drift_injection']:.3f}")
    print(f"mean post-injection drift  S   : {summary['drift_steered']:.3f}")
    print(f"steering drift reduction       : {summary['drift_reduction_pct']:.1f}%")
    print(f"safety: clean-run corrections  : {summary['clean_run_corrections']} "
          f"(must be 0)  tokens unchanged: {summary['clean_run_unchanged']}")
    verdict = "[OK]" if summary["verdict"] else "[FAIL]"
    print("-" * 66)
    print(f"VERDICT {verdict}  steering detects the in-decode belief jump and pulls "
          f"the recurrent trajectory back toward baseline, while leaving a healthy "
          f"run untouched")
    print("=" * 66)


def maybe_plot(summary: dict, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping plot")
        return

    xi = [s for s, _ in summary["inject_curve"]]
    yi = [d for _, d in summary["inject_curve"]]
    xs = [s for s, _ in summary["steer_curve"]]
    ys = [d for _, d in summary["steer_curve"]]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(xi, yi, "o-", color="crimson", label="injection only")
    ax.plot(xs, ys, "o-", color="steelblue", label="injection + steering")
    ax.axvline(summary["inject_step"], ls="--", color="grey", label="belief jump")
    ax.set_xlabel("decode step")
    ax.set_ylabel("recurrent-state drift from baseline (L2)")
    ax.set_title("L3 steering in real decoding: corrects an in-decode belief jump")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="L3 causal steering inside real decoding")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=18)
    p.add_argument("--inject-step", type=int, default=8, dest="inject_step")
    p.add_argument("--noise", type=float, default=5.0)
    p.add_argument("--prompt-len", type=int, default=6, dest="prompt_len")
    p.add_argument("--d-model", type=int, default=104, dest="d_model")
    p.add_argument("--n-layers", type=int, default=2, dest="n_layers")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-path", type=str,
                   default="artifacts/causal_decoding_drift.png", dest="plot_path")
    args = p.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
