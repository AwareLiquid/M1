"""
examples/demo_imagination.py -- L4 latent "mental simulation" rollout.

What this demo shows
--------------------
The world model (``PredictiveStateHead``) learns a ONE-step latent transition.
:class:`mt_lnn.imagination.LatentImagination` iterates that *same* learned map
forward in latent space to "imagine" how the next several steps unfold -- the
in-your-head simulation the project's roadmap is after. The demo proves two
things, in ASCII, on CPU, in ~a few seconds:

  PART 1 -- REAL-MODEL INTEGRATION (zero coupling)
    Build a live ``MTLNNModel`` with the world model enabled, run a forward,
    tap its ``final_norm`` hidden state, and roll an imagined trajectory off it.
    The imagination driver attaches to the real model with no change to the
    backbone (it never imports ``model.py``) and adds zero parameters. We print
    the per-step confidence + novelty so you can see confidence decay with the
    horizon.

  PART 2 -- IT ACTUALLY COMPOSES (the verdict)
    Train the world-model head on a known *rotating* latent dynamic, then
    compare the multi-step imagined trajectory against:
      * the TRUE k-step future (ground truth), and
      * a STATIC "assume nothing changes" baseline.
    Because the world genuinely moves, the static baseline degrades with the
    horizon while the composed rollout keeps tracking the truth. The verdict is
    ``imagined_cos > static_cos`` by a clear margin -- the rollout is a genuine
    composition of the learned one-step map, not a wrapper.

Honest scope
------------
On an *untrained* backbone (Part 1) the imagined trajectory off the model's own
hidden state is not meaningful content -- Part 1 only proves the integration
surface + the confidence machinery. The quantitative "it works" claim lives in
Part 2, where the head has actually learned a dynamic. Token-level payoff on a
real task is for a trained ``serve.pt``; this harness carries over to it.

Decoupling
----------
Uses only public APIs: ``MTLNNModel`` (+ a forward hook on ``final_norm`` that
the demo owns) and ``mt_lnn.imagination.LatentImagination`` (imports only torch;
never ``model.py``). Zero trainable parameters added by the rollout.
Deterministic given ``--seed``.

Run::

    python examples/demo_imagination.py
    python examples/demo_imagination.py --horizon 6 --seed 3
"""

from __future__ import annotations

import argparse
import math
from typing import Dict

import torch
import torch.nn.functional as F

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from mt_lnn.world_model import PredictiveStateHead
from mt_lnn.imagination import LatentImagination


# ---------------------------------------------------------------------------
# Part 1 -- attach imagination to a live MTLNNModel
# ---------------------------------------------------------------------------

def build_model(*, seed: int = 0, d_model: int = 104, n_layers: int = 2,
                max_seq_len: int = 96) -> MTLNNModel:
    """A small MT-LNN decoder with the world model (Phase C) enabled."""
    torch.manual_seed(seed)
    cfg = MTLNNConfig(
        vocab_size=64, d_model=d_model, n_layers=n_layers, n_heads=13,
        n_kv_heads=1, d_head=8, max_seq_len=max_seq_len, gwtb_n_heads=1,
        dropout=0.0, attention_dropout=0.0,
        use_world_model=True,
    )
    return MTLNNModel(cfg).eval()


def hidden_from_forward(model: MTLNNModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Run a forward and capture the post-``final_norm`` hidden state.

    The demo owns this hook; the model needs no modification. Returns
    (B, T, d_model).
    """
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        captured["h"] = out.detach()

    handle = model.final_norm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids, use_cache=False)
    finally:
        handle.remove()
    return captured["h"]


def part1_real_model(args) -> dict:
    model = build_model(seed=args.seed, d_model=args.d_model, n_layers=args.n_layers)
    head = model.world_model_head
    assert head is not None, "world model must be enabled"

    g = torch.Generator().manual_seed(args.seed + 1)
    prompt = torch.randint(0, 64, (1, args.prompt_len), generator=g)
    hidden = hidden_from_forward(model, prompt)                 # (1, T, d)

    n_params_before = sum(p.numel() for p in model.parameters())
    imag = LatentImagination(head, horizon=args.horizon,
                             trust_decay=args.trust_decay,
                             novelty_penalty=args.novelty_penalty)
    traj = imag.imagine(hidden)
    n_params_after = sum(p.numel() for p in model.parameters())

    return {
        "horizon": traj.horizon,
        "confidence": traj.confidence[0].tolist(),
        "novelty": traj.novelty[0].tolist(),
        "drift": float(traj.confidence_weighted_drift()[0]),
        "added_params": n_params_after - n_params_before,
        "imag_params": imag.n_parameters,
        "conf_decreases": bool(traj.confidence[0, -1] < traj.confidence[0, 0]),
    }


# ---------------------------------------------------------------------------
# Part 2 -- does the rollout actually compose? (trained head, rotating world)
# ---------------------------------------------------------------------------

def _rotation(d_model: int, theta: float) -> torch.Tensor:
    A = torch.zeros(d_model, d_model)
    c, s = math.cos(theta), math.sin(theta)
    for i in range(0, d_model - 1, 2):
        A[i, i] = c; A[i, i + 1] = -s
        A[i + 1, i] = s; A[i + 1, i + 1] = c
    if d_model % 2:
        A[-1, -1] = 1.0
    return A


def _true_future(x0: torch.Tensor, A: torch.Tensor, H: int) -> torch.Tensor:
    xs, x = [], x0
    for _ in range(H):
        x = x @ A.T
        xs.append(x)
    return torch.stack(xs, dim=1)


def _train_head(head, A, d_model, *, steps, seed):
    opt = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=1e-2)
    g = torch.Generator().manual_seed(seed)
    head.train()
    for _ in range(steps):
        x0 = torch.randn(64, d_model, generator=g)
        seqx, x = [x0], x0
        for _ in range(7):
            x = x @ A.T
            seqx.append(x)
        _, loss = head(torch.stack(seqx, dim=1), compute_loss=True)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()


def part2_composition(args) -> dict:
    torch.manual_seed(args.seed)
    d_model, proj_dim, H = 16, 8, args.horizon
    head = PredictiveStateHead(d_model, proj_ratio=proj_dim / d_model,
                               hidden_ratio=1.0, warmup_steps=50)
    A = _rotation(d_model, theta=args.theta)
    _train_head(head, A, d_model, steps=args.train_steps, seed=args.seed)

    imag = LatentImagination(head, horizon=H, trust_decay=1.0, novelty_penalty=0.0)
    g = torch.Generator().manual_seed(args.seed + 7)
    x0 = torch.randn(16, d_model, generator=g)
    traj = imag.imagine(x0)

    with torch.no_grad():
        z_true = F.normalize(head.target_proj(_true_future(x0, A, H)), dim=-1)
        z0 = F.normalize(head.online_proj(x0), dim=-1)
    cos_imag = (traj.latents * z_true).sum(dim=-1).mean(dim=0)        # (H,)
    cos_static = (z0.unsqueeze(1) * z_true).sum(dim=-1).mean(dim=0)   # (H,)

    return {
        "horizon": H,
        "cos_imag": cos_imag.tolist(),
        "cos_static": cos_static.tolist(),
        "imag_mean": float(cos_imag.mean()),
        "static_mean": float(cos_static.mean()),
        "far_imag": float(cos_imag[-1]),
        "far_static": float(cos_static[-1]),
    }


def run_demo(args) -> dict:
    p1 = part1_real_model(args)
    p2 = part2_composition(args)
    verdict = (
        p1["added_params"] == 0
        and p1["conf_decreases"]
        and p2["far_imag"] > p2["far_static"] + 0.05
        and p2["imag_mean"] > p2["static_mean"]
    )
    return {"part1": p1, "part2": p2, "verdict": bool(verdict)}


def print_report(summary: dict) -> None:
    p1, p2 = summary["part1"], summary["part2"]
    line = "=" * 64
    print(line)
    print("L4 LATENT IMAGINATION ROLLOUT -- mental simulation demo")
    print(line)

    print("\n[Part 1] Integration onto a live MTLNNModel (world model ON)")
    print(f"  horizon                 : {p1['horizon']}")
    print(f"  params added by rollout : {p1['added_params']} (driver: {p1['imag_params']})")
    print(f"  confidence-weighted drift: {p1['drift']:.4f}")
    print("  step :  confidence   novelty")
    for t, (c, n) in enumerate(zip(p1["confidence"], p1["novelty"]), start=1):
        print(f"   {t:>2}  :   {c:7.4f}   {n:7.4f}")
    print(f"  confidence decays with horizon : {p1['conf_decreases']}")

    print("\n[Part 2] Does it compose? (head trained on a rotating world)")
    print("  step :  imagined_cos   static_cos   (vs TRUE k-step future)")
    for t, (ci, cs) in enumerate(zip(p2["cos_imag"], p2["cos_static"]), start=1):
        mark = "  <-- imagination ahead" if ci > cs else ""
        print(f"   {t:>2}  :   {ci:9.4f}    {cs:9.4f}{mark}")
    print(f"  mean cos  imagined={p2['imag_mean']:.4f}  static={p2['static_mean']:.4f}")
    print(f"  far horizon imagined={p2['far_imag']:.4f}  static={p2['far_static']:.4f}")

    print("\n" + line)
    status = "[OK]" if summary["verdict"] else "[FAIL]"
    print(f"VERDICT {status}: rollout adds 0 params, confidence decays, and the "
          f"composed\n        imagination tracks the true future better than "
          f"'nothing changes'.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="L4 latent imagination rollout demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--prompt-len", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=104)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--trust-decay", type=float, default=0.85)
    ap.add_argument("--novelty-penalty", type=float, default=0.5)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--train-steps", type=int, default=800)
    args = ap.parse_args()

    summary = run_demo(args)
    print_report(summary)


if __name__ == "__main__":
    main()
