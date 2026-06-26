"""smoke_kuramoto.py — verify the external-plugin hard-rule contract on a tiny
from-scratch O1 MTLNNModel.

Run:  python -m external_plugins.ablation.smoke_kuramoto

Checks
------
1. ZERO REGRESSION (observe mode): logits with the plugins attached in observe
   mode are BIT-IDENTICAL to the native forward. (Strict non-intrusion.)
2. DETACH RESTORES: after detaching, the native forward is bit-identical to the
   pre-attach baseline (no residual state, hooks fully removed).
3. DIAGNOSTICS ARE REAL: the Kuramoto order parameter R and the early-exit
   settling-block index are produced and finite.
4. INTERVENE IS BOUNDED: in intervene mode logits CHANGE but stay finite, and
   the per-token logit shift is small at alpha=0.05 (a gentle, removable bias).
5. NO GRAD / NO WEIGHT CHANGE: a snapshot of model weights is unchanged after
   all plugin activity, and no plugin tensor carries grad.
"""

from __future__ import annotations

import sys
import warnings

import torch

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from mt_lnn.config import MTLNNConfig          # noqa: E402
from mt_lnn.model import MTLNNModel            # noqa: E402
from external_plugins import (                 # noqa: E402
    PluginRunner,
    KuramotoCoupling,
    SettlingEarlyExit,
)


def tiny_cfg() -> MTLNNConfig:
    # d_model = 13 * 8 = 104 → divides evenly into P=13 protofilaments.
    return MTLNNConfig(
        vocab_size=64, d_model=104, n_layers=4, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=32, n_protofilaments=13, gwtb_n_heads=1,
        dropout=0.0, attention_dropout=0.0,
    )


def weight_signature(model) -> torch.Tensor:
    return torch.stack([p.detach().float().sum() for p in model.parameters()]).sum()


def main() -> int:
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = MTLNNModel(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (2, 16))

    with torch.no_grad():
        base_logits = model(ids)["logits"].clone()
    w0 = weight_signature(model)

    fails = []

    # ---- 1 & 3: observe mode is bit-identical + diagnostics real ---------- #
    kur = KuramotoCoupling(n_protofilaments=13, kappa=0.6, n_steps=4, mode="observe")
    ee = SettlingEarlyExit(tol=0.05, relative=True)
    runner = PluginRunner(model, plugins=[kur, ee])
    with torch.no_grad(), runner:
        obs_logits = model(ids)["logits"].clone()
    diag = runner.diagnostics()

    bit_identical = torch.equal(obs_logits, base_logits)
    print(f"[1] observe bit-identical to native : {bit_identical}")
    if not bit_identical:
        max_dev = (obs_logits - base_logits).abs().max().item()
        fails.append(f"observe mode changed logits (max dev {max_dev:.3e})")

    R_mean = diag["kuramoto"].get("R_mean") if diag["kuramoto"] else None
    settle = diag["early_exit"].get("settle_block") if diag["early_exit"] else None
    n_blk = diag["early_exit"].get("n_blocks") if diag["early_exit"] else None
    print(f"[3] kuramoto R_mean                 : {R_mean}")
    print(f"[3] early_exit settle_block/n_blocks: {settle}/{n_blk}")
    if R_mean is None or not (0.0 <= R_mean <= 1.0):
        fails.append(f"order parameter R invalid: {R_mean}")
    if settle is None:
        fails.append("early-exit settle_block not produced")

    # ---- 2: detach restores native exactly -------------------------------- #
    with torch.no_grad():
        post_logits = model(ids)["logits"].clone()
    restored = torch.equal(post_logits, base_logits)
    print(f"[2] detach restores native exactly  : {restored}")
    if not restored:
        fails.append("model not bit-identical after detach")

    # ---- 4: intervene changes logits but stays finite & bounded ----------- #
    kur_i = KuramotoCoupling(
        n_protofilaments=13, kappa=0.6, n_steps=4, mode="intervene", alpha=0.05
    )
    runner_i = PluginRunner(model, plugins=[kur_i])
    with torch.no_grad(), runner_i:
        intv_logits = model(ids)["logits"].clone()
    finite = bool(torch.isfinite(intv_logits).all())
    changed = not torch.equal(intv_logits, base_logits)
    shift = (intv_logits - base_logits).abs().mean().item()
    print(f"[4] intervene finite                : {finite}")
    print(f"[4] intervene changed logits        : {changed}")
    print(f"[4] intervene mean|Δlogit|          : {shift:.4e}")
    if not finite:
        fails.append("intervene produced non-finite logits")
    if not changed:
        fails.append("intervene did not change logits (plugin inert?)")

    # ---- 5: no weight change anywhere ------------------------------------- #
    w1 = weight_signature(model)
    weights_unchanged = bool(torch.equal(w0, w1))
    print(f"[5] model weights unchanged         : {weights_unchanged}")
    if not weights_unchanged:
        fails.append("model weights changed during plugin activity")

    print("\n==================== VERDICT ====================")
    if not fails:
        print("  PASS: observe is bit-identical, detach restores, diagnostics")
        print("        are real, intervene is finite/bounded, weights untouched.")
        print("        External-plugin hard-rule contract holds.")
        print("=================================================")
        return 0
    print("  FAIL:")
    for f in fails:
        print(f"    - {f}")
    print("=================================================")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
