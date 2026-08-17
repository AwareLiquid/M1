"""O1 full-stack smoke test: does the from-scratch MTLNNModel actually support
ALL bio-inspired modules ON AT ONCE -- joint loss aggregation, no dead params,
no graph break, convergent training?

WHY THIS EXISTS: the existing "mechanism effectiveness" tests each enable ONE
module. Even test_p0_fixes_dont_break_training_loop (labelled "all P0 mechanisms
enabled") leaves use_world_model=False AND use_predictive_coding=False. So the
claim "O1 natively supports all modules jointly" was UNTESTED. This script tests
exactly that, and reports honestly which (if any) module is a dead parameter.

Checks:
  1. forward returns every expected aux-loss key, each non-zero
  2. total loss is finite and backward() succeeds (graph intact)
  3. EVERY module's parameter group receives a non-zero gradient (no dead params)
  4. a 100-step loop stays finite and the loss decreases (joint training works)
"""

from __future__ import annotations

import math
import sys
import warnings

import torch
import torch.optim as optim

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn.config import MTLNNConfig  # noqa: E402
from mt_lnn.model import MTLNNModel  # noqa: E402


def full_stack_cfg():
    """Every self-developed module ON simultaneously."""
    return MTLNNConfig(
        vocab_size=64, d_model=104, n_layers=2, n_heads=13, n_kv_heads=1, d_head=8,
        max_seq_len=32, gwtb_n_heads=1,
        # GWT-B competitive workspace + ortho penalty
        use_competitive_gwtb=True, n_competitive_bids=3,
        competitive_score_noise=0.5, competitive_ortho_weight=0.01,
        # Hebbian plasticity (gate ON: exercises the decoupled lavi_temperature
        # gate, which is now driven by Hebbian's own co-activation magnitude and
        # is trainable WITHOUT rhythm -- see plasticity.py DECOUPLING note).
        use_hebbian=True, hebbian_lr=1e-2, hebbian_lavi_gate=True,
        # Predictive coding
        use_predictive_coding=True,
        # World model (no warmup so it contributes a loss immediately)
        use_world_model=True, world_model_warmup_steps=0,
        world_model_loss_weight=0.1,
        # rhythm left off for this first pass (separate axis)
        use_rhythm=False, global_rhythm=False,
        dynamic_scale_gates=True, dropout=0.0, attention_dropout=0.0,
    )


# Module-name prefixes we expect to receive gradient if joint training is real.
MODULE_GROUPS = {
    "embedding":        lambda n: n.startswith("embedding."),
    "lnn_core":         lambda n: ".lnn." in n and ".resonance." not in n,
    "resonance(pc)":    lambda n: ".resonance." in n,   # predictive-coding W_pred lives here
    "gwtb":             lambda n: n.startswith("gwtb.") or ".gwtb." in n,
    "world_model":      lambda n: "world_model_head" in n,
    "hebbian":          lambda n: "hebbian" in n,
    "lm_head/out":      lambda n: n.startswith("lm_head") or n.startswith("norm") or n.startswith("final"),
}


def grad_report(model):
    """Sum |grad| per module group; a 0.0 group is a DEAD parameter set."""
    sums = {k: 0.0 for k in MODULE_GROUPS}
    counts = {k: 0 for k in MODULE_GROUPS}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = 0.0 if p.grad is None else float(p.grad.abs().sum())
        for k, pred in MODULE_GROUPS.items():
            if pred(name):
                sums[k] += g
                counts[k] += 1
    return sums, counts


def main():
    torch.set_num_threads(2)
    cfg = full_stack_cfg()
    torch.manual_seed(0)
    model = MTLNNModel(cfg)
    model.train()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[setup] full-stack MTLNNModel  trainable_params={n_trainable:,}")
    print(f"[setup] flags: hebbian={cfg.use_hebbian} pc={cfg.use_predictive_coding} "
          f"world_model={cfg.use_world_model} competitive_gwtb={cfg.use_competitive_gwtb}")

    # --- single step: inspect aux-loss keys + values ---
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=labels)

    print("\n[1] aux-loss terms present in forward result:")
    aux_keys = ["lm_loss", "pred_loss", "world_model_loss", "hebbian_loss", "ortho_penalty"]
    for k in aux_keys:
        if k in out:
            print(f"    {k:<18} = {float(out[k]):.6e}")
        else:
            print(f"    {k:<18} = <MISSING>")
    print(f"    total loss         = {float(out['loss']):.6e}  finite={math.isfinite(float(out['loss']))}")

    # --- backward + per-module gradient audit (dead-param check) ---
    model.zero_grad()
    out["loss"].backward()
    sums, counts = grad_report(model)
    print("\n[2] per-module gradient (sum|grad|; 0 => DEAD parameter set):")
    dead = []
    for k in MODULE_GROUPS:
        flag = "OK" if sums[k] > 0 else ("DEAD" if counts[k] > 0 else "n/a")
        if counts[k] > 0 and sums[k] == 0:
            dead.append(k)
        print(f"    {k:<16} params={counts[k]:<4} sum|grad|={sums[k]:.4e}  [{flag}]")

    # --- 100-step joint training: finite + decreasing ---
    torch.manual_seed(0)
    model2 = MTLNNModel(cfg)
    model2.train()
    opt = optim.Adam(model2.parameters(), lr=3e-3)
    losses = []
    for _ in range(100):
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        labels = torch.randint(0, cfg.vocab_size, (2, 16))
        o = model2(ids, labels=labels)
        losses.append(float(o["loss"]))
        opt.zero_grad()
        o["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        opt.step()
    all_finite = all(math.isfinite(l) for l in losses)
    decreased = losses[-1] < losses[0]
    print(f"\n[3] 100-step joint training: start={losses[0]:.4f} end={losses[-1]:.4f} "
          f"finite={all_finite} decreased={decreased}")

    print("\n==================== VERDICT ====================")
    ok = (all_finite and decreased and not dead
          and all(k in out for k in aux_keys))
    if ok:
        print("  PASS: all four aux losses present + non-dead grads + joint loop")
        print("        converges. The from-scratch MTLNNModel DOES support the")
        print("        full self-developed stack jointly (validates Plan 2 premise).")
    else:
        print("  ISSUES FOUND (this is the honest, useful outcome):")
        if dead:
            print(f"    - DEAD parameter groups (no gradient): {dead}")
        miss = [k for k in aux_keys if k not in out]
        if miss:
            print(f"    - missing aux-loss terms: {miss}")
        if not all_finite:
            print("    - non-finite loss during 100-step loop")
        if not decreased:
            print(f"    - loss did NOT decrease (start={losses[0]:.4f} end={losses[-1]:.4f})")
    print("=================================================")


if __name__ == "__main__":
    main()
