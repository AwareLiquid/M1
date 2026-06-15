"""
experiments/liquid_pc/run.py — end-to-end validation of the integrated
predictive-coding liquid core vs parameter-matched baselines.

This is the SECOND axis (effectiveness validation), the one the roadmap legend
says is empty. It is deliberately small, seeded and CPU-runnable so anyone can
reproduce the numbers. It answers one honest question:

    At a matched parameter budget and the IDENTICAL next-step-MSE objective, does
    embedding predictive coding into the liquid core's continuous-time dynamics
    predict a multi-timescale signal better than a GRU / LSTM / Transformer?

It reports:
  * #params per model (transparency on the "matched budget" claim);
  * 1-step test MSE;
  * H-step autoregressive rollout MSE (the harder, longer-horizon test);
  * a persistence reference (predict x_{t+1}=x_t) for interpretable scale;
  * a continual-learning probe: train on regime A, then on regime B, and measure
    how much the A-error degrades (less degradation = less forgetting).

Whatever the result — advantage, parity, or no advantage — it is printed and
saved verbatim. Validation means finding out, not confirming.

Run:  python -m experiments.liquid_pc.run
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.liquid_pc.data import make_signal, train_test_split, two_regimes
from experiments.liquid_pc.model import build_models, count_params


# --------------------------------------------------------------------------- #
# training / evaluation helpers                                               #
# --------------------------------------------------------------------------- #
def train(model: nn.Module, x: torch.Tensor, *, epochs: int, batch: int,
          lr: float, seed: int = 0, ss_max: float = 0.0,
          ewc_lambda: float = 0.0, replay_x: torch.Tensor | None = None,
          replay_batch: int = 16, replay_weight: float = 1.0) -> None:
    """Train a next-step predictor with MSE on (x[:, :-1] -> x[:, 1:]).

    ``ss_max`` enables *scheduled sampling* (Bengio et al. 2015): with a
    probability that ramps linearly from 0 to ``ss_max`` over training, each
    input position is replaced by the model's OWN (teacher-forced, detached)
    prediction of that position rather than the ground-truth value. This injects
    the model's errors into its inputs during training so it learns to recover
    from them — directly targeting autoregressive *rollout* compounding error.
    The cheap 2-pass approximation is used (one teacher-forced pass to source the
    predictions, one mixed-input pass for the gradient), applied IDENTICALLY to
    every model so the comparison stays fair.

    ``replay_x`` enables *experience replay / rehearsal* (the systems-
    consolidation / hippocampal-replay analogue): a small buffer of an OLD task's
    sequences is interleaved each step, adding ``replay_weight`` * MSE on a random
    replay minibatch to the loss. Unlike EWC (which protects weights in parameter
    space), replay revisits the old data directly — the strongest-known continual-
    learning lever, but it stores raw old data. Applied identically to every model
    (it is a training protocol, not an architectural advantage), so PCLiquidCore
    and the RNN baseline are rehearsed the same way for a fair comparison.
    """
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x.shape[0]
    n_replay = replay_x.shape[0] if replay_x is not None else 0
    model.train()
    for ep in range(epochs):
        p = ss_max * (ep / max(1, epochs - 1)) if ss_max > 0.0 else 0.0
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = x[idx]
            inp, tgt = xb[:, :-1], xb[:, 1:]
            opt.zero_grad()
            if p > 0.0:
                # pass 1: teacher-forced predictions to mix in (no grad).
                with torch.no_grad():
                    yhat = model(inp)                    # (B, L, d): yhat[:,t]=pred x[t+1]
                B, L, d = inp.shape
                mask = (torch.rand(B, L - 1, 1, generator=g) < p)
                repl = yhat[:, :-1]                       # predicts inp positions 1..L-1
                inp = inp.clone()
                inp[:, 1:] = torch.where(mask, repl, inp[:, 1:])
            pred = model(inp)
            loss = nn.functional.mse_loss(pred, tgt)
            if ewc_lambda > 0.0 and hasattr(model, "ewc_loss"):
                loss = loss + model.ewc_loss(ewc_lambda)
            if n_replay > 0 and replay_weight > 0.0:
                ridx = torch.randint(0, n_replay, (min(replay_batch, n_replay),),
                                     generator=g)
                rb = replay_x[ridx]
                rpred = model(rb[:, :-1])
                loss = loss + replay_weight * nn.functional.mse_loss(
                    rpred, rb[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()


@torch.no_grad()
def one_step_mse(model: nn.Module, x: torch.Tensor) -> float:
    model.eval()
    pred = model(x[:, :-1])
    return nn.functional.mse_loss(pred, x[:, 1:]).item()


@torch.no_grad()
def rollout_mse(model: nn.Module, x: torch.Tensor, *, prime: int) -> float:
    """H-step autoregressive rollout error, batched across the whole test set.

    Feed the first ``prime`` real steps, then for the remaining horizon feed the
    model's OWN predictions back in and score against ground truth. Same generic
    procedure for every model, so the comparison is fair.
    """
    model.eval()
    B, T, d = x.shape
    seq = x[:, :prime, :].clone()                        # primed context
    preds = []
    for t in range(prime, T):
        nxt = model(seq)[:, -1:, :]                      # predict next from history
        preds.append(nxt)
        seq = torch.cat([seq, nxt], dim=1)               # feed prediction back
    roll = torch.cat(preds, dim=1)                       # (B, T-prime, d)
    return nn.functional.mse_loss(roll, x[:, prime:, :]).item()


@torch.no_grad()
def generate_synthetic(model: nn.Module, *, n_syn: int, seq_len: int, d_in: int,
                       seed: int = 0, warmup: int = 16,
                       noise_std: float = 1.0) -> torch.Tensor:
    """DREAM synthetic sequences by free-running the model from a noise seed.

    The model just trained on a task is rolled out autoregressively: seeded with
    ``warmup`` steps of noise, it then predicts ``seq_len`` further steps, each fed
    back as the next input. The warmup is discarded and the model-generated
    continuation is returned. This is *generative replay / pseudo-rehearsal*
    (Robins 1995; Shin et al. 2017): a snapshot of the model dreams up
    old-task-like data, so a later task can rehearse WITHOUT storing any raw old
    data. Because the sequence is built from the model's own predictions,
    next-step MSE on it distils the snapshot (``x[t+1]`` IS its prediction at t).

    It is a GENERIC tool — any next-step predictor (PCLiquidCore OR a GRU) can be
    rolled out this way — so PCLiquidCore and the RNN baseline are dreamed
    identically and the comparison of dream QUALITY stays fair.
    """
    g = torch.Generator().manual_seed(seed)
    seq = noise_std * torch.randn(n_syn, warmup, d_in, generator=g)
    model.eval()
    for _ in range(seq_len):
        nxt = model(seq)[:, -1:, :]
        seq = torch.cat([seq, nxt], dim=1)
    return seq[:, warmup:, :].contiguous()               # (n_syn, seq_len, d_in)


def persistence_mse(x: torch.Tensor) -> float:
    """Naive reference: predict x_{t+1} = x_t."""
    return nn.functional.mse_loss(x[:, :-1], x[:, 1:]).item()


# --------------------------------------------------------------------------- #
# experiments                                                                 #
# --------------------------------------------------------------------------- #
def main(
    *,
    n_seq: int = 320,
    seq_len: int = 96,
    d_in: int = 1,
    epochs: int = 40,
    batch: int = 64,
    lr: float = 3e-3,
    prime: int = 48,
    seed: int = 0,
    ss_max: float = 0.0,
    tag: str = "",
    pc_kwargs: Dict | None = None,
) -> Dict:
    torch.manual_seed(seed)
    t0 = time.time()

    # ---- main task: one multi-timescale regime --------------------------- #
    x = make_signal(n_seq, seq_len, d_in, seed=seed)
    x_tr, x_te = train_test_split(x, 0.8)
    persist = persistence_mse(x_te)

    rows = {}
    for name, model in build_models(d_in, pc_kwargs=pc_kwargs).items():
        torch.manual_seed(seed)                          # same init RNG draw order
        # rebuild so every model starts from the same seed state
        model = build_models(d_in, pc_kwargs=pc_kwargs)[name]
        n_params = count_params(model)
        train(model, x_tr, epochs=epochs, batch=batch, lr=lr, seed=seed, ss_max=ss_max)
        rows[name] = {
            "params": n_params,
            "one_step_mse": one_step_mse(model, x_te),
            "rollout_mse": rollout_mse(model, x_te, prime=prime),
        }
        print(f"[main] {name:14s} params={n_params:6d} "
              f"1step={rows[name]['one_step_mse']:.5f} "
              f"rollout={rows[name]['rollout_mse']:.5f}")

    # ---- continual-learning probe: train A, then B, re-measure A --------- #
    a, b = two_regimes(n_seq, seq_len, d_in, seed=seed)
    a_tr, a_te = train_test_split(a, 0.8)
    b_tr, _ = train_test_split(b, 0.8)
    forget = {}
    for name in build_models(d_in, pc_kwargs=pc_kwargs):
        torch.manual_seed(seed)
        model = build_models(d_in, pc_kwargs=pc_kwargs)[name]
        train(model, a_tr, epochs=epochs, batch=batch, lr=lr, seed=seed, ss_max=ss_max)
        a_err_before = one_step_mse(model, a_te)
        train(model, b_tr, epochs=epochs, batch=batch, lr=lr, seed=seed + 1, ss_max=ss_max)
        a_err_after = one_step_mse(model, a_te)
        forget[name] = {
            "A_before": a_err_before,
            "A_after_B": a_err_after,
            "forgetting": a_err_after - a_err_before,
        }
        print(f"[forget] {name:14s} A_before={a_err_before:.5f} "
              f"A_after_B={a_err_after:.5f} "
              f"forgetting={forget[name]['forgetting']:+.5f}")

    report = {
        "config": dict(n_seq=n_seq, seq_len=seq_len, d_in=d_in, epochs=epochs,
                       batch=batch, lr=lr, prime=prime, seed=seed, ss_max=ss_max,
                       pc_kwargs=pc_kwargs or {}),
        "persistence_mse": persist,
        "main": rows,
        "forgetting": forget,
        "seconds": round(time.time() - t0, 1),
    }
    _save(report, tag=tag)
    _print_summary(report)
    return report


def _save(report: Dict, *, tag: str = "") -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, f"report{tag}.json"), "w") as f:
        json.dump(report, f, indent=2)
    lines = ["# PC-Liquid-Core validation report", ""]
    lines.append(f"Config: `{report['config']}`  |  runtime {report['seconds']}s")
    lines.append("")
    lines.append(f"Persistence reference (predict x_t+1=x_t) 1-step MSE: "
                 f"**{report['persistence_mse']:.5f}**")
    lines.append("")
    lines.append("## Main task (multi-timescale signal)")
    lines.append("")
    lines.append("| model | #params | 1-step MSE | rollout MSE |")
    lines.append("|---|---:|---:|---:|")
    for k, v in report["main"].items():
        lines.append(f"| {k} | {v['params']} | {v['one_step_mse']:.5f} | "
                     f"{v['rollout_mse']:.5f} |")
    lines.append("")
    lines.append("## Continual-learning probe (train A -> train B, re-measure A)")
    lines.append("")
    lines.append("| model | A before | A after B | forgetting (lower=better) |")
    lines.append("|---|---:|---:|---:|")
    for k, v in report["forgetting"].items():
        lines.append(f"| {k} | {v['A_before']:.5f} | {v['A_after_B']:.5f} | "
                     f"{v['forgetting']:+.5f} |")
    lines.append("")
    with open(os.path.join(here, f"report{tag}.md"), "w") as f:
        f.write("\n".join(lines))


def _print_summary(report: Dict) -> None:
    main = report["main"]
    best_1 = min(main, key=lambda k: main[k]["one_step_mse"])
    best_r = min(main, key=lambda k: main[k]["rollout_mse"])
    print("\n=== SUMMARY ===")
    print(f"best 1-step:  {best_1}  ({main[best_1]['one_step_mse']:.5f})")
    print(f"best rollout: {best_r}  ({main[best_r]['rollout_mse']:.5f})")
    pc = main.get("PCLiquidCore", {})
    if pc:
        print(f"PCLiquidCore: 1-step={pc['one_step_mse']:.5f} "
              f"rollout={pc['rollout_mse']:.5f}  (wins 1-step: {best_1=='PCLiquidCore'}, "
              f"wins rollout: {best_r=='PCLiquidCore'})")


if __name__ == "__main__":
    main()
