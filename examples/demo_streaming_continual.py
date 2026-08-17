"""
examples/demo_streaming_continual.py -- streaming continual learning on the REAL
MTLNNModel: does experience replay stop catastrophic forgetting?

The whole P2 streaming story in one runnable file. A tiny MT-LNN is trained on a
*sequence* of tasks, one after another, the way a deployed model meets a
non-stationary world: it never gets to shuffle all tasks together. We run the
exact same schedule twice --

  * NAIVE  -- train each task in turn, then move on (no rehearsal);
  * REPLAY -- the same, but every step also rehearses a minibatch drawn from a
    bounded :class:`mt_lnn.replay.ReservoirBuffer` of past examples.

-- fill the T x T accuracy matrix for each, and score them with the standard
catastrophic-forgetting metrics in :mod:`mt_lnn.continual_eval`. The verdict is
honest and quantitative: replay must *learn each task just as well* (comparable
learning accuracy) while *forgetting far less* (lower forgetting measure, higher
final accuracy). If replay did not clearly beat naive, the demo FAILS.

The task family (jointly learnable, yet naively forgotten)
----------------------------------------------------------
Vocabulary = ``V`` data tokens + ``T`` task-id tokens. A length-``L`` sequence for
task ``t`` is ``[TASK_t, x0, x1, ...]`` with ``x_{k+1} = (x_k + shift_t) mod V``
-- a task-specific cyclic shift. The leading task token *identifies* the rule, so
a single model CAN satisfy every task at once (a joint optimum exists); but a
model trained only on later tasks drifts off the earlier rules -- exactly the
forgetting replay is meant to cure. Accuracy on a task = next-token argmax
correctness on its (deterministic) data positions.

Honesty / scope
---------------
* The backbone, optimiser and loss are the real ones (``MTLNNModel(config)``,
  ``AdamW``, the model's internal next-token cross-entropy). The replay buffer and
  the metrics are the just-shipped zero-parameter operators -- replay adds **no
  model parameters**, only rehearsal data.
* It is a *small synthetic* benchmark chosen so the effect is visible in seconds
  on CPU; it demonstrates the machinery and the metric, not a SOTA CL number.

ASCII-only output. Run:  python examples/demo_streaming_continual.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.config import MTLNNConfig                       # noqa: E402
from mt_lnn.continual_eval import evaluate_continual         # noqa: E402
from mt_lnn.model import MTLNNModel                           # noqa: E402
from mt_lnn.replay import ReservoirBuffer                     # noqa: E402


# --------------------------------------------------------------------------- #
# task family                                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaskSpec:
    """A cyclic-shift task: prefix token ``task_tok``, rule ``+shift (mod V)``."""

    index: int
    shift: int
    task_tok: int


def build_tasks(n_tasks: int, v_data: int) -> List[TaskSpec]:
    """``n_tasks`` distinct non-zero shifts over a shared ``v_data`` token block.

    Task token ``t`` is the id ``v_data + t`` (the task-id tokens sit above the
    data tokens). Shifts are ``1, 2, ...`` so the rules genuinely differ.
    """
    return [TaskSpec(index=t, shift=t + 1, task_tok=v_data + t) for t in range(n_tasks)]


def vocab_size_for(n_tasks: int, v_data: int) -> int:
    return v_data + n_tasks


def make_batch(task: TaskSpec, n_seq: int, seq_len: int, v_data: int,
               gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``n_seq`` sequences ``[TASK_t, x0, x1, ...]`` of length ``seq_len``.

    Returns ``(input_ids, labels)`` both ``(n_seq, seq_len)`` long, with labels
    ALIGNED to inputs (the model shifts internally for next-token loss).
    """
    x0 = torch.randint(0, v_data, (n_seq,), generator=gen)
    cols = [torch.full((n_seq,), task.task_tok, dtype=torch.long)]
    cur = x0
    for _ in range(seq_len - 1):
        cols.append(cur.clone())
        cur = (cur + task.shift) % v_data
    ids = torch.stack(cols, dim=1)          # (n_seq, seq_len)
    return ids, ids.clone()


@torch.no_grad()
def accuracy_on_task(model: MTLNNModel, task: TaskSpec, *, n_seq: int,
                     seq_len: int, v_data: int, gen: torch.Generator) -> float:
    """Next-token argmax accuracy on the deterministic data positions of a task.

    Position 0 (the task token predicting the random seed ``x0``) is excluded --
    ``x0`` is unpredictable; every later position obeys the cyclic rule.
    """
    model.eval()
    ids, labels = make_batch(task, n_seq, seq_len, v_data, gen)
    logits = model(ids)["logits"]                       # (B, T, V)
    pred = logits[:, :-1, :].argmax(dim=-1)             # predicts tokens 1..T-1
    tgt = labels[:, 1:]
    # keep positions >= 1 (drop the task-token -> x0 prediction at index 0)
    pred, tgt = pred[:, 1:], tgt[:, 1:]
    return float((pred == tgt).float().mean())


# --------------------------------------------------------------------------- #
# model                                                                       #
# --------------------------------------------------------------------------- #
def build_model(vocab: int, seq_len: int, seed: int) -> MTLNNModel:
    torch.manual_seed(seed)
    cfg = MTLNNConfig(
        vocab_size=vocab,
        d_model=104,         # 8 * 13 (n_heads=13 is the fixed protofilament count;
        d_head=8,            #         8 keeps the protofilament einsum aligned)
        n_layers=2,
        n_heads=13,
        n_kv_heads=1,
        gwtb_n_heads=1,      # d_gw=13 is prime; one workspace head keeps GWTB valid
        max_seq_len=max(seq_len, 16),
        dropout=0.0,
        attention_dropout=0.0,
    )
    return MTLNNModel(cfg)


# --------------------------------------------------------------------------- #
# the two training schedules                                                  #
# --------------------------------------------------------------------------- #
def train_sequence(regime: str, tasks: List[TaskSpec], *, v_data: int,
                   seq_len: int, steps: int, batch: int, lr: float,
                   replay_cap: int, replay_batch: int, seed: int,
                   ) -> Tuple[List[List[float]], List[float]]:
    """Train across the task sequence under one regime; return (R, diag).

    ``regime`` is ``"naive"`` (no rehearsal) or ``"replay"`` (interleave a
    reservoir minibatch each step). ``R[i][j]`` is accuracy on task ``j`` after
    finishing training on task ``i``; ``diag[i] = R[i][i]``.
    """
    vocab = vocab_size_for(len(tasks), v_data)
    model = build_model(vocab, seq_len, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_gen = torch.Generator().manual_seed(seed + 1)
    eval_gen_seed = seed + 999
    buf = ReservoirBuffer(capacity=replay_cap, seed=seed + 7) if regime == "replay" else None

    R: List[List[float]] = []
    for i, task in enumerate(tasks):
        model.train()
        for _ in range(steps):
            ids, labels = make_batch(task, batch, seq_len, v_data, train_gen)
            if buf is not None:
                buf.add_batch(input_ids=ids, labels=labels)
            opt.zero_grad()
            loss = model(ids, labels=labels)["loss"]
            if buf is not None and len(buf) >= replay_batch:
                r = buf.sample(replay_batch)
                loss = loss + model(r["input_ids"], labels=r["labels"])["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # snapshot: accuracy on EVERY task after finishing task i
        row = [
            accuracy_on_task(model, tj, n_seq=64, seq_len=seq_len, v_data=v_data,
                             gen=torch.Generator().manual_seed(eval_gen_seed + j))
            for j, tj in enumerate(tasks)
        ]
        R.append(row)
    diag = [R[k][k] for k in range(len(tasks))]
    return R, diag


# --------------------------------------------------------------------------- #
# driver                                                                      #
# --------------------------------------------------------------------------- #
def run_demo(args) -> Dict:
    tasks = build_tasks(args.n_tasks, args.v_data)
    common = dict(
        tasks=tasks, v_data=args.v_data, seq_len=args.seq_len, steps=args.steps,
        batch=args.batch, lr=args.lr, replay_cap=args.replay_cap,
        replay_batch=args.replay_batch, seed=args.seed,
    )
    naive_R, _ = train_sequence("naive", **common)
    replay_R, _ = train_sequence("replay", **common)

    naive = evaluate_continual(naive_R)
    replay = evaluate_continual(replay_R)

    # the headline checks: replay learns tasks about as well (LA close), but
    # forgets far less (lower FM) and ends more accurate overall (higher ACC).
    learns_comparably = replay.learning_accuracy >= naive.learning_accuracy - 0.15
    forgets_less = replay.forgetting <= naive.forgetting - 0.10
    more_accurate = replay.average_accuracy >= naive.average_accuracy + 0.10
    naive_actually_forgot = naive.forgetting >= 0.15      # the problem is real
    ok = bool(learns_comparably and forgets_less and more_accurate
              and naive_actually_forgot)

    return {
        "tasks": tasks, "naive_R": naive_R, "replay_R": replay_R,
        "naive": naive, "replay": replay,
        "checks": {
            "naive_actually_forgot": naive_actually_forgot,
            "replay_learns_comparably": learns_comparably,
            "replay_forgets_less": forgets_less,
            "replay_more_accurate": more_accurate,
        },
        "ok": ok, "args": args,
    }


# --------------------------------------------------------------------------- #
# reporting (ASCII only)                                                      #
# --------------------------------------------------------------------------- #
def _fmt_matrix(R: List[List[float]]) -> List[str]:
    n = len(R)
    head = "        " + "".join(f"  task{j}" for j in range(n))
    rows = [head]
    for i, row in enumerate(R):
        cells = "".join(f"  {v:5.2f}" for v in row)
        rows.append(f"after t{i}{cells}")
    return rows


def print_report(s: Dict) -> None:
    a = s["args"]
    line = "=" * 68
    print(line)
    print("  STREAMING CONTINUAL LEARNING  --  replay vs catastrophic forgetting")
    print(line)
    print(f"  tasks={a.n_tasks}  v_data={a.v_data}  seq_len={a.seq_len}  "
          f"steps/task={a.steps}  batch={a.batch}")
    print(f"  replay: reservoir capacity={a.replay_cap}  rehearsal batch={a.replay_batch}")
    print("")
    print("  accuracy matrix  R[i,j] = acc on task j after training task i")
    print("  -- NAIVE (no rehearsal) --")
    for r in _fmt_matrix(s["naive_R"]):
        print("    " + r)
    print("  -- REPLAY (reservoir rehearsal) --")
    for r in _fmt_matrix(s["replay_R"]):
        print("    " + r)
    print("")

    def _scores(tag, rep):
        print(f"  {tag:7s}  ACC={rep.average_accuracy:5.2f}  "
              f"LA={rep.learning_accuracy:5.2f}  "
              f"BWT={rep.backward_transfer:+5.2f}  "
              f"FM(forgetting)={rep.forgetting:5.2f}")

    _scores("NAIVE", s["naive"])
    _scores("REPLAY", s["replay"])
    print("")
    for name, passed in s["checks"].items():
        print(f"    [{'OK' if passed else 'XX'}] {name}")
    print("")
    verdict = "OK" if s["ok"] else "FAIL"
    print(f"  VERDICT [{verdict}]  "
          + ("replay preserves old tasks while learning new ones"
             if s["ok"] else "replay did not clearly beat naive forgetting"))
    print(line)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Streaming continual learning demo.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-tasks", dest="n_tasks", type=int, default=3)
    p.add_argument("--v-data", dest="v_data", type=int, default=12)
    p.add_argument("--seq-len", dest="seq_len", type=int, default=9)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--replay-cap", dest="replay_cap", type=int, default=256)
    p.add_argument("--replay-batch", dest="replay_batch", type=int, default=32)
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    print_report(run_demo(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
