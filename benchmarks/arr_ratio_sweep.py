"""O-series hybrid-ratio Pareto sweep — quality vs KV memory, 0:1 → 3:1.

The O-series (pure recurrent ARR, every attention layer replaced) is an
all-or-nothing bet: it buys O(1) carried state at 2.15x the teacher's PPL
(25.4 vs 11.8). The frontier has since converged on a compromise — keep a
slice of the layers as full attention:

  * Qwen3-Next   75% Gated DeltaNet + 25% gated attention (3:1)
  * Kimi Linear  3:1 KDA (linear) : full MLA, up to 75% KV-cache reduction
  * GLM-5.3-Flash 3 linear + 1 sparse-attention layer, periodically

This script sweeps attention ratio {0, 1:8, 1:4, 1:2} at a FIXED teacher and
a FIXED distillation budget and reports, per ratio:

  1. val PPL (and the convergence rate toward the teacher's PPL), and
  2. the carried-memory account in TWO columns that never merge:
       constant recurrent state  +  O(T) KV of the surviving attention
     layers, at fp16 and at 2-bit (bit-packed, KIVI-style scale metadata —
     the same `kv_bytes` formula benchmarks/kv_frontier.py uses).

HONEST BOUNDARY, ENFORCED IN THE OUTPUT: back-filling k of the layers with
attention makes the model NO LONGER O(1). Every table this script writes
carries the state column and the KV column separately, and the ratio-0 row
(the pure-ARR control) is the only one whose KV column is zero.

Reading rules were fixed BEFORE any number existed (no post-hoc gates):
  * primary "improvement" = PPL reduction vs the ratio-0 control,
    `(ppl_0 - ppl_k) / ppl_0`, on seed means;
  * the gate on ratio 1:4 is >= 15%. Missing it is recorded as the negative
    result "back-fill does not pay", not re-negotiated;
  * a secondary column reports gap closure `(ppl_0 - ppl_k)/(ppl_0 - ppl_t)`
    so a reader who prefers that definition can check it without a rerun;
  * 2 seeds is EXPLORATORY. Nothing here goes into RESULTS.md until the
    winning ratio has >= 3 seeds (see `promotion` in the JSON).

Modes:
    python benchmarks/arr_ratio_sweep.py --dry_run   # print the plan only
    python benchmarks/arr_ratio_sweep.py --smoke     # tiny, CPU, minutes
    python benchmarks/arr_ratio_sweep.py             # remote, GPU, hours

Full mode shells out to benchmarks/distill_arr.py once per (ratio, seed) so
the distillation code path is the SAME one that produced the existing
O-series numbers; --keep_ratio is the only knob that moves. Artifacts land in
benchmarks/results/: arr_ratio_<ratio>.json (per ratio, all seeds),
arr_ratio_pareto.json and arr_ratio_pareto.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from benchmarks.kv_frontier import kv_bytes                      # one formula
from mt_lnn.arr import (convert_to_arr, iter_mixer_parameters,
                        parse_ratio, plan_hybrid_layers,
                        probe_return_convention, recurrent_state_elems)

RATIOS = ("0", "1:8", "1:4", "1:2")
SEEDS = (0, 1)
CONTEXTS = (512, 2048, 8192, 32768, 131072, 1048576)
REF_CONTEXT = 32768
KV_BITS = (16, 2)                  # the two mandated memory columns
STATE_ELEM_BYTES = 2               # recurrent state carried in fp16
GATE_RATIO = "1:4"                 # pre-registered decision point
GATE_MIN_GAIN = 0.15               # >=15% PPL reduction vs the ratio-0 control
PROMOTION_SEEDS = 3                # seeds required before RESULTS.md
TEACHER_PPL_REF = 11.8             # WikiText-2 teacher PPL, O-series line
_BOUNDARY_NOTE = (
    "回填注意力后本模型**不再是 O(1)**：携带内存 = 恒定循环状态（不随 T 增长）"
    " + 保留注意力层的 O(T) KV。两列分列，永不合并；只有 ratio 0（纯 ARR）的 "
    "KV 列为 0。KV 列按 benchmarks/kv_frontier.py 的同一公式计费（bit-packed "
    "+ KIVI 非对称量化 scale 元数据），不享受双重标准。")


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.dry_run:
        _print_commands(args)
        return 0
    runs = _smoke_runs(args) if args.smoke else _distill_runs(args)
    ledger = _ledger(runs, args)
    _write(ledger, args.out_dir)
    _report(ledger)
    return 0


def _distill_runs(args):
    """Full sweep: one distill_arr.py process per (ratio, seed), same budget.

    Subprocess isolation is deliberate — the conversion is in-place and
    irreversible, so every (ratio, seed) needs a fresh student anyway, and
    going through the real entry point guarantees the distillation recipe
    (two-stage MOHAWK-lite, tau, alpha, grad-accum) cannot drift between the
    control row and the hybrid rows.
    """
    runs = []
    for ratio in args.ratios:
        for seed in args.seeds:
            t0 = time.time()
            run = _distill_run(args, ratio, seed)
            run["seconds"] = round(time.time() - t0, 1)
            runs.append(run)
            print(f"[sweep] ratio {ratio} seed {seed} -> val PPL "
                  f"{run['val_ppl']:.3f} ({run['seconds']}s)", flush=True)
    return runs


def _distill_run(args, ratio, seed):
    """Run one (ratio, seed) and fold distill_arr.py's JSON into a ledger row."""
    out_dir = os.path.join(args.work_dir, f"ratio_{_tag(ratio)}_seed{seed}")
    cmd = [sys.executable, os.path.join(_ROOT, "benchmarks", "distill_arr.py"),
           "--model", args.model, "--keep_ratio", ratio,
           "--steps_a", str(args.steps_a), "--steps_b", str(args.steps_b),
           "--seq_len", str(args.seq_len), "--batch", str(args.batch),
           "--grad_accum", str(args.grad_accum), "--seed", str(seed),
           "--d_proto", str(args.d_proto), "--proj_rank", str(args.proj_rank),
           "--fw_dim", str(args.fw_dim), "--out_dir", out_dir]
    if args.no_cache:
        cmd.append("--no_cache")
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=_ROOT)
    with open(os.path.join(out_dir, "arr_result.json")) as f:
        res = json.load(f)
    return _record(ratio, seed, res, args)


def _smoke_runs(args):
    """CPU smoke: a randomly initialised teacher, one KD step per ratio.

    The teacher carries no knowledge, so smoke PPL is NOT a quality signal —
    this exists to prove the pipeline runs end-to-end at every ratio
    (subset conversion, hybrid forward, KD step, PPL eval, memory ledger)
    in minutes, on a laptop, before a GPU is booked.
    """
    runs = []
    for ratio in args.ratios:
        run = _smoke_run(args, ratio)
        runs.append(run)
        print(f"[smoke] ratio {ratio}: keep {run['attention_layers']} | "
              f"val PPL {run['val_ppl']:.2f} | state "
              f"{_mb(run['memory']['recurrent_state_bytes'])} MB", flush=True)
    return runs


def _smoke_run(args, ratio):
    torch.manual_seed(args.seed)
    teacher = _tiny_teacher(args)
    student = copy.deepcopy(teacher)
    keep, to_convert = plan_hybrid_layers(args.n_layers, parse_ratio(ratio))
    converted = convert_to_arr(
        student, layer_indices=to_convert, d_proto=args.d_proto,
        proj_rank=args.proj_rank, fast_weight_dim=args.fw_dim,
        n_protofilaments=args.n_protofilaments,
        n_time_scales=args.n_time_scales,
    )
    if args.no_cache:
        student.config.use_cache = False
    probe_return_convention(student)
    gen = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, args.vocab, (args.batch, args.seq_len), generator=gen)
    _smoke_train(teacher, student, ids, args, ratio)
    return _record(ratio, args.seed, {
        "student_ppl_final": _ppl(student, args, gen),
        "teacher_ppl": _ppl(teacher, args, gen),
        "converted_layers": converted, "mixer_params": 0, "model": "smoke",
        "attention_layers": keep, "n_layers": args.n_layers,
        "d_head": args.d_model // args.n_heads, "n_kv_heads": args.n_kv_heads,
    }, args)


def _smoke_train(teacher, student, ids, args, ratio):
    """A handful of KD+CE steps on the mixers — gradient plumbing, not quality."""
    opt = torch.optim.AdamW(iter_mixer_parameters(student), lr=args.lr)
    loss = None
    try:
        for _ in range(args.smoke_steps):
            loss = _kd_loss(teacher, student, ids, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    except (IndexError, TypeError, AttributeError) as exc:
        raise SystemExit(
            f"hybrid forward failed at ratio {ratio}: {exc!r}\n"
            f"retry with --no_cache: if that fixes it, the pinned transformers "
            f"version cannot consume a recurrent layer's None cache slot. "
            f"Record it in the Phase B log either way.")
    return None if loss is None else loss.item()


def _ledger(runs, args):
    """Group runs by ratio, attach the memory account, apply the fixed gates."""
    by_ratio = {}
    for run in runs:
        by_ratio.setdefault(run["ratio"], []).append(run)
    control = _mean_ppl(by_ratio.get(RATIOS[0], []))
    rows = [_ratio_row(ratio, grouped, control) for ratio, grouped
            in sorted(by_ratio.items(), key=lambda kv: parse_ratio(kv[0]))]
    return {"meta": _meta(args, control), "rows": rows,
            "promotion": _promotion(rows), "pareto_md": _markdown(rows)}


def _ratio_row(ratio, grouped, control):
    """One Pareto row: seed-mean quality + the two-column memory account."""
    ppls = [g["val_ppl"] for g in grouped]
    mean = sum(ppls) / len(ppls)
    teacher = grouped[0]["teacher_ppl"]
    # 子集运行（--ratios 不含 "0"）没有对照行 — 增益如实记 None，
    # _verdict 会打 "no ratio-0 control" 标。2026-08-31 A100 事故回归点。
    has_control = control is not None
    row = {
        "ratio": ratio, "ratio_f": round(parse_ratio(ratio), 6),
        "seeds": [g["seed"] for g in grouped], "val_ppl": round(mean, 4),
        "val_ppl_per_seed": [round(p, 4) for p in ppls],
        "val_ppl_spread": round(max(ppls) - min(ppls), 4),
        "teacher_ppl": round(teacher, 4),
        "ppl_gain_vs_control": (_ratio(control - mean, control)
                                if has_control else None),
        "gap_closed_vs_teacher": (_ratio(control - mean, control - teacher)
                                  if has_control else None),
        "memory": grouped[0]["memory"], "runs": grouped,
    }
    row["verdict"] = _verdict(row)
    return row


def _memory_row(n_layers, n_attn, d_head, n_kv, state_elems, contexts):
    """Carried bytes at every context: constant state + O(T) KV, both bit widths.

    `kv_bytes` is benchmarks/kv_frontier.py's own function — the KV column is
    charged by the same formula (and the same 2-bit KIVI scale metadata) that
    the frontier ledger charges every opponent, so our hybrid is not quietly
    billed at fp16 while the competition is billed at 2-bit.
    """
    geo = {"n_layers": n_attn, "d_head": d_head}
    n_recurrent = n_layers - n_attn
    state = state_elems * n_recurrent * STATE_ELEM_BYTES
    kv = {str(b): {str(t): kv_bytes(t, b, n_kv, geo) for t in contexts}
          for b in KV_BITS}
    return {
        "n_layers": n_layers, "attention_layers": n_attn,
        "recurrent_layers": n_recurrent,
        "recurrent_state_bytes": state,          # flat in T — the O(1) part
        "kv_bytes": kv,                          # O(T) — the part we gave back
        "total_bytes": {b: {t: state + v for t, v in per_t.items()}
                        for b, per_t in kv.items()},
        "mb_at_ref": {b: _mb(state + per_t[str(REF_CONTEXT)])
                      for b, per_t in kv.items()},
        "state_mb": _mb(state), "contexts": list(contexts),
    }


def _verdict(row):
    """Pre-registered read-out. Written before the numbers, never tuned to them."""
    gain = row["ppl_gain_vs_control"]
    if row["ratio"] == RATIOS[0]:
        return "control — pure ARR, the only O(1) row (KV column is 0)"
    if gain is None:
        return "no ratio-0 control in this run — gain undefined"
    if row["ratio"] == GATE_RATIO:
        ok = gain >= GATE_MIN_GAIN
        return (f"{'PASS' if ok else 'NEGATIVE RESULT'}: 1:4 PPL 改善 "
                f"{gain:.1%} {'≥' if ok else '<'} {GATE_MIN_GAIN:.0%} 门限"
                + ("" if ok else " → 回填无效，记负结果"))
    return f"exploratory: PPL 改善 {gain:.1%}（不进 RESULTS.md）"


def _promotion(rows):
    """2 seeds is exploratory. RESULTS.md needs the winner at >= 3 seeds."""
    ranked = [r for r in rows if r["ppl_gain_vs_control"] is not None]
    best = max(ranked, key=lambda r: r["ppl_gain_vs_control"], default=None)
    if best is None:
        return {"best_ratio": None, "promotable": False,
                "reason": "no ratio with a computable gain"}
    n = len(best["seeds"])
    return {"best_ratio": best["ratio"], "seeds_run": n,
            "seeds_required": PROMOTION_SEEDS, "promotable": n >= PROMOTION_SEEDS,
            "reason": ("seed 数不足，探索性数字不得进 RESULTS.md"
                       if n < PROMOTION_SEEDS else
                       "可进 RESULTS.md（仍需先在 BENCHMARKS/PRODUCT_LINES 落表）")}


def _markdown(rows):
    """The Pareto table, with the memory account split into its two columns."""
    return "\n".join([
        "# ARR hybrid-ratio Pareto — quality vs carried memory", "",
        _BOUNDARY_NOTE, "", _pareto_table(rows), "",
        "## Carried MB vs context — fp16 KV column", "",
        _context_table(rows, 16), "",
        "## Carried MB vs context — 2-bit KV column", "",
        _context_table(rows, 2), ""])


def _pareto_table(rows):
    out = ["| ratio | attn layers | recurrent layers | val PPL | PPL 改善 vs 0 "
           "| 向 teacher 收敛 | 恒定循环状态 MB | KV MB @fp16 | KV MB @2bit "
           f"| 合计 @fp16 | 合计 @2bit (T={REF_CONTEXT}) | 判定 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m, gain, gap = r["memory"], r["ppl_gain_vs_control"], r["gap_closed_vs_teacher"]
        out.append(
            f"| {r['ratio']} | {m['attention_layers']}/{m['n_layers']} "
            f"| {m['recurrent_layers']} | {r['val_ppl']} | "
            f"{'—' if gain is None else format(gain, '.1%')} | "
            f"{'—' if gap is None else format(gap, '.1%')} | "
            f"**{m['state_mb']}** | {_mb(m['kv_bytes']['16'][str(REF_CONTEXT)])} "
            f"| {_mb(m['kv_bytes']['2'][str(REF_CONTEXT)])} "
            f"| {m['mb_at_ref']['16']} | {m['mb_at_ref']['2']} | {r['verdict']} |")
    return "\n".join(out)


def _context_table(rows, bits):
    ctxs = rows[0]["memory"]["contexts"] if rows else CONTEXTS
    out = ["| ratio | " + " | ".join(f"T={t}" for t in ctxs) + " |",
           "|" + "---|" * (1 + len(ctxs))]
    for r in rows:
        per_t = r["memory"]["total_bytes"][str(bits)]
        out.append("| " + r["ratio"] + " | " +
                   " | ".join(str(_mb(per_t[str(t)])) for t in ctxs) + " |")
    return "\n".join(out)


def _write(ledger, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for row in ledger["rows"]:
        path = os.path.join(out_dir, f"arr_ratio_{_tag(row['ratio'])}.json")
        with open(path, "w") as f:
            json.dump(row, f, indent=2)
    with open(os.path.join(out_dir, "arr_ratio_pareto.json"), "w") as f:
        json.dump({"meta": ledger["meta"], "rows": ledger["rows"],
                   "promotion": ledger["promotion"]}, f, indent=2)
    with open(os.path.join(out_dir, "arr_ratio_pareto.md"), "w") as f:
        f.write(ledger["pareto_md"])


def _report(ledger):
    print(f"\nteacher PPL (reference) {ledger['meta']['teacher_ppl_reference']} "
          f"| control (ratio 0) PPL {ledger['meta']['control_ppl']}")
    for r in ledger["rows"]:
        m = r["memory"]
        print(f"  ratio {r['ratio']:>4s}  PPL {r['val_ppl']:>9.3f}  "
              f"state {m['state_mb']:>8} MB  "
              f"KV@fp16 {_mb(m['kv_bytes']['16'][str(REF_CONTEXT)]):>10} MB  "
              f"KV@2bit {_mb(m['kv_bytes']['2'][str(REF_CONTEXT)]):>9} MB  "
              f"| {r['verdict']}")
    print(f"\npromotion: {ledger['promotion']}")


def _tiny_teacher(args):
    """A randomly initialised Llama — no download, no knowledge, smoke only."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=args.vocab, hidden_size=args.d_model,
                      intermediate_size=2 * args.d_model,
                      num_hidden_layers=args.n_layers,
                      num_attention_heads=args.n_heads,
                      num_key_value_heads=args.n_kv_heads,
                      max_position_embeddings=max(args.seq_len * 2, 512))
    model = LlamaForCausalLM(cfg)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = False
    return model.eval()


def _kd_loss(teacher, student, ids, args):
    """One stage-B KD+CE step — the same objective distill_arr.py minimises."""
    with torch.no_grad():
        t_logits = teacher(input_ids=ids).logits
    s_logits = student(input_ids=ids).logits
    tau = args.tau
    kl = F.kl_div(F.log_softmax(s_logits.float() / tau, dim=-1),
                  F.log_softmax(t_logits.float() / tau, dim=-1),
                  log_target=True, reduction="batchmean") * tau * tau / ids.shape[1]
    ce = F.cross_entropy(s_logits[:, :-1].reshape(-1, s_logits.shape[-1]),
                         ids[:, 1:].reshape(-1))
    return args.kd_alpha * kl + (1.0 - args.kd_alpha) * ce


@torch.no_grad()
def _ppl(model, args, gen):
    nll, tok = 0.0, 0
    for _ in range(args.eval_batches):
        ids = torch.randint(0, args.vocab, (args.batch, args.seq_len),
                            generator=gen)
        loss = model(input_ids=ids, labels=ids).loss.float().item()
        nll += loss * ids[:, 1:].numel()
        tok += ids[:, 1:].numel()
    return math.exp(nll / max(tok, 1))


def _record(ratio, seed, res, args):
    """Fold a distill_arr.py result (or a smoke result) into one run record."""
    keep = res["attention_layers"]
    n_layers = res["n_layers"]
    memory = _memory_row(n_layers, len(keep), res["d_head"], res["n_kv_heads"],
                         recurrent_state_elems(args.n_protofilaments,
                                               args.d_proto, args.n_time_scales,
                                               args.fw_dim), args.contexts)
    return {"ratio": ratio, "seed": seed, "model": res.get("model"),
            "val_ppl": float(res["student_ppl_final"]),
            "teacher_ppl": float(res["teacher_ppl"]),
            "ppl_after_stage_a": res.get("student_ppl_after_a"),
            "attention_layers": keep, "n_layers": n_layers,
            "n_recurrent_layers": n_layers - len(keep),
            "d_head": res["d_head"], "n_kv_heads": res["n_kv_heads"],
            "mixer_params": res.get("mixer_params"), "memory": memory}


def _meta(args, control):
    return {"ratios": list(args.ratios), "seeds": list(args.seeds),
            "mode": "smoke" if args.smoke else "full", "model": args.model,
            "contexts": list(args.contexts), "kv_bits": list(KV_BITS),
            "state_elem_bytes": STATE_ELEM_BYTES, "control_ppl": control,
            "gate_ratio": GATE_RATIO, "gate_min_gain": GATE_MIN_GAIN,
            "promotion_seeds": PROMOTION_SEEDS,
            "teacher_ppl_reference": TEACHER_PPL_REF,
            "boundary": _BOUNDARY_NOTE.splitlines()[0]}


def _print_commands(args):
    print(f"# plan: {len(args.ratios)} ratios x {len(args.seeds)} seeds = "
          f"{len(args.ratios) * len(args.seeds)} distill_arr.py runs")
    for ratio in args.ratios:
        for seed in args.seeds:
            print(f"python benchmarks/distill_arr.py --model {args.model} "
                  f"--keep_ratio {ratio} --steps_a {args.steps_a} "
                  f"--steps_b {args.steps_b} --seed {seed} "
                  f"--out_dir {args.work_dir}/ratio_{_tag(ratio)}_seed{seed}")


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _add_sweep_args(ap)
    _add_smoke_args(ap)
    args = ap.parse_args()
    args.ratios = [r.strip() for r in args.ratios.split(",") if r.strip()]
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.contexts = [int(c) for c in args.contexts.split(",") if c.strip()]
    return args


def _add_sweep_args(ap):
    """Knobs for the remote full sweep — mostly pass-throughs to distill_arr.py."""
    ap.add_argument("--ratios", default=",".join(RATIOS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--contexts", default=",".join(str(c) for c in CONTEXTS))
    ap.add_argument("--out_dir",
                    default=os.path.join(_ROOT, "benchmarks", "results"))
    ap.add_argument("--work_dir",
                    default=os.path.join(_ROOT, "benchmarks", "arr_ratio_out"))
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--steps_a", type=int, default=1000)
    ap.add_argument("--steps_b", type=int, default=2000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--d_proto", type=int, default=96)
    ap.add_argument("--proj_rank", type=int, default=384)
    ap.add_argument("--fw_dim", type=int, default=96)
    ap.add_argument("--n_protofilaments", type=int, default=13)
    ap.add_argument("--n_time_scales", type=int, default=5)
    ap.add_argument("--no_cache", action="store_true",
                    help="force use_cache off; use if the pinned transformers "
                         "version cannot take a recurrent layer's None cache "
                         "slot (quality accounting is unaffected)")
    ap.add_argument("--dry_run", action="store_true", help="print the plan only")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random teacher, CPU, no downloads")


def _add_smoke_args(ap):
    """Smoke-only geometry — a full run reads all of it from the teacher's config."""
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_kv_heads", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=128)
    ap.add_argument("--smoke_steps", type=int, default=1)
    ap.add_argument("--eval_batches", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--kd_alpha", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)


def _mean_ppl(grouped):
    ppls = [g["val_ppl"] for g in grouped]
    return sum(ppls) / len(ppls) if ppls else None


def _ratio(num, den):
    return None if not den else round(num / den, 6)


def _mb(n_bytes):
    return round(n_bytes / 2**20, 3)


def _tag(ratio):
    return str(ratio).replace(":", "_").replace("/", "_")


if __name__ == "__main__":
    raise SystemExit(main())
