"""
benchmarks/state_tracking_a5.py -- the A5 word problem: the NC1 boundary test.

What this tests, and why it is the RIGHT next experiment
--------------------------------------------------------
benchmarks/state_tracking_parity.py showed that adding input-dependent AND
negative eigenvalues takes the liquid core from 0.000 to 1.000 on parity with
perfect length extrapolation. That fixed a PARAMETERISATION defect. It did NOT
change the complexity class, because parity is already in TC0.

A5 (the alternating group on 5 elements, order 60) is the real boundary:
Barrington 1989 (JCSS 38:150-164) -- the word problem for any NON-SOLVABLE
group is NC1-complete under AC0 reductions. A5 is the smallest such group.
Merrill, Petty & Sabharwal (ICML 2024) Cor 4.7: assuming TC0 != NC1, NO
log-precision SSM with a diagonal or input-independent transition can solve it.

Our fix is still DIAGONAL. So the prediction is:

    liquid_both (parity solver)  -> STILL FAILS A5
    lstm (nonlinear recurrence)  -> SOLVES A5

Escaping would need a NON-DIAGONAL input-dependent transition (Thm 5.2 / IDS4 /
DeltaProduct Householder products), which we did not build. If the liquid arm
unexpectedly SUCCEEDS, the algebra audit is wrong and must be re-derived before
anything is claimed.

The LSTM control is not optional
--------------------------------
Merrill et al. train a single-layer RNN that learns A5 at arbitrary length. If
our LSTM control ALSO fails here, the training pipeline is broken and NOTHING in
this file is interpretable -- a liquid-core failure would then be uninformative.
Read the control first, always.

Group construction is self-generated (not jopetty/word-problem) because we only
compare M1 arms against each other here; absolute numbers are therefore NOT
comparable to published tables. `verify_group` asserts the axioms so a silently
wrong Cayley table cannot masquerade as a negative result.

Run:  py -3.11 benchmarks/state_tracking_a5.py --steps 6000 --seeds 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import permutations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt_lnn.config import MTLNNConfig       # noqa: E402
from mt_lnn.mt_lnn_layer import MTLNNLayer  # noqa: E402


# --------------------------------------------------------------------------- #
# A5 as an explicit Cayley table                                              #
# --------------------------------------------------------------------------- #
def _parity(p):
    """Permutation parity: number of inversions mod 2."""
    n, inv = len(p), 0
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                inv += 1
    return inv % 2


def build_a5():
    """Even permutations of 5 elements + composition table. |A5| = 60."""
    elems = [p for p in permutations(range(5)) if _parity(p) == 0]
    idx = {p: i for i, p in enumerate(elems)}
    n = len(elems)
    table = torch.empty(n, n, dtype=torch.long)
    for i, a in enumerate(elems):
        for j, b in enumerate(elems):
            table[i, j] = idx[tuple(a[b[k]] for k in range(5))]   # (a∘b)(k)
    return elems, table


def verify_group(elems, table):
    """Assert the axioms — a wrong table would fake a negative result."""
    n = len(elems)
    assert n == 60, f"|A5| must be 60, got {n}"
    ident = elems.index(tuple(range(5)))
    assert all(int(table[ident, j]) == j for j in range(n)), "identity broken"
    assert all(int(table[i, ident]) == i for i in range(n)), "identity broken"
    for i in range(n):                                   # every row a bijection
        assert len(set(table[i].tolist())) == n, f"row {i} not a permutation"
    assert any(int(table[i, j]) != int(table[j, i])      # non-abelian
               for i in range(n) for j in range(n)), "A5 must be non-abelian"
    return ident


# --------------------------------------------------------------------------- #
# models                                                                       #
# --------------------------------------------------------------------------- #
class LiquidProbe(nn.Module):
    def __init__(self, cfg, n_classes, n_layers):
        super().__init__()
        self.embed = nn.Embedding(n_classes, cfg.d_model)
        self.layers = nn.ModuleList([MTLNNLayer(cfg) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, n_classes)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            out, _ = layer(h, None, position_offset=0, use_scan=True)
            h = h + out
        return self.head(self.norm(h))


class LSTMProbe(nn.Module):
    """Positive control: nonlinear recurrence, which theory says CAN do A5."""

    def __init__(self, d_model, n_classes, n_layers):
        super().__init__()
        self.embed = nn.Embedding(n_classes, d_model)
        self.rnn = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h, _ = self.rnn(self.embed(x))
        return self.head(self.norm(h))


def make_batch(bs, lo, hi, table_cpu, device, gen):
    """Random A5 words; label_t = prefix product g_1∘...∘g_t.

    The prefix loop runs entirely on CPU (table_cpu) and transfers ONCE at the
    end: indexing a CUDA table with CPU indices inside the loop forces a
    host<->device sync per step and dominates runtime.
    """
    t = int(torch.randint(lo, hi + 1, (1,), generator=gen).item())
    x = torch.randint(0, table_cpu.shape[0], (bs, t), generator=gen)
    y = torch.empty_like(x)
    acc = x[:, 0].clone()
    y[:, 0] = acc
    for k in range(1, t):
        acc = table_cpu[acc, x[:, k]]
        y[:, k] = acc
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def run_one(arm, seed, args, table, device):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n_classes = table.shape[0]

    if arm == "lstm_control":
        model = LSTMProbe(args.d_model, n_classes, args.probe_layers).to(device)
    else:
        cfg = MTLNNConfig(
            vocab_size=n_classes, d_model=args.d_model, n_layers=1, n_heads=4,
            n_kv_heads=2, d_head=args.d_model // 4, max_seq_len=args.test_len + 8,
            n_protofilaments=args.n_proto, n_time_scales=args.n_scales,
            tau_max=args.tau_max,
            # main 旗标映射（2026-08-15）：分支的
            # allow_negative_decay + input_dependent_decay（both）=
            # main 的 selective_decay（每步带符号输入相关转移，两者 superset）
            selective_decay=(arm in ("liquid_both", "liquid_ndit",
                                     "liquid_deltap")),
            # E5e: 参数化切换（默认 tanh = 历史；exp 已验证恢复长度外推）
            selective_decay_mode=getattr(args, "sel_mode", "tanh"),
            # NDIT (M2, 2026-08-15): Householder 非对角输入相关转移 —
            # A5 需要非对角（Merrill Cor 4.7），对角修复无论参数化都解不了
            use_householder_transition=(arm == "liquid_ndit"),
            # DeltaProduct NDIT (§4.5 路线修正): 非对合稠密低秩修正
            use_deltaproduct_transition=(arm == "liquid_deltap"),
        )
        model = LiquidProbe(cfg, n_classes, args.probe_layers).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for _ in range(args.steps):
        x, y = make_batch(args.batch, 1, args.train_len, table, device, gen)
        loss = F.cross_entropy(model(x).reshape(-1, n_classes), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    out = {"final_loss": float(loss),
           "params": sum(p.numel() for p in model.parameters())}
    with torch.no_grad():
        for name, (lo, hi) in {
            "in_dist": (1, args.train_len),
            "extrapolate": (args.train_len + 1, args.test_len),
        }.items():
            seq_ok = tok_ok = tok_n = seq_n = 0
            for _ in range(args.eval_batches):
                x, y = make_batch(args.batch, lo, hi, table, device, gen)
                pred = model(x).argmax(-1)
                seq_ok += (pred == y).all(dim=1).sum().item()
                tok_ok += (pred == y).sum().item()
                tok_n += y.numel()
                seq_n += x.shape[0]
            out[name] = seq_ok / seq_n
            out[name + "_tok"] = tok_ok / tok_n
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_len", type=int, default=16)
    p.add_argument("--test_len", type=int, default=48)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_proto", type=int, default=4)
    p.add_argument("--n_scales", type=int, default=4)
    p.add_argument("--probe_layers", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--tau_max", type=float, default=200.0)
    p.add_argument("--eval_batches", type=int, default=8)
    p.add_argument("--sel-mode", choices=["tanh", "exp"], default="tanh",
                   help="selective transition parameterisation (E5e)")
    p.add_argument("--arms", nargs="+", default=None,
                   help="subset of lstm_control/liquid_legacy/liquid_both/"
                        "liquid_ndit")
    p.add_argument("--out", type=str,
                   default="benchmarks/results/state_tracking_a5.json")
    args = p.parse_args()

    elems, table = build_a5()
    verify_group(elems, table)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # table stays on CPU: it is only used to BUILD labels (see make_batch)
    print(f"A5 verified: |G|=60, non-abelian, NC1-complete (Barrington 1989)")
    print(f"device={device}  train<=({args.train_len})  test=({args.train_len+1}..{args.test_len})")
    print(f"per-token chance = 1/60 = 0.0167\n")

    ARMS = ["lstm_control", "liquid_legacy", "liquid_both", "liquid_ndit",
            "liquid_deltap"]
    if getattr(args, "arms", None) is not None:
        ARMS = [a for a in ARMS if a in args.arms]
        if not ARMS:
            raise SystemExit(f"--arms {args.arms} matched nothing")
    results = {}
    for arm in ARMS:
        runs = []
        for s in range(args.seeds):
            t0 = time.time()
            r = run_one(arm, s, args, table, device)
            r["seed"] = s
            r["secs"] = round(time.time() - t0, 1)
            runs.append(r)
            print(f"[{arm}] seed={s} in_dist={r['in_dist']:.3f} "
                  f"(tok {r['in_dist_tok']:.3f})  extrap={r['extrapolate']:.3f} "
                  f"(tok {r['extrapolate_tok']:.3f})  {r['secs']}s")
        mean = lambda k: sum(x[k] for x in runs) / len(runs)
        results[arm] = {"runs": runs,
                        "in_dist_mean": mean("in_dist"),
                        "in_dist_tok_mean": mean("in_dist_tok"),
                        "extrapolate_mean": mean("extrapolate"),
                        "extrapolate_tok_mean": mean("extrapolate_tok")}
        print(f"[{arm}] MEAN in_dist={mean('in_dist'):.3f} "
              f"(tok {mean('in_dist_tok'):.3f})  extrap={mean('extrapolate'):.3f}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"wrote {out}")

    ctrl = results.get("lstm_control", {}).get("in_dist_tok_mean", None)
    both = results.get("liquid_both", {}).get("in_dist_tok_mean", None)
    print("\n=== VERDICT (per-token acc in-distribution; chance = 0.0167) ===")
    for a in ARMS:
        r = results[a]
        print(f"  {a:15s} in_dist_tok={r['in_dist_tok_mean']:.3f}  "
              f"extrap_tok={r['extrapolate_tok_mean']:.3f}")
    print()
    ndit = results.get("liquid_ndit", {}).get("in_dist_tok_mean", None)
    deltap = results.get("liquid_deltap", {}).get("in_dist_tok_mean", None)
    if ctrl is not None and ctrl < 0.5:
        print("  !! LSTM CONTROL FAILED -- the pipeline, not the theory, is at "
              "fault. Nothing else in this run is interpretable.")
    elif deltap is not None and deltap >= 0.5:
        print("  -> DELTAPRODUCT VERDICT: the non-involutory dense correction "
              "SOLVES A5 - first NC1-level expressivity gain on the liquid "
              "core. Proceed to M3 (parity regression + extrapolation).")
    elif deltap is not None and both is not None and deltap <= both + 0.02:
        print("  -> DELTAPRODUCT FAILED (no gain over diagonal). The dense "
              "correction did not open - check dp_g energy; the bottleneck "
              "is elsewhere (width/budget/interaction design).")
    elif both is not None and both < 0.5 and ndit is not None and ndit >= 0.5:
        print("  -> NDIT VERDICT: the diagonal fix fails A5 AND the "
              "Householder non-diagonal transition SOLVES it - first "
              "NC1-level expressivity gain on the liquid core. Proceed to "
              "M3 (parity regression + length extrapolation).")
    elif both is not None and both < 0.5 and ndit is not None and ndit < 0.5:
        print("  -> NDIT FAILED A5 too. The rotation was learned inert or the "
              "budget/width is insufficient - check the Q_t off-diagonal "
              "energy before any expressivity claim. Back to the design doc.")
    elif both is not None and both < 0.5:
        print("  -> AS PREDICTED: the diagonal fix solves parity but NOT A5. "
              "Consistent with Merrill et al. Cor 4.7 (diagonal => TC0 => no "
              "NC1-hard word problem). Escaping needs a NON-DIAGONAL "
              "input-dependent transition.")
    elif both is not None:
        print("  -> UNEXPECTED: the diagonal arm solved A5. The algebra audit "
              "is wrong somewhere -- re-derive BEFORE claiming anything.")
    else:
        print("  (partial run - no full verdict; see result rows)")


if __name__ == "__main__":
    main()
