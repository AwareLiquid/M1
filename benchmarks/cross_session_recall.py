"""Cross-SESSION recall — does snapshot/restore make recall survive a process?

The cross-window benchmark proved the fast-weight (F,z) carries key->value
recall across a dropped KV cache WITHIN one generation (0.56). But that state
is reset every request and never leaves the process. This benchmark asks the
next question: if we SNAPSHOT (F,z) to disk at the end of session 1 and
RESTORE it into a FRESH model in session 2, does the recall survive?

Protocol (per trial), reusing cross_window_recall's exact binding generator:
  SESSION 1 (write): fresh streams, feed segment A = K1 V1 ... Kn Vn -> writes
    (F,z). snapshot_adapter_streams -> torch.save to disk.
  --- build a BRAND-NEW model object (same trained weights, (F,z)=None by
      construction) + torch.load the snapshot: the volatile state provably did
      not ride along in RAM ---
  SESSION 2 (query): reset (zeros), restore the loaded snapshot, feed segment B
    = the keys re-queried in a FRESH attention context. Read logits at key
    positions -> cross_session accuracy.

Controls that give the number meaning (must score ~= chance = 1/|val range|):
  C1 no-restore : session 2 without restore (streams stay zero). If this beats
                  chance the harness leaks and the number is invalid.
  C2 wrong-sess : restore a DIFFERENT trial's snapshot. Proves recall is bound
                  to the RIGHT session, not "any non-zero F helps".

Reference: within_window = the in-process two-segment recall (state carried in
RAM, no snapshot) — the round-trip loss is cross_session minus this.

Retrieval is ORACLE here (restore the correct session by id): this isolates
the (F,z) round-trip fidelity from the separate question of content-addressed
retrieval (e5 keys), which is a follow-on once this number is real.

Usage:
    python benchmarks/cross_session_recall.py --config mt_v2s --steps 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from benchmarks.cross_window_recall import (  # DLL-order safe (imports datasets)
    make_batch,
    recall_loss_and_acc,
    set_stream_mode,
    setup,
    two_segment_forward,
)
from mt_lnn.llama_adapter import (
    _iter_all_adapters,
    count_trainable_parameters,
    reset_adapter_streams,
    restore_adapter_streams,
    snapshot_adapter_streams,
)


def build_model(args, device, dtype):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    m.config.use_cache = False
    m, _ = setup(args.config_setup, m, args.lora_r, args.lora_alpha)
    if args.state_scale_init > 0:
        for a in _iter_all_adapters(m):
            a.scale.data.fill_(args.state_scale_init)
            if getattr(a, "fw_scale", None) is not None:
                a.fw_scale.data.fill_(args.state_scale_init)
    return m.to(device)


def train_recall(m, args, device, dtype):
    """Mixed-protocol recall training (75% cross-window / 25% in-window),
    gradients through carried state — same recipe that reached 0.56."""
    set_stream_mode(m, True, train_through=True)
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=args.lr)
    scaler = (torch.amp.GradScaler("cuda")
              if device == "cuda" and dtype == torch.float16 else None)
    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # Train with train_batch (batch=1 recall training is too noisy to
        # converge — it stalls at ~0.03 vs 0.6 at batch 4). The session
        # snapshot/eval separately runs B=1 to dodge the batch-shape trap;
        # training batch and eval batch are independent.
        seg_a, seg_b = make_batch(args.train_batch, args.n_pairs, args.key_lo,
                                  args.key_hi, args.val_lo, args.val_hi, g)
        seg_a, seg_b = seg_a.to(device), seg_b.to(device)
        cross = torch.rand((), generator=g).item() < 0.75
        with torch.amp.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
            if cross:
                logits_b = two_segment_forward(m, seg_a, seg_b, reset_adapter_streams)
            else:
                reset_adapter_streams(m)
                full = m(input_ids=torch.cat([seg_a, seg_b], dim=1),
                         use_cache=False).logits
                logits_b = full[:, seg_a.shape[1]:, :]
            loss, acc = recall_loss_and_acc(logits_b, seg_b)
        (scaler.scale(loss).backward() if scaler else loss.backward())
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in m.parameters() if p.requires_grad], 1.0)
        (scaler.step(opt), scaler.update()) if scaler else opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            dt = max(time.time() - t0, 1e-3)
            print(f"[train {args.config}] {step}/{args.steps} loss {loss.item():.4f}"
                  f" acc {acc:.3f} | {args.log_every/dt:.2f} it/s", flush=True)
            t0 = time.time()
    m.eval()
    set_stream_mode(m, True, train_through=False)   # detached stream for eval


@torch.no_grad()
def write_session(m, seg_a, path):
    """Session 1: write bindings into (F,z), snapshot to disk. Returns the
    in-RAM snapshot too (for the within-window in-process reference)."""
    reset_adapter_streams(m)
    with torch.amp.autocast("cuda", dtype=next(m.parameters()).dtype,
                            enabled=seg_a.is_cuda):
        m(input_ids=seg_a, use_cache=False)
    snap = snapshot_adapter_streams(m)
    torch.save(snap, path)
    return snap


@torch.no_grad()
def read_session(m, seg_b, snap):
    """Session 2 on a (possibly fresh) model: reset, optionally restore, query."""
    reset_adapter_streams(m)
    if snap is not None:
        restore_adapter_streams(m, snap)
    with torch.amp.autocast("cuda", dtype=next(m.parameters()).dtype,
                            enabled=seg_b.is_cuda):
        logits_b = m(input_ids=seg_b, use_cache=False).logits
    return recall_loss_and_acc(logits_b, seg_b)


def _tag_encoder(dim):
    """Deterministic per-session-tag encoder (no e5 download): clean, distinct
    keys so 'key' retrieval mode measures the store round-trip + recall, not
    e5-over-random-ids key quality (which real session TEXT, not these random
    token ids, would exercise in production)."""
    from mt_lnn.fast_weight_store import id_key
    return lambda tag: id_key(tag, dim)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--config", default="mt_v2s",
                    help="mt_v2 / mt_v2s / mt_v2_nofw (maps to cross_window setup)")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=1)   # EVAL/snapshot: B=1 dodges the batch-shape trap
    ap.add_argument("--train_batch", type=int, default=4,   # recall training needs B>1 to converge
                    help="training batch (independent of the B=1 eval/snapshot)")
    ap.add_argument("--n_pairs", type=int, default=8)
    ap.add_argument("--eval_trials", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--state_scale_init", type=float, default=0.1)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--key_lo", type=int, default=5000)
    ap.add_argument("--key_hi", type=int, default=6000)
    ap.add_argument("--val_lo", type=int, default=7000)
    ap.add_argument("--val_hi", type=int, default=8000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrieval", choices=["oracle", "key"], default="oracle",
                    help="oracle: restore correct session by id (isolates "
                         "round-trip fidelity). key: content-addressed recall "
                         "through FastWeightSessionStore (adds retrieval_top1).")
    ap.add_argument("--out_dir", default="benchmarks/cross_session_out")
    args = ap.parse_args()
    # cross_window_recall.setup expects the "*_only" config names.
    args.config_setup = {"mt_v2": "mt_v2_only", "mt_v2s": "mt_v2s_only",
                         "mt_v2_nofw": "mt_v2_nofw"}.get(args.config, args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
             else (torch.float16 if device == "cuda" else torch.float32))
    chance = 1.0 / (args.val_hi - args.val_lo)
    print(f"device={device} dtype={dtype} config={args.config} chance={chance:.4f}",
          flush=True)

    m = build_model(args, device, dtype)
    print(f"trainable {count_trainable_parameters(m):,}", flush=True)
    if args.steps > 0:
        train_recall(m, args, device, dtype)
    trained_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    def fresh_twin():
        tw = build_model(args, device, dtype)
        tw.load_state_dict(trained_sd)          # trained weights; (F,z)=None by construction
        tw.eval()
        return tw

    g = torch.Generator().manual_seed(10_000 + args.seed)
    accs = {"within_window": [], "cross_session": [], "c1_no_restore": [],
            "c2_wrong_session": []}
    retrieval_ok = []       # key mode only: did content-addressing return session 1?

    # 'key' mode routes session 1 -> durable content-addressed store (e5-like
    # tag key) -> session 2 recalls by key. 'oracle' restores the correct
    # snapshot by id (isolates round-trip fidelity from retrieval).
    use_store = args.retrieval == "key"
    store = enc = None
    if use_store:
        from mt_lnn.fast_weight_store import (FastWeightSessionStore,
                                              build_session_key)
        enc = _tag_encoder(384)

    with tempfile.TemporaryDirectory() as d:
        if use_store:
            store = FastWeightSessionStore(db_path=os.path.join(d, "fw.db"),
                                           key_dim=384)
        for trial in range(args.eval_trials):
            a1, b1 = make_batch(args.batch, args.n_pairs, args.key_lo, args.key_hi,
                                args.val_lo, args.val_hi, g)
            a2, b2 = make_batch(args.batch, args.n_pairs, args.key_lo, args.key_hi,
                                args.val_lo, args.val_hi, g)   # a distractor session
            a1, b1, a2 = a1.to(device), b1.to(device), a2.to(device)

            # within-window reference: same model, state carried in RAM.
            _, acc_win = recall_loss_and_acc(
                two_segment_forward(m, a1, b1, reset_adapter_streams), b1)
            accs["within_window"].append(acc_win)

            # session 1 writes; snapshot goes to disk (oracle) or store (key).
            p1 = os.path.join(d, "s1.pt")
            snap1 = write_session(m, a1, p1)
            p2 = os.path.join(d, "s2.pt")
            snap2 = write_session(m, a2, p2)     # distractor session's snapshot
            if use_store:
                sid1, sid2 = f"t{trial}-s1", f"t{trial}-s2"
                store.write_session(sid1, build_session_key(sid1, enc), snap1)
                store.write_session(sid2, build_session_key(sid2, enc), snap2)

            # session 2: fresh model, restore session 1's state, query b1.
            # oracle: correct snapshot straight from disk (round-trip fidelity).
            # key: content-addressed recall THROUGH the store. retrieval_top1
            #   verifies the store returned session 1 among the N stored (an
            #   exact-key round-trip among distractors — e5 semantic
            #   discrimination on similar text is covered by the store unit test).
            tw = fresh_twin()
            if use_store:
                hits = store.recall_session(build_session_key(sid1, enc),
                                            top_k=1, center=False)
                got_snap, _, got_sid, _ = hits[0]
                retrieval_ok.append(1.0 if got_sid == sid1 else 0.0)
                _, acc_x = read_session(tw, b1, got_snap)
            else:
                _, acc_x = read_session(tw, b1, torch.load(p1, weights_only=False))
            accs["cross_session"].append(acc_x)

            # C1: fresh model, no restore -> must be chance (state really gone).
            _, acc_c1 = read_session(fresh_twin(), b1, None)
            accs["c1_no_restore"].append(acc_c1)

            # C2: restore the WRONG (distractor) session, query b1 -> chance.
            # key mode routes C2 THROUGH the store (recall the distractor's key)
            # so the whole control path exercises content-addressing, not disk.
            if use_store:
                d_hits = store.recall_session(build_session_key(sid2, enc),
                                              top_k=1, center=False)
                wrong_snap = d_hits[0][0]
            else:
                wrong_snap = torch.load(p2, weights_only=False)
            _, acc_c2 = read_session(fresh_twin(), b1, wrong_snap)
            accs["c2_wrong_session"].append(acc_c2)
        if store is not None:
            store.close()

    res = {k: sum(v) / len(v) for k, v in accs.items()}
    if retrieval_ok:
        res["retrieval_top1"] = sum(retrieval_ok) / len(retrieval_ok)
    res.update({"config": args.config, "chance": chance, "steps": args.steps,
                "n_pairs": args.n_pairs, "retrieval": args.retrieval})
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,
              f"cross_session_{args.config}_{args.steps}.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print(f"CROSS-SESSION RECALL | {args.config} | chance={chance:.4f}", flush=True)
    print("=" * 60, flush=True)
    keys = ["within_window", "cross_session", "c1_no_restore", "c2_wrong_session"]
    if "retrieval_top1" in res:
        keys.append("retrieval_top1")
    for k in keys:
        print(f"  {k:<18} {res[k]:.3f}", flush=True)
    delta = res["within_window"] - res["cross_session"]
    print(f"\n  round-trip loss (within - cross_session): {delta:+.3f}", flush=True)
    # Honest verdict: PASS requires (1) the model actually LEARNED recall
    # (within_window clears chance), (2) the round-trip RECOVERED most of it —
    # cross_session >= 0.7*within_window, so a 96%-lossy restore is NOT a pass
    # even though it clears chance — and (3) both controls at chance. Tying the
    # gate to within_window (not an absolute bar) is what enforces "lossless".
    trained = res["within_window"] > 5 * chance
    lossless = res["cross_session"] >= 0.7 * res["within_window"]
    controls_ok = (res["c1_no_restore"] < 3 * chance
                   and res["c2_wrong_session"] < 3 * chance)
    verdict = "PASS" if (trained and lossless and controls_ok) else "CHECK"
    if not trained:
        verdict += " (undertrained: within_window at chance — bridge claim needs a trained model)"
    print(f"  verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
