"""Parametric memory engine — four-competency benchmark (MemoryAgentBench axes).

Extends the cross-window recall protocol (the 0.56 anchor) to the four
MemoryAgentBench (arXiv:2507.05257) competencies, so the parametric-memory
runtime and its LM-level mechanism get comparable numbers on every axis:

  D1 accurate_retrieval      in-window recall (attention can see the writes;
                             control column — every competent config passes)
  D2 test_time_learning      cross-window recall after the KV cache is DROPPED
                             (the 0.56-anchor measurement; structural zero for
                             baseline/lora_only)
  D3 conflict_resolution     write k->v1, then k->v2 (both across the window);
                             the answer must be the LATEST value. Mechanism
                             prediction: delta corrects the binding, sum blends
                             both writes (recency-blind). Reported as measured.
  D4 cross_session_long_range session 1 writes -> snapshot to disk -> a REAL
                             SUBPROCESS restores it into a fresh model (same
                             trained weights) -> session 2 recalls.

Configs (LM level): baseline / lora_only (structural-zero controls), mt_v2
(sum write rule), mt_v2_delta (delta write rule). Plus a zero-dependency
pure-Python BM25 store as the external-library route's representative
(Mem0/Letta shape: memory = retrievable records, storage grows with history)
on the same binding task, with its retrieval latency and store-bytes curves
as a function of the number of entries — next to the parametric runtime's
constant state_bytes (and its honest capacity curve).

Protocol discipline: >=3 seeds per config, per-seed JSONs + a mean+-std
summary; resume-safe (existing per-(config, seed) JSONs are skipped).

Scale discipline: TinyLlama-1.1B (the anchor's model), steps <= 3000 by
default (the 0.56 anchor used 8000 — every JSON records its own steps, and
the docs compare at the recorded budget, never silently). --smoke runs a
tiny random-init Llama on CPU end-to-end including the subprocess path.

Usage:
    python benchmarks/parametric_memory_bench.py --smoke
    python benchmarks/parametric_memory_bench.py --configs mt_v2 --seeds 0
    python benchmarks/parametric_memory_bench.py            # full grid
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from benchmarks.attribution_ablation import setup as attr_setup  # DLL-order guard
from benchmarks.cross_window_recall import (
    make_batch,
    recall_loss_and_acc,
    set_stream_mode,
    setup as recall_setup,
    two_segment_forward,
)

CONFIGS = ["baseline", "lora_only", "mt_v2", "mt_v2_delta"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------------------
# Binding generators (D3 conflict; D1/D2 reuse cross_window_recall.make_batch)
# ---------------------------------------------------------------------------

def make_conflict_batch(bsz, n_pairs, key_lo, key_hi, val_lo, val_hi, g):
    """Same keys written TWICE with different values. Returns
    (seg_a1, seg_a2, seg_b): a1 = K1 V1 ... Kn Vn, a2 = K1 W1 ... Kn Wn
    (W_i != V_i per position), b = keys re-queried shuffled, teacher-forced
    with the LATEST value W."""
    keys = torch.stack([
        torch.randperm(key_hi - key_lo, generator=g)[:n_pairs] + key_lo
        for _ in range(bsz)
    ])
    vals1 = torch.randint(val_lo, val_hi, (bsz, n_pairs), generator=g)
    vals2 = torch.randint(val_lo, val_hi, (bsz, n_pairs), generator=g)
    vals2 = torch.where(vals2 == vals1, (vals2 + 1) % (val_hi - val_lo) + val_lo,
                        vals2)
    order = torch.stack([torch.randperm(n_pairs, generator=g) for _ in range(bsz)])
    seg = lambda ks, vs: torch.stack([ks, vs], dim=-1).reshape(bsz, 2 * n_pairs)
    seg_a1, seg_a2 = seg(keys, vals1), seg(keys, vals2)
    seg_b = seg(torch.gather(keys, 1, order), torch.gather(vals2, 1, order))
    return seg_a1, seg_a2, seg_b


@torch.no_grad()
def conflict_forward(model, seg_a1, seg_a2, seg_b, reset_fn, cross: bool):
    """cross=True: KV dropped between all three segments (both writes reach the
    model ONLY through the adapter state; attention sees segment b fresh).
    cross=False: one contiguous [a1|a2|b] forward (in-window sanity)."""
    if not cross:
        reset_fn(model)
        full = model(input_ids=torch.cat([seg_a1, seg_a2, seg_b], dim=1),
                     use_cache=False).logits
        logits_b = full[:, seg_a1.shape[1] + seg_a2.shape[1]:, :]
        return recall_loss_and_acc(logits_b, seg_b)
    reset_fn(model)
    model(input_ids=seg_a1, use_cache=False)                  # write v1
    model(input_ids=seg_a2, use_cache=False)                  # overwrite -> v2
    logits_b = model(input_ids=seg_b, use_cache=False).logits
    return recall_loss_and_acc(logits_b, seg_b)


# ---------------------------------------------------------------------------
# Model build / train (anchor protocol: 75% cross-window mix, lr 1e-3,
# state_scale_init 0.1 — BENCHMARKS.md "Cross-window associative recall")
# ---------------------------------------------------------------------------

def build_model(args, device, dtype):
    from transformers import AutoModelForCausalLM
    from mt_lnn.llama_adapter import _iter_all_adapters

    torch.manual_seed(args.seed)
    m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    m.config.use_cache = False
    m, _ = recall_setup(args.config, m, args.lora_r, args.lora_alpha)
    for a in _iter_all_adapters(m):
        a.scale.data.fill_(args.state_scale_init)
        if getattr(a, "fw_scale", None) is not None:
            a.fw_scale.data.fill_(args.state_scale_init)
    return m.to(device)


def train_recall(m, args, device, dtype, reset_fn):
    from mt_lnn.llama_adapter import count_trainable_parameters

    n_train = count_trainable_parameters(m)
    if n_train == 0 or args.steps == 0:
        return n_train
    set_stream_mode(m, True, train_through=True)
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=args.lr)
    scaler = (torch.amp.GradScaler("cuda")
              if device == "cuda" and dtype == torch.float16 else None)
    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        seg_a, seg_b = make_batch(args.batch, args.n_pairs, args.key_lo,
                                  args.key_hi, args.val_lo, args.val_hi, g)
        seg_a, seg_b = seg_a.to(device), seg_b.to(device)
        cross = torch.rand((), generator=g).item() < args.cross_frac
        with torch.amp.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
            if cross:
                logits_b = two_segment_forward(m, seg_a, seg_b, reset_fn)
            else:
                reset_fn(m)
                full = m(input_ids=torch.cat([seg_a, seg_b], dim=1),
                         use_cache=False).logits
                logits_b = full[:, seg_a.shape[1]:, :]
            loss, acc = recall_loss_and_acc(logits_b, seg_b)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in m.parameters() if p.requires_grad], 1.0)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            dt = max(time.time() - t0, 1e-3)
            print(f"[train {args.config}] {step}/{args.steps} loss "
                  f"{loss.item():.4f} acc {acc:.3f} | {args.log_every/dt:.2f} it/s",
                  flush=True)
            t0 = time.time()
    m.eval()
    set_stream_mode(m, True, train_through=False)
    return n_train


# ---------------------------------------------------------------------------
# In-process dimensions D1-D3
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_d1_d3(m, args, device, g_eval, reset_fn):
    m.eval()
    accs = {"in_window": [], "cross_window": [], "conflict_cross": [],
            "conflict_in_window": []}
    for _ in range(args.eval_batches):
        a, b = make_batch(args.batch, args.n_pairs, args.key_lo, args.key_hi,
                          args.val_lo, args.val_hi, g_eval)
        a, b = a.to(device), b.to(device)
        reset_fn(m)
        full = m(input_ids=torch.cat([a, b], dim=1), use_cache=False).logits
        _, acc = recall_loss_and_acc(full[:, a.shape[1]:, :], b)
        accs["in_window"].append(acc)
        logits_b = two_segment_forward(m, a, b, reset_fn)
        _, acc = recall_loss_and_acc(logits_b, b)
        accs["cross_window"].append(acc)

        a1, a2, cb = make_conflict_batch(args.batch, args.n_pairs, args.key_lo,
                                         args.key_hi, args.val_lo, args.val_hi,
                                         g_eval)
        a1, a2, cb = a1.to(device), a2.to(device), cb.to(device)
        _, acc = conflict_forward(m, a1, a2, cb, reset_fn, cross=True)
        accs["conflict_cross"].append(acc)
        _, acc = conflict_forward(m, a1, a2, cb, reset_fn, cross=False)
        accs["conflict_in_window"].append(acc)
    return {k: sum(v) / len(v) for k, v in accs.items()}


# ---------------------------------------------------------------------------
# D4 cross-session: parent writes snapshots; a REAL subprocess restores
# ---------------------------------------------------------------------------

def _reset_fn(model):
    from mt_lnn.arr import reset_mixer_streams
    from mt_lnn.llama_adapter import reset_adapter_streams
    reset_adapter_streams(model)
    reset_mixer_streams(model)


def save_adapter_params(m, path):
    sd = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}
    torch.save(sd, path)


@torch.no_grad()
def parent_write_snapshots(m, args, device, g_eval, workdir):
    """Session 1 per trial: fresh streams -> write segment A -> snapshot to
    disk. Returns (trial_manifest, within_window_accs)."""
    from mt_lnn.llama_adapter import snapshot_adapter_streams
    manifest = []
    for trial in range(args.session_trials):
        a, b = make_batch(1, args.n_pairs, args.key_lo, args.key_hi,
                          args.val_lo, args.val_hi, g_eval)
        a, b = a.to(device), b.to(device)
        _, acc_win = recall_loss_and_acc(two_segment_forward(m, a, b, _reset_fn),
                                         b)
        _reset_fn(m)
        m(input_ids=a, use_cache=False)                 # session-1 write
        snap_path = os.path.join(workdir, f"snap_{trial}.pt")
        torch.save(snapshot_adapter_streams(m), snap_path)
        torch.save({"seg_b": b.cpu(), "snap": snap_path,
                    "within_window": acc_win},
                   os.path.join(workdir, f"trial_{trial}.pt"))
        manifest.append(os.path.join(workdir, f"trial_{trial}.pt"))
    return manifest


def run_cross_session_child(args, out_json):
    """Session 2 in a FRESH PROCESS: rebuild the model (frozen base + trained
    adapter weights from disk), restore each trial's snapshot, recall. The
    volatile (F, z) provably did not ride along in RAM."""
    from mt_lnn.llama_adapter import restore_adapter_streams

    device = _pick_device()
    m = build_model(args, device, torch.float32)
    sd = torch.load(args.adapter_sd, map_location="cpu", weights_only=False)
    m.load_state_dict(sd, strict=False)   # trainable params only; base is frozen
    m.eval()
    set_stream_mode(m, True, train_through=False)

    accs, no_restore = [], []
    for trial_path in sorted(args.trials):
        t = torch.load(trial_path, map_location="cpu", weights_only=False)
        seg_b = t["seg_b"].to(device)
        _reset_fn(m)
        if t.get("restore", True):
            restore_adapter_streams(m, torch.load(t["snap"], map_location="cpu",
                                                  weights_only=False))
        _, acc = recall_loss_and_acc(
            m(input_ids=seg_b, use_cache=False).logits, seg_b)
        accs.append(acc)
        _reset_fn(m)                                     # no-restore control
        _, acc0 = recall_loss_and_acc(
            m(input_ids=seg_b, use_cache=False).logits, seg_b)
        no_restore.append(acc0)
    within = [torch.load(p, map_location="cpu", weights_only=False)["within_window"]
              for p in sorted(args.trials)]
    out = {"cross_session": sum(accs) / len(accs),
           "cross_session_trials": accs,
           "no_restore_control": sum(no_restore) / len(no_restore),
           "within_window_reference": sum(within) / len(within)}
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[child {args.config}] cross_session {out['cross_session']:.3f} | "
          f"no-restore {out['no_restore_control']:.3f}", flush=True)


def spawn_cross_session_child(args, device, manifest, adapter_sd_path, workdir):
    out_json = os.path.join(workdir, "child_out.json")
    cmd = [sys.executable, os.path.abspath(__file__), "--child",
           "--model", args.model, "--config", args.config,
           "--seed", str(args.seed),
           "--lora_r", str(args.lora_r), "--lora_alpha", str(args.lora_alpha),
           "--state_scale_init", str(args.state_scale_init),
           "--adapter_sd", adapter_sd_path,
           "--out", out_json, "--trials"] + manifest
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")               # model is cached
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["TMPDIR"] = workdir
    subprocess.run(cmd, check=True, env=env)
    with open(out_json) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# BM25 external-store control (pure Python, zero deps)
# ---------------------------------------------------------------------------

class BM25Store:
    """Minimal upsert BM25 with an inverted index (the Mem0/Letta route:
    memory = retrievable records + index). One doc per key, tokens
    [K<key>, V<value>]; rewriting a key REPLACES the doc (update semantics,
    as Mem0 does). Query scores candidate docs by BM25 over the query term.
    The index is part of the store — everything that must persist lives in
    store_bytes, which is the column that grows linearly with entries."""

    def __init__(self):
        self.doc_len = {}       # key -> length (2 for every doc here)
        self.values = {}        # key -> value token ("V<value>")
        self.postings = {}      # term -> set of keys
        self.total_len = 0

    def write(self, key, value):
        self._remove(key)
        toks = [f"K{key}", f"V{value}"]
        self.doc_len[key] = len(toks)
        self.values[key] = f"V{value}"
        self.total_len += len(toks)
        for t in toks:
            self.postings.setdefault(t, set()).add(key)

    def _remove(self, key):
        if key not in self.doc_len:
            return
        for t in [f"K{key}", self.values[key]]:
            s = self.postings.get(t)
            if s is not None:
                s.discard(key)
                if not s:
                    del self.postings[t]
        self.total_len -= self.doc_len.pop(key)
        del self.values[key]

    def query(self, key):
        """Value token ("V<value>") of the highest-scoring doc for K<key>."""
        cands = self.postings.get(f"K{key}")
        if not cands:
            return None
        n = len(self.doc_len)
        avgdl = self.total_len / n
        df = len(cands)
        best, best_s = None, -1.0
        for k2 in cands:
            tf = 1                      # term appears once per doc by design
            dl = self.doc_len[k2]
            s = _bm25_idf(n, df) * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
            if s > best_s:
                best, best_s = k2, s
        return self.values[best]


def _bm25_idf(n, df):
    import math
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


def bm25_benchmark(n_entries_list, n_queries=100, seed=0):
    """Accuracy on the binding task + latency and store bytes vs N entries.
    Task aligned with the LM protocol: keys from a large space, values from a
    1000-wide space (chance = 1/1000)."""
    g = torch.Generator().manual_seed(seed)
    keys_all = torch.randperm(900000, generator=g)[:max(n_entries_list)]
    vals_all = torch.randint(0, 1000, (len(keys_all),), generator=g)
    store = BM25Store()
    rows = []
    written = 0
    for n_target in n_entries_list:
        while written < n_target:
            store.write(int(keys_all[written]), int(vals_all[written]))
            written += 1
        probe = min(n_queries, written)
        hits, lat = 0, []
        for _ in range(probe):
            j = int(torch.randint(0, written, (1,), generator=g))
            t0 = time.perf_counter()
            got = store.query(int(keys_all[j]))
            lat.append(time.perf_counter() - t0)
            hits += int(got == f"V{int(vals_all[j])}")
        rows.append({
            "n_entries": written,
            "accuracy": hits / probe,
            "query_latency_ms_mean": 1000 * sum(lat) / len(lat),
            "query_latency_ms_median": 1000 * sorted(lat)[len(lat) // 2],
            "store_bytes": len(pickle.dumps(store)),
        })
    # conflict semantics: rewrite 100 keys, all queries must return LATEST
    conflicts, c_hits = 100, 0
    for i in range(conflicts):
        j = int(keys_all[i])
        new_v = (int(vals_all[i]) + 1) % 1000
        store.write(int(j), new_v)
        c_hits += int(store.query(int(j)) == f"V{new_v}")
    return {"curve": rows, "conflict_latest_accuracy": c_hits / conflicts}


# ---------------------------------------------------------------------------
# Parametric runtime curves (O(1) state vs O(n) external store)
# ---------------------------------------------------------------------------

def runtime_curves(n_entries_list, dims=(128, 512, 2048), n_queries=100, seed=0):
    """ParametricMemory accuracy / state_bytes / recall latency vs N writes.
    Task aligned with the LM protocol + BM25 control: keys from a large space
    (vocab table), values decoded against a 1000-wide value range (chance
    1/1000). state_bytes is constant by construction; accuracy vs N traces
    the honest capacity boundary (~sqrt(d/N) read SNR for random unit keys)."""
    from mt_lnn.parametric_memory import ParametricMemory

    out = []
    for d in dims:
        mem = ParametricMemory(d_mem=d, vocab_size=1_000_000, seed=seed,
                               update_rule="sum")
        g = torch.Generator().manual_seed(seed)
        keys = torch.randperm(1_000_000, generator=g)[:max(n_entries_list) + 100]
        vals = torch.randint(900_000, 901_000, (len(keys),), generator=g)
        written = 0
        for n_target in n_entries_list:
            while written < n_target:
                mem.write("s", key=int(keys[written]), value=int(vals[written]))
                written += 1
            hits, lat = 0, []
            for _ in range(n_queries):
                j = int(torch.randint(0, written, (1,), generator=g))
                t0 = time.perf_counter()
                got = mem.recall("s", int(keys[j]), top_k=1,
                                 candidates=range(900_000, 901_000))[0][0]
                lat.append(time.perf_counter() - t0)
                hits += int(got == int(vals[j]))
            out.append({
                "d_mem": d, "n_writes": written, "accuracy": hits / n_queries,
                "recall_latency_ms_mean": 1000 * sum(lat) / len(lat),
                "state_bytes": mem.state_bytes("s"),
            })
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _pick_device():
    forced = os.environ.get("BENCH_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_lm_config(args, device, dtype, workroot):
    from mt_lnn.llama_adapter import _iter_all_adapters

    out_path = os.path.join(
        RESULTS_DIR, f"parametric_memory_{args.config}_s{args.seed}.json")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} exists", flush=True)
        with open(out_path) as f:
            return json.load(f)

    workdir = os.path.join(workroot, f"{args.config}_s{args.seed}")
    os.makedirs(workdir, exist_ok=True)
    torch.manual_seed(args.seed)
    m = build_model(args, device, dtype)
    t0 = time.time()
    trainable = train_recall(m, args, device, dtype, _reset_fn)
    train_s = time.time() - t0
    print(f"[{args.config}] trainable {trainable:,} trained in {train_s:.0f}s",
          flush=True)

    g_eval = torch.Generator().manual_seed(10_000 + args.seed)
    dims = eval_d1_d3(m, args, device, g_eval, _reset_fn)

    manifest = parent_write_snapshots(m, args, device, g_eval, workdir)
    adapter_sd = os.path.join(workdir, "adapter_sd.pt")
    save_adapter_params(m, adapter_sd)
    del m
    if device == "cuda":
        torch.cuda.empty_cache()

    d4 = spawn_cross_session_child(args, device, manifest, adapter_sd, workdir)

    res = {
        "config": args.config, "seed": args.seed, "trainable": trainable,
        "steps": args.steps, "model": args.model, "device": device,
        "chance": 1.0 / (args.val_hi - args.val_lo),
        "dims": {
            "accurate_retrieval": dims["in_window"],
            "test_time_learning": dims["cross_window"],
            "conflict_resolution": dims["conflict_cross"],
            "conflict_in_window_sanity": dims["conflict_in_window"],
            "cross_session_long_range": d4["cross_session"],
            "cross_session_no_restore_control": d4["no_restore_control"],
            "cross_session_within_window_reference": d4["within_window_reference"],
        },
        "meta": {k: v for k, v in vars(args).items()
                 if isinstance(v, (int, float, str))},
        "train_seconds": train_s,
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{args.config} s{args.seed}] D1 {res['dims']['accurate_retrieval']:.3f}"
          f" D2 {res['dims']['test_time_learning']:.3f}"
          f" D3 {res['dims']['conflict_resolution']:.3f}"
          f" D4 {res['dims']['cross_session_long_range']:.3f}", flush=True)
    return res


def aggregate(configs, seeds):
    import statistics as st
    summary = {}
    for cfg in configs:
        per_seed = []
        for s in seeds:
            p = os.path.join(RESULTS_DIR, f"parametric_memory_{cfg}_s{s}.json")
            if os.path.exists(p):
                with open(p) as f:
                    per_seed.append(json.load(f))
        if not per_seed:
            continue
        entry = {"n_seeds": len(per_seed), "per_seed": []}
        for dim in per_seed[0]["dims"]:
            vals = [r["dims"][dim] for r in per_seed]
            entry[dim] = {"mean": st.mean(vals), "std": st.pstdev(vals),
                          "per_seed": vals}
        summary[cfg] = entry
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--config", default="",
                    help="single config (child mode / one-off runs)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seed", type=int, default=0,
                    help="single seed (child mode / one-off runs)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_pairs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cross_frac", type=float, default=0.75,
                    help="anchor protocol: 75%% cross-window training mix")
    ap.add_argument("--state_scale_init", type=float, default=0.1)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--key_lo", type=int, default=5000)
    ap.add_argument("--key_hi", type=int, default=6000)
    ap.add_argument("--val_lo", type=int, default=7000)
    ap.add_argument("--val_hi", type=int, default=8000)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--session_trials", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--out_dir", default=RESULTS_DIR)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random Llama, CPU, ~2 min end-to-end")
    ap.add_argument("--skip_bm25", action="store_true")
    # child-mode flags (spawned by spawn_cross_session_child)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--adapter_sd", default="")
    ap.add_argument("--trials", nargs="*", default=[])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.child:
        if args.model == "tiny-random":
            _register_tiny_random()
        run_cross_session_child(args, args.out)
        return

    if args.smoke:
        args.model = "tiny-random"
        args.steps, args.eval_batches = 30, 2
        args.session_trials, args.n_pairs = 2, 4
        args.seeds = "0"
        args.log_every = 10
        # tiny-random vocab is 512 — keep key/value ids inside it
        args.key_lo, args.key_hi = 100, 300
        args.val_lo, args.val_hi = 300, 500

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    wanted = [c for c in args.configs.split(",") if c]

    os.makedirs(args.out_dir, exist_ok=True)
    device = _pick_device()
    dtype = (torch.bfloat16 if device == "cuda" else torch.float32)
    args.device = device

    if args.model == "tiny-random":
        _register_tiny_random()

    print(f"device={device} dtype={dtype} model={args.model} steps={args.steps} "
          f"configs={wanted} seeds={seeds}", flush=True)

    import tempfile
    results = []
    with tempfile.TemporaryDirectory() as workroot:
        for cfg in wanted:
            for s in seeds:
                args.config, args.seed = cfg, s
                results.append(run_lm_config(args, device, dtype, workroot))

    summary_path = os.path.join(args.out_dir, "parametric_memory_summary.json")
    summary = {"protocol": {k: v for k, v in vars(args).items()
                            if isinstance(v, (int, float, str))},
               "lm_configs": aggregate(wanted, seeds)}
    if not args.skip_bm25:
        bm25_path = os.path.join(args.out_dir, "parametric_memory_bm25.json")
        if os.path.exists(bm25_path):
            with open(bm25_path) as f:
                bm25 = json.load(f)
        else:
            bm25 = bm25_benchmark([100, 1000, 10000])
            with open(bm25_path, "w") as f:
                json.dump(bm25, f, indent=2)
        summary["bm25_external_store"] = bm25
        rt_path = os.path.join(args.out_dir, "parametric_memory_runtime.json")
        if os.path.exists(rt_path):
            with open(rt_path) as f:
                rt = json.load(f)
        else:
            rt = {"curve": runtime_curves([100, 500, 1000, 2000, 5000, 10000])}
            with open(rt_path, "w") as f:
                json.dump(rt, f, indent=2)
        summary["parametric_runtime"] = rt
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78, flush=True)
    print(f"PARAMETRIC MEMORY 4-COMPETENCY | {args.model} | steps={args.steps} "
          f"| chance={1.0/(args.val_hi-args.val_lo):.4f}", flush=True)
    print("=" * 78, flush=True)
    hdr = (f"{'config':<13}{'D1 in-win':>10}{'D2 x-win':>10}"
           f"{'D3 conflict':>12}{'D4 x-session':>14}")
    print(hdr, flush=True)
    for cfg, entry in summary["lm_configs"].items():
        d = entry
        print(f"{cfg:<13}"
              f"{d['accurate_retrieval']['mean']:>10.3f}"
              f"{d['test_time_learning']['mean']:>10.3f}"
              f"{d['conflict_resolution']['mean']:>12.3f}"
              f"{d['cross_session_long_range']['mean']:>14.3f}", flush=True)
    if not args.skip_bm25:
        b = summary["bm25_external_store"]["curve"][-1]
        print(f"\nbm25 @ {b['n_entries']} entries: acc {b['accuracy']:.3f} | "
              f"store {b['store_bytes']/1e6:.2f} MB | "
              f"latency {b['query_latency_ms_mean']*1000:.1f} us", flush=True)
        rt = summary["parametric_runtime"]["curve"]
        for r in rt:
            if r["n_writes"] in (100, 10000) or r["n_writes"] == rt[-1]["n_writes"]:
                print(f"runtime d={r['d_mem']} @ {r['n_writes']} writes: "
                      f"acc {r['accuracy']:.3f} | state {r['state_bytes']/1e3:.0f} KB"
                      f" | latency {r['recall_latency_ms_mean']*1000:.1f} us",
                      flush=True)


def _register_tiny_random():
    """--smoke builds a tiny random LlamaForCausalLM (no download, CPU-fast)
    by intercepting AutoModelForCausalLM.from_pretrained for model='tiny-random'."""
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    orig = AutoModelForCausalLM.from_pretrained
    cfg = LlamaConfig(vocab_size=512, hidden_size=96, intermediate_size=192,
                      num_hidden_layers=8, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=256)

    def fake_from_pretrained(name, *a, **kw):
        if name != "tiny-random":
            return orig(name, *a, **kw)
        torch.manual_seed(0)
        return LlamaForCausalLM(cfg)
    AutoModelForCausalLM.from_pretrained = fake_from_pretrained


if __name__ == "__main__":
    main()
