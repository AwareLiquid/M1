"""Verification for the 2026-07-14 bug-fix batch (deep review DEEP_REVIEW_2026_07_14.md).

Covers: C-1 hamiltonian no_grad, H-2 sparse selection consistency, MTP indexing,
GWTB zero-init invariant, pad_mask None-parity, world-model buffer persistence,
fast-weight UPSERT, KB key_dim guard, atomic session save, weights_only KB.
Run:  .venv/Scripts/python.exe _verify_fixes_2026_07_14.py
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS" if cond else "FAIL"), name, detail)


def tiny_cfg(**kw):
    base = dict(vocab_size=128, max_seq_len=64, d_model=52, n_layers=2,
                n_heads=13, n_kv_heads=1, d_head=4, n_protofilaments=13,
                map_hidden_dim=8, gwtb_compression_ratio=4, gwtb_n_heads=1)
    base.update(kw)
    return MTLNNConfig(**base)


# ---- 1. C-1: hamiltonian head under no_grad (forward + generate) -----------
cfg = tiny_cfg(use_hamiltonian_world_model=True)
model = MTLNNModel(cfg)
ids = torch.randint(0, 128, (2, 12))
try:
    with torch.no_grad():
        out = model(ids)
    check("C-1 hamiltonian forward under no_grad", torch.isfinite(out["logits"]).all().item())
except Exception as e:
    check("C-1 hamiltonian forward under no_grad", False, repr(e))
try:
    gen = model.generate(ids[:, :4], max_new_tokens=3)
    check("C-1 hamiltonian generate()", True)
except Exception as e:
    check("C-1 hamiltonian generate()", False, repr(e))
# training path still produces a finite loss + grads
model.train()
out = model(ids, labels=ids)
out["loss"].backward()
check("C-1 hamiltonian train loss finite+backward", torch.isfinite(out["loss"]).item())

# ---- 2. H-2: sparse selection — prefill vs incremental consistency ---------
cfg = tiny_cfg(sparse_resonance_kernel=True, sparse_resonance_top_k=2)
model = MTLNNModel(cfg).eval()
ids = torch.randint(0, 128, (1, 10))
with torch.no_grad():
    full = model(ids)["logits"]
    cache = None
    incr = []
    for t in range(ids.shape[1]):
        o = model(ids[:, t:t + 1], cache=cache, use_cache=True)
        cache = o["cache"]
        incr.append(o["logits"])
    incr = torch.cat(incr, dim=1)
diff = (full - incr).abs().max().item()
check("H-2 sparse prefill==incremental", diff < 1e-4, f"max diff {diff:.2e}")
# batch independence: sample 0 alone vs in a batch of 2
with torch.no_grad():
    a = model(ids)["logits"]
    b = model(torch.cat([ids, torch.randint(0, 128, (1, 10))], 0))["logits"][:1]
bdiff = (a - b).abs().max().item()
check("H-2 sparse batch-independent", bdiff < 1e-4, f"max diff {bdiff:.2e}")

# ---- 3. MTP: head k=1 no longer duplicates the main CE ---------------------
cfg = tiny_cfg(use_mtp_heads=True, mtp_lookahead=2, mtp_loss_weight=0.1)
model = MTLNNModel(cfg).train()
ids = torch.randint(0, 128, (2, 12))
out = model(ids, labels=ids)
check("MTP aux loss computed", "mtp_loss" in out and torch.isfinite(out["mtp_loss"]).item())
# indexing sanity: with T <= K+1 the block must be skipped, not crash
out_small = model(ids[:, :3], labels=ids[:, :3])
check("MTP short-seq guard (T<=K+1 skips)", "mtp_loss" not in out_small)

# ---- 4. GWTB competitive zero-init invariant --------------------------------
cfg = tiny_cfg(use_gwtb=True, gwtb_competitive=True) if hasattr(MTLNNConfig, "gwtb_competitive") else None
try:
    from mt_lnn.gwtb import CompetitiveGWTBLayer
    import dataclasses
    fields = {f.name for f in dataclasses.fields(MTLNNConfig)}
    flag = next((f for f in ("use_competitive_gwtb", "gwtb_competitive", "competitive_gwtb") if f in fields), None)
    if flag:
        m2 = MTLNNModel(tiny_cfg(**{flag: True}))
        if isinstance(m2.gwtb, CompetitiveGWTBLayer):
            z = all((p.fc2.weight == 0).all() and (p.fc2.bias == 0).all() for p in m2.gwtb.bid_projectors)
            z = z and (m2.gwtb.score_head[-1].weight == 0).all() and (m2.gwtb.score_head[-1].bias == 0).all()
            check("GWTB zero-init survives global init", bool(z))
        else:
            check("GWTB zero-init survives global init", False, "flag did not build CompetitiveGWTBLayer")
    else:
        print("SKIP GWTB competitive flag not found in config")
except Exception as e:
    check("GWTB zero-init survives global init", False, repr(e))

# ---- 5. pad_mask=None parity (default path bit-identical) ------------------
cfg = tiny_cfg()
model = MTLNNModel(cfg).eval()
ids = torch.randint(0, 128, (2, 8))
with torch.no_grad():
    o1 = model(ids)["logits"]
    o2 = model(ids, pad_mask=torch.ones(2, 8, dtype=torch.bool))["logits"]
    o3 = model(ids)["logits"]
check("pad_mask None-parity (determinism)", (o1 == o3).all().item())
check("pad_mask all-True ≈ None", (o1 - o2).abs().max().item() < 1e-5,
      f"max diff {(o1 - o2).abs().max().item():.2e}")

# ---- 6. world-model surprise buffer persists through state_dict ------------
import dataclasses
fields = {f.name for f in dataclasses.fields(MTLNNConfig)}
wm_flag = next((f for f in ("use_world_model", "use_world_model_head") if f in fields), None)
if wm_flag:
    m = MTLNNModel(tiny_cfg(**{wm_flag: True})).train()
    m(ids, labels=ids)
    val = m.world_model_head.last_pred_error.item()
    sd = m.state_dict()
    check("WM buffer in state_dict", "world_model_head.last_pred_error" in sd)
    m2 = MTLNNModel(tiny_cfg(**{wm_flag: True}))
    m2.load_state_dict(sd)  # strict=True
    check("WM buffer restored", abs(m2.world_model_head.last_pred_error.item() - val) < 1e-6)
    # old-checkpoint compat: drop the buffer keys, strict load must still work
    sd2 = {k: v for k, v in sd.items() if "last_pred_error" not in k}
    m3 = MTLNNModel(tiny_cfg(**{wm_flag: True}))
    try:
        m3.load_state_dict(sd2)
        check("WM old-checkpoint strict load", True)
    except Exception as e:
        check("WM old-checkpoint strict load", False, repr(e))
else:
    print("SKIP world model flag not found")

# ---- 7. fast-weight store UPSERT: 40 writes -> latest snapshot -------------
from mt_lnn.fast_weight_store import FastWeightSessionStore, id_key
from mt_lnn.knowledge_memory import PersistentKnowledgeMemory

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
    db = os.path.join(td, "fw.db")
    store = FastWeightSessionStore(db_path=db, key_dim=16)
    key = id_key("sess-A", 16)
    for i in range(40):
        store.write_session("sess-A", key, {"step": torch.tensor(float(i))}, surprise=0.5)
    got = store.recall_session(key, expected_session_id="sess-A")
    snap = got[0][0] if got else None
    ok = snap is not None and abs(float(snap["step"]) - 39.0) < 1e-6
    check("UPSERT recall == latest (write #40)", ok,
          f"got step={float(snap['step']) if snap is not None else None}")
    n_rows = len(store.kb.all_meta())
    check("UPSERT single row per session", n_rows == 1, f"rows={n_rows}")
    store.kb.close()

# ---- 8. KB key_dim guard ----------------------------------------------------
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
    db = os.path.join(td, "kb.db")
    kb = PersistentKnowledgeMemory(16, db_path=db)
    kb.write(torch.randn(16), content="hello", meta={"a": 1})
    kb.close()
    try:
        PersistentKnowledgeMemory(32, db_path=db)
        check("KB key_dim mismatch fail-fast", False, "no error raised")
    except ValueError:
        check("KB key_dim mismatch fail-fast", True)
    # reopening with the right dim still works + weights_only round-trip
    kb2 = PersistentKnowledgeMemory(16, db_path=db)
    res = kb2.query(torch.randn(16), top_k=1)
    check("KB weights_only round-trip",
          len(res) >= 1 and any("hello" in str(r) for r in [res[0]]))
    kb2.close()

# ---- 9. atomic session save round-trip --------------------------------------
from mt_lnn.session_state import HFSessionState, load_session, save_session

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "s.json")
    s = HFSessionState(session_id="x")
    s.evidence_log.append({"k": "v"})
    save_session(s, p)
    save_session(s, p)  # overwrite path (os.replace onto existing target)
    s2 = load_session(p)
    check("atomic save round-trip", s2.evidence_log == [{"k": "v"}])
    leftovers = [f for f in os.listdir(td) if f.endswith(".tmp")]
    check("atomic save no tmp leftovers", not leftovers)

print("\n==== SUMMARY:", len(PASS), "pass /", len(FAIL), "fail ====")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
