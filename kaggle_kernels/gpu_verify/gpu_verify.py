#!/usr/bin/env python3
"""MT-LNN GPU verification + performance benchmark (Kaggle, branch physics-informed-head).

Phase 1  clone + deps
Phase 2  CPU fix-verification suite (scripts/diagnostics/_verify_fixes_2026_07_14.py, 19 checks)
Phase 3  GPU correctness: forward/backward/generate, hamiltonian no_grad,
         sparse prefill==incremental, pad_mask parity, fp16 autocast
Phase 4  GPU performance: default 125M config - train-step throughput +
         peak memory at T=512/1024/2048, decode tokens/s with cache
Phase 5  pytest fast subset (non-fatal, summary only)
"""
import os
import subprocess
import sys
import time

BRANCH = "physics-informed-head"
REPO = "https://github.com/everest-an/M1.git"
DIR = "/tmp/M1"  # NOT /kaggle/working: the clone must not become an output artifact

print("=== Phase 1: clone + deps ===", flush=True)
if not os.path.exists(DIR):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, DIR],
                   check=True, timeout=300)
os.chdir(DIR)
# Kaggle's preinstalled torch is built for sm_70+; the assigned P100 is sm_60.
# torch 2.4.1+cu121 still ships sm_60 kernels (same pin the repo's minimal_test
# kernel uses).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"],
               check=True, timeout=900)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "einops", "peft", "hypothesis", "pytest"], check=True, timeout=600)
sys.path.insert(0, DIR)

import torch  # noqa: E402

assert torch.cuda.is_available(), "no CUDA on this kernel"
DEV = "cuda"
GPU_NAME = torch.cuda.get_device_name(0)
print(f"torch {torch.__version__} | GPU: {GPU_NAME}", flush=True)

# ---------------------------------------------------------------- Phase 2
print("\n=== Phase 2: CPU fix-verification suite ===", flush=True)
r = subprocess.run([sys.executable, "scripts/diagnostics/_verify_fixes_2026_07_14.py"],
                   capture_output=True, text=True, timeout=1200)
tail = "\n".join(r.stdout.strip().splitlines()[-25:])
print(tail, flush=True)
PHASE2_OK = r.returncode == 0

# ---------------------------------------------------------------- Phase 3
print("\n=== Phase 3: GPU correctness ===", flush=True)
from mt_lnn.config import MTLNNConfig  # noqa: E402
from mt_lnn.model import MTLNNModel  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS" if cond else "FAIL"), name, detail, flush=True)


def tiny_cfg(**kw):
    base = dict(vocab_size=128, max_seq_len=64, d_model=52, n_layers=2,
                n_heads=13, n_kv_heads=1, d_head=4, n_protofilaments=13,
                map_hidden_dim=8, gwtb_compression_ratio=4, gwtb_n_heads=1)
    base.update(kw)
    return MTLNNConfig(**base)


torch.manual_seed(0)

# 3.1 default tiny model: forward/backward/generate on GPU
m = MTLNNModel(tiny_cfg()).to(DEV)
ids = torch.randint(0, 128, (2, 16), device=DEV)
try:
    m.train()
    out = m(ids, labels=ids)
    out["loss"].backward()
    check("gpu forward+backward finite", torch.isfinite(out["loss"]).item(),
          f"loss={out['loss'].item():.3f}")
except Exception as e:
    check("gpu forward+backward finite", False, repr(e))
try:
    m.eval()
    m.generate(ids[:, :4], max_new_tokens=4)
    check("gpu generate()", True)
except Exception as e:
    check("gpu generate()", False, repr(e))

# 3.2 hamiltonian head under no_grad on GPU (fix C-1)
try:
    mh = MTLNNModel(tiny_cfg(use_hamiltonian_world_model=True)).to(DEV).eval()
    with torch.no_grad():
        o = mh(ids)
    mh.generate(ids[:, :4], max_new_tokens=3)
    check("gpu hamiltonian no_grad + generate", torch.isfinite(o["logits"]).all().item())
except Exception as e:
    check("gpu hamiltonian no_grad + generate", False, repr(e))

# 3.3 sparse selection: prefill == incremental on GPU (fix H-2)
try:
    ms = MTLNNModel(tiny_cfg(sparse_resonance_kernel=True,
                             sparse_resonance_top_k=2)).to(DEV).eval()
    seq = torch.randint(0, 128, (1, 10), device=DEV)
    with torch.no_grad():
        full = ms(seq)["logits"]
        cache, chunks = None, []
        for t in range(seq.shape[1]):
            o = ms(seq[:, t:t + 1], cache=cache, use_cache=True)
            cache = o["cache"]
            chunks.append(o["logits"])
        incr = torch.cat(chunks, dim=1)
    d = (full - incr).abs().max().item()
    check("gpu sparse prefill==incremental", d < 1e-3, f"max diff {d:.2e}")
except Exception as e:
    check("gpu sparse prefill==incremental", False, repr(e))

# 3.4 pad_mask parity on GPU
try:
    mp = MTLNNModel(tiny_cfg()).to(DEV).eval()
    seq = torch.randint(0, 128, (2, 8), device=DEV)
    with torch.no_grad():
        a = mp(seq)["logits"]
        b = mp(seq, pad_mask=torch.ones(2, 8, dtype=torch.bool, device=DEV))["logits"]
    d = (a - b).abs().max().item()
    check("gpu pad_mask all-True == None", d < 1e-4, f"max diff {d:.2e}")
except Exception as e:
    check("gpu pad_mask all-True == None", False, repr(e))

# 3.5 fp16 autocast forward finite (incl. GTP fp32-clock fix at long offset)
try:
    ma = MTLNNModel(tiny_cfg(max_seq_len=4096)).to(DEV).eval()
    seq = torch.randint(0, 128, (1, 64), device=DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        o1 = ma(seq)["logits"]
        # long position offset via cache to exercise the GTP clock path
        cache = ma(seq, use_cache=True)["cache"]
        o2 = ma(seq[:, :1], cache=cache, use_cache=True)["logits"]
    check("gpu fp16 autocast finite", (torch.isfinite(o1).all() and torch.isfinite(o2).all()).item())
except Exception as e:
    check("gpu fp16 autocast finite", False, repr(e))

# ---------------------------------------------------------------- Phase 4
print("\n=== Phase 4: GPU performance (default ~125M config) ===", flush=True)
perf = {}
cfg = MTLNNConfig()  # shipped defaults
model = MTLNNModel(cfg).to(DEV)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"params: {n_params:.1f}M | d_model={cfg.d_model} layers={cfg.n_layers}", flush=True)

import gc


def train_bench(model, B, T):
    """One (B, T) training-throughput measurement. Returns (tok/s, peak MB)."""
    model.train()
    x = torch.randint(0, cfg.vocab_size, (B, T), device=DEV)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):  # warmup
        out = model(x, labels=x)
        out["loss"].backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 6
    for _ in range(iters):
        out = model(x, labels=x)
        out["loss"].backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return B * T * iters / (time.perf_counter() - t0), torch.cuda.max_memory_allocated() / 2**20


for T in (512, 1024, 2048):
    for B in (4, 1):
        oomed = False
        try:
            tput, mem = train_bench(model, B, T)
            perf[f"train T={T}"] = f"B={B}: {tput:,.0f} tok/s, peak {mem:,.0f} MB"
            print(f"  train B={B} T={T}: {tput:,.0f} tok/s | peak mem {mem:,.0f} MB", flush=True)
            break
        except torch.cuda.OutOfMemoryError:
            perf[f"train T={T}"] = f"OOM at B={B}" + ("" if B > 1 else " (unusable)")
            print(f"  train B={B} T={T}: OOM", flush=True)
            oomed = True
        # Recovery must happen OUTSIDE the except block: while the exception is
        # live, its traceback frames pin train_bench's locals (x, out, the
        # partial forward graph) AND module attrs hold aux tensors with graphs
        # (resonance.last_pred_error, _hebb_signal...). Only after the except
        # scope closes are those references dropped - v4 rebuilt the model
        # inside the handler and OOMed on the rebuild itself.
        if oomed:
            model.zero_grad(set_to_none=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = MTLNNModel(cfg).to(DEV)

# decode throughput: prefill 512, decode 128 with cache - on a FRESH eval
# model so no training-phase allocation can contaminate the peak-mem reading.
try:
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = MTLNNModel(cfg).to(DEV)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, cfg.vocab_size, (1, 512), device=DEV)
    with torch.no_grad():
        out = model(x, use_cache=True)
        cache = out["cache"]
        nxt = out["logits"][:, -1:].argmax(-1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 128
        for _ in range(N):
            o = model(nxt, cache=cache, use_cache=True)
            cache = o["cache"]
            nxt = o["logits"][:, -1:].argmax(-1)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / 2**20
    perf["decode"] = f"{N / dt:,.1f} tok/s, peak {mem:,.0f} MB"
    print(f"  decode (prefill 512 + 128 new): {N / dt:,.1f} tok/s | peak mem {mem:,.0f} MB", flush=True)
except Exception as e:
    perf["decode"] = f"ERROR {e!r}"
    print(f"  decode ERROR: {e!r}", flush=True)

del model
torch.cuda.empty_cache()

# ---------------------------------------------------------------- Phase 5
print("\n=== Phase 5: pytest fast subset (non-fatal) ===", flush=True)
r = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-m", "not slow", "-q",
     "-p", "no:cacheprovider"],
    capture_output=True, text=True, timeout=3600,
)
lines = (r.stdout + r.stderr).strip().splitlines()
print("\n".join(lines[-30:]), flush=True)
# Env-noise diagnosis: the v2/v3 runs showed ModuleNotFound failures that do
# not reproduce locally - print ONE full traceback to identify the module.
if lines and " failed" in lines[-1]:
    r2 = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_streaming_state.py", "-x", "-q",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=600,
    )
    tb = (r2.stdout + r2.stderr).strip().splitlines()
    print("\n--- first failing traceback (env diagnosis) ---", flush=True)
    print("\n".join(tb[-25:]), flush=True)

# ---------------------------------------------------------------- summary
print("\n" + "=" * 60)
print(f"GPU: {GPU_NAME}")
print(f"Phase 2 (CPU fix suite): {'OK' if PHASE2_OK else 'FAILED'}")
print(f"Phase 3 (GPU correctness): {len(PASS)} pass / {len(FAIL)} fail",
      ("FAILED: " + ", ".join(FAIL)) if FAIL else "")
print("Phase 4 (perf):")
for k, v in perf.items():
    print(f"  {k}: {v}")
print(f"Phase 5 (pytest): {'see tail above'}")
print("=" * 60)
if FAIL or not PHASE2_OK:
    sys.exit(1)
print("ALL GPU CHECKS PASSED")
