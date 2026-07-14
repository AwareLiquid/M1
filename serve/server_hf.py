#!/usr/bin/env python3
"""
serve/server_hf.py -- MT-LNN adapter demo server (multilingual).

Loads a frozen HuggingFace causal LM (default: Qwen2.5-0.5B-Instruct)
with MT-LNN residual adapters attached, then serves the same REST API as
serve/server.py so the frontend at serve/static/index.html works unchanged.

Instruct models (those with a tokenizer.chat_template) are used in chat mode:
prompts are automatically wrapped with apply_chat_template so the model
responds as an assistant rather than doing raw text continuation.

Run (fresh adapters, Qwen2.5-0.5B-Instruct -- multilingual):
    python -m uvicorn serve.server_hf:app --host 0.0.0.0 --port 8000

Run (explicit model / adapter checkpoint):
    BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    ADAPTER_CKPT=checkpoints/adapter_002000.pt \
        python -m uvicorn serve.server_hf:app --host 0.0.0.0 --port 8000

Environment variables:
    BASE_MODEL      HF model id or local path
                    (default: Qwen/Qwen2.5-0.5B-Instruct)
    ADAPTER_CKPT    Adapter checkpoint from train_llama_mt_adapter.py (optional)
    MT_EVERY        Attach an MT adapter after every N decoder layers (default: 4)
    MT_PROTO        Number of protofilaments per MT adapter (default: 13)
    SYSTEM_PROMPT   System message for instruct models
                    (default: "You are AwareLiquid, a helpful multilingual
                     assistant powered by MT-LNN liquid neural dynamics.")
    DEVICE          cpu | cuda (default: auto)
    MAX_NEW_TOKENS_CAP  Hard cap per request (default: 512)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_NEW_CAP = int(os.environ.get("MAX_NEW_TOKENS_CAP", "512"))

# Honest default identity. When NO trained MT-LNN adapter is loaded, the model
# is just the frozen HF base model (the residual adapters are identity-like
# no-ops), so it must NOT claim to be "powered by MT-LNN". We only attach the
# MT-LNN identity when a real ADAPTER_CKPT is loaded (see _startup). The default
# below is a neutral, honest baseline-assistant prompt.
_DEFAULT_SYSTEM = (
    "你是一个乐于助人的多语言助手。"
    "请始终使用用户所用的语言回复：用户用中文提问就用中文回答，用英文提问就用英文回答。\n"
    "You are a helpful multilingual assistant. Always reply in the same "
    "language the user writes in."
)

# Identity used ONLY when a trained MT-LNN adapter checkpoint is actually loaded.
_MTLNN_SYSTEM = (
    "你是 AwareLiquid，一个在冻结基座模型上叠加 MT-LNN 液态神经网络残差适配器的多语言助手。"
    "请始终使用用户所用的语言回复。\n"
    "You are AwareLiquid, a multilingual assistant that augments a frozen base "
    "model with trained MT-LNN liquid-dynamics residual adapters. Always reply "
    "in the same language the user writes in."
)

app = FastAPI(title="AwareLiquid HF-Adapter Server", version="1.0")
_STATE: dict = {}

app.add_middleware(GZipMiddleware, minimum_size=1024)

_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none';"
)

class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        path = request.url.path
        if re.search(r'\.(svg|png|jpg|ico|woff2?)$', path):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif path.startswith("/v1") or path in ("/health",):
            response.headers["Cache-Control"] = "no-store"
        elif path in ("/", "/about", "/research"):
            response.headers["Cache-Control"] = "no-cache"
        return response

app.add_middleware(_SecurityHeadersMiddleware)

if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def index():
    idx = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx, media_type="text/html")
    raise HTTPException(404, "frontend not built")


def _static_page(name: str) -> FileResponse:
    """Serve a standalone static HTML page (about/research) by base name."""
    path = os.path.join(_STATIC_DIR, f"{name}.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    raise HTTPException(404, f"{name} page not built")


@app.get("/about")
def about():
    return _static_page("about")


@app.get("/api")
def api_docs():
    return _static_page("api")


@app.get("/research")
def research():
    return _static_page("research")


@app.get("/privacy")
def privacy():
    return _static_page("privacy")


@app.get("/terms")
def terms():
    return _static_page("terms")


@app.get("/demo")
def demo():
    return _static_page("demo")


def _text_file(name: str) -> FileResponse:
    path = os.path.join(_STATIC_DIR, name)
    if os.path.exists(path):
        return FileResponse(path, media_type="text/plain; charset=utf-8")
    raise HTTPException(404, f"{name} not found")


@app.get("/llms.txt")
def llms_txt():
    return _text_file("llms.txt")


@app.get("/llms-full.txt")
def llms_full_txt():
    return _text_file("llms-full.txt")


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    path_404 = os.path.join(_STATIC_DIR, "404.html")
    if os.path.exists(path_404):
        content = open(path_404, encoding="utf-8").read()
        return HTMLResponse(content=content, status_code=404)
    return HTMLResponse("<h1>404 — Not Found</h1>", status_code=404)


@app.on_event("startup")
def _startup() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mt_lnn.llama_adapter import attach_mt_adapters, count_trainable_parameters

    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    adapter_ckpt = os.environ.get("ADAPTER_CKPT", "")
    mt_every = int(os.environ.get("MT_EVERY", "4"))
    mt_proto = int(os.environ.get("MT_PROTO", "13"))
    # RECIPE controls how the adapter graph is built so it MATCHES the checkpoint:
    #   phase5b -> MT adapters (every 4th layer) + PEFT LoRA on q/k/v/o
    #              (this is what train_llama_mt_adapter.py / the validated runs used,
    #               and what the checkpoints under checkpoints/llama_mt_adapter need)
    #   mt_only -> MT adapters only, no LoRA (legacy/no-checkpoint baseline path)
    recipe = os.environ.get("RECIPE", "phase5b" if adapter_ckpt else "mt_only").lower()
    device_str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # torch defaults its CPU intra-op thread pool to the DETECTED physical-core
    # count, which inside a container often comes back as 2 even on a 4-vCPU box
    # -- so OMP_NUM_THREADS alone does NOT raise it. Set it explicitly so CPU
    # GEMMs (the decode bottleneck) use every vCPU. THREADS env overrides; else
    # use OMP_NUM_THREADS, else os.cpu_count().
    if device_str != "cuda":
        n_threads = int(os.environ.get("THREADS")
                        or os.environ.get("OMP_NUM_THREADS")
                        or (os.cpu_count() or 1))
        torch.set_num_threads(n_threads)
        print(f"[serve] torch CPU threads set to {torch.get_num_threads()}")

    print(f"[serve] loading base model: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Pick the CPU dtype carefully. bf16 HALVES memory (~4.4 GB -> ~2.2 GB for a
    # 1.1B base) which matters on small-RAM boxes, BUT bf16 is only fast when the
    # CPU has hardware bf16 (AVX512-BF16 or AMX). On CPUs without it (e.g. plain
    # Skylake/AVX2) torch emulates bf16 in software -> SLOWER than fp32 (observed
    # ~14 s/token). So: bf16 only when the hardware supports it, else fp32 (the
    # safe CPU default, RAM permitting). CUDA keeps fp16. CPU_DTYPE env forces a
    # specific dtype (e.g. CPU_DTYPE=bfloat16 to trade speed for memory).
    if device_str == "cuda":
        dtype = torch.float16
    else:
        env_dt = os.environ.get("CPU_DTYPE")
        if env_dt:
            dtype = getattr(torch, env_dt)
        else:
            try:
                with open("/proc/cpuinfo") as f:
                    _flags = f.read()
                _hw_bf16 = ("avx512_bf16" in _flags) or ("amx_bf16" in _flags)
            except OSError:
                _hw_bf16 = False
            dtype = torch.bfloat16 if _hw_bf16 else torch.float32
    print(f"[serve] CPU/GPU dtype = {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    # Set use_cache on the raw base config BEFORE wrapping (PEFT proxies config).
    model.config.use_cache = True

    # Load the checkpoint FIRST so we can rebuild the exact adapter graph it was
    # trained with. The validated checkpoints are PEFT-wrapped
    # (base_model.model.model.*.{mt_adapter,lora_*}); those names only reproduce
    # if we (1) attach MT adapters then (2) wrap with PEFT LoRA using the SAME
    # hyperparameters -- exactly as train_llama_mt_adapter.py does. Building from
    # the checkpoint's own saved `args` guarantees the names line up.
    ck = None
    cargs: dict = {}
    if adapter_ckpt and os.path.exists(adapter_ckpt):
        ck = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
        cargs = ck.get("args", {}) or {}

    # The checkpoint's `args` dict does NOT record whether predictive coding was
    # enabled, but the resonance bank's W_pred tensors are only present in the
    # state_dict when it was. Detect them directly so we rebuild the SAME graph
    # the checkpoint was trained with -- otherwise those 6 tensors land in
    # `unexpected` and the honest guard refuses to claim the adapter is active.
    # W_pred is inference-inert (MTLNNLayer skips the predictive-coding branch
    # outside training), so enabling it changes nothing about generation output.
    ck_sd_preview = (ck or {}).get("state_dict", {}) if ck is not None else {}
    want_pc = any("W_pred" in k for k in ck_sd_preview)
    if want_pc:
        print("[serve] checkpoint carries resonance W_pred -> rebuilding graph "
              "with predictive coding to match (inference-inert).")

    # (1) MT adapters -- use checkpoint args when present, else env/defaults.
    # Adapter GENERATION comes from the checkpoint's recorded --adapter flag
    # (train_llama_mt_adapter.py), overridable via ADAPTER_KIND env for
    # checkpoint-less baselines. v2 = mt_lnn.mt_lnn_v2 (bottleneck factorized
    # projections + fast-weight memory, ~8.4M vs v1's 62.8M on TinyLlama).
    adapter_kind = str(
        cargs.get("adapter") or os.environ.get("ADAPTER_KIND", "v1")
    ).lower()
    if adapter_kind == "v2":
        from mt_lnn.mt_lnn_v2 import attach_mt_v2_adapters
        wrapped = attach_mt_v2_adapters(
            model,
            every=int(cargs.get("mt_every", mt_every)),
            n_protofilaments=int(cargs.get("mt_proto", mt_proto)),
            d_proto=int(cargs.get("v2_d_proto", 64)),
            n_time_scales=int(cargs.get("mt_scales", 5)),
            proj_rank=int(cargs.get("v2_rank", 128)),
            init_scale=float(cargs.get("mt_init_scale", 1e-3)),
            dropout=float(cargs.get("mt_dropout", 0.0)),
            selective_decay=bool(cargs.get("v2_selective", False)),
            use_fast_weight=not bool(cargs.get("v2_no_fw", False)),
            fast_weight_dim=int(cargs.get("v2_fw_dim", 64)),
            fast_weight_heads=int(cargs.get("v2_fw_heads", 1)),
        )
    else:
        wrapped = attach_mt_adapters(
            model,
            every=int(cargs.get("mt_every", mt_every)),
            n_protofilaments=int(cargs.get("mt_proto", mt_proto)),
            n_time_scales=int(cargs.get("mt_scales", 5)),
            map_hidden_dim=int(cargs.get("mt_map_hidden", 64)),
            dropout=float(cargs.get("mt_dropout", 0.0)),
            init_scale=float(cargs.get("mt_init_scale", 1e-3)),
            use_scan=not bool(cargs.get("mt_no_scan", False)),
            use_predictive_coding=want_pc,
        )
    print(f"[serve] attached MT adapters ({adapter_kind}) to layers {wrapped}")

    # (2) PEFT LoRA -- only if the checkpoint used it (cargs['lora']) or RECIPE
    # explicitly asks for phase5b. get_peft_model returns a NEW PeftModel that we
    # MUST keep (this is the object whose state_dict carries the base_model.model.
    # prefix the checkpoint expects).
    want_lora = bool(cargs.get("lora", recipe == "phase5b"))
    if want_lora:
        from peft import LoraConfig, get_peft_model
        targets = cargs.get("lora_targets", "q_proj,k_proj,v_proj,o_proj")
        if isinstance(targets, str):
            targets = targets.split(",")
        lcfg = LoraConfig(
            r=int(cargs.get("lora_r", 8)),
            lora_alpha=int(cargs.get("lora_alpha", 16)),
            lora_dropout=float(cargs.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, lcfg)
        print(f"[serve] applied PEFT LoRA (r={lcfg.r}, alpha={lcfg.lora_alpha}, targets={targets})")

    trainable = count_trainable_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"[serve] {trainable:,} adapter params / {total:,} total ({100*trainable/total:.3f}%)")

    adapter_loaded = False
    if ck is not None:
        sd = ck.get("state_dict", {})
        missing, unexpected = model.load_state_dict(sd, strict=False)
        adapter_keys = [k for k in sd if "mt_adapter" in k or "lora_" in k]
        matched = len(sd) - len(unexpected)
        # HONEST GUARD: only claim the adapter is active if the checkpoint tensors
        # actually mapped onto the model graph. A recipe/name mismatch shows up as
        # a non-zero `unexpected` count -> the tensors were silently dropped and we
        # are really serving the bare base model, so we must NOT claim MT-LNN.
        adapter_loaded = len(adapter_keys) > 0 and len(unexpected) == 0
        print(f"[serve] loaded checkpoint (step {ck.get('step', '?')}, base "
              f"{ck.get('model', '?')}): {matched}/{len(sd)} tensors matched; "
              f"missing(base)={len(missing)} unexpected={len(unexpected)}")
        if adapter_loaded:
            print(f"[serve] adapter ACTIVE: {len(adapter_keys)} MT/LoRA tensors loaded.")
        else:
            print(f"[serve] WARNING: {len(unexpected)} checkpoint tensors did NOT map "
                  f"onto the model graph -- recipe mismatch. Serving the BARE BASE "
                  f"MODEL as a labeled baseline (NOT claiming MT-LNN identity).")
    else:
        print("[serve] no ADAPTER_CKPT -- serving as a labeled BASELINE "
              "(frozen base model; identity-like no-op adapters).")

    # Force a uniform dtype across the whole graph. PEFT can create LoRA tensors
    # in fp32 even on a bf16 base; loading the fp32 checkpoint above is fine, but
    # we downcast everything here so all matmuls run in one dtype (and the bf16
    # memory saving actually applies to the LoRA/MT tensors too).
    model = model.to(device=device_str, dtype=dtype).eval()
    torch.set_grad_enabled(False)

    # Optional INT8 dynamic quantization (CPU only). Decode on CPU is memory-
    # bandwidth bound: int8 weights are ~1/4 the fp32 bytes and AVX2 fbgemm int8
    # GEMM beats fp32 -- a meaningful speedup on GPU-less boxes. We scope it to
    # the MLP projections (gate/up/down) ONLY. Those are plain nn.Linear, are NOT
    # wrapped by LoRA (which targets q/k/v/o) nor by the MT adapter, and account
    # for ~70% of the per-token weight bandwidth -- so they carry most of the
    # speedup. Quantizing the LoRA-wrapped attention projections instead breaks
    # PEFT, which reads base_layer.weight.dtype (a quantized Linear's .weight is a
    # method, not a tensor -> 'function' has no attribute 'dtype'). The resonance
    # einsums stay fp32, so the trained adapter dynamics are unchanged; this is an
    # inference-time optimization, not a model change. Gated by env so it is
    # trivially reversible (QUANTIZE=0 + restart) if quality regresses.
    if device_str == "cpu" and os.environ.get("QUANTIZE", "0").lower() in ("1", "true", "yes"):
        import torch.ao.quantization as tq
        targets = {
            name for name, m in model.named_modules()
            if isinstance(m, torch.nn.Linear)
            and name.split(".")[-1] in ("gate_proj", "up_proj", "down_proj")
        }
        if targets:
            model = tq.quantize_dynamic(model, targets, dtype=torch.qint8)
            print(f"[serve] INT8 dynamic quantization applied to {len(targets)} MLP Linear layers.")
        else:
            print("[serve] QUANTIZE set but no MLP gate/up/down Linear layers found; skipped.")

    # Streaming recurrent state (train/serve consistency fix): during KV-cached
    # decode each T=1 forward used to hit the MT adapter with ZERO state, so the
    # recurrence it was trained with degraded to a per-token gated FFN. With
    # streaming on, adapters carry state across decode steps exactly like the
    # training-time full-sequence scan (parity-tested to 2e-7 in
    # tests/test_streaming_state.py). Endpoints reset the stream at each
    # request start. STREAM_STATE=0 restores the old stateless behaviour.
    if os.environ.get("STREAM_STATE", "1").lower() in ("1", "true", "yes"):
        from mt_lnn.llama_adapter import set_adapter_streaming
        n_streaming = set_adapter_streaming(model, True)
        if n_streaming:
            print(f"[serve] adapter streaming state ENABLED on {n_streaming} adapters.")

    # Detect instruct / chat mode from the tokenizer's chat_template field.
    use_chat = getattr(tok, "chat_template", None) is not None
    # Pick identity honestly: the AwareLiquid/MT-LNN identity is only claimed
    # when a trained adapter is actually loaded; otherwise this is a labeled
    # baseline (the frozen base model). An explicit SYSTEM_PROMPT env overrides.
    default_system = _MTLNN_SYSTEM if adapter_loaded else _DEFAULT_SYSTEM
    system_prompt = os.environ.get("SYSTEM_PROMPT", default_system)
    print(f"[serve] chat_template={'yes' if use_chat else 'no'} | "
          f"identity={'mtlnn' if adapter_loaded else 'baseline'} | system_prompt set")

    # Optional self-thinking decode (mt_lnn.thinking): routes each token through
    # the deliberation policy (LOCAL / SELF_CRITIQUE self-consistency vote /
    # CLOUD-degrade) and returns a per-step thinking trace. Off by default so the
    # default decode path is unchanged; THINKING=1 turns it on globally and each
    # request can still override via the `think` field. This is the FIRST higher
    # cognitive module wired into serving -- it is honest about what it does (a
    # token-level self-consistency re-decode on uncertain steps, no extra model
    # forwards), and it does NOT fabricate facts (CLOUD degrades to self-critique
    # when no cloud client is wired).
    thinking_enabled = os.environ.get("THINKING", "0").lower() in ("1", "true", "yes")
    # Entropy thresholds gate the self-critique band (low <= H < high -> re-decode
    # via self-consistency vote). deliberation.py's library defaults (3.0/5.0) are
    # too high for this 1.1B's actual next-token entropy scale -- measured mean
    # entropy on confident answers is ~0.1-1.0 nats, so 3.0 leaves the mechanism
    # INERT (every token routes LOCAL). 0.6/4.0 was tuned on held-out prompts to
    # actually engage self-critique on uncertain tokens (verified: 8-23 critique
    # steps, 4-9 token revisions per ~35-token answer) without firing on
    # high-confidence spans. Override per-model via env.
    think_low = float(os.environ.get("THINK_ENTROPY_LOW", "0.6"))
    think_high = float(os.environ.get("THINK_ENTROPY_HIGH", "4.0"))
    think_samples = int(os.environ.get("THINK_SAMPLES", "5"))
    print(f"[serve] self-thinking decode: {'ON' if thinking_enabled else 'off'} "
          f"(low={think_low} high={think_high} samples={think_samples})")

    # Optional causal-consistency check (mt_lnn.causality.CausalConsistencyChecker).
    # Feeds the per-step hidden state to a TRAJECTORY monitor; a broken trajectory
    # (consistency < floor) forces SELF_CRITIQUE even on low-entropy tokens. The
    # wiring is mechanically correct and gated on its own env (applied inside the
    # THINKING path, since it feeds the same router).
    #
    # HONEST CALIBRATION NOTE -- OFF by default, and NOT enabled in M1 prod.
    # This detector was designed for the NATIVE MT-LNN recurrent state (continuous
    # LTC/ODE dynamics, where a trajectory jump is a real semantic break). A frozen
    # HF Transformer (TinyLlama/Qwen) recomputes its hidden state fresh every token,
    # so per-step novelty is high regardless of meaning. A local calibration
    # (_calib_causal.py) measured the score distribution on a coherent prompt vs a
    # deliberate topic-switch prompt for BOTH methods: cosine saturates (no
    # separation), and subspace gives ~0.1 for BOTH coherent and switch text (the
    # switch is not less consistent) -> there is NO floor that discriminates on this
    # base, and floor=0.3 just fires on ~every token (noise, not signal). So on the
    # M1 HF-adapter stack this is provided for experimentation / the native O1 path
    # it was built for, and we do NOT claim it catches hallucinations here. Enable
    # only with CAUSAL_CHECK=1 if you have re-calibrated for your model.
    causal_check = os.environ.get("CAUSAL_CHECK", "0").lower() in ("1", "true", "yes")
    causal_method = os.environ.get("CAUSAL_METHOD", "subspace")
    causal_floor = float(os.environ.get("CAUSAL_FLOOR", "0.3"))
    causal_window = int(os.environ.get("CAUSAL_WINDOW", "8"))
    print(f"[serve] causal-consistency check: {'ON' if causal_check else 'off'} "
          f"(method={causal_method} floor={causal_floor} window={causal_window})")

    # Optional persistent declarative memory (mt_lnn.knowledge_memory).
    # A local, content-addressable long-term KNOWLEDGE store (SQLite) that is
    # queried by similarity and pulled in on demand -- the complement to the
    # model's procedural weights and the live recurrent/working state. Text is
    # encoded to a key by mean-pooling the model's final hidden state; retrieval
    # uses the store's anisotropy-robust CENTERED cosine (plain cosine on
    # LM-pooled keys collapses to a near-universal nearest neighbour -- verified
    # 3/6 wrong vs 4/4 correct after centering, see _sweep_encoder.py). Enabled by
    # setting KB_PATH to a db file (":memory:" for ephemeral). When unset, all the
    # memory endpoints report disabled and completions are unaffected.
    #
    # HONEST SCOPE: this is a real retrieval-augmentation tier, but it only helps
    # when the store actually CONTAINS relevant facts -- an EMPTY store is a strict
    # no-op (we never fabricate a "fact"). The mean-pooled-hidden-state key is a
    # lightweight semantic index, not a dedicated sentence-embedding model; swap in
    # a real embedder for large/precision-critical bases.
    kb = None
    kb_key_dim = None
    kb_path = os.environ.get("KB_PATH", "").strip()
    kb_score_floor = float(os.environ.get("KB_SCORE_FLOOR", "0.15"))
    kb_top_k = int(os.environ.get("KB_TOP_K", "3"))
    if kb_path:
        from mt_lnn.knowledge_memory import PersistentKnowledgeMemory
        kb_max = os.environ.get("KB_MAX_ENTRIES")
        # Probe-encode to learn the key dimension from THIS model/graph (robust to
        # PEFT/quantization wrapping, which can hide config.hidden_size).
        probe = _encode_key_with("dimension probe", model, tok, device_str)
        kb_key_dim = int(probe.numel())
        kb = PersistentKnowledgeMemory(
            key_dim=kb_key_dim, db_path=kb_path,
            max_entries=int(kb_max) if kb_max else None,
        )
        print(f"[serve] declarative memory: ON (db={kb_path} key_dim={kb_key_dim} "
              f"entries={len(kb)} score_floor={kb_score_floor} top_k={kb_top_k})")
    else:
        print("[serve] declarative memory: off (set KB_PATH to enable)")

    # Optional automatic EPISODIC CONVERSATION memory (mt_lnn.conversation_memory).
    # The "remembers you" tier: the user's own statements are stored as they are
    # said and the relevant ones are recalled on later turns (persisted to SQLite,
    # so it survives restarts). Distinct from the declarative KB above in BOTH
    # purpose and encoder: this uses a dedicated sentence-embedding model
    # (mt_lnn.sentence_encoder, multilingual-e5-small by default) because the chat
    # model's mean-pooled hidden state cannot separate first-person paraphrases
    # (_diag_conv_mem.py showed "I love hiking" always loses to the anisotropic
    # universal nearest neighbour). Its own store / key_dim, so it does not disturb
    # the KB.
    #
    # TYPED CARDS (mt_lnn.memory_cards, borrowed from the Awareness-Market design):
    # when enabled, each stored statement is tagged with a card_type (identity /
    # preference / emotion / plan / relationship / health / detail) and recall
    # boosts a hit whose type matches the query's classified type. This is what
    # rescues bilingual recall RANKING: under e5, several same-language statements
    # create "magnet" sentences that win raw cosine regardless of meaning (a
    # Chinese allergy question would otherwise recall the emotion sentence); the
    # type boost promotes the correct-type memory back to the top (verified 0/3 ->
    # 3/3 in _test_memory_cards.py). HONEST BOUND: the typing fixes ranking, NOT
    # off-topic relevance gating -- e5's compressed high-cosine band means no
    # absolute score_floor cleanly rejects an off-topic Chinese query, and off-
    # topic queries are themselves confidently mis-typed. Floor is e5-calibrated
    # (~0.75 for English; weaker for Chinese -- documented, not over-claimed).
    #
    # HONEST SCOPE: this is AUTOMATED RETRIEVAL (RAG) over past user utterances --
    # it makes the assistant recall what you told it, and the card_type is a
    # lightweight semantic TAG (it embeds near emotional/health/... phrasings), NOT
    # affect or intent understanding. It adds NO reasoning, emotion, or self-
    # awareness; a 0.5B model with typed episodic recall is still a 0.5B model.
    # Off by default; set CONV_MEMORY=1 + CONV_DB_PATH.
    conv_mem = None
    conv_enabled = os.environ.get("CONV_MEMORY", "0").lower() in ("1", "true", "yes")
    conv_db = os.environ.get("CONV_DB_PATH", "").strip()
    conv_floor = float(os.environ.get("CONV_SCORE_FLOOR", "0.75"))
    conv_top_k = int(os.environ.get("CONV_TOP_K", "3"))
    # Typed cards on by default when memory is on (the "情感等类型" feature); set
    # CONV_TYPED=0 to fall back to the untyped (purely-similarity) behaviour.
    conv_typed = os.environ.get("CONV_TYPED", "1").lower() in ("1", "true", "yes")
    conv_type_boost = float(os.environ.get("CONV_TYPE_BOOST", "0.15"))
    if conv_enabled and conv_db:
        from mt_lnn.knowledge_memory import PersistentKnowledgeMemory
        from mt_lnn.conversation_memory import EpisodicConversationMemory
        from mt_lnn.sentence_encoder import SentenceEncoder
        conv_model_id = os.environ.get("CONV_MODEL", "intfloat/multilingual-e5-small")
        conv_enc = SentenceEncoder(model_id=conv_model_id, device="cpu")
        conv_max = os.environ.get("CONV_MAX_ENTRIES")
        conv_store = PersistentKnowledgeMemory(
            key_dim=conv_enc.dim, db_path=conv_db,
            max_entries=int(conv_max) if conv_max else None,
        )
        # Optional typed-card classifier (same shared encoder, asymmetric).
        conv_clf = None
        if conv_typed:
            from mt_lnn.memory_cards import MemoryCardClassifier
            conv_clf = MemoryCardClassifier(
                conv_enc.as_fn(is_query=False),
                query_encode_fn=conv_enc.as_fn(is_query=True))
        # e5 is isotropic enough for English -> no centering; asymmetric query
        # encoder (prefix on the query side) widens the relevant/off-topic gap.
        conv_mem = EpisodicConversationMemory(
            conv_store, conv_enc.as_fn(is_query=False),
            score_floor=conv_floor, center=False,
            query_encode_fn=conv_enc.as_fn(is_query=True),
            classifier=conv_clf, type_boost=conv_type_boost,
        )
        print(f"[serve] conversation memory: ON (db={conv_db} "
              f"model={conv_model_id} key_dim={conv_enc.dim} "
              f"entries={len(conv_store)} floor={conv_floor} top_k={conv_top_k} "
              f"typed={'on' if conv_typed else 'off'} boost={conv_type_boost})")
    elif conv_enabled and not conv_db:
        print("[serve] conversation memory: requested but CONV_DB_PATH unset -> off")
    else:
        print("[serve] conversation memory: off (set CONV_MEMORY=1 + CONV_DB_PATH)")

    _STATE.update(
        model=model, tok=tok, device=device_str,
        base_model=base_model,
        adapter_ckpt=adapter_ckpt or None,
        n_params=total,
        adapter_params=trainable,
        use_chat=use_chat,
        system_prompt=system_prompt,
        adapter_loaded=adapter_loaded,
        is_baseline=not adapter_loaded,
        thinking_enabled=thinking_enabled,
        think_low=think_low,
        think_high=think_high,
        think_samples=think_samples,
        causal_check=causal_check,
        causal_method=causal_method,
        causal_floor=causal_floor,
        causal_window=causal_window,
        kb=kb,
        kb_key_dim=kb_key_dim,
        kb_path=kb_path or None,
        kb_score_floor=kb_score_floor,
        kb_top_k=kb_top_k,
        conv_mem=conv_mem,
        conv_db=conv_db or None,
        conv_score_floor=conv_floor,
        conv_top_k=conv_top_k,
        ready=True,
    )
    print(f"[serve] ready | {total/1e9:.2f}B params | device={device_str}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = Field(64, ge=1)
    do_sample: bool = True
    temperature: float = Field(0.8, gt=0.0)
    top_k: int = 0
    top_p: float = 0.9
    stop_at_eos: bool = True
    # Per-request override for self-thinking decode. None -> use the server
    # default (THINKING env). True/False -> force on/off for this request.
    think: Optional[bool] = None
    # Per-request toggle for declarative-memory retrieval augmentation. None ->
    # on whenever a store is loaded; True/False -> force. A no-op when no store is
    # loaded or the store has no hit above the score floor (never fabricates).
    use_memory: Optional[bool] = None
    # Cross-session fast-weight memory: when FW_SESSION_STORE=1 AND a session_id
    # is supplied, the adapter's (F,z) recurrent state is RESTORED from the prior
    # turn of this session before generating and SNAPSHOTTED back after — the
    # "M1 remembers this conversation across requests" feature, built on the
    # lossless snapshot/restore proven in benchmarks/cross_session_recall.py.
    # A no-op when the store is off or the session_id is unseen.
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Cross-session fast-weight memory (env-gated; default off -> zero regression)
# ---------------------------------------------------------------------------

_FW_KEY_DIM = 256

# The served model is a SINGLE shared instance whose MT adapters hold MUTABLE
# per-request streaming state (_stream_h/_stream_fw). FastAPI runs the sync
# generation endpoints on an anyio threadpool, so without serialization two
# concurrent requests interleave per-token reads/writes of that shared state and
# corrupt each other's decode (and, with the session store, persist cross-session
# garbage). This lock serializes ALL model-touching generation. Held across a
# stream's whole token loop — the shared adapter state cannot be time-shared, so
# concurrent streams must run one at a time anyway.
_MODEL_LOCK = threading.Lock()
_FW_STORE_LOCK = threading.Lock()


def _gen_lock():
    """FastAPI yield-dependency that serializes a whole request on the shared
    model: setup acquires _MODEL_LOCK, teardown releases it AFTER the response
    (incl. a streaming body) is fully sent. Applied to the non-inline
    generation endpoints so the shared adapter streaming state is never touched
    by two requests at once. (threading.Lock, not asyncio: sync endpoints run on
    the anyio threadpool; a Lock may be released by a different thread than
    acquired, which is safe here.)"""
    _MODEL_LOCK.acquire()
    try:
        yield
    finally:
        _MODEL_LOCK.release()


def _fw_store():
    """Lazily create the (F,z) session store on first use, thread-safe
    (double-checked). FW_SESSION_STORE=1 enables it; otherwise every session
    call is a clean per-request reset (the unchanged default)."""
    if _STATE.get("fw_store_init"):
        return _STATE.get("fw_store")
    with _FW_STORE_LOCK:
        if _STATE.get("fw_store_init"):
            return _STATE.get("fw_store")
        _STATE["fw_store"] = None
        enabled = os.environ.get("FW_SESSION_STORE", "0").lower() in ("1", "true", "yes")
        _STATE["fw_store_init"] = True
        if enabled:
            try:
                from mt_lnn.fast_weight_store import FastWeightSessionStore
                path = os.environ.get("FW_SESSION_DB", "fw_sessions.sqlite")
                _STATE["fw_store"] = FastWeightSessionStore(db_path=path, key_dim=_FW_KEY_DIM)
                print(f"[serve] cross-session fast-weight store ENABLED at {path}")
            except Exception as e:   # never let memory wiring break generation
                print(f"[serve] fast-weight store init failed ({e}); disabled")
    return _STATE.get("fw_store")


def _session_enter(model, req) -> None:
    """Restore this session's fast-weight (F,z) if stored, else a clean reset.
    Always leaves the adapters in a valid state for a fresh generation."""
    from mt_lnn.llama_adapter import (reset_adapter_streams,
                                      restore_adapter_streams)
    reset_adapter_streams(model)
    store = _fw_store()
    sid = getattr(req, "session_id", None)
    if store is not None and sid:
        try:
            from mt_lnn.fast_weight_store import id_key
            hits = store.recall_session(id_key(sid, _FW_KEY_DIM),
                                        expected_session_id=sid, center=False)
            if hits:
                restore_adapter_streams(model, hits[0][0])
        except Exception as e:
            print(f"[serve] session restore failed for {sid!r}: {e}")


def _session_exit(model, req, surprise: float = 0.0) -> None:
    """Snapshot this session's fast-weight state so a later request resumes it.
    ``surprise`` (mean generated-token entropy, a live uncertainty signal) is
    recorded so offline consolidate() can keep 'what surprised the model'
    rather than the longest session."""
    store = _fw_store()
    sid = getattr(req, "session_id", None)
    if store is not None and sid:
        try:
            from mt_lnn.fast_weight_store import id_key
            from mt_lnn.llama_adapter import snapshot_adapter_streams
            store.write_session(sid, id_key(sid, _FW_KEY_DIM),
                                snapshot_adapter_streams(model),
                                surprise=float(surprise))
        except Exception as e:
            print(f"[serve] session snapshot failed for {sid!r}: {e}")


def _token_entropy(logits) -> float:
    """Shannon entropy (nats) of one next-token distribution — a per-step
    surprise/uncertainty scalar (the same signal the deliberation router gates
    on)."""
    logp = torch.log_softmax(logits.float().reshape(-1), dim=-1)
    return float(-(logp.exp() * logp).sum().item())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok" if _STATE.get("ready") else "starting",
        "device": _STATE.get("device"),
        "multimodal": False,
    }


@app.get("/v1/model")
def model_info():
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    return {
        "n_params": _STATE["n_params"],
        "adapter_params": _STATE["adapter_params"],
        "base_model": _STATE["base_model"],
        "adapter_ckpt": _STATE["adapter_ckpt"],
        "device": _STATE["device"],
        "chat_mode": _STATE.get("use_chat", False),
        # Honest labeling: True when no trained MT-LNN adapter is loaded, i.e.
        # this is the frozen base model served as a comparison baseline.
        "is_baseline": _STATE.get("is_baseline", True),
        "adapter_loaded": _STATE.get("adapter_loaded", False),
        "thinking_enabled": _STATE.get("thinking_enabled", False),
        "causal_check": _STATE.get("causal_check", False),
        "memory_enabled": _STATE.get("kb") is not None,
        "memory_entries": len(_STATE["kb"]) if _STATE.get("kb") is not None else 0,
        "conv_memory_enabled": _STATE.get("conv_mem") is not None,
        "conv_memory_entries": (
            len(_STATE["conv_mem"]) if _STATE.get("conv_mem") is not None else 0),
        "multimodal": False,
        "vision_tower": None,
        "multimodal_error": None,
    }


class MemoryWriteRequest(BaseModel):
    content: str                       # the fact / knowledge text to store
    key: Optional[str] = None          # text to index by (defaults to content)
    meta: Optional[dict] = None        # optional picklable metadata


class MemoryQueryRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1)


@app.get("/v1/memory/stats")
def memory_stats():
    """Declarative-memory store status. Honest about being disabled when no
    KB_PATH is configured (rather than pretending an empty store exists)."""
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    kb = _STATE.get("kb")
    if kb is None:
        return {"enabled": False, "reason": "KB_PATH not set"}
    return {
        "enabled": True,
        "entries": len(kb),
        "key_dim": _STATE.get("kb_key_dim"),
        "db_path": _STATE.get("kb_path"),
        "score_floor": _STATE.get("kb_score_floor"),
        "top_k": _STATE.get("kb_top_k"),
    }


@app.post("/v1/memory/write")
def memory_write(req: MemoryWriteRequest, _lock=Depends(_gen_lock)):
    """Store a fact, indexed by its (model-encoded) key. Returns the row id.

    Locked: _encode_key runs a forward on the SHARED model under
    adapter_streaming_paused, which globally flips/restores the adapters'
    streaming state — interleaving with a generation (or another memory
    request) corrupts that state."""
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    kb = _STATE.get("kb")
    if kb is None:
        raise HTTPException(400, "declarative memory disabled (set KB_PATH)")
    if not req.content.strip():
        raise HTTPException(400, "content must be non-empty")
    key_text = req.key if (req.key and req.key.strip()) else req.content
    row_id = kb.write(_encode_key(key_text), req.content, meta=req.meta)
    return {"id": row_id, "entries": len(kb)}


@app.post("/v1/memory/query")
def memory_query(req: MemoryQueryRequest, _lock=Depends(_gen_lock)):
    """Retrieve the top-k stored facts most similar to the query (centered,
    anisotropy-robust scoring). Empty list when the store is empty.

    Locked: same reason as /v1/memory/write — the key encode forward toggles
    the shared adapters' streaming state."""
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    kb = _STATE.get("kb")
    if kb is None:
        raise HTTPException(400, "declarative memory disabled (set KB_PATH)")
    if len(kb) == 0:
        return {"hits": []}
    hits = kb.query(_encode_key(req.query), top_k=req.top_k, center=True)
    return {"hits": [{"content": str(c), "score": round(float(s), 3),
                      "meta": mt} for c, s, mt in hits]}


def _sample_next_token(logits: torch.Tensor, req: "CompletionRequest") -> int:
    """Pick the next token id from the final-position ``logits``.

    A pure function of the logits and the decoding knobs (greedy when
    ``do_sample`` is False, else temperature + top-k + nucleus sampling), so the
    streaming generator's selection logic is unit-testable without loading a
    model. ``logits`` is ``(1, vocab)``.
    """
    if not req.do_sample:
        return int(logits.argmax(-1).item())
    lg = logits / max(req.temperature, 1e-6)
    if req.top_k > 0:
        topk_vals, _ = lg.topk(min(req.top_k, lg.shape[-1]))
        lg = lg.masked_fill(lg < topk_vals[:, -1:], float("-inf"))
    if req.top_p < 1.0:
        sorted_lg, sorted_idx = lg.sort(descending=True)
        probs = sorted_lg.softmax(-1)
        # shift by one so the most probable token is always kept
        remove = probs.cumsum(-1) - probs > req.top_p
        sorted_lg = sorted_lg.masked_fill(remove, float("-inf"))
        lg = lg.scatter(1, sorted_idx, sorted_lg)
    return int(torch.multinomial(torch.softmax(lg, -1), 1).item())


@torch.no_grad()
def _encode_key_with(text: str, model, tok, device: str) -> torch.Tensor:
    """Encode *text* to a fixed-length key vector by mean-pooling the model's
    final hidden state over the (truncated) token sequence.

    Used both for the startup dimension probe and for memory write/query. Runs a
    single no-cache forward with output_hidden_states; cheap for the short fact /
    query strings a declarative store holds. Returns a CPU float32 (d_model,)
    tensor -- PersistentKnowledgeMemory L2-normalises it on write/query.
    """
    ids = tok(text or " ", return_tensors="pt", truncation=True,
              max_length=256).input_ids.to(device)
    # Paused: this side forward must not read or clobber a generation's
    # streaming adapter state.
    from mt_lnn.llama_adapter import adapter_streaming_paused
    with adapter_streaming_paused(model):
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][0].mean(dim=0).float().cpu()   # (d_model,)


def _encode_key(text: str) -> torch.Tensor:
    """State-reading wrapper around :func:`_encode_key_with` for the endpoints."""
    return _encode_key_with(text, _STATE["model"], _STATE["tok"], _STATE["device"])


def _build_input_ids(prompt: str) -> torch.Tensor:
    """Encode prompt, applying chat_template for instruct models."""
    tok, device = _STATE["tok"], _STATE["device"]
    if _STATE.get("use_chat"):
        messages = [
            {"role": "system", "content": _STATE["system_prompt"]},
            {"role": "user",   "content": prompt},
        ]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tok(text, return_tensors="pt").input_ids
    else:
        ids = tok.encode(prompt or "", return_tensors="pt")
        if ids.shape[1] == 0:
            ids = torch.tensor([[tok.bos_token_id or 1]])
    return ids.to(device)


def _augment_with_memory(req: "CompletionRequest") -> Tuple[str, dict]:
    """Build the memory-augmented prompt for *req* and the per-request memory
    metadata, shared by the buffered and streaming completion endpoints so both
    surfaces behave identically.

    Two strictly-additive, honest tiers:
      1. Declarative KB (mt_lnn.knowledge_memory) -- prepends recalled facts above
         the score floor (anisotropy-robust centered query). Absent/empty store or
         no hit -> prompt untouched (never fabricates).
      2. Episodic conversation memory (mt_lnn.conversation_memory) -- recalls the
         user's own earlier statements relevant to THIS turn, then observes the
         current turn for future ones. Recall runs on the PAST store; observe
         happens AFTER recall so the user never "recalls" what they just said.
         observe() is wrapped so a memory write can never break a completion.

    Returns ``(aug_prompt, meta)`` where meta carries memory_used / memory_hits /
    conv_memory_used / conv_memory_hits for the response body.
    """
    aug_prompt = req.prompt
    meta = {"memory_used": False, "memory_hits": [],
            "conv_memory_used": False, "conv_memory_hits": []}

    kb = _STATE.get("kb")
    want_mem = req.use_memory if req.use_memory is not None else (kb is not None)
    if want_mem and kb is not None and len(kb) > 0 and req.prompt.strip():
        floor = _STATE.get("kb_score_floor", 0.15)
        hits = kb.query(_encode_key(req.prompt),
                        top_k=_STATE.get("kb_top_k", 3), center=True)
        kept = [(c, s) for c, s, _ in hits if s >= floor]
        if kept:
            meta["memory_used"] = True
            meta["memory_hits"] = [{"content": str(c), "score": round(float(s), 3)}
                                   for c, s in kept]
            facts = "\n".join(f"- {c}" for c, _ in kept)
            aug_prompt = (
                "Use the following retrieved facts if relevant to answer the "
                f"question. If they are not relevant, ignore them.\n{facts}\n\n"
                f"{req.prompt}"
            )

    conv_mem = _STATE.get("conv_mem")
    want_conv = req.use_memory if req.use_memory is not None else (conv_mem is not None)
    if want_conv and conv_mem is not None and req.prompt.strip():
        recalled = conv_mem.recall(req.prompt, top_k=_STATE.get("conv_top_k", 3))
        if recalled:
            meta["conv_memory_used"] = True
            meta["conv_memory_hits"] = [{"content": c, "score": round(s, 3)}
                                        for c, s in recalled]
            lines = "\n".join(f"- {c}" for c, _ in recalled)
            aug_prompt = (
                "The user told you earlier:\n" + lines +
                "\nUse this to stay consistent and personal if relevant.\n\n"
                f"{aug_prompt}"
            )
        try:
            conv_mem.observe(req.prompt)
        except Exception as exc:  # never let memory writes break a completion
            print(f"[serve] conv-memory observe failed: {exc}")

    return aug_prompt, meta


@app.post("/v1/completions")
def completions(req: CompletionRequest, _lock=Depends(_gen_lock)):
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    if req.max_new_tokens > MAX_NEW_CAP:
        raise HTTPException(400, f"max_new_tokens exceeds cap {MAX_NEW_CAP}")
    model, tok = _STATE["model"], _STATE["tok"]

    aug_prompt, mem_meta = _augment_with_memory(req)
    memory_used = mem_meta["memory_used"]
    memory_hits = mem_meta["memory_hits"]
    conv_used = mem_meta["conv_memory_used"]
    conv_hits = mem_meta["conv_memory_hits"]

    ids = _build_input_ids(aug_prompt)
    t0 = time.time()

    # Restore this session's carried (F,z) state if a session_id is supplied
    # and the store is on; otherwise a clean per-request reset (the default).
    _session_enter(model, req)

    # Self-thinking decode path (mt_lnn.thinking) -- per-request `think`
    # overrides the server default. Routes each token through the deliberation
    # policy and attaches a thinking trace summary. Falls back transparently to
    # the plain generate() path below when off.
    want_think = req.think if req.think is not None else _STATE.get("thinking_enabled", False)
    if want_think:
        from mt_lnn.thinking import generate_with_thinking
        from mt_lnn.deliberation import DeliberationRouter, RouterThresholds
        router = DeliberationRouter(thresholds=RouterThresholds(
            low=_STATE.get("think_low", 3.0),
            high=_STATE.get("think_high", 5.0),
            consistency_floor=_STATE.get("causal_floor", 0.3),
        ))
        text, trace = generate_with_thinking(
            model, tok, req.prompt,
            input_ids=ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            n_critique_samples=_STATE.get("think_samples", 5),
            router=router,
            device=_STATE["device"],
            consistency_check=_STATE.get("causal_check", False),
            consistency_method=_STATE.get("causal_method", "subspace"),
            consistency_window=_STATE.get("causal_window", 8),
        )
        dt = time.time() - t0
        n_new = len(trace.steps)
        ents = [s.entropy for s in trace.steps if s.entropy is not None]
        _session_exit(model, req, surprise=(sum(ents) / len(ents)) if ents else 0.0)
        return {
            "text": text,
            "tokens": [s.token_id for s in trace.steps if s.token_id >= 0],
            "n_new_tokens": n_new,
            "elapsed_s": round(dt, 4),
            "tok_per_s": round(n_new / dt, 2) if dt > 0 else None,
            "thinking": trace.summary(),
            "memory_used": memory_used,
            "memory_hits": memory_hits,
            "conv_memory_used": conv_used,
            "conv_memory_hits": conv_hits,
        }

    want_session = _fw_store() is not None and getattr(req, "session_id", None)
    gen = model.generate(
        ids,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature,
        top_k=req.top_k if req.top_k > 0 else None,
        top_p=req.top_p,
        eos_token_id=tok.eos_token_id if req.stop_at_eos else None,
        pad_token_id=tok.pad_token_id,
        # only pay for scores when we need the surprise signal for a session
        return_dict_in_generate=bool(want_session),
        output_scores=bool(want_session),
    )
    out = gen.sequences if want_session else gen
    new_ids = out[0, ids.shape[1]:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    dt = time.time() - t0
    surprise = 0.0
    if want_session and gen.scores:
        surprise = sum(_token_entropy(s) for s in gen.scores) / len(gen.scores)
    _session_exit(model, req, surprise=surprise)   # persist (F,z) + surprise
    return {
        "text": text,
        "tokens": new_ids.tolist(),
        "n_new_tokens": len(new_ids),
        "elapsed_s": round(dt, 4),
        "tok_per_s": round(len(new_ids) / dt, 2) if dt > 0 else None,
        "memory_used": memory_used,
        "memory_hits": memory_hits,
        "conv_memory_used": conv_used,
        "conv_memory_hits": conv_hits,
    }


@app.post("/v1/completions/stream")
def completions_stream(req: CompletionRequest, _lock=Depends(_gen_lock)):
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    if req.max_new_tokens > MAX_NEW_CAP:
        raise HTTPException(400, f"max_new_tokens exceeds cap {MAX_NEW_CAP}")
    model, tok = _STATE["model"], _STATE["tok"]
    # Same memory augmentation as the buffered endpoint, so streaming "remembers
    # you" too. The recall/observe must run before the stream opens (observe is
    # one write, recall one query -- both cheap), then the augmented prompt drives
    # generation. The recalled metadata is emitted as the FIRST SSE event so a
    # client can show "recalled: ..." before any token arrives; it carries an
    # explicit "event":"memory" marker to distinguish it from token events (which
    # have token/text), keeping older token-only clients compatible.
    aug_prompt, mem_meta = _augment_with_memory(req)
    ids = _build_input_ids(aug_prompt)
    eos_id = tok.eos_token_id if req.stop_at_eos else None

    def _gen():
        # _MODEL_LOCK is held for the whole request by the _gen_lock dependency
        # (spans _augment_with_memory above AND this stream, released after the
        # response is fully sent). The try/finally guarantees the (F,z) snapshot
        # runs even on client disconnect (GeneratorExit), so the turn's state is
        # not lost.
        yield f"data: {json.dumps({'event': 'memory', **mem_meta})}\n\n"
        _session_enter(model, req)
        past = None
        cur_ids = ids
        ent_sum, ent_n = 0.0, 0
        try:
            for _ in range(req.max_new_tokens):
                with torch.no_grad():
                    out = model(cur_ids, past_key_values=past, use_cache=True)
                logits = out.logits[:, -1, :]
                past = out.past_key_values
                ent_sum += _token_entropy(logits); ent_n += 1
                tid = _sample_next_token(logits, req)
                piece = tok.decode([tid], skip_special_tokens=True)
                yield f"data: {json.dumps({'token': tid, 'text': piece})}\n\n"
                if eos_id is not None and tid == eos_id:
                    break
                cur_ids = torch.tensor([[tid]], device=_STATE["device"])
        finally:
            _session_exit(model, req,
                          surprise=(ent_sum / ent_n) if ent_n else 0.0)
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# OpenAI-compatible surface (/v1/chat/completions + /v1/models)
#
# This is the GATEWAY contract: relay panels (new-api / one-api class) speak
# the OpenAI schema to their upstream channels. The native endpoints above
# keep M1's own UI features (memory augmentation, thinking traces); this
# surface is deliberately plain — the caller owns the conversation, we honor
# their messages verbatim (no memory injection), reset the recurrent state
# per request, and answer in OpenAI chunk/response format.
# ---------------------------------------------------------------------------

_OAI_MODEL_ID = os.environ.get("OAI_MODEL_ID", "awareliquid-m1")


@app.get("/v1/models")
def oai_models():
    return {"object": "list", "data": [{
        "id": _OAI_MODEL_ID, "object": "model", "owned_by": "awareliquid",
    }]}


@app.post("/v1/chat/completions")
def oai_chat_completions(body: dict, _lock=Depends(_gen_lock)):
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "messages required")
    model, tok, device = _STATE["model"], _STATE["tok"], _STATE["device"]

    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens")
                     or 256)
    max_tokens = max(1, min(max_tokens, MAX_NEW_CAP))
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.95))
    do_sample = temperature > 0
    stream = bool(body.get("stream", False))

    # Honor caller-supplied system prompt; add ours only when absent.
    if not any(m.get("role") == "system" for m in messages):
        messages = ([{"role": "system", "content": _STATE["system_prompt"]}]
                    + messages)
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    except Exception as exc:
        raise HTTPException(400, f"bad messages: {exc}")
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    n_prompt = ids.shape[1]
    eos_id = tok.eos_token_id
    rid = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    from mt_lnn.llama_adapter import reset_adapter_streams
    reset_adapter_streams(model)

    class _P:  # adapter for _sample_next_token's attribute access
        pass
    sp = _P()
    sp.do_sample, sp.temperature, sp.top_k, sp.top_p = do_sample, temperature, 0, top_p

    if stream:
        def _gen():
            head = {"id": rid, "object": "chat.completion.chunk",
                    "created": created, "model": _OAI_MODEL_ID,
                    "choices": [{"index": 0,
                                 "delta": {"role": "assistant", "content": ""},
                                 "finish_reason": None}]}
            yield f"data: {json.dumps(head)}\n\n"
            past, cur = None, ids
            finish = "length"
            for _ in range(max_tokens):
                with torch.no_grad():
                    out = model(cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                tid = _sample_next_token(out.logits[:, -1, :], sp)
                if eos_id is not None and tid == eos_id:
                    finish = "stop"
                    break
                piece = tok.decode([tid], skip_special_tokens=True)
                chunk = {"id": rid, "object": "chat.completion.chunk",
                         "created": created, "model": _OAI_MODEL_ID,
                         "choices": [{"index": 0, "delta": {"content": piece},
                                      "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                cur = torch.tensor([[tid]], device=device)
            tail = {"id": rid, "object": "chat.completion.chunk",
                    "created": created, "model": _OAI_MODEL_ID,
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": finish}]}
            yield f"data: {json.dumps(tail)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_gen(), media_type="text/event-stream")

    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=eos_id, pad_token_id=tok.pad_token_id,
        )
    new_ids = out[0, n_prompt:]
    finish = ("stop" if eos_id is not None and len(new_ids)
              and new_ids[-1].item() == eos_id else "length")
    answer = tok.decode(new_ids, skip_special_tokens=True)
    return {
        "id": rid, "object": "chat.completion", "created": created,
        "model": _OAI_MODEL_ID,
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": n_prompt,
                  "completion_tokens": int(len(new_ids)),
                  "total_tokens": n_prompt + int(len(new_ids))},
    }
