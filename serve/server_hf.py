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
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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


@app.get("/research")
def research():
    return _static_page("research")


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
    print(f"[serve] attached MT adapters to layers {wrapped}")

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

    # Detect instruct / chat mode from the tokenizer's chat_template field.
    use_chat = getattr(tok, "chat_template", None) is not None
    # Pick identity honestly: the AwareLiquid/MT-LNN identity is only claimed
    # when a trained adapter is actually loaded; otherwise this is a labeled
    # baseline (the frozen base model). An explicit SYSTEM_PROMPT env overrides.
    default_system = _MTLNN_SYSTEM if adapter_loaded else _DEFAULT_SYSTEM
    system_prompt = os.environ.get("SYSTEM_PROMPT", default_system)
    print(f"[serve] chat_template={'yes' if use_chat else 'no'} | "
          f"identity={'mtlnn' if adapter_loaded else 'baseline'} | system_prompt set")

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
        "multimodal": False,
        "vision_tower": None,
        "multimodal_error": None,
    }


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


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    if req.max_new_tokens > MAX_NEW_CAP:
        raise HTTPException(400, f"max_new_tokens exceeds cap {MAX_NEW_CAP}")
    model, tok = _STATE["model"], _STATE["tok"]
    ids = _build_input_ids(req.prompt)
    t0 = time.time()
    out = model.generate(
        ids,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature,
        top_k=req.top_k if req.top_k > 0 else None,
        top_p=req.top_p,
        eos_token_id=tok.eos_token_id if req.stop_at_eos else None,
        pad_token_id=tok.pad_token_id,
    )
    new_ids = out[0, ids.shape[1]:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    dt = time.time() - t0
    return {
        "text": text,
        "tokens": new_ids.tolist(),
        "n_new_tokens": len(new_ids),
        "elapsed_s": round(dt, 4),
        "tok_per_s": round(len(new_ids) / dt, 2) if dt > 0 else None,
    }


@app.post("/v1/completions/stream")
def completions_stream(req: CompletionRequest):
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    if req.max_new_tokens > MAX_NEW_CAP:
        raise HTTPException(400, f"max_new_tokens exceeds cap {MAX_NEW_CAP}")
    model, tok = _STATE["model"], _STATE["tok"]
    ids = _build_input_ids(req.prompt)
    eos_id = tok.eos_token_id if req.stop_at_eos else None

    def _gen():
        past = None
        cur_ids = ids
        for _ in range(req.max_new_tokens):
            with torch.no_grad():
                out = model(cur_ids, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            past = out.past_key_values
            tid = _sample_next_token(logits, req)
            piece = tok.decode([tid], skip_special_tokens=True)
            yield f"data: {json.dumps({'token': tid, 'text': piece})}\n\n"
            if eos_id is not None and tid == eos_id:
                break
            cur_ids = torch.tensor([[tid]], device=_STATE["device"])

        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")
