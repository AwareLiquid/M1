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
    device_str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"[serve] loading base model: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if device_str == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = True

    wrapped = attach_mt_adapters(model, every=mt_every, n_protofilaments=mt_proto, n_time_scales=5)
    print(f"[serve] attached MT adapters to {wrapped} decoder layers")
    trainable = count_trainable_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"[serve] {trainable:,} adapter params / {total:,} total ({100*trainable/total:.3f}%)")

    adapter_loaded = False
    if adapter_ckpt and os.path.exists(adapter_ckpt):
        ck = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", {})
        missing, unexpected = model.load_state_dict(sd, strict=False)
        adapter_keys = [k for k in sd if "mt_adapter" in k or "lora_" in k]
        adapter_loaded = len(adapter_keys) > 0
        print(f"[serve] loaded {len(adapter_keys)} adapter tensors from {adapter_ckpt} "
              f"(step {ck.get('step', '?')}); missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[serve] no ADAPTER_CKPT -- using freshly initialized (identity-like) "
              "adapters; serving as a labeled BASELINE (frozen base model).")

    model = model.to(device_str).eval()
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
