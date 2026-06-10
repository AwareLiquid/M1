#!/usr/bin/env python3
"""
serve/server.py — minimal production-style inference server for the native
MT-LNN model (not the HF-adapter demo).

It wraps :meth:`mt_lnn.model.MTLNNModel.generate` behind a small FastAPI app
with health, model-info, and text/token completion endpoints, plus optional
token streaming (Server-Sent Events). Designed to run in the Docker image at
the repo root (`Dockerfile`).

Run locally
-----------
    # serve a trained checkpoint (config is embedded in the .pt)
    CKPT_PATH=checkpoints/final.pt TOKENIZER=gpt2 \
        python -m uvicorn serve.server:app --host 0.0.0.0 --port 8000

    # smoke without a checkpoint / tokenizer (byte-level, fresh tiny model)
    SMALL=1 python -m uvicorn serve.server:app --port 8000

Endpoints
---------
    GET  /health            → {"status": "ok", ...}
    GET  /v1/model          → model + tokenizer info
    POST /v1/completions     → {"prompt": "...", "max_new_tokens": 64, ...}
    POST /v1/completions/stream  (SSE: one JSON event per token)

Environment
-----------
    CKPT_PATH   checkpoint .pt with embedded config (else a fresh model)
    TOKENIZER   HF tokenizer id or local dir (default: gpt2)
    SMALL       "1" → tiny byte-level model (vocab 256), no HF tokenizer
    DEVICE      cpu | cuda  (default: auto)
    MAX_NEW_TOKENS_CAP  hard upper bound per request (default: 1024)
    ENABLE_MULTIMODAL   "1" → load a CLIP vision tower and expose
                        POST /v1/multimodal/completions (needs transformers,
                        Pillow, and a one-time CLIP weight download)
    CLIP_MODEL          CLIP id for the vision tower
                        (default: openai/clip-vit-base-patch32)
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import io
import json
import os
import sys
import time
from typing import List, Optional

# Repo root on path so `mt_lnn` imports resolve when run as `serve.server`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Static frontend (v0): minimal streaming chat UI served at the domain root so
# awareliquid.ai/ is a real interactive demo, not a 404. Same-origin → the page
# uses relative /v1/* paths, working both at localhost:8088 and via the tunnel.
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel

MAX_NEW_CAP = int(os.environ.get("MAX_NEW_TOKENS_CAP", "1024"))


# ---------------------------------------------------------------------------
# Tokenizer abstraction: HF BPE for real checkpoints, byte-level for smoke.
# ---------------------------------------------------------------------------

class _ByteTokenizer:
    """Trivial reversible byte-level tokenizer (vocab 256) for SMALL mode."""
    vocab_size = 256
    eos_token_id = None

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int], **_) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


def _load_tokenizer(small: bool):
    if small:
        return _ByteTokenizer()
    from transformers import AutoTokenizer
    name = os.environ.get("TOKENIZER", "gpt2")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_model(small: bool) -> MTLNNModel:
    ckpt_path = os.environ.get("CKPT_PATH")
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg_dict = ckpt.get("config")
        if cfg_dict is None:
            raise RuntimeError(f"{ckpt_path} has no embedded config.")
        # Only constructor args — d_proto / d_proto_total are field(init=False),
        # derived in __post_init__; passing them would raise TypeError.
        valid = {f.name for f in dataclasses.fields(MTLNNConfig) if f.init}
        cfg = MTLNNConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
        model = MTLNNModel(cfg)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[serve] loaded {ckpt_path} (step {ckpt.get('step', '?')}); "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        return model.eval()

    if small:
        cfg = MTLNNConfig(
            vocab_size=256, d_model=104, n_layers=2, n_heads=13, n_kv_heads=1,
            d_head=8, max_seq_len=256, gwtb_n_heads=1,
        )
    else:
        cfg = MTLNNConfig(
            vocab_size=50257, d_model=832, n_layers=12, n_heads=13, n_kv_heads=1,
            d_head=64, max_seq_len=2048, gwtb_n_heads=1,
        )
    print("[serve] no CKPT_PATH — built a FRESH (untrained) model.")
    return MTLNNModel(cfg).eval()


# ---------------------------------------------------------------------------
# App + lifespan state
# ---------------------------------------------------------------------------

app = FastAPI(title="MT-LNN Inference Server", version="1.0")
_STATE: dict = {}

# Serve the v0 frontend. Mount static assets under /static and the index at /.
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def index():
    idx = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx, media_type="text/html")
    raise HTTPException(404, "frontend not built")


@app.on_event("startup")
def _startup() -> None:
    small = os.environ.get("SMALL", "0") == "1"
    device = os.environ.get(
        "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.set_grad_enabled(False)
    model = _build_model(small).to(device)
    tok = _load_tokenizer(small)
    _STATE.update(
        model=model, tok=tok, device=device, small=small,
        n_params=sum(p.numel() for p in model.parameters()),
        ready=True,
    )
    print(f"[serve] ready | {_STATE['n_params']/1e6:.1f}M params | device={device}")

    # Optional multimodal frontend (CLIP vision tower → projector → inputs_embeds).
    _STATE["mm_ready"] = False
    _STATE["mm_error"] = None
    if os.environ.get("ENABLE_MULTIMODAL", "0") == "1":
        try:
            from mt_lnn.multimodal import CLIPModalityEncoder
            clip_name = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
            enc = CLIPModalityEncoder(
                d_model=model.config.d_model, model_name=clip_name
            ).to(device).eval()
            _STATE["mm_encoder"] = enc
            _STATE["mm_model"] = clip_name
            _STATE["mm_ready"] = True
            print(f"[serve] multimodal ready | vision tower={clip_name}")
        except Exception as e:  # noqa: BLE001 - degrade gracefully, keep text serving
            _STATE["mm_error"] = f"{type(e).__name__}: {e}"
            print(f"[serve] multimodal DISABLED ({_STATE['mm_error']})")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    prompt: str = ""
    input_ids: Optional[List[int]] = None
    max_new_tokens: int = Field(64, ge=1)
    do_sample: bool = True
    temperature: float = Field(0.8, gt=0.0)
    top_k: int = 0
    top_p: float = 0.9
    stop_at_eos: bool = True


def _encode(req: CompletionRequest) -> torch.Tensor:
    tok = _STATE["tok"]
    if req.input_ids is not None:
        ids = req.input_ids
    else:
        ids = tok.encode(req.prompt)
    if not ids:
        ids = [0]
    return torch.tensor([ids], dtype=torch.long, device=_STATE["device"])


def _eos_id(stop: bool) -> Optional[int]:
    return getattr(_STATE["tok"], "eos_token_id", None) if stop else None


class MultimodalRequest(BaseModel):
    prompt: str = ""
    images: List[str] = Field(default_factory=list)  # base64-encoded PNG/JPEG
    max_new_tokens: int = Field(64, ge=1)
    do_sample: bool = True
    temperature: float = Field(0.8, gt=0.0)
    top_k: int = 0
    top_p: float = 0.9
    stop_at_eos: bool = True


def _decode_images(b64_list: List[str]):
    """base64 strings → list of PIL RGB images."""
    from PIL import Image  # lazy: only needed for multimodal requests
    imgs = []
    for i, b64 in enumerate(b64_list):
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]  # strip data-URI prefix
        try:
            raw = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(400, f"image[{i}] is not valid base64: {e}")
        try:
            imgs.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"image[{i}] could not be decoded: {e}")
    return imgs


@torch.no_grad()
def _generate_from_embeds(fused, req) -> List[int]:
    """Prefill a fused (1, S, d_model) sequence, then autoregress text tokens."""
    model, tok = _STATE["model"], _STATE["tok"]
    eos = _eos_id(req.stop_at_eos)
    out = model(inputs_embeds=fused, use_cache=True)
    cache = out["cache"]
    logits = out["logits"][:, -1, :]
    new_ids: List[int] = []
    for _ in range(req.max_new_tokens):
        if req.do_sample:
            lg = logits / max(req.temperature, 1e-6)
            lg = model._filter_top_k(lg, req.top_k)
            lg = model._filter_top_p(lg, req.top_p)
            nxt = torch.multinomial(torch.softmax(lg, dim=-1), 1)
        else:
            nxt = logits.argmax(dim=-1, keepdim=True)
        tid = int(nxt.item())
        new_ids.append(tid)
        if eos is not None and tid == eos:
            break
        o = model(nxt, cache=cache, use_cache=True)
        cache = o["cache"]
        logits = o["logits"][:, -1, :]
    return new_ids


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok" if _STATE.get("ready") else "starting",
            "device": _STATE.get("device"),
            "multimodal": bool(_STATE.get("mm_ready"))}


@app.get("/v1/model")
def model_info():
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    m = _STATE["model"]
    return {
        "n_params": _STATE["n_params"],
        "d_model": m.config.d_model,
        "n_layers": m.config.n_layers,
        "vocab_size": m.config.vocab_size,
        "max_seq_len": m.config.max_seq_len,
        "device": _STATE["device"],
        "tokenizer": "byte" if _STATE["small"]
        else os.environ.get("TOKENIZER", "gpt2"),
        "multimodal": bool(_STATE.get("mm_ready")),
        "vision_tower": _STATE.get("mm_model"),
        "multimodal_error": _STATE.get("mm_error"),
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    if req.max_new_tokens > MAX_NEW_CAP:
        raise HTTPException(400, f"max_new_tokens exceeds cap {MAX_NEW_CAP}")
    model, tok = _STATE["model"], _STATE["tok"]
    ids = _encode(req)
    t0 = time.time()
    out = model.generate(
        ids, max_new_tokens=req.max_new_tokens, do_sample=req.do_sample,
        temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        eos_token_id=_eos_id(req.stop_at_eos),
    )
    new_ids = out[0, ids.shape[1]:].tolist()
    dt = time.time() - t0
    return {
        "text": tok.decode(new_ids, skip_special_tokens=True),
        "tokens": new_ids,
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
    eos = _eos_id(req.stop_at_eos)
    ids = _encode(req)

    def _gen():
        # Prefill, then emit one token per step (greedy/sampled) over the cache.
        out = model(ids, use_cache=True)
        cache = out["cache"]
        logits = out["logits"][:, -1, :]
        for _ in range(req.max_new_tokens):
            if req.do_sample:
                lg = logits / max(req.temperature, 1e-6)
                lg = model._filter_top_k(lg, req.top_k)
                lg = model._filter_top_p(lg, req.top_p)
                nxt = torch.multinomial(torch.softmax(lg, dim=-1), 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            tid = int(nxt.item())
            piece = tok.decode([tid], skip_special_tokens=True)
            yield f"data: {json.dumps({'token': tid, 'text': piece})}\n\n"
            if eos is not None and tid == eos:
                break
            o = model(nxt, cache=cache, use_cache=True)
            cache = o["cache"]
            logits = o["logits"][:, -1, :]
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/v1/multimodal/completions")
def multimodal_completions(req: MultimodalRequest):
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    if not _STATE.get("mm_ready"):
        raise HTTPException(
            503,
            "multimodal not enabled; start the server with ENABLE_MULTIMODAL=1 "
            f"(last error: {_STATE.get('mm_error')})",
        )
    if req.max_new_tokens > MAX_NEW_CAP:
        raise HTTPException(400, f"max_new_tokens exceeds cap {MAX_NEW_CAP}")
    if not req.images:
        raise HTTPException(400, "provide at least one image (base64)")

    model, tok, enc = _STATE["model"], _STATE["tok"], _STATE["mm_encoder"]
    device = _STATE["device"]

    images = _decode_images(req.images)
    t0 = time.time()
    with torch.no_grad():
        img_tok = enc(images)                       # (n_images, N, d_model)
        n_img = img_tok.shape[0]
        img_tok = img_tok.reshape(1, n_img * img_tok.shape[1], img_tok.shape[2])
        text_ids = tok.encode(req.prompt) if req.prompt else []
        if text_ids:
            ids = torch.tensor([text_ids], dtype=torch.long, device=device)
            txt_tok = model.embed_tokens(ids)        # (1, T, d_model)
            fused = torch.cat([img_tok, txt_tok], dim=1)
        else:
            fused = img_tok
        new_ids = _generate_from_embeds(fused, req)

    dt = time.time() - t0
    return {
        "text": tok.decode(new_ids, skip_special_tokens=True),
        "tokens": new_ids,
        "n_images": n_img,
        "n_image_tokens": int(img_tok.shape[1]),
        "n_prompt_tokens": len(req.prompt and tok.encode(req.prompt) or []),
        "n_new_tokens": len(new_ids),
        "elapsed_s": round(dt, 4),
        "tok_per_s": round(len(new_ids) / dt, 2) if dt > 0 else None,
    }
