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
    SESSION_DB  path to a SQLite file for cross-session recurrent-state
                persistence (Gap 4). When set, a completion request carrying a
                "session_id" continues from that session's saved LNN h_prev and
                writes the updated state back, so a conversation keeps its
                recurrent memory across separate HTTP requests / processes.
                Unset (default) → the server is fully stateless, exactly as
                before. Backed by mt_lnn.memory.SessionMemory (demo/single-node).
    KNOWLEDGE_DB  path to the SQLite file for the long-term content-addressable
                knowledge store that POST /v1/sleep consolidates sessions into
                (Gap 4, NREM sleep). Defaults to a sibling of SESSION_DB
                (``<SESSION_DB>.knowledge.db``). Only used by /v1/sleep; requires
                SESSION_DB to be set. Backed by
                mt_lnn.knowledge_memory.PersistentKnowledgeMemory.
    GRAPH_DB    path to the SQLite file for the relational GRAPH memory that
                POST /v1/sleep?build_graph=true sediments sessions into (nodes +
                cosine-linked edges, traversable by spreading activation).
                Defaults to a sibling of SESSION_DB (``<SESSION_DB>.graph.db``).
                Backed by mt_lnn.graph_memory.GraphKnowledgeMemory.
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
import threading
import time
from typing import List, Optional

# Repo root on path so `mt_lnn` imports resolve when run as `serve.server`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import (FileResponse, StreamingResponse, HTMLResponse,
                               RedirectResponse, JSONResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

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

# The served model is a SINGLE shared instance. FastAPI runs the sync endpoints
# on an anyio threadpool, so without serialization two concurrent requests
# interleave forwards on the shared model (recurrent LNN state, session
# save/load, and /v1/sleep's in-place adapter downscaling) and corrupt each
# other. This lock serializes ALL model-touching work; held across a stream's
# whole token loop — the shared state cannot be time-shared, so concurrent
# streams must run one at a time anyway. (Same pattern as serve/server_hf.py.)
_MODEL_LOCK = threading.Lock()


def _gen_lock():
    """FastAPI yield-dependency that serializes a whole request on the shared
    model: setup acquires _MODEL_LOCK, teardown releases it AFTER the response
    (incl. a streaming body) is fully sent. Applied to every endpoint that
    runs a model forward / generation or mutates live weights, so the shared
    model state is never touched by two requests at once. (threading.Lock,
    not asyncio: sync endpoints run on the anyio threadpool; a Lock may be
    released by a different thread than acquired, which is safe here.)"""
    _MODEL_LOCK.acquire()
    try:
        yield
    finally:
        _MODEL_LOCK.release()

# ── Middleware: gzip compression ──────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1024)

# ── Middleware: security headers + CSP ────────────────────────────────────────
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
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # Cache static assets aggressively, HTML/API not at all
        path = request.url.path
        if re.search(r'\.(svg|png|jpg|ico|woff2?)$', path):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif path.startswith("/v1") or path in ("/health", "/v1/model"):
            response.headers["Cache-Control"] = "no-store"
        elif path.endswith(".html") or path in ("/", "/about", "/research"):
            response.headers["Cache-Control"] = "no-cache"
        return response

app.add_middleware(_SecurityHeadersMiddleware)

# Serve the v0 frontend. Mount static assets under /static and the index at /.
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


@app.get("/privacy")
def privacy():
    return _static_page("privacy")


@app.get("/terms")
def terms():
    return _static_page("terms")


@app.get("/demo")
def demo():
    return _static_page("demo")


@app.get("/api")
def api_page():
    # The homepage nav links here (index.html); without this route it 404s.
    return _static_page("api")


# ---------------------------------------------------------------------------
# Partner referral tracking
# ---------------------------------------------------------------------------
# A partner (e.g. clawhunt) links to /partners/<name>. Each hit is counted
# SERVER-SIDE — reliable, unlike a client JS beacon that ad-blockers drop — then
# the visitor is 302'd to the homepage with ?ref=<name> so downstream analytics
# and the client can attribute the session too. Counts persist to a JSONL file
# (one line per visit) plus an aggregate JSON, so a restart never loses history.

_PARTNER_DIR = os.environ.get(
    "PARTNER_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "partners"))
_PARTNER_LOCK = threading.Lock()
# Only allow a conservative slug so the path can never be abused for traversal
# or injected into the redirect target.
_PARTNER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _record_partner_hit(name: str, request: "Request") -> None:
    os.makedirs(_PARTNER_DIR, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partner": name,
        # Referrer/UA are useful for spotting bots vs real clicks; no PII beyond
        # what any web server already logs.
        "referer": request.headers.get("referer", ""),
        "ua": request.headers.get("user-agent", "")[:300],
        "ip": (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
               or (request.client.host if request.client else "")),
    }
    with _PARTNER_LOCK:
        with open(os.path.join(_PARTNER_DIR, "hits.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        agg_path = os.path.join(_PARTNER_DIR, "counts.json")
        try:
            with open(agg_path, encoding="utf-8") as f:
                agg = json.load(f)
        except (FileNotFoundError, ValueError):
            agg = {}
        agg[name] = int(agg.get(name, 0)) + 1
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)


@app.get("/partners/{name}")
def partner_referral(name: str, request: Request):
    """Count an inbound partner click, then redirect to the site homepage."""
    slug = name.lower()
    if not _PARTNER_RE.match(slug):
        # Unknown/malformed slug: don't count it, just send them home.
        return RedirectResponse(url="/", status_code=302)
    try:
        _record_partner_hit(slug, request)
    except Exception:
        # Tracking must never break the redirect the visitor came for.
        pass
    return RedirectResponse(url=f"/?ref={slug}", status_code=302)


@app.get("/partners")
def partner_stats(request: Request):
    """Aggregate referral counts. Optionally gate with PARTNER_STATS_TOKEN."""
    token = os.environ.get("PARTNER_STATS_TOKEN")
    if token and request.query_params.get("token") != token:
        raise HTTPException(401, "stats token required")
    agg_path = os.path.join(_PARTNER_DIR, "counts.json")
    try:
        with open(agg_path, encoding="utf-8") as f:
            counts = json.load(f)
    except (FileNotFoundError, ValueError):
        counts = {}
    return JSONResponse({"counts": counts, "total": sum(counts.values())})


def _text_file(name: str, media_type: str = "text/plain") -> FileResponse:
    path = os.path.join(_STATIC_DIR, name)
    if os.path.exists(path):
        return FileResponse(path, media_type=media_type)
    raise HTTPException(404, f"{name} not found")


@app.get("/llms.txt")
def llms_txt():
    return _text_file("llms.txt", "text/plain; charset=utf-8")


@app.get("/llms-full.txt")
def llms_full_txt():
    return _text_file("llms-full.txt", "text/plain; charset=utf-8")


@app.get("/robots.txt")
def robots_txt():
    # Lives in static/ but StaticFiles is only mounted at /static, so crawlers
    # hitting the canonical /robots.txt got the 404 page instead.
    return _text_file("robots.txt", "text/plain; charset=utf-8")


@app.get("/sitemap.xml")
def sitemap_xml():
    return _text_file("sitemap.xml", "application/xml")


def _wants_json(request: Request) -> bool:
    """True when the caller is an API client rather than a browser.

    Without this, a legitimate API error such as
    HTTPException(404, "session persistence disabled (set SESSION_DB)") was
    replaced by the full 404.html page, so every SDK call blew up with
    "SyntaxError: Unexpected token '<'" and the real reason was lost.
    """
    path = request.url.path
    if path.startswith("/v1/") or path.startswith("/adapter/v1/"):
        return True
    accept = request.headers.get("accept", "")
    return "application/json" in accept and "text/html" not in accept


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Serve the branded 404 page to browsers, JSON to API clients."""
    if _wants_json(request):
        detail = getattr(exc, "detail", None) or "Not Found"
        return JSONResponse({"detail": detail}, status_code=404)
    path_404 = os.path.join(_STATIC_DIR, "404.html")
    if os.path.exists(path_404):
        with open(path_404, encoding="utf-8") as fh:
            content = fh.read()
        return HTMLResponse(content=content, status_code=404)
    return HTMLResponse("<h1>404 — Not Found</h1>", status_code=404)


@app.on_event("startup")
def _startup() -> None:
    small = os.environ.get("SMALL", "0") == "1"
    device = os.environ.get(
        "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.set_grad_enabled(False)
    model = _build_model(small).to(device)
    tok = _load_tokenizer(small)
    # Cross-session recurrent-state persistence (Gap 4). Empty string → OFF, so
    # the server stays stateless and bit-identical to before unless SESSION_DB is
    # explicitly provided AND a request carries a session_id.
    session_db = os.environ.get("SESSION_DB", "").strip()
    # Long-term knowledge store for the NREM sleep bridge (POST /v1/sleep).
    # Defaults to a sibling of SESSION_DB so a single SESSION_DB is enough to get
    # working+long-term memory; only read by /v1/sleep.
    knowledge_db = os.environ.get("KNOWLEDGE_DB", "").strip()
    if not knowledge_db and session_db:
        knowledge_db = session_db + ".knowledge.db"
    # Relational graph store for POST /v1/sleep?build_graph=true.
    graph_db = os.environ.get("GRAPH_DB", "").strip()
    if not graph_db and session_db:
        graph_db = session_db + ".graph.db"
    _STATE.update(
        model=model, tok=tok, device=device, small=small,
        n_params=sum(p.numel() for p in model.parameters()),
        session_db=session_db,
        knowledge_db=knowledge_db,
        graph_db=graph_db,
        ready=True,
    )
    print(f"[serve] ready | {_STATE['n_params']/1e6:.1f}M params | device={device}"
          + (f" | session_db={session_db}" if session_db else " | stateless"))

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
    # Cross-session recurrent-state persistence (Gap 4): when set AND the server
    # was started with SESSION_DB, the request continues from this session's saved
    # LNN h_prev state and writes the updated state back afterwards. Omitted (None)
    # → stateless, identical to the pre-persistence behaviour.
    session_id: Optional[str] = None


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


# ---------------------------------------------------------------------------
# Cross-session recurrent-state persistence (Gap 4)
# ---------------------------------------------------------------------------
# Backed by mt_lnn.memory.SessionMemory via the model's save_state/load_state
# convenience wrappers. The hot path loads the session's persisted per-layer LNN
# h_prev, prefills the prompt as a *continuation* of that recurrent state
# (use_lnn_recurrence=True), autoregresses over the cache, then writes the
# updated h_prev back. KV caches are intentionally NOT persisted (they are
# position-tied to a token sequence); only the recurrent state crosses requests.

def _session_enabled(req: "CompletionRequest") -> bool:
    """A request uses persistence iff it carries a session_id AND the server was
    started with SESSION_DB. Otherwise the stateless path is used (zero change)."""
    return bool(getattr(req, "session_id", None)) and bool(_STATE.get("session_db"))


def _session_prior_tokens(session_id: str) -> int:
    """Cumulative token_count stored for this session (0 if new)."""
    from mt_lnn.memory import SessionMemory
    with SessionMemory(_STATE["session_db"]) as mem:
        info = mem.session_info(session_id)
    return int(info["token_count"]) if info else 0


def _sample_next(model, logits, req) -> torch.Tensor:
    """One decoding step shared by the stateless and session paths."""
    if req.do_sample:
        lg = logits / max(req.temperature, 1e-6)
        lg = model._filter_top_k(lg, req.top_k)
        lg = model._filter_top_p(lg, req.top_p)
        return torch.multinomial(torch.softmax(lg, dim=-1), 1)
    return logits.argmax(dim=-1, keepdim=True)


@torch.no_grad()
def _complete_with_session(req: "CompletionRequest", ids: torch.Tensor):
    """Generate while continuing from (and then updating) the session's persisted
    recurrent state. Returns (new_ids, total_token_count)."""
    from mt_lnn.model import ModelCacheStruct
    model = _STATE["model"]
    sid, db, device = req.session_id, _STATE["session_db"], _STATE["device"]
    eos = _eos_id(req.stop_at_eos)

    # Load prior per-layer h_prev (None KV) — fresh cache if the session is new.
    cache = model.load_state(sid, db_path=db, device=device) or ModelCacheStruct()
    prior_tokens = _session_prior_tokens(sid)

    # Prefill the prompt as a continuation of the recurrent state.
    out = model(ids, cache=cache, use_cache=True, use_lnn_recurrence=True)
    cache = out["cache"]
    logits = out["logits"][:, -1, :]

    new_ids: List[int] = []
    for _ in range(req.max_new_tokens):
        nxt = _sample_next(model, logits, req)
        tid = int(nxt.item())
        new_ids.append(tid)
        if eos is not None and tid == eos:
            break
        o = model(nxt, cache=cache, use_cache=True, use_lnn_recurrence=True)
        cache = o["cache"]
        logits = o["logits"][:, -1, :]

    total_tokens = prior_tokens + int(ids.shape[1]) + len(new_ids)
    model.save_state(sid, cache, token_count=total_tokens, db_path=db)
    return new_ids, total_tokens


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


@app.get("/v1/sessions")
def list_sessions():
    """List persisted sessions (newest first). 404 if persistence is disabled."""
    db = _STATE.get("session_db")
    if not db:
        raise HTTPException(404, "session persistence disabled (set SESSION_DB)")
    from mt_lnn.memory import SessionMemory
    with SessionMemory(db) as mem:
        return {"sessions": mem.list_sessions()}


@app.get("/v1/sessions/{session_id}")
def session_info(session_id: str):
    """Metadata (token_count, updated_at) for one session without loading tensors."""
    db = _STATE.get("session_db")
    if not db:
        raise HTTPException(404, "session persistence disabled (set SESSION_DB)")
    from mt_lnn.memory import SessionMemory
    with SessionMemory(db) as mem:
        info = mem.session_info(session_id)
    if info is None:
        raise HTTPException(404, f"session {session_id!r} not found")
    return info


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str):
    """Forget a session's persisted recurrent state."""
    db = _STATE.get("session_db")
    if not db:
        raise HTTPException(404, "session persistence disabled (set SESSION_DB)")
    from mt_lnn.memory import SessionMemory
    with SessionMemory(db) as mem:
        deleted = mem.delete(session_id)
    if not deleted:
        raise HTTPException(404, f"session {session_id!r} not found")
    return {"deleted": True, "session_id": session_id}


@app.post("/v1/sleep")
def sleep_consolidate(
    consolidate_fraction: float = 1.0,
    seed: int = 0,
    downscale_factor: float = 1.0,
    build_graph: bool = False,
    link_threshold: float = 0.55,
    link_quantile: Optional[float] = None,
    _lock=Depends(_gen_lock),
):
    """Run one sleep pass: NREM consolidation, then optional SHY downscaling.

    Stage 1 (NREM, always) -- replay persisted sessions and consolidate the
    salient ones, BY CONTENT, into the long-term knowledge store. This is the
    fast->slow (working->long-term) transfer: after a pass a session's pooled
    recurrent signature becomes a content-addressable key in the durable
    PersistentKnowledgeMemory, so a past session can be recalled by similarity.
    Reuses the EXISTING SleepWakeConsolidator.nrem_replay policy unchanged (see
    mt_lnn.session_consolidation). Idempotent-safe: a no-op all-zero summary when
    there is nothing consolidatable.

    Stage 1b (GRAPH, opt-in) -- when ``build_graph`` is true, additionally
    sediment the same sessions into a relational graph store (GRAPH_DB): each
    session is a node, auto-linked to already-consolidated sessions whose
    recurrent signature is within the link cut (edge weight = the similarity).
    The cut is either the fixed ``link_threshold`` cosine, or -- when
    ``link_quantile`` is given (in [0, 1]) -- ADAPTIVE: the threshold is the
    ``link_quantile``-th percentile of the sessions' own pairwise-cosine band, so
    it tracks the corpus/encoder instead of a hand-tuned absolute (e.g.
    ``link_quantile=0.9`` keeps the top ~10% strongest relations). The cut used is
    echoed back as ``graph.link_threshold``. Sleep then leaves behind not just
    isolated memories but the relations between them, traversable later by
    spreading activation. Reuses the same nrem_replay policy; reported under a
    ``graph`` block.

    Stage 2 (SHY, opt-in) -- when ``downscale_factor`` < 1.0, multiplicatively
    renormalise the served model's TRAINABLE MT-adapter weights downward
    (Synaptic Homeostasis Hypothesis), preserving the relative weight pattern
    while shrinking total synaptic weight. The frozen base is never touched. This
    MUTATES the live served weights in place, so it is OFF by default
    (downscale_factor == 1.0 is a no-op). Acts only on the M1 adapter route; on a
    from-scratch model with no MT adapters it is a safe all-zero no-op.

    Requires SESSION_DB (404 otherwise). Returns the NREM summary plus, when SHY
    ran, a ``downscale`` block {n_adapters, factor, n_tensors, pre_norm, post_norm}.
    """
    db = _STATE.get("session_db")
    if not db:
        raise HTTPException(404, "session persistence disabled (set SESSION_DB)")
    if not (0.0 <= consolidate_fraction <= 1.0):
        raise HTTPException(400, "consolidate_fraction must be in [0, 1]")
    if not (0.0 < downscale_factor <= 1.0):
        raise HTTPException(400, "downscale_factor must be in (0, 1]")

    from mt_lnn.session_consolidation import consolidate_sessions
    summary = consolidate_sessions(
        db, _STATE["knowledge_db"],
        consolidate_fraction=consolidate_fraction,
        seed=seed, device=_STATE["device"],
    )
    result = {"knowledge_db": _STATE["knowledge_db"], **summary}

    # Stage 1b: relational graph sedimentation (opt-in).
    if build_graph:
        if not (0.0 <= link_threshold <= 1.0):
            raise HTTPException(400, "link_threshold must be in [0, 1]")
        if link_quantile is not None and not (0.0 <= link_quantile <= 1.0):
            raise HTTPException(400, "link_quantile must be in [0, 1]")
        from mt_lnn.session_consolidation import consolidate_sessions_to_graph
        graph_summary = consolidate_sessions_to_graph(
            db, _STATE["graph_db"],
            consolidate_fraction=consolidate_fraction,
            seed=seed, link_threshold=link_threshold,
            link_quantile=link_quantile, device=_STATE["device"],
        )
        result["graph"] = {"graph_db": _STATE["graph_db"], **graph_summary}

    # Stage 2: synaptic homeostasis on the adapter route (opt-in, mutates weights).
    if downscale_factor < 1.0:
        from mt_lnn.adapter_homeostasis import downscale_adapters
        report = downscale_adapters(_STATE["model"], factor=downscale_factor)
        result["downscale"] = {
            "n_adapters": report.n_adapters,
            "factor": report.factor,
            "n_tensors": report.n_tensors,
            "pre_norm": report.pre_norm,
            "post_norm": report.post_norm,
        }
    return result


@app.post("/v1/completions")
def completions(req: CompletionRequest, _lock=Depends(_gen_lock)):
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    if req.max_new_tokens > MAX_NEW_CAP:
        raise HTTPException(400, f"max_new_tokens exceeds cap {MAX_NEW_CAP}")
    model, tok = _STATE["model"], _STATE["tok"]
    ids = _encode(req)
    t0 = time.time()
    if _session_enabled(req):
        # Continue from (and update) the persisted recurrent state for this session.
        new_ids, total_tokens = _complete_with_session(req, ids)
    else:
        out = model.generate(
            ids, max_new_tokens=req.max_new_tokens, do_sample=req.do_sample,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
            eos_token_id=_eos_id(req.stop_at_eos),
        )
        new_ids = out[0, ids.shape[1]:].tolist()
        total_tokens = None
    dt = time.time() - t0
    resp = {
        "text": tok.decode(new_ids, skip_special_tokens=True),
        "tokens": new_ids,
        "n_new_tokens": len(new_ids),
        "elapsed_s": round(dt, 4),
        "tok_per_s": round(len(new_ids) / dt, 2) if dt > 0 else None,
    }
    if total_tokens is not None:
        resp["session_id"] = req.session_id
        resp["session_token_count"] = total_tokens
    return resp


@app.post("/v1/completions/stream")
def completions_stream(req: CompletionRequest, _lock=Depends(_gen_lock)):
    if not _STATE.get("ready"):
        raise HTTPException(503, "model not ready")
    if req.max_new_tokens > MAX_NEW_CAP:
        raise HTTPException(400, f"max_new_tokens exceeds cap {MAX_NEW_CAP}")
    model, tok = _STATE["model"], _STATE["tok"]
    eos = _eos_id(req.stop_at_eos)
    ids = _encode(req)
    session = _session_enabled(req)

    def _gen():
        # Prefill, then emit one token per step (greedy/sampled) over the cache.
        # When a session is active, the prefill CONTINUES from the persisted
        # recurrent state and the updated state is written back at the end.
        # _MODEL_LOCK is held for the whole request by the _gen_lock dependency
        # (released after the response is fully sent). The try/finally
        # guarantees the session state write-back runs even on client
        # disconnect (GeneratorExit), so the turn's state is not lost.
        if session:
            from mt_lnn.model import ModelCacheStruct
            cache = (model.load_state(req.session_id, db_path=_STATE["session_db"],
                                      device=_STATE["device"])
                     or ModelCacheStruct())
            prior_tokens = _session_prior_tokens(req.session_id)
            out = model(ids, cache=cache, use_cache=True, use_lnn_recurrence=True)
        else:
            cache = None
            out = model(ids, use_cache=True)
        cache = out["cache"]
        logits = out["logits"][:, -1, :]
        n_new = 0
        try:
            for _ in range(req.max_new_tokens):
                nxt = _sample_next(model, logits, req)
                tid = int(nxt.item())
                n_new += 1
                piece = tok.decode([tid], skip_special_tokens=True)
                yield f"data: {json.dumps({'token': tid, 'text': piece})}\n\n"
                if eos is not None and tid == eos:
                    break
                o = model(nxt, cache=cache, use_cache=True,
                          use_lnn_recurrence=True) if session else \
                    model(nxt, cache=cache, use_cache=True)
                cache = o["cache"]
                logits = o["logits"][:, -1, :]
        finally:
            if session:
                total = prior_tokens + int(ids.shape[1]) + n_new
                model.save_state(req.session_id, cache,
                                 token_count=total, db_path=_STATE["session_db"])
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/v1/multimodal/completions")
def multimodal_completions(req: MultimodalRequest, _lock=Depends(_gen_lock)):
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
