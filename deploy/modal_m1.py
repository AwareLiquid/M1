"""AwareLiquid **M1** (TinyLlama-1.1B-Chat + trained MT-LNN adapter) on Modal.

Serverless GPU (scale-to-zero) wrapper around ``serve/server_hf.py`` -- the SAME
FastAPI app the Vultr box runs, but on a T4 instead of CPU. It exposes the exact
``/v1/*`` chat API (and the static frontend) the site already uses, so the
model menu's ``/adapter`` route can be pointed here for GPU-speed generation.

WHY THIS EXISTS
---------------
The prod box is CPU-only (4 vCPU / 8 GB, no GPU). M1 1.1B + self-thinking decode
on CPU is slow. Modal gives on-demand T4 GPU: you pay per-second only while it is
actually generating, it scales to zero when idle (~free under the $30/mo credit),
and the container stays warm ``SCALEDOWN`` seconds after each request.

ONE-TIME SETUP (run locally; your Modal secret never needs to touch a file)
---------------------------------------------------------------------------
    pip install modal
    modal setup                     # opens a browser to authenticate

    # upload the 129 MB trained adapter into the checkpoint Volume:
    modal volume create awareliquid-m1-ckpt
    modal volume put awareliquid-m1-ckpt \
        checkpoints/llama_mt_adapter/llama_mt_adapter_003000.pt \
        /llama_mt_adapter_003000.pt

DEPLOY
------
    modal deploy deploy/modal_m1.py
    # -> prints a public URL like https://<you>--awareliquid-m1-web.modal.run
    # test:  curl https://<...>.modal.run/v1/model
    #        open  https://<...>.modal.run           (full chat UI on GPU)

The TinyLlama-1.1B base (~2.2 GB) downloads from HF on the FIRST cold start and
is cached in the ``awareliquid-m1-hf`` Volume, so later cold starts are fast.

WIRING INTO awareliquid.ai (after the Modal URL works)
------------------------------------------------------
Point the site's ``/adapter/*`` route at the Modal URL instead of the local
adapter container: in ``deploy/Caddyfile`` replace ``reverse_proxy adapter:8000``
with ``reverse_proxy https://<...>.modal.run`` (+ ``header_up Host {upstream_hostport}``),
then reload Caddy. Left as a separate step so we verify the GPU demo standalone
first.
"""
from __future__ import annotations

import modal

APP_NAME = "awareliquid-m1"
ADAPTER_FNAME = "llama_mt_adapter_003000.pt"

# Container mount points for the two persistent Volumes.
CKPT_DIR = "/ckpt"     # holds the uploaded adapter .pt
HF_DIR = "/hf"         # HuggingFace cache (base-model weights persist here)

# GPU: T4 (16 GB) is plenty for a 1.1B fp16 model. Bump to "L4"/"A10" for more
# throughput; there is no benefit going higher at this size.
GPU = "T4"
# Stay warm this many seconds after the last request, then scale to zero. Larger
# = fewer cold starts but more idle GPU-seconds billed. 300s (5 min) is a good
# demo default.
SCALEDOWN = 300

app = modal.App(APP_NAME)

# Persistent Volumes (created on first use / by `modal volume create`).
ckpt_vol = modal.Volume.from_name(f"{APP_NAME}-ckpt", create_if_missing=True)
hf_vol = modal.Volume.from_name(f"{APP_NAME}-hf", create_if_missing=True)

# Image: CUDA torch wheel (PyPI torch on linux ships the CUDA build) + the serve
# stack, with the repo's mt_lnn package and the serve/ dir (server_hf.py + the
# static frontend) baked in. server_hf.py is env-driven, so all model config is
# passed via .env(...) below and read at FastAPI startup.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "numpy<2",
        "transformers>=4.44,<5",
        "tokenizers>=0.19",
        "peft>=0.11",
        "accelerate>=0.30",
        "fastapi[standard]>=0.115",
    )
    .env(
        {
            # --- match the Vultr adapter service EXACTLY so tensors map 1:1 ---
            "BASE_MODEL": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "ADAPTER_CKPT": f"{CKPT_DIR}/{ADAPTER_FNAME}",
            "RECIPE": "phase5b",           # MT adapters + LoRA (checkpoint's recipe)
            "DEVICE": "cuda",              # fp16 on GPU; skips the CPU-only INT8 path
            "MAX_NEW_TOKENS_CAP": "512",
            # self-thinking decode (cheap on GPU) -- same knobs prod uses
            "THINKING": "1",
            "THINK_ENTROPY_LOW": "0.6",
            "THINK_ENTROPY_HIGH": "4.0",
            "THINK_SAMPLES": "5",
            # cache base-model weights on the persistent Volume
            "HF_HOME": HF_DIR,
        }
    )
    .add_local_dir("mt_lnn", "/root/mt_lnn")
    .add_local_dir("serve", "/root/serve")
)


@app.cls(
    image=image,
    gpu=GPU,
    volumes={CKPT_DIR: ckpt_vol, HF_DIR: hf_vol},
    scaledown_window=SCALEDOWN,
    min_containers=0,          # scale to zero when idle (no cost). Set to 1 to
                               # pin one warm GPU (e.g. right before a live demo).
    timeout=600,
)
class M1:
    @modal.asgi_app()
    def web(self):
        # Import inside the container (not at deploy time on the laptop). server_hf
        # is a top-level module under /root/serve; it imports mt_lnn from /root.
        # Its FastAPI startup event builds TinyLlama + attaches the MT-LNN adapter
        # from ADAPTER_CKPT during container boot, so the model is warm before the
        # first request is served (the cold start pays the load, not the caller).
        import sys

        for p in ("/root", "/root/serve"):
            if p not in sys.path:
                sys.path.insert(0, p)
        import server_hf

        return server_hf.app
