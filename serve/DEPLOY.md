# Deploying awareliquid.ai on your own server (no Modal)

The site and the interactive demo are served by a **single** process,
`serve/server.py` (FastAPI). It serves the static pages (`/`, `/about`,
`/research`, …) **and** the model API (`/v1/model`, `/v1/completions`,
`/v1/completions/stream`) on relative paths. The front-end talks only to those
relative paths, so there is **no external inference service and no Modal
dependency** — if this one process is up, the whole site works; if it is down,
the front-end now shows a clean "Demo offline" state instead of hanging.

Verified locally (2026-07-24): `SMALL=1` boot serves `/health`, `/v1/model`,
`/` and a real `/v1/completions` end-to-end.

## 0. One-line smoke test (no weights, no tokenizer)

```bash
pip install -r serve/requirements-serve.txt          # + a torch wheel for your platform
SMALL=1 python -m uvicorn serve.server:app --host 0.0.0.0 --port 8000
# then: curl localhost:8000/health   and open http://<server>:8000/
```
`SMALL=1` runs a tiny random byte-level model — output is gibberish, but it
proves the server, the static site and the demo wiring are all live. Use it to
confirm the box is set up before loading real weights.

## 1. Serving a real model

`serve/server.py` serves a **native MTLNNModel** whose config is embedded in the
checkpoint:

```bash
CKPT_PATH=checkpoints/<your_native_mtlnn>.pt TOKENIZER=gpt2 \
  python -m uvicorn serve.server:app --host 0.0.0.0 --port 8000
```

The front-end probes two bases and shows whichever answers:
- `/adapter/v1/model` → the **M1** option (TinyLlama-1.1B + trained adapter).
  This is the HF-adapter path (`serve/server_hf.py`) and needs the TinyLlama
  base downloaded plus the adapter weight
  (`checkpoints/llama_mt_adapter/llama_mt_adapter_v2s_003000.pt`, in the repo).
- `/v1/model` (root) → the **O1** option (48M from-scratch native model).

### ⚠ Weights status (read before wiring the model selector)

| model | weight file | where it is |
|---|---|---|
| M1 adapter | `llama_mt_adapter_v2s_003000.pt` | in the repo + on HF `EverestAn/MT-LNN` (older `_000500` there); needs TinyLlama-1.1B base |
| **O1 48M native** | — | **not in the repo and not on HF.** Most likely only inside the (now-disabled) Modal workspace or a training box. |

So today you can bring up **M1** (adapter path) but **not O1** until its 48M
checkpoint is located. Until then the O1 option will fail its probe and the
front-end simply falls back to M1 or, if neither is up, the clean offline state.
Do not advertise O1 in the demo selector as live until its weights are served.

## 2. Behind Caddy (typical)

Point Caddy at the FastAPI process and let it serve everything:

```
awareliquid.ai {
    reverse_proxy 127.0.0.1:8000
}
```

If you want the two model bases (`/adapter/*` and `/*`) backed by two different
processes (M1 on one, O1 on another), route `/adapter/*` to the adapter server
and everything else to the native server:

```
awareliquid.ai {
    handle_path /adapter/* {
        reverse_proxy 127.0.0.1:8001   # server_hf.py (M1 adapter)
    }
    reverse_proxy 127.0.0.1:8000       # server.py (O1 native + static site)
}
```

## 3. Keep it running

`systemd` unit (simplest; survives reboots and crashes):

```ini
# /etc/systemd/system/awareliquid.service
[Unit]
Description=AwareLiquid site + demo
After=network.target

[Service]
WorkingDirectory=/opt/M1
Environment=CKPT_PATH=checkpoints/<your>.pt
Environment=TOKENIZER=gpt2
ExecStart=/usr/bin/python -m uvicorn serve.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```
```bash
systemctl enable --now awareliquid
```
A crash restarts in 3s; there is no scale-to-zero cold start, so the front-end's
probe budget was reduced from 160s (Modal-era) to ~9s.

## 4. When O1 weights turn up → the browser (WebGPU) option

Once the 48M O1 checkpoint is available, it can be shipped to the page itself and
run in-browser with onnxruntime-web (WebGPU) — then the demo cannot go down at
all, no server required:

```bash
py -3.11 benchmarks/export_o1_for_browser.py --ckpt <o1.pt> --int8
# → serve/static/model/o1.int8.onnx (~66 MB projected) + manifest.json
```
Feasibility is already proven (`benchmarks/check_onnx_webgpu_feasibility.py`:
export + run + numerics max|onnx−torch| ≈ 3.6e-07). The front-end wiring for the
WebGPU path is the remaining work and is gated only on the weights.

## 5. Partner referral links (clawhunt etc.)

`GET /partners/<name>` counts an inbound partner click **server-side** and 302s
the visitor to `/?ref=<name>`. The homepage stores `aw_ref` in localStorage so a
later conversion can be attributed, and strips the param from the URL.

- Give the partner this link: `https://awareliquid.ai/partners/clawhunt`
- Counts persist to `data/partners/counts.json` (aggregate) and
  `data/partners/hits.jsonl` (one line per visit, with UA/referer/IP for
  bot-filtering). This directory is gitignored — it is runtime data.
- Read the totals: `GET /partners` → `{"counts":{...},"total":N}`.
  Protect it in production by setting `PARTNER_STATS_TOKEN=<secret>` and calling
  `/partners?token=<secret>`.
- New partners need no code change — any lowercase slug `[a-z0-9_-]` works and is
  counted separately; malformed/underscore-abusing slugs are rejected (404, not
  counted) so the path cannot be used for traversal.
- Pure-static fallback: `serve/static/partners/clawhunt.html` does a
  meta-refresh + `sendBeacon` so the link still redirects if the site is ever
  served without the FastAPI backend (server-side counting is unavailable then).
