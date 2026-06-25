# Go-live on Vultr: MT-LNN server + domain (awareliquid.ai)

End-to-end guide to put the native MT-LNN inference server online behind HTTPS on
your own domain. The app stack (server + TLS reverse proxy) is fully defined in
`deploy/docker-compose.prod.yml` + `deploy/Caddyfile`; the only manual steps are
provisioning the box and pointing DNS at it (those need your Vultr/registrar
account, which only you can do).

---

## 1. Which Vultr plan

The model is ~125M params, CPU inference, autoregressive generation. RAM need is
modest (weights ~0.5 GB fp32 + KV cache + torch/python ~1.5 GB), but token/s on CPU
is bound by **sustained single-thread + a couple of cores**. So favour dedicated
CPU over the cheapest shared burstable plan.

| Tier | Vultr product | Specs | Notes |
|---|---|---|---|
| **Start here** | Cloud Compute - **High Performance** (AMD EPYC + NVMe) | 2 vCPU / 4 GB | Cheapest viable. Good single-thread; fine for a demo / low traffic. |
| **Recommended** | **Optimized Cloud Compute - General Purpose** (dedicated vCPU) | 2-4 vCPU / 8 GB | No noisy-neighbour throttling -> predictable latency under sustained generation. Best price/perf for go-live. |
| Overkill for now | Optimized / 8 vCPU+ | 8 vCPU / 16 GB | Only if you see real concurrent traffic. Scale up later, don't pre-buy. |

- **Region**: closest to your users (latency on streaming is noticeable).
- **OS**: Ubuntu 24.04 LTS x64.
- **Do NOT** pick a GPU plan -- the CPU image is what we built; GPU adds cost with
  no benefit at this size/traffic.

> Pricing changes; verify the live monthly cost in the Vultr console before you
> deploy. Start small -- you can resize the instance up without rebuilding.

---

## 2. Provision the instance (you do this in the Vultr console)

1. **Deploy New Server** -> Cloud Compute (High Performance) or Optimized Cloud
   Compute -> pick the plan from the table -> Ubuntu 24.04 -> add your SSH key.
2. After it boots, note the **public IPv4**.
3. **Firewall** (Vultr console -> Firewall, or `ufw` on the box): allow
   `22` (SSH), `80` (HTTP, needed for the TLS challenge), `443` (HTTPS). Block the
   rest. The app port 8000 stays internal -- never expose it directly.

---

## 3. Point the domain at the box (DNS)

`awareliquid.ai` is registered at **Alibaba Cloud** and uses **Alibaba Cloud DNS**
("Cloud DNS / 云解析 DNS"). Two distinct consoles are involved -- records live in
*Cloud DNS*, but the nameserver delegation is changed in the *Domains* console.

**3a. A records (Alibaba Cloud DNS / 云解析 DNS console):**

| Type | Host | Value | TTL |
|---|---|---|---|
| A | `@`   | `<your server IPv4>` | 300 (10 min) |
| A | `www` | `<your server IPv4>` | 300 (10 min) |

**3b. Nameserver delegation (Alibaba "Domains / 域名" console -> the domain ->
Manage -> Modify DNS Servers).** The records in 3a only take effect once the
domain's authoritative NS point at Alibaba. If the NS still read a third party
(e.g. `*.ns.cloudflare.com`), switch them to the Alibaba-assigned pair shown in
the Cloud DNS console, typically:

```
ns7.alidns.com
ns8.alidns.com
```

> If the NS already point to a **non-Alibaba** DNS host you prefer to keep (e.g.
> Cloudflare), DON'T switch -- just add the two A records there instead, and set
> them **DNS-only (grey cloud)** so Caddy's HTTP-01 challenge isn't intercepted by
> the proxy. Migrate any existing MX/TXT/CNAME records before changing NS.

Wait for propagation, then confirm BOTH before the next step (or Caddy's cert
request fails):

```bash
nslookup -type=NS awareliquid.ai 8.8.8.8   # -> ns7/ns8.alidns.com
nslookup awareliquid.ai 8.8.8.8            # -> <your server IPv4>
```

---

## 4. Bring the stack up (on the server, over SSH)

```bash
# install docker + compose plugin
curl -fsSL https://get.docker.com | sh

# get the code
git clone https://github.com/everest-an/M1.git
cd M1

# drop the trained checkpoint in (see step 5). The stack runs a FRESH untrained
# model until this file exists, so you can also start now and add it later.
mkdir -p checkpoints
# cp /path/to/serve.pt checkpoints/m2_final.pt

# launch: builds the CPU image, starts mtlnn + caddy, Caddy auto-issues the cert
docker compose -f deploy/docker-compose.prod.yml up -d --build

# watch it come up (Caddy logs the cert issuance; mtlnn logs "ready | ...M params")
docker compose -f deploy/docker-compose.prod.yml logs -f
```

Then open **https://awareliquid.ai** -- the streaming chat UI should load, and
`https://awareliquid.ai/health` should return `{"status":"ok",...}`.

---

## 5. The trained checkpoint

The public model comes from the Kaggle M2 pretrain run. When that kernel finishes:

```bash
# locally, download the kernel output, then copy the slim server checkpoint up:
kaggle kernels output muningan/awareliquid-m2-pretrain -p ./m2_out
scp ./m2_out/checkpoints/serve.pt <user>@<server-ip>:~/M1/checkpoints/m2_final.pt
```

On the server, load it without a full restart:

```bash
docker compose -f deploy/docker-compose.prod.yml restart mtlnn
```

`serve.pt` is the slim (config + weights, no optimizer) checkpoint the kernel
writes specifically so it downloads over a flaky proxy and drops straight in here.

---

## 6. Operate

| Task | Command (from `~/M1`) |
|---|---|
| Tail logs | `docker compose -f deploy/docker-compose.prod.yml logs -f` |
| Reload new checkpoint | `docker compose -f deploy/docker-compose.prod.yml restart mtlnn` |
| Update **frontend only** (HTML/CSS/JS/SVG) | `git pull` |
| Update **mtlnn Python** (`server.py` / `mt_lnn` / deps) | `git pull && docker compose -f deploy/docker-compose.prod.yml up -d --build` |
| Update **adapter Python** (`server_hf.py` / `mt_lnn`) | `git pull && docker compose -f deploy/docker-compose.prod.yml restart adapter` |
| Stop everything | `docker compose -f deploy/docker-compose.prod.yml down` |

> **Pick the lightest update path -- and mind what is mounted vs baked.** The
> `mtlnn` service mounts ONLY `../serve/static:/app/serve/static:ro`; everything
> else it runs (`serve/server.py` and the `mt_lnn` package) is COPYed into the
> image at build time, NOT mounted.
>
> - **Frontend-only** (a new page, edited HTML, an added chart SVG): `StaticFiles`
>   reads from the bind-mounted dir per request, so the change goes live the
>   instant `git pull` updates the working tree -- **no restart, no rebuild**. A
>   bare `restart mtlnn` would NOT help here, and would NOT pick up a `server.py`
>   change either (the old image is reused).
> - **mtlnn Python or deps** (`server.py`, anything under `mt_lnn`,
>   `requirements-serve.txt`, the `Dockerfile`): the code is baked into the image,
>   so you must **`up -d --build`** -- a plain `restart` reruns the same old image.
> - **adapter service** is different: it bind-mounts all of `../serve` + `../mt_lnn`,
>   so `restart adapter` is enough even for `server_hf.py` / `mt_lnn` edits (no
>   rebuild). It is profile-gated, so it only matters if you launched it.

**Sanity check from your laptop:**

```bash
curl https://awareliquid.ai/v1/model
curl -X POST https://awareliquid.ai/v1/completions \
  -H 'content-type: application/json' \
  -d '{"prompt":"The microtubule is","max_new_tokens":40}'
```

---

## 7. Notes & honest caveats

- **CPU generation is slow-ish.** A 125M model on 2-4 CPU cores will do roughly a
  few-to-low-tens of tokens/sec. Fine for a demo; for snappy interactive use at
  traffic, you'd move to a GPU box (swap the Dockerfile base for an
  `nvidia/cuda` runtime + the CUDA torch wheel, and run with `DEVICE=cuda`).
- **`MAX_NEW_TOKENS_CAP=512`** bounds per-request work so one client can't peg the
  box. Raise/lower in the compose env.
- **The model quality** is whatever the M2 run achieved (last clean run: val PPL
  ~136 at 5000 steps). It is a small from-scratch model -- the demo shows the
  architecture works end to end, not GPT-class fluency. Set expectations on the
  landing page accordingly.
- Caddy persists its issued certs in the `caddy_data` volume, so restarts don't
  re-hit Let's Encrypt rate limits.
