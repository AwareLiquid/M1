#!/usr/bin/env bash
# deploy/golive.sh -- one-command server-side go-live for the MT-LNN demo.
#
# Run this ON the Vultr box (after you SSH in yourself). It is idempotent: safe
# to re-run after a code change or a checkpoint upload.
#
#   ssh root@<server-ip>
#   curl -fsSL https://raw.githubusercontent.com/AwareLiquid/M1/main/deploy/golive.sh | bash
#   # ...or, if the repo is already cloned:  bash ~/M1/deploy/golive.sh
#
# What it does, in order:
#   1. installs Docker + compose plugin if missing
#   2. clones the repo (or git pull if already present)
#   3. opens ufw for 22/80/443 if ufw is active
#   4. brings up the prod stack (mtlnn + caddy) -- runs a FRESH untrained model
#      until checkpoints/m2_final.pt exists, so the service is live immediately
#   5. waits for the app health check and prints next steps
#
# It NEVER handles any secret: no passwords, no keys. The checkpoint is uploaded
# separately from your laptop via scp (see the printed instructions at the end).
set -euo pipefail

REPO_URL="https://github.com/AwareLiquid/M1.git"
REPO_DIR="${REPO_DIR:-$HOME/M1}"
COMPOSE_FILE="deploy/docker-compose.prod.yml"
DOMAIN="awareliquid.ai"

log() { printf '\n\033[1;36m[golive]\033[0m %s\n' "$*"; }

# --- 1. Docker -------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  log "Docker not found -- installing via get.docker.com ..."
  curl -fsSL https://get.docker.com | sh
else
  log "Docker present: $(docker --version)"
fi
if ! docker compose version >/dev/null 2>&1; then
  log "ERROR: 'docker compose' plugin unavailable. Install docker-compose-plugin and re-run." >&2
  exit 1
fi

# --- 2. Repo ---------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  log "Repo exists at $REPO_DIR -- pulling latest ..."
  git -C "$REPO_DIR" pull --ff-only
else
  log "Cloning repo into $REPO_DIR ..."
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
mkdir -p checkpoints

# --- 3. Firewall (only if ufw is active) -----------------------------------
if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -q "Status: active"; then
  log "ufw active -- ensuring 22/80/443 are allowed ..."
  ufw allow 22/tcp  || true
  ufw allow 80/tcp  || true
  ufw allow 443/tcp || true
else
  log "ufw inactive/absent -- ensure 22/80/443 are open in the Vultr firewall."
fi

# --- 4. Bring up the stack -------------------------------------------------
if [ -f checkpoints/m2_final.pt ]; then
  log "Found checkpoints/m2_final.pt -- the trained model will load."
else
  log "No checkpoints/m2_final.pt yet -- starting with a FRESH untrained model."
  log "Upload it later from your laptop, then: docker compose -f $COMPOSE_FILE restart mtlnn"
fi
log "Building image + starting mtlnn + caddy ..."
docker compose -f "$COMPOSE_FILE" up -d --build

# --- 5. Wait for app health ------------------------------------------------
log "Waiting for the app health check (up to ~90s) ..."
ok=0
for _ in $(seq 1 30); do
  if docker exec mtlnn_prod python -c \
      "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)" \
      >/dev/null 2>&1; then
    ok=1; break
  fi
  sleep 3
done
if [ "$ok" = "1" ]; then
  log "App is healthy. mtlnn is serving on the internal :8000 (Caddy is the public edge)."
else
  log "App not healthy yet -- check logs: docker compose -f $COMPOSE_FILE logs --tail=80 mtlnn"
fi

cat <<EOF

============================================================
 Stack is up. Remaining steps (you run these):
------------------------------------------------------------
 1) Upload the model FROM YOUR LAPTOP (separate terminal):
      scp /path/to/tiny_serve.pt root@<server-ip>:$REPO_DIR/checkpoints/m2_final.pt
    then on the server:
      docker compose -f $COMPOSE_FILE restart mtlnn

 2) DNS must resolve before HTTPS works:
      nslookup $DOMAIN 8.8.8.8         # -> this server's IP
    Once it does, Caddy auto-issues the TLS cert (ports 80/443 must be open).

 3) Verify:
      curl https://$DOMAIN/health
      open  https://$DOMAIN

 Logs:    docker compose -f $COMPOSE_FILE logs -f
 Reload:  docker compose -f $COMPOSE_FILE restart mtlnn
============================================================
EOF
