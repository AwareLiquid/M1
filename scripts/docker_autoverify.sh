#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/docker_autoverify.sh
#
# Polls the Docker daemon every 5 minutes until it is reachable, then builds
# the `mtlnn` image, runs it, and verifies the HTTP endpoints. Everything is
# appended to docker_verification.log at the repo root.
#
# The container is started in SMALL mode (-e SMALL=1) so the infra check needs
# no checkpoint and no tokenizer download — it validates that the image builds,
# boots, and serves. Swap to a real CKPT_PATH once a trained model is available.
#
# Run detached:
#     nohup bash scripts/docker_autoverify.sh >/dev/null 2>&1 &
# ---------------------------------------------------------------------------
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 99
LOG="$ROOT/docker_verification.log"
NAME="mtlnn_verify"
PORT="${PORT:-8000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-72}"     # 72 x 5min = 6h ceiling
RETRY_SECS="${RETRY_SECS:-300}"        # 5 minutes

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "==== docker autoverify started (pid $$) ===="

# Try to nudge Docker Desktop awake (no-op if already running / not installed).
if command -v powershell.exe >/dev/null 2>&1; then
  powershell.exe -NoProfile -Command \
    "Start-Process 'C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe' -ErrorAction SilentlyContinue" \
    >/dev/null 2>&1 || true
fi

# 1) Wait for the daemon, polling every RETRY_SECS.
up=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if timeout 15 docker info >/dev/null 2>&1; then
    log "daemon UP on attempt ${attempt}/${MAX_ATTEMPTS}"
    up=1
    break
  fi
  log "daemon not reachable (attempt ${attempt}/${MAX_ATTEMPTS}); retrying in ${RETRY_SECS}s"
  sleep "$RETRY_SECS"
done

if [ "$up" -ne 1 ]; then
  log "FAILED: Docker daemon never became reachable within the time ceiling."
  exit 1
fi

docker version --format 'client={{.Client.Version}} server={{.Server.Version}}' 2>>"$LOG" | tee -a "$LOG"

# 2) Build the image.
log "building image 'mtlnn' ..."
if docker build -t mtlnn . >>"$LOG" 2>&1; then
  log "build OK"
else
  log "FAILED: docker build (see log above)"
  exit 2
fi

# 3) Run the container (SMALL mode: no checkpoint / tokenizer download).
docker rm -f "$NAME" >/dev/null 2>&1 || true
CID="$(docker run -d --name "$NAME" -e SMALL=1 -p "${PORT}:8000" mtlnn 2>>"$LOG")"
if [ -z "$CID" ]; then
  log "FAILED: docker run did not return a container id"
  exit 3
fi
log "container started: $CID"

# 4) Wait for /health to report ready.
ready=0
for i in $(seq 1 40); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 3
done
if [ "$ready" -ne 1 ]; then
  log "FAILED: /health never became reachable; dumping container logs"
  docker logs "$NAME" >>"$LOG" 2>&1
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  exit 4
fi

# 5) Verify endpoints.
{
  echo "================ ENDPOINT VERIFICATION ================"
  echo "----- GET /health -----"
  curl -s "http://localhost:${PORT}/health"; echo
  echo "----- GET /v1/model -----"
  curl -s "http://localhost:${PORT}/v1/model"; echo
  echo "----- POST /v1/completions -----"
  curl -s -X POST "http://localhost:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Hello world","max_new_tokens":20,"do_sample":false}'; echo
  echo "----- POST /v1/completions/stream (first lines) -----"
  curl -s -N -X POST "http://localhost:${PORT}/v1/completions/stream" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Hello world","max_new_tokens":10,"do_sample":false}' | head -20; echo
  echo "======================================================"
} >>"$LOG" 2>&1

log "endpoint verification complete"

# 6) Capture container logs and clean up.
echo "----- container logs -----" >>"$LOG"
docker logs "$NAME" >>"$LOG" 2>&1
docker rm -f "$NAME" >/dev/null 2>&1 || true
log "==== docker autoverify DONE (image 'mtlnn' built & verified) ===="
exit 0
