#!/bin/bash
set -euo pipefail

# Full deploy: backend + frontend + treatise. Use when the solver changed.
#
# Works around the two known blockers (see the deploy-env-gotchas notes):
#   1. The backend cannot be cross-compiled on an Apple Silicon Mac (`cross` wants
#      an x86_64 host toolchain that cannot run on ARM). Instead, Dockerfile.backend
#      is multi-stage and compiles the binary INSIDE a linux/amd64 container, built
#      here with `docker build --platform linux/amd64` (QEMU emulation, slow but
#      reliable). No host Rust toolchain is involved.
#   2. The server has `docker compose` v2 (space), not `docker-compose` (hyphen).
#
# The oracle.bin (3 GB) is NOT baked into the image; it is a server-side volume
# mount (see docker-compose.yml). Only the neutral + four risk-detent tables and
# the percentile table are baked. The Cloudflare purge at the end is mandatory:
# without it the edge serves stale assets and the deploy looks green while users
# see the old build. Cloudflare creds live on the server at ~/.cloudflare.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/deploy"
BUILD_DIR="$DEPLOY_DIR/build"
SERVER_USER="daniel.langkilde"
SERVER_HOST="35.217.49.95"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH="ssh -i $SSH_KEY -o ConnectTimeout=15 $SERVER_USER@$SERVER_HOST"
SCP="scp -i $SSH_KEY"

echo "== Build frontend + treatise =="
cd "$PROJECT_ROOT/frontend" && npm run build >/dev/null 2>&1
cd "$PROJECT_ROOT/treatise" && npm run build >/dev/null 2>&1
echo "  ok"

echo "== Assemble backend context (source + baked tables, no oracle) =="
BCTX="$BUILD_DIR/backend"
rm -rf "$BCTX"; mkdir -p "$BCTX/data/strategy_tables"
cp "$PROJECT_ROOT/solver/Cargo.toml" "$PROJECT_ROOT/solver/Cargo.lock" "$BCTX/"
cp -r "$PROJECT_ROOT/solver/src" "$BCTX/src"
cp "$DEPLOY_DIR/Dockerfile.backend" "$BCTX/Dockerfile"
cp "$PROJECT_ROOT/data/strategy_tables/all_states.bin" "$BCTX/data/strategy_tables/"
for T in -0.080 -0.040 0.040 0.080; do
  SRC="$PROJECT_ROOT/data/strategy_tables/all_states_theta_${T}.bin"
  [ -f "$SRC" ] && cp "$SRC" "$BCTX/data/strategy_tables/" || echo "  WARNING: detent $T missing"
done
[ -f "$PROJECT_ROOT/data/strategy_tables/percentiles.bin" ] && \
  cp "$PROJECT_ROOT/data/strategy_tables/percentiles.bin" "$BCTX/data/strategy_tables/"
echo "  ok"

echo "== Assemble frontend context =="
FCTX="$BUILD_DIR/frontend"
rm -rf "$FCTX"; mkdir -p "$FCTX/apps/play"
cp -r "$PROJECT_ROOT/frontend/dist/"* "$FCTX/apps/play/"
mkdir -p "$FCTX/apps/profile/css" "$FCTX/apps/profile/js" "$FCTX/apps/profile/data"
cp "$PROJECT_ROOT/profiler/index.html" "$FCTX/apps/profile/index.html"
cp -r "$PROJECT_ROOT/profiler/css/"* "$FCTX/apps/profile/css/"
cp -r "$PROJECT_ROOT/profiler/js/"* "$FCTX/apps/profile/js/"
cp -r "$PROJECT_ROOT/profiler/data/"* "$FCTX/apps/profile/data/"
mkdir -p "$FCTX/apps/probabilities/css" "$FCTX/apps/probabilities/js"
cp "$PROJECT_ROOT/probabilities/index.html" "$FCTX/apps/probabilities/index.html"
cp -r "$PROJECT_ROOT/probabilities/css/"* "$FCTX/apps/probabilities/css/"
cp -r "$PROJECT_ROOT/probabilities/js/"* "$FCTX/apps/probabilities/js/"
rm -f "$FCTX/apps/probabilities/js/"*.test.js
cp "$DEPLOY_DIR/nginx.conf" "$FCTX/nginx.conf"
cp "$DEPLOY_DIR/entrypoint.sh" "$FCTX/entrypoint.sh"
cp "$DEPLOY_DIR/Dockerfile.frontend" "$FCTX/Dockerfile"
echo "  ok"

echo "== Build images (linux/amd64; backend compiles inside, slow under emulation) =="
docker build --platform linux/amd64 -t yatzy-backend "$BCTX"
docker build --platform linux/amd64 -t yatzy-frontend "$FCTX"
echo "  backend:  $(docker image inspect yatzy-backend --format='{{.Size}}' | awk '{printf "%.1f MB", $1/1048576}')"
echo "  frontend: $(docker image inspect yatzy-frontend --format='{{.Size}}' | awk '{printf "%.1f MB", $1/1048576}')"

echo "== Save + compress + transfer =="
docker save yatzy-backend yatzy-frontend | gzip > "$BUILD_DIR/yatzy-images.tar.gz"
echo "  $(du -h "$BUILD_DIR/yatzy-images.tar.gz" | cut -f1)"
$SSH "mkdir -p ~/yatzy"
$SCP "$DEPLOY_DIR/docker-compose.yml" "$SERVER_USER@$SERVER_HOST:~/yatzy/docker-compose.yml"
$SCP "$BUILD_DIR/yatzy-images.tar.gz" "$SERVER_USER@$SERVER_HOST:~/yatzy-images.tar.gz"
echo "  ok"

echo "== Load + restart (docker compose v2) =="
$SSH 'set -e; docker load < ~/yatzy-images.tar.gz >/dev/null; cd ~/yatzy && docker compose up -d --force-recreate; rm -f ~/yatzy-images.tar.gz; docker image prune -f >/dev/null'
echo "  ok"

echo "== Sync treatise + shared (volume mount serving /shared to every UI) =="
rsync -az -e "ssh -i $SSH_KEY" --delete \
  --exclude='node_modules/' --exclude='package*.json' --exclude='build.mjs' --exclude='sections/*.md' \
  "$PROJECT_ROOT/treatise/" "$SERVER_USER@$SERVER_HOST:~/yatzy-treatise/"
rsync -az -e "ssh -i $SSH_KEY" "$PROJECT_ROOT/shared/" "$SERVER_USER@$SERVER_HOST:~/yatzy-treatise/shared/"
echo "  ok"

echo "== Purge Cloudflare (edge serves stale assets otherwise) =="
$SSH bash -s <<'PURGE'
if [ ! -f ~/.cloudflare ]; then echo "  ~/.cloudflare not found, skipping"; exit 0; fi
CF_ZONE_ID=$(sed -n '1p' ~/.cloudflare); CF_TOKEN=$(sed -n '2p' ~/.cloudflare)
R=$(curl -s -X POST "https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/purge_cache" \
  -H "Authorization: Bearer ${CF_TOKEN}" -H "Content-Type: application/json" --data '{"purge_everything":true}')
echo "$R" | grep -q '"success":true\|"success": true' && echo "  purged" || { echo "  PURGE FAILED"; echo "$R" | head -c 300; }
PURGE

echo "== Health + status =="
$SSH 'cd ~/yatzy && docker compose ps --format "{{.Service}}\t{{.Status}}"; echo -n "backend detents: "; docker exec yatzy-backend curl -sf http://localhost:9000/evaluate -H "content-type: application/json" -d "{\"dice\":[1,2,3,4,5],\"upper_score\":0,\"scored_categories\":0,\"rerolls_remaining\":2}" | grep -o "\"detents\"" | head -1 || echo "(no detents!)"'
echo "DEPLOY DONE"
