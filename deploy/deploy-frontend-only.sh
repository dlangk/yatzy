#!/bin/bash
set -euo pipefail

# Frontend-only deploy: use when no solver code changed.
#
# Skips the backend cross-compile, which fails on an ARM Mac ("toolchain
# stable-x86_64-unknown-linux-gnu may not be able to run on this system"), and
# leaves the running backend container alone via --no-deps. Mirrors the frontend
# half of deploy.sh, and uses `docker compose` v2, since the server has no
# `docker-compose` binary.
#
# The Cloudflare purge at the end is NOT optional: without it the edge keeps
# serving stale HTML and the deploy looks green while users see the old build.
# Credentials live on the server at ~/.cloudflare, not on the Mac.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/deploy"
BUILD_DIR="$DEPLOY_DIR/build"
SERVER_USER="daniel.langkilde"
SERVER_HOST="35.217.49.95"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH="ssh -i $SSH_KEY -o ConnectTimeout=15 $SERVER_USER@$SERVER_HOST"
SCP="scp -i $SSH_KEY"

echo "── Build frontend"
cd "$PROJECT_ROOT/frontend" && npm run build >/dev/null 2>&1
echo "  ok"

echo "── Build treatise"
cd "$PROJECT_ROOT/treatise" && npm run build >/dev/null 2>&1
echo "  ok"

echo "── Assemble frontend context"
FRONTEND_CTX="$BUILD_DIR/frontend"
rm -rf "$FRONTEND_CTX"
mkdir -p "$FRONTEND_CTX/apps"

mkdir -p "$FRONTEND_CTX/apps/play"
cp -r "$PROJECT_ROOT/frontend/dist/"* "$FRONTEND_CTX/apps/play/"

mkdir -p "$FRONTEND_CTX/apps/profile/css" "$FRONTEND_CTX/apps/profile/js" "$FRONTEND_CTX/apps/profile/data"
cp "$PROJECT_ROOT/profiler/index.html" "$FRONTEND_CTX/apps/profile/index.html"
cp -r "$PROJECT_ROOT/profiler/css/"* "$FRONTEND_CTX/apps/profile/css/"
cp -r "$PROJECT_ROOT/profiler/js/"* "$FRONTEND_CTX/apps/profile/js/"
cp -r "$PROJECT_ROOT/profiler/data/"* "$FRONTEND_CTX/apps/profile/data/"

mkdir -p "$FRONTEND_CTX/apps/probabilities/css" "$FRONTEND_CTX/apps/probabilities/js"
cp "$PROJECT_ROOT/probabilities/index.html" "$FRONTEND_CTX/apps/probabilities/index.html"
cp -r "$PROJECT_ROOT/probabilities/css/"* "$FRONTEND_CTX/apps/probabilities/css/"
cp -r "$PROJECT_ROOT/probabilities/js/"* "$FRONTEND_CTX/apps/probabilities/js/"
rm -f "$FRONTEND_CTX/apps/probabilities/js/"*.test.js

cp "$DEPLOY_DIR/nginx.conf" "$FRONTEND_CTX/nginx.conf"
cp "$DEPLOY_DIR/entrypoint.sh" "$FRONTEND_CTX/entrypoint.sh"
cp "$DEPLOY_DIR/Dockerfile.frontend" "$FRONTEND_CTX/Dockerfile"
echo "  ok"

echo "── Build image (linux/amd64)"
docker build --platform linux/amd64 -t yatzy-frontend "$FRONTEND_CTX" >/dev/null
echo "  $(docker image inspect yatzy-frontend --format='{{.Size}}' | awk '{printf "%.1f MB", $1/1048576}')"

echo "── Save + compress"
docker save yatzy-frontend | gzip > "$BUILD_DIR/yatzy-frontend.tar.gz"
echo "  $(du -h "$BUILD_DIR/yatzy-frontend.tar.gz" | cut -f1)"

echo "── Transfer"
$SCP "$BUILD_DIR/yatzy-frontend.tar.gz" "$SERVER_USER@$SERVER_HOST:~/yatzy-frontend.tar.gz"
echo "  ok"

echo "── Load + restart (docker compose v2, --no-deps so the backend is untouched)"
$SSH 'set -e; docker load < ~/yatzy-frontend.tar.gz >/dev/null; cd ~/yatzy && docker compose up -d --no-deps yatzy-frontend; rm -f ~/yatzy-frontend.tar.gz'
echo "  ok"

echo "── Sync treatise + shared (volume mount that serves /shared/nav.js to every UI)"
rsync -az -e "ssh -i $SSH_KEY" --delete "$PROJECT_ROOT/treatise/" "$SERVER_USER@$SERVER_HOST:~/yatzy-treatise/"
rsync -az -e "ssh -i $SSH_KEY" "$PROJECT_ROOT/shared/" "$SERVER_USER@$SERVER_HOST:~/yatzy-treatise/shared/"
echo "  ok"

echo "── Purge Cloudflare (edge serves stale HTML otherwise)"
$SSH bash -s <<'PURGE'
if [ ! -f ~/.cloudflare ]; then echo "  ~/.cloudflare not found, skipping"; exit 0; fi
CF_ZONE_ID=$(sed -n '1p' ~/.cloudflare)
CF_TOKEN=$(sed -n '2p' ~/.cloudflare)
R=$(curl -s -X POST "https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/purge_cache" \
  -H "Authorization: Bearer ${CF_TOKEN}" -H "Content-Type: application/json" \
  --data '{"purge_everything":true}')
echo "$R" | grep -q '"success":true\|"success": true' && echo "  purged" || { echo "  PURGE FAILED"; echo "$R" | head -c 300; }
PURGE

echo "── Status"
$SSH 'cd ~/yatzy && docker compose ps --format "{{.Service}}\t{{.Status}}"'
echo "DEPLOY DONE"
