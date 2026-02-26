#!/bin/bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/deploy"
BUILD_DIR="$DEPLOY_DIR/build"
SERVER_USER="daniel.langkilde"
SERVER_HOST="35.217.49.95"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH="ssh -i $SSH_KEY $SERVER_USER@$SERVER_HOST"
SCP="scp -i $SSH_KEY"
TARGET="x86_64-unknown-linux-gnu"

echo "=== Yatzy Deploy ==="
echo "Project root: $PROJECT_ROOT"

# ── Step 1: Cross-compile Rust solver ────────────────────────────────────────
echo ""
echo "── Step 1: Cross-compile solver for $TARGET"
cd "$PROJECT_ROOT/solver"
cross build --release --target "$TARGET"
echo "✓ Solver compiled"

# ── Step 2: Build frontend ───────────────────────────────────────────────────
echo ""
echo "── Step 2: Build frontend"
cd "$PROJECT_ROOT/frontend"
npm run build
echo "✓ Frontend built"

# ── Step 2b: Build treatise (Markdown → HTML) ──────────────────────────────
echo ""
echo "── Step 2b: Build treatise sections"
cd "$PROJECT_ROOT/treatise"
npm run build
echo "✓ Treatise built"

# ── Step 3: Verify strategy table ────────────────────────────────────────────
echo ""
echo "── Step 3: Verify strategy table"
STRATEGY_TABLE="$PROJECT_ROOT/data/strategy_tables/all_states.bin"
if [ ! -f "$STRATEGY_TABLE" ]; then
    echo "ERROR: Strategy table not found at $STRATEGY_TABLE"
    echo "Run 'just precompute' first."
    exit 1
fi
echo "✓ Strategy table exists ($(du -h "$STRATEGY_TABLE" | cut -f1))"

# ── Step 4: Assemble build contexts ─────────────────────────────────────────
echo ""
echo "── Step 4: Assemble build contexts"
rm -rf "$BUILD_DIR"

# Backend context
BACKEND_CTX="$BUILD_DIR/backend"
mkdir -p "$BACKEND_CTX/data/strategy_tables"
cp "$PROJECT_ROOT/solver/target/$TARGET/release/yatzy" "$BACKEND_CTX/yatzy"
cp "$STRATEGY_TABLE" "$BACKEND_CTX/data/strategy_tables/all_states.bin"
cp "$DEPLOY_DIR/Dockerfile.backend" "$BACKEND_CTX/Dockerfile"
echo "✓ Backend context assembled"

# Frontend context
FRONTEND_CTX="$BUILD_DIR/frontend"
mkdir -p "$FRONTEND_CTX/html"

# Treatise at root
cp -r "$PROJECT_ROOT/treatise/"* "$FRONTEND_CTX/html/"

# Game UI at /play/
mkdir -p "$FRONTEND_CTX/html/play"
cp -r "$PROJECT_ROOT/frontend/dist/"* "$FRONTEND_CTX/html/play/"

# Profiler at /profile/
mkdir -p "$FRONTEND_CTX/html/profile/css" "$FRONTEND_CTX/html/profile/js" "$FRONTEND_CTX/html/profile/data"
cp "$PROJECT_ROOT/blog/profile.html" "$FRONTEND_CTX/html/profile/index.html"
# Fix "Back to article" link to point to treatise root
sed -i '' 's|href="index.html"|href="/yatzy/"|g' "$FRONTEND_CTX/html/profile/index.html"
cp -r "$PROJECT_ROOT/blog/css/"* "$FRONTEND_CTX/html/profile/css/"
cp -r "$PROJECT_ROOT/blog/js/"* "$FRONTEND_CTX/html/profile/js/"
cp -r "$PROJECT_ROOT/blog/data/"* "$FRONTEND_CTX/html/profile/data/"

# Nginx config and entrypoint
cp "$DEPLOY_DIR/nginx.conf" "$FRONTEND_CTX/nginx.conf"
cp "$DEPLOY_DIR/entrypoint.sh" "$FRONTEND_CTX/entrypoint.sh"
cp "$DEPLOY_DIR/Dockerfile.frontend" "$FRONTEND_CTX/Dockerfile"
echo "✓ Frontend context assembled"

# ── Step 5: Build Docker images ─────────────────────────────────────────────
echo ""
echo "── Step 5: Build Docker images"
docker build --platform linux/amd64 -t yatzy-backend "$BACKEND_CTX"
docker build --platform linux/amd64 -t yatzy-frontend "$FRONTEND_CTX"
echo "✓ Docker images built"

# Show image sizes
echo "  Backend:  $(docker image inspect yatzy-backend --format='{{.Size}}' | awk '{printf "%.1f MB", $1/1048576}')"
echo "  Frontend: $(docker image inspect yatzy-frontend --format='{{.Size}}' | awk '{printf "%.1f MB", $1/1048576}')"

# ── Step 6: Save and compress images ────────────────────────────────────────
echo ""
echo "── Step 6: Save and compress images"
docker save yatzy-backend yatzy-frontend | gzip > "$BUILD_DIR/yatzy-images.tar.gz"
echo "✓ Images saved ($(du -h "$BUILD_DIR/yatzy-images.tar.gz" | cut -f1))"

# ── Step 7: Transfer to server ──────────────────────────────────────────────
echo ""
echo "── Step 7: Transfer to server"
$SCP "$BUILD_DIR/yatzy-images.tar.gz" "$SERVER_USER@$SERVER_HOST:~/yatzy-images.tar.gz"
$SSH "mkdir -p ~/yatzy"
$SCP "$DEPLOY_DIR/docker-compose.yml" "$SERVER_USER@$SERVER_HOST:~/yatzy/docker-compose.yml"
echo "✓ Files transferred"

# ── Step 8: Deploy on server ────────────────────────────────────────────────
echo ""
echo "── Step 8: Deploy on server"
$SSH bash -s <<'REMOTE'
set -euo pipefail
echo "Loading Docker images..."
docker load < ~/yatzy-images.tar.gz
echo "Restarting containers..."
cd ~/yatzy
docker-compose down || true
docker-compose up -d
echo "Cleaning up..."
rm ~/yatzy-images.tar.gz
echo "Waiting for containers to start..."
sleep 5
REMOTE
echo "✓ Containers deployed"

# ── Step 9: Health checks ───────────────────────────────────────────────────
echo ""
echo "── Step 9: Health checks"
$SSH bash -s <<'HEALTHCHECK'
set -e
echo -n "Backend health: "
docker exec yatzy-backend curl -sf http://localhost:9000/health && echo " OK" || echo " FAIL"
echo -n "Treatise:       "
docker exec yatzy-frontend wget -q --spider http://localhost:8090/ && echo "OK" || echo "FAIL"
echo -n "Game UI:        "
docker exec yatzy-frontend wget -q --spider http://localhost:8090/play/ && echo "OK" || echo "FAIL"
echo -n "Profiler:       "
docker exec yatzy-frontend wget -q --spider http://localhost:8090/profile/ && echo "OK" || echo "FAIL"
echo -n "External:       "
curl -sk -H "Host: langkilde.se" -o /dev/null -w "%{http_code}" https://localhost/yatzy/ && echo " OK" || echo " FAIL"
HEALTHCHECK

echo ""
echo "=== Deploy complete ==="
echo "  Treatise:  https://langkilde.se/yatzy/"
echo "  Game:      https://langkilde.se/yatzy/play/"
echo "  Profiler:  https://langkilde.se/yatzy/profile/"
echo "  API:       https://langkilde.se/yatzy/api/health"
