#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_USER="daniel.langkilde"
SERVER_HOST="35.217.49.95"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "=== Treatise Deploy ==="

# Build markdown → HTML
echo "── Building treatise"
cd "$PROJECT_ROOT/treatise"
npm run build
echo "✓ Built"

# Rsync to server (delta transfer, ~2 seconds)
echo "── Syncing to server"
rsync -az --delete \
  -e "ssh -i $SSH_KEY" \
  --exclude='node_modules/' \
  --exclude='package*.json' \
  --exclude='build.mjs' \
  --exclude='sections/*.md' \
  "$PROJECT_ROOT/treatise/" \
  "$SERVER_USER@$SERVER_HOST:/home/$SERVER_USER/yatzy-treatise/"
echo "✓ Synced"

# Health check
echo "── Health check"
ssh -i "$SSH_KEY" "$SERVER_USER@$SERVER_HOST" \
  'docker exec yatzy-frontend wget -q --spider http://localhost:8090/ && echo "✓ Treatise OK" || echo "✗ FAIL"'

# Purge Cloudflare cache
echo "── Purging Cloudflare cache"
ssh -i "$SSH_KEY" "$SERVER_USER@$SERVER_HOST" bash -s <<'PURGE'
if [ ! -f ~/.cloudflare ]; then
  echo "✗ ~/.cloudflare not found — skipping cache purge"
  exit 0
fi
CF_ZONE_ID=$(sed -n '1p' ~/.cloudflare)
CF_TOKEN=$(sed -n '2p' ~/.cloudflare)
CF_RESPONSE=$(curl -s -X POST "https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/purge_cache" \
  -H "Authorization: Bearer ${CF_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{"purge_everything":true}')
if echo "$CF_RESPONSE" | grep -q '"success":true\|"success": true'; then
  echo "✓ Cache purged"
else
  echo "✗ Cache purge may have failed"
  echo "$CF_RESPONSE"
fi
PURGE

echo ""
echo "=== Done — https://langkilde.se/yatzy/ ==="
