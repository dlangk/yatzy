Deploy the yatzy system to production at langkilde.se/yatzy.

Server: ssh -i ~/.ssh/id_ed25519 daniel.langkilde@35.217.49.95

## Deploy workflows

| Scenario | Command | Time |
|----------|---------|------|
| Treatise only | `just deploy-treatise` | ~3 seconds |
| Full system | `just deploy` | ~3 minutes (includes treatise rsync) |

## Pre-deploy checks

1. Run `cargo test` in `solver/` — all 182 tests must pass.
2. Run `npm test` in `frontend/` — all 46 tests must pass.
3. Run `npm run lint` in `frontend/` and `cargo clippy` in `solver/` — no errors.
4. Run `just bench-check` — no performance regressions.

## Build

5. Cross-compile solver: `cd solver && cross build --release --target x86_64-unknown-linux-gnu`
6. Build frontend: `cd frontend && npm run build`
7. Build treatise: `cd treatise && npm run build`
8. Verify strategy table exists: `data/strategy_tables/all_states.bin` (if not, run `just precompute`)

## Package

9. Assemble build contexts in `deploy/build/`:
   - `deploy/build/backend/`: copy cross-compiled binary + Dockerfile.backend + data/strategy_tables/all_states.bin
   - `deploy/build/frontend/`: copy frontend/dist/* to apps/play/, profiler assets to apps/profile/ (sed "Back to article" link to /yatzy/), copy nginx.conf + entrypoint.sh + Dockerfile.frontend
10. Build Docker images: `docker build --platform linux/amd64 -t yatzy-backend deploy/build/backend` and same for frontend.
11. Save images: `docker save yatzy-backend yatzy-frontend | gzip > deploy/build/yatzy-images.tar.gz`

## Deploy

12. Transfer: `scp -i ~/.ssh/id_ed25519 deploy/build/yatzy-images.tar.gz daniel.langkilde@35.217.49.95:~/yatzy-images.tar.gz`
13. Transfer compose: `scp -i ~/.ssh/id_ed25519 deploy/docker-compose.yml daniel.langkilde@35.217.49.95:~/yatzy/docker-compose.yml`
14. SSH and deploy:
    - `docker load < ~/yatzy-images.tar.gz`
    - `cd ~/yatzy && docker-compose down && docker-compose up -d`
    - `rm ~/yatzy-images.tar.gz`
15. Rsync treatise to `~/yatzy-treatise/` (volume-mounted into nginx container at `/usr/share/nginx/html`)

## Architecture

- **Treatise**: served from host volume mount (`~/yatzy-treatise` → `/usr/share/nginx/html`)
- **Game UI + Profiler**: baked into Docker image at `/usr/share/nginx/apps/`
- Fast treatise deploys: `just deploy-treatise` rsyncs directly, no Docker rebuild needed

## Post-deploy verification

16. SSH health checks (containers aren't exposed on host, use docker exec):
    - `docker exec yatzy-backend curl -sf http://localhost:9000/health` (backend responds)
    - `docker exec yatzy-frontend wget -q --spider http://localhost:8090/` (treatise HTML)
    - `docker exec yatzy-frontend wget -q --spider http://localhost:8090/play/` (game UI HTML)
    - `docker exec yatzy-frontend wget -q --spider http://localhost:8090/profile/` (profiler HTML)
17. External checks:
    - `curl -sf https://langkilde.se/yatzy/ | head -5` (treatise loads through nginx)
    - `curl -sf https://langkilde.se/yatzy/api/health` (API responds through nginx)
18. Report status: container health, image sizes, and endpoint responses.
