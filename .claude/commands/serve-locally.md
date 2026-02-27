Start the local development servers for testing all three UIs.

## Steps

1. Build the treatise (Markdown → HTML): `cd treatise && npm run build`
2. Check if backend is already running: `curl -sf http://localhost:9000/health`
3. If not running, start backend in background: `YATZY_BASE_PATH=. solver/target/release/yatzy`
4. Check if Vite is already running: `curl -sf -o /dev/null http://localhost:5173/yatzy/play/`
5. If not running, start Vite in background: `cd frontend && npm run dev`
6. Verify all endpoints respond:
   - `http://localhost:5173/yatzy/` (treatise)
   - `http://localhost:5173/yatzy/play/` (game UI)
   - `http://localhost:5173/yatzy/profile/` (profiler)
   - `http://localhost:5173/yatzy/api/health` (API proxy)
7. Report which URLs are ready.

## Notes

- Two servers total: backend (Rust, port 9000) + Vite (port 5173)
- Vite serves all three UIs and proxies `/yatzy/api/` to the backend
- URL structure mirrors production at `langkilde.se/yatzy/*`
- If ports are occupied, report which process owns them instead of failing
- Always rebuild treatise before serving (Markdown may have changed)
