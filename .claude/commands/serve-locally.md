Start the local development servers for testing all three UIs.

IMPORTANT: All commands MUST use absolute paths. Background bash commands do NOT inherit the working directory. Use the repo root `/Users/langkilde/dev/yatzy` in all paths.

## Steps

1. Build the treatise (Markdown → HTML): `cd /Users/langkilde/dev/yatzy/treatise && npm run build`
2. Check if backend is already running: `curl -sf http://localhost:9000/health`
3. Check if Vite is already running: `curl -sf -o /dev/null http://localhost:5173/yatzy/play/`
4. Start whichever servers are not running (in parallel, both as background tasks):
   - Backend: `cd /Users/langkilde/dev/yatzy && YATZY_BASE_PATH=/Users/langkilde/dev/yatzy solver/target/release/yatzy`
   - Vite: `cd /Users/langkilde/dev/yatzy/frontend && npm run dev`
5. Wait 3 seconds, then verify all endpoints respond:
   - `http://localhost:5173/yatzy/` (treatise)
   - `http://localhost:5173/yatzy/play/` (game UI)
   - `http://localhost:5173/yatzy/profile/` (profiler)
   - `http://localhost:5173/yatzy/api/health` (API proxy)
6. Report which URLs are ready.

## Notes

- Two servers total: backend (Rust, port 9000) + Vite (port 5173)
- Vite serves all three UIs and proxies `/yatzy/api/` to the backend
- URL structure mirrors production at `langkilde.se/yatzy/*`
- If ports are occupied, report which process owns them instead of failing
- Always rebuild treatise before serving (Markdown may have changed)
- Steps 2+3 can run in parallel. Step 4 servers can start in parallel. Step 5 must wait for step 4.
