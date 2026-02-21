---
paths:
  - "solver/src/server.rs"
  - "solver/src/api_computations.rs"
  - "solver/src/bin/server.rs"
---

# API Server Rules

The web server layer MUST NEVER panic on user input. Handlers must catch errors and return HTTP 400/500. This is the opposite of the HPC core, where unwrap()/expect() on invariants is fine.

- Gameplay endpoints (`/evaluate`, `/state_value`) are latency-critical: no blocking, no unnecessary allocation.
- The server is stateless — all state lives in `Arc<YatzyContext>` loaded at startup via mmap.
- CPU-heavy computation (density evolution) uses `tokio::spawn_blocking`.
- Response types must derive `Serialize`.
- When adding/changing an endpoint, update `solver/CLAUDE.md` API Reference section.
- Keep error handling minimal but correct — return sensible HTTP status codes.
