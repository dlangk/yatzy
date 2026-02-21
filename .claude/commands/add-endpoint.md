Add a new API endpoint to the Rust backend: $ARGUMENTS

1. Read `solver/CLAUDE.md` to understand existing endpoint patterns.
2. Add request/response types in `solver/src/server.rs` (derive `Deserialize`/`Serialize`).
3. Implement the handler function in `solver/src/server.rs`.
4. If computation is non-trivial, add the logic to `solver/src/api_computations.rs`.
5. Register the route in `create_router_with_state()`.
6. Classify latency: `gameplay-critical` (must be <1ms) or `batch-ok` (can take seconds).
7. If gameplay-critical, ensure no blocking calls. If batch-ok, use `tokio::spawn_blocking`.
8. Run `cargo test` to verify nothing breaks.
9. Run `just bench-check` to verify no performance regression.
10. Update `solver/CLAUDE.md` API Reference section with the new endpoint.
11. Update `frontend/src/api.ts` if the frontend will call it.
