# Repo Review — 2026-04-06

## Summary

The yatzy repo is in good shape. All four components (solver, frontend, treatise, analytics) have consistent API contracts, clean dependency graphs, and zero code debt markers. One genuine gap: the analytics Python package has no CI coverage. Everything else passes cleanly across all five audit phases. This is a healthy, mature codebase with intentional structure.

## Findings

### CRITICAL

_None._

### MODERATE

#### Analytics component has no CI coverage
- **Location:** `.github/workflows/ci.yml` — no analytics steps present
- **Evidence:** CI covers Rust (format, clippy, 185 tests), TypeScript (eslint, typecheck, 46 tests), and treatise build (`node build.mjs`). Analytics (`analytics/`) is absent entirely.
- **Impact:** Python analysis pipelines can silently break between runs. No automated check catches import errors, API changes, or broken CLI commands.
- **Recommendation:** Add a CI job: `cd analytics && uv venv && uv pip install -e . && uv run ruff check . && uv run pytest` (if tests exist). At minimum add `ruff check`.

### LOW

#### `timing` feature in `solver/Cargo.toml` is undocumented
- **Location:** `solver/Cargo.toml` line with `timing = []`; `solver/src/state_computation.rs:367`, `solver/src/widget_solver.rs`
- **Evidence:** Feature exists and is used via `#[cfg(feature = "timing")]` guards around profiling output. But it is not mentioned in `solver/CLAUDE.md` or any justfile recipe.
- **Impact:** Future contributors may not know it exists or how to enable it.
- **Recommendation:** Add a one-line note to `solver/CLAUDE.md` under build commands: `cargo build --features timing` enables per-level timing output.

## Cross-component coherence status

| Concern | Status |
|---------|--------|
| API endpoints (Rust ↔ TypeScript) | All 4 endpoints match: `/health`, `/state_value`, `/evaluate`, `/density` |
| Wire format naming | Rust uses snake_case; TypeScript converts at boundary in `api.ts` — correct |
| Shared files (`shared/nav.js`, `shared/dice.js`, `shared/dice.css`) | Referenced correctly from all three UIs |
| CLAUDE.md cross-references | Root, solver, frontend, treatise, profiler, analytics all consistent |
| Vite proxy ↔ backend port | Both use port 9000; proxy rewrite strips `/yatzy/api` prefix correctly |
| Deploy paths | `deploy.sh` + `nginx.conf` + `docker-compose.yml` all aligned with URL structure |
| Lock files | `Cargo.lock`, `frontend/package-lock.json`, `treatise/package-lock.json`, `analytics/uv.lock` all present |

## Recommendations

**Quick wins (< 30 min):**
- Add a note to `solver/CLAUDE.md` documenting the `--features timing` build flag

**Medium effort (a few hours):**
- Add analytics CI job to `.github/workflows/ci.yml` — `ruff check` at minimum, `pytest` if test coverage grows

## What was checked and found clean

- No orphaned/ghost files (legacy C code in `legacy/` is explicitly documented as reference-only)
- No duplicate source files
- No generated artifacts tracked in git
- No TODO/FIXME/HACK/XXX markers anywhere in source
- No large commented-out code blocks
- No unused direct dependencies in any component
- No API endpoint/consumer mismatches (every endpoint has a frontend caller; every frontend call hits a live endpoint)
- No naming drift between components beyond the intentional snake_case ↔ camelCase boundary at the API layer
- `.DS_Store` is in `.gitignore` and not tracked
