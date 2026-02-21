# Target Structure

## Guiding Principle

The current structure is **already good**. The solver is fast, the components are cleanly separated, and the data flow is well-defined. Phase 4 should be conservative — fix genuine organizational problems without introducing churn.

## Rust Backend (`solver/`)

### Current State
- Single crate, 28 binaries, ~18K LOC
- Well-organized core modules (widget_solver, batched_solver, simd, state_computation)
- Simulation subsystem cleanly separated (simulation/, density/, profiling/, rosetta/, scenarios/)
- Hot paths heavily optimized with NEON SIMD

### Proposed Changes

**No structural changes.** The crate organization is sound:
- Core DP: `widget_solver.rs`, `batched_solver.rs`, `simd.rs`, `state_computation.rs`
- Game logic: `constants.rs`, `types.rs`, `dice_mechanics.rs`, `game_mechanics.rs`
- Infrastructure: `storage.rs`, `phase0_tables.rs`
- API: `server.rs`, `api_computations.rs`
- Simulation: `simulation/` (engine, lockstep, sweep, statistics, etc.)
- Analysis: `density/`, `profiling/`, `rosetta/`, `scenarios/`

**Minor improvements:**
1. Add `// PERF: intentional` comments to the 4 batched solver variants explaining why they're separate
2. Add `// PERF: intentional` to lockstep vs sequential simulation
3. The 28 binaries are fine as-is — each is a self-contained tool

### What NOT To Do
- Do NOT create a workspace — the single crate compiles efficiently with shared deps
- Do NOT extract a CLI framework — manual arg parsing is simpler for this use case
- Do NOT merge binaries — each has a distinct purpose and independent CLI

## Frontend (`frontend/`)

### Current State
- React 19 + TypeScript + Vite
- Clean: App.tsx, reducer.ts, types.ts, api.ts
- Tests: 34 (mask + reducer)

### Proposed Changes

**No structural changes.** The frontend is lean and well-organized. The overhaul plan's Phase 6 proposes migrating to vanilla TypeScript + D3.js, but that's a separate phase decision.

## Analytics (`analytics/`)

### Current State
- Python package with Click CLI (1,501 LOC cli.py)
- 14 core modules + 25 plot modules
- ~20K LOC total

### Proposed Changes

**Defer to Phase 7.** The overhaul plan schedules analytics restructuring (pandas→Polars, cli.py split) for Phase 7. No structural changes now.

## Blog (`blog/`)

### Current State
- Static site, vanilla JS, no build system
- Game code duplicated from frontend (see duplication-audit.md)

### Proposed Changes

**No structural changes.** The blog's static nature is intentional — no build step means easy deployment.

## Overall Assessment

The current project structure is mature and well-organized. The main organizational debt is:
1. cli.py monolith → Phase 7
2. Frontend framework → Phase 6 (if approved)
3. Solver `// PERF:` annotations → Phase 5

**Phase 4 action items are limited to:**
1. Add `// PERF: intentional` comments to batched solver variants
2. Add `// PERF: intentional` comments to lockstep/sequential simulation
3. Write this audit documentation (done)

This is a conservative outcome — the codebase doesn't need major restructuring.
