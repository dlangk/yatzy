# Duplication Audit

## Cross-Component Duplication

### 1. Frontend ↔ Blog Game Logic (~280 LOC duplicated)

The blog game (`blog/js/game/`) is a vanilla JS reimplementation of the React frontend game logic.

| File Pair | Frontend | Blog | Overlap | Blog Status |
|-----------|----------|------|---------|-------------|
| reducer.ts / reducer.js | 355 | 290 | 82% (238 lines) | Subset (no trajectory) |
| api.ts / api.js | 64 | 22 | 100% of blog | Subset (2 of 4 endpoints) |
| mask.ts / mask.js | 67 | 46 | 98% | Identical logic |
| constants.ts / constants.js | 27 | 11 | 100% of blog | Subset |

**Source of truth:** Rust solver is authoritative for game rules. Frontend/blog should call the API, not reimplement. However, the frontend reducer manages local UI state (which dice are held, which category is selected) — this is legitimately client-side logic.

**Verdict:** The blog game intentionally lacks trajectory features and density calls. The core game reducer logic (~238 LOC) is genuinely duplicated. However, the blog is a static site with no build system, so it can't import from the React frontend. **Acceptable duplication** — the alternative (sharing a build artifact) would add complexity disproportionate to the risk.

**Action:** Document as known duplication. If game rules change, update both places. Tests exist for both (frontend: 34, blog: 25).

### 2. Constants Duplication (3 codebases)

Game constants are independently defined in:
1. `solver/src/constants.rs` — canonical (CATEGORY_COUNT=15, UPPER_BONUS=50, UPPER_SCORE_CAP=63)
2. `frontend/src/types.ts` + `constants.ts` — TypeScript
3. `blog/js/game/constants.js` — vanilla JS

**Risk:** If Scandinavian Yatzy rules change (they won't — it's a fixed game), three places need updating.

**Verdict:** Low risk. The game rules are a fixed specification. **Leave duplicated.**

### 3. Binary Format Specs (Rust ↔ Python)

Binary file format specs duplicated between:
1. `solver/src/storage.rs` + `simulation/raw_storage.rs` — Rust writer/reader
2. `analytics/src/yatzy_analysis/io.py` + `config.py` — Python reader

**Risk:** If the Rust writer changes format version, the Python reader breaks.

**Verdict:** This is inherent in cross-language binary I/O. The format is versioned (magic + version header) and very stable (version 5, unchanged for months). **Leave as-is.** The version check will catch mismatches.

## Intra-Component Duplication

### 4. Rust Binary Context Initialization

25/28 binaries share the same pattern:
```rust
let mut ctx = YatzyContext::new_boxed();
phase0_tables::precompute_lookup_tables(&mut ctx);
load_all_state_values(&mut ctx, &state_file_path(theta));
```

9 binaries duplicate a `set_working_directory()` function verbatim.

**Verdict:** This is ~5 lines per binary. Extracting a helper would save a trivial amount of code while adding indirection. **Leave duplicated** — the pattern is simple and grep-able. The `set_working_directory` function could be extracted to a shared utility, but it's not worth a separate phase.

### 5. Rust Binary CLI Parsing

24/28 binaries parse `--theta`, 20 parse `--games`, 18 parse `--output`. 10 define their own `parse_args()` function.

**Verdict:** Each binary has slightly different argument sets. A shared CLI framework (clap) would reduce boilerplate but add a dependency and change the binary interface. **Not worth changing** — manual parsing is simple and each binary is self-contained.

### 6. Analytics cli.py Monolith (1,501 LOC)

40+ Click commands in one file. Could be split into command groups.

**Verdict:** This is a real smell. However, the overhaul plan doesn't include analytics restructuring until Phase 7. **Defer to Phase 7.**

## Near-Duplicates

### 7. Four Solver Variants (batched_solver.rs)

`solve_widget_batched()`, `solve_widget_batched_risk()`, `solve_widget_batched_utility()`, `solve_widget_batched_max()` are near-duplicates with different inner-loop math.

**Verdict:** These are **intentionally duplicated hot-path code**. Each variant is optimized for its specific case (EV vs risk vs utility vs max). Unifying behind a generic interface would add branch prediction overhead. **Keep separate. Mark with `// PERF: intentional`.**

### 8. Lockstep vs Sequential Simulation

`simulation/lockstep.rs` and `simulation/engine.rs` both simulate games but with very different strategies (horizontal vs vertical).

**Verdict:** **Intentionally separate** — different performance characteristics. Lockstep is 4.5x faster but more complex.

## Dead Code Assessment

| Module | Status | Used By |
|--------|--------|---------|
| `simulation/strategy.rs` (743 LOC) | USED | 3 binaries (human_baseline, multiplayer, underdog_sweep) |
| `analytics/density.py` (305 LOC) | USED | cli.py (alternative density data pipeline) |
| `analytics/adaptive.py` (91 LOC) | USED | cli.py adaptive command |

No dead code found — all flagged modules are actively used.

## Summary

| Category | Items | Action |
|----------|-------|--------|
| Frontend/Blog game duplication | ~280 LOC | Leave (different build systems) |
| Constants duplication | ~50 LOC across 3 codebases | Leave (fixed game spec) |
| Binary format specs | ~100 LOC across 2 languages | Leave (versioned, stable) |
| Binary context init | ~5 LOC × 25 binaries | Leave (trivial) |
| Binary CLI parsing | ~20 LOC × 10 binaries | Leave (each binary is unique) |
| cli.py monolith | 1,501 LOC | Defer to Phase 7 |
| Solver variants | 4 × ~200 LOC | Keep (PERF: intentional) |
| Dead code | None found | — |
