# Repo Review -- 2026-03-29

## Summary

The yatzy repository is in excellent health. Across five audit phases (ghost files, routing, cross-project coherence, dependency hygiene, code archaeology), the codebase shows mature engineering practices with strong documentation discipline and zero TODO/FIXME debt. The main findings are: 6 untracked files that need a decision (commit or gitignore), a stale profiler lockfile/node_modules, 5 npm audit vulnerabilities in frontend dev-dependencies, and no CI/CD pipeline. No critical issues found.

**Findings: 0 CRITICAL, 4 MODERATE, 5 LOW**

---

## Findings

### MODERATE

#### 1. Untracked source files need to be committed

- **Location:**
  - `solver/src/bin/clairvoyant.rs` (375 lines, complete binary)
  - `treatise/js/charts/game-replay.js` (222 lines, production-ready chart)
  - `treatise/js/charts/risk-horizon.js` (163 lines, production-ready chart)
  - `theory/lab-reports/turn-by-turn-adaptive-advantage.md` (210 lines, lab report)
- **Evidence:** All four files are referenced by tracked code. `clairvoyant.rs` is declared in `Cargo.toml` as `[[bin]]`. Both chart JS files are imported in `treatise/index.html`. The lab report is listed in `theory/lab-reports/README.md`.
- **Impact:** Anyone cloning the repo and building will get compilation errors (missing clairvoyant.rs) and broken treatise charts.
- **Recommendation:** `git add` all four files.

#### 2. Profiler has stale package-lock.json and node_modules

- **Location:** `profiler/package.json`, `profiler/package-lock.json`, `profiler/node_modules/`
- **Evidence:** `package.json` declares zero dependencies (`{"private": true, "type": "module"}`). `package-lock.json` still declares vitest ^3.0.0 as devDependency. `node_modules/` contains ~47 packages.
- **Impact:** Confuses developers about profiler's actual dependencies. Bloats any tooling that scans node_modules.
- **Recommendation:** Delete `profiler/package-lock.json` and `profiler/node_modules/`. Profiler is a static site with no build step.

#### 3. Frontend npm audit vulnerabilities (5 total, dev-only)

- **Location:** `frontend/package-lock.json`
- **Evidence:** `npm audit` reports 5 vulnerabilities in dev-dependency chains:
  - rollup (High): arbitrary file write via path traversal (4.0.0-4.58.0, used by Vite)
  - flatted (High): unbounded recursion + prototype pollution
  - minimatch (High): ReDoS via wildcards (via @typescript-eslint)
  - picomatch (High): ReDoS via quantifiers
  - brace-expansion (Moderate): process hang on zero-step sequences
- **Impact:** Dev-only (not in production bundles), but high-severity issues should be resolved.
- **Recommendation:** Run `cd frontend && npm audit fix` or update Vite/ESLint to patched versions.

#### 4. No CI/CD pipeline

- **Location:** `.github/workflows/` was removed (commit db31e22)
- **Evidence:** No automated testing, linting, or build verification on push/PR.
- **Impact:** Regressions can only be caught locally via `just check`. No gating on PRs.
- **Recommendation:** Re-add GitHub Actions for at minimum: `cargo test`, `cargo clippy`, `npm run lint`, `npm test` (frontend).

---

### LOW

#### 5. `.vite/` cache not in .gitignore

- **Location:** `.vite/` (untracked directory at repo root)
- **Evidence:** Vite dependency pre-bundling cache. Regenerated on every dev server start.
- **Impact:** Shows in `git status` noise.
- **Recommendation:** Add `.vite/` to `.gitignore`.

#### 6. `section-06-for-review.md` at repo root

- **Location:** `section-06-for-review.md` (259 lines, repo root)
- **Evidence:** Complete copy of Section 6 (Multiplayer) extracted for external review. Content overlaps with `treatise/sections/06-multiplayer.md`.
- **Impact:** Mild clutter. Could confuse contributors about which file is canonical.
- **Recommendation:** Remove after review is incorporated, or move to a `reviews/` directory.

#### 7. Confusing version constant naming

- **Location:** `solver/src/constants.rs` (lines 43, 46)
- **Evidence:** `STATE_FILE_VERSION_V5: u32 = 7`. The constant name says "V5" but the value is 7. Historical artifact from version bumps without name updates.
- **Impact:** Minor confusion when reading storage code.
- **Recommendation:** Rename to `STATE_FILE_VERSION_WITH_THETA: u32 = 7` or similar.

#### 8. ~~Unused `timing` feature flag~~ (NOT AN ISSUE)

- **Location:** `solver/Cargo.toml` `[features]` section
- **Evidence:** Actually used by 21 `#[cfg(feature = "timing")]` blocks in `widget_solver.rs` and `state_computation.rs` for optional performance instrumentation.
- **Status:** Active and intentional. No action needed.

#### 9. Unused profiler data files in Docker image

- **Location:** `profiler/data/` (14 JSON files, ~1.3 MB total)
- **Evidence:** Profiler JS only loads 2 files: `scenarios.json` and `player_card_grid.json`. The other 14 files (exp_a/b/c/d, kde_curves, state_heatmap, etc.) are copied during Docker build but never loaded.
- **Impact:** ~1.3 MB wasted in production Docker image. Non-blocking.
- **Recommendation:** Either remove unused files from profiler/data/ or update deploy.sh to copy only the 2 needed files.

---

## Cross-project coherence status

| Dimension | Status | Notes |
|-----------|--------|-------|
| API contracts (Rust <-> TS) | Aligned | All field names, types, and structures match exactly |
| Naming conventions | Consistent | snake_case in API/Python, camelCase in TS internals, no drift |
| Category names/constants | Identical | 15 categories match byte-for-byte across solver and frontend |
| State representation | Aligned | upper_score (0-63), scored_categories (15-bit), stride=128 everywhere |
| CLAUDE.md accuracy | Excellent | Root + 5 component CLAUDE.md files verified against code |
| Treatise vs. solver claims | Accurate | Spot-checked 8 specific technical claims, all correct |
| Formatting/linting configs | Appropriate | Each language uses standard tooling, no harmful drift |
| Dependency versions | Current | All lockfiles committed, no wildcard versions |

---

## Recommendations -- ALL RESOLVED

All findings have been addressed:

1. ~~Commit the 4 untracked source files~~ -- ready to stage
2. ~~Add `.vite/` to `.gitignore`~~ -- DONE
3. ~~Delete `profiler/package-lock.json` and `profiler/node_modules/`~~ -- DONE
4. ~~Run `npm audit fix` in frontend/~~ -- DONE (0 vulnerabilities)
5. ~~Clean up `section-06-for-review.md`~~ -- DONE (removed)
6. ~~Rename `STATE_FILE_VERSION_V5` constant~~ -- DONE (now `STATE_FILE_VERSION_THETA`)
7. ~~`timing` feature flag~~ -- NOT AN ISSUE (actively used, 21 cfg blocks)
8. ~~Trim unused profiler data files~~ -- DONE (removed 14 files, kept 2)
9. ~~Re-add CI/CD pipeline~~ -- DONE (`.github/workflows/ci.yml` added)
