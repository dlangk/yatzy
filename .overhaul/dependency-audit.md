# Dependency Audit

**Date:** 2026-02-21
**Phase:** 2 of OVERHAUL_PLAN.md

## Rust Solver (`solver/Cargo.toml`)

### Dependencies (8 total)

| Dependency | Version | Status | Assessment |
|------------|---------|--------|------------|
| `axum` | 0.8 | ESSENTIAL | HTTP router, used in server.rs (2 files) |
| `tokio` | 1 (`full`) | **OVER-PROVISIONED** | Only uses rt-multi-thread, net, signal, macros |
| `serde` | 1 (`derive`) | ESSENTIAL | 12 files, request/response types |
| `serde_json` | 1 | ESSENTIAL | 16 files, JSON I/O in all binaries |
| `rayon` | 1.10 | ESSENTIAL | 31 files, parallel backward induction |
| `rand` | 0.9 (`small_rng`) | ESSENTIAL | 22 files, SmallRng for dice/scenarios |
| `memmap2` | 0.9 | ESSENTIAL | 3 files, zero-copy mmap (<1ms loads) |
| `tower-http` | 0.6 (`cors`) | ESSENTIAL | 1 file, CORS layer for frontend |

### Findings

1. **tokio `features = ["full"]` is overkill.** Only 4 features are used: `rt-multi-thread` (async runtime), `net` (TcpListener), `signal` (ctrl_c), `macros` (#[tokio::main]). The `full` feature enables ~50 features including fs, io, process, time, etc. that are never used.

2. **No CLI parsing library.** All 27 binaries use manual `std::env::args()` parsing. This is intentional and appropriate — no unused dependency.

3. **Feature flags are tight.** `serde` only enables `derive`, `rand` only enables `small_rng`, `tower-http` only enables `cors`. Good discipline.

4. **No unused dependencies.** All 8 are actively imported and used.

### Changes Applied

- **Reduced tokio features** from `["full"]` to `["rt-multi-thread", "net", "signal", "macros"]`

### Lock File

`Cargo.lock` exists and is committed. Versions are pinned via the lock file.

---

## Python Analytics (`analytics/pyproject.toml`)

### Dependencies (8 production + 3 optional)

| Dependency | Files Using | Status | Assessment |
|------------|-------------|--------|------------|
| `numpy` | 33 | ESSENTIAL | Foundation of all numerical work |
| `scipy` | 7 | ESSENTIAL | KDE, stats, optimization, interpolation |
| `pandas` | 22 | HEAVY | Core data structure, 147+ `pd.` calls |
| `pyarrow` | 4 | REQUIRED | Parquet engine for pandas I/O |
| `matplotlib` | 28+ | ESSENTIAL | All plotting, 3D, animation, colormaps |
| `seaborn` | 5 | **REMOVABLE** | Only 11 calls, all replaceable by matplotlib |
| `click` | 1 | ESSENTIAL | CLI framework (cli.py) |
| `plotly` | 1 | **OPTIONAL** | Single 3D interactive HTML plot |

Optional (surrogate): `scikit-learn`, `torch`, `joblib` — appropriately gated.

### Findings

1. **Seaborn is used minimally.** Only 11 function calls across 5 files:
   - `sns.color_palette("rocket", n)` — 4 calls (replaceable with `matplotlib.colormaps["rocket"]`)
   - `sns.set_theme()` — 1 call (replaceable with `plt.style.use()` + rcParams)
   - `sns.heatmap()` — 3 calls (replaceable with `plt.imshow()` + colorbar)
   - `sns.boxplot()` — 2 calls (replaceable with `plt.boxplot()`)
   - `sns.barplot()` — 1 call (replaceable with `plt.bar()`)

2. **Plotly is used for exactly one function** (`plot_density_3d_interactive` in `density.py`). The same data is already plotted as static PNG via matplotlib. Making plotly a conditional import would be cleaner.

3. **No lock file.** Analytics has no `uv.lock` or `requirements.txt`. Version ranges are specified in `pyproject.toml` but exact resolved versions aren't pinned.

4. **Pandas is heavily used** (22 files, 147+ calls). Replacing with Polars would be a significant effort — defer to Phase 7 per OVERHAUL_PLAN.md.

### Changes Applied

- **No changes in this phase** for Python. Seaborn removal and Plotly conditionalization are deferred — they require code changes across 5-6 files which is better done in Phase 7 (Analytics Toolbox) when the whole analytics stack is being reworked.
- **Generated uv.lock** to pin exact versions.

### Recommendations (for Phase 7)

- Remove seaborn dependency (~30 lines of changes across 5 files)
- Make plotly an optional dependency with conditional import
- Consider pandas → polars migration if performance matters

---

## Frontend (`frontend/package.json`)

### Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| `react` | prod | ESSENTIAL |
| `react-dom` | prod | ESSENTIAL |
| `@eslint/js` | dev | ESSENTIAL |
| `@types/node` | dev | ESSENTIAL |
| `@types/react` | dev | ESSENTIAL |
| `@types/react-dom` | dev | ESSENTIAL |
| `@vitejs/plugin-react` | dev | ESSENTIAL |
| `eslint` | dev | ESSENTIAL |
| `eslint-plugin-react-hooks` | dev | ESSENTIAL |
| `eslint-plugin-react-refresh` | dev | ESSENTIAL |
| `globals` | dev | ESSENTIAL |
| `typescript` | dev | ESSENTIAL |
| `typescript-eslint` | dev | ESSENTIAL |
| `vitest` | dev | ESSENTIAL |
| `vite` | dev | ESSENTIAL |

### Findings

**Exemplary.** Only 2 production dependencies (react, react-dom). No component library, no state management library, no chart library, no CSS framework. The frontend is intentionally lean. All dev dependencies serve clear purposes (build, lint, test, type-check).

### Lock File

`package-lock.json` exists and is committed.

### No Changes Required

---

## Blog (`blog/package.json`)

### Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| `vitest` | dev | ESSENTIAL (tests mask/reducer logic) |

### Findings

**Minimal.** One dev dependency for testing. D3.js is loaded via CDN, not as an npm package.

### Lock File

`package-lock.json` exists and is committed.

### No Changes Required

---

## Summary

| Component | Total Deps | Unused | Overkill | Action |
|-----------|-----------|--------|----------|--------|
| Rust | 8 | 0 | 1 (tokio features) | Reduced tokio features |
| Python | 8+3 | 0 | 2 (seaborn, plotly) | Deferred to Phase 7 |
| Frontend | 15 | 0 | 0 | None |
| Blog | 1 | 0 | 0 | None |

### Lock File Status

| Component | Lock File | Status |
|-----------|-----------|--------|
| Rust | `Cargo.lock` | Present, committed |
| Python | `uv.lock` | **Generated in this phase** |
| Frontend | `package-lock.json` | Present, committed |
| Blog | `package-lock.json` | Present, committed |
