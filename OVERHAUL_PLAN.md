# Yatzy HPC Codebase Overhaul — Claude Code Prompt

You are performing a systematic overhaul of a high-performance computing system for exploring Yatzy (the dice game). The system has three major components with **different optimization goals**:

1. **Rust Backend** — Performance is sacred. Never trade computational efficiency for aesthetics.
2. **Frontend** — Pure HTML + TypeScript + D3.js. No framework. Lean and zero-dependency beyond D3.
3. **Analytics Pipeline** — Flexibility and modularity. Build a reusable toolbox.

Work in phases. At the end of each phase, write a status checkpoint to `/.overhaul/phase-N-complete.md` summarizing what was done, decisions made, and any open questions. If a checkpoint file already exists, skip that phase and resume from the next incomplete one.

A **primary goal** of this overhaul is to produce documentation so thorough that an AI coding agent (you, in future sessions) can instantly orient itself in the codebase — finding the right file, understanding the right abstraction, calling the right function with the right parameters, without trial and error. Every phase should contribute to this goal.

---

## Pre-Flight Checklist — Resolve Before Running

Fill in these values before starting. The agent must NOT guess any of these — if a value is still `[TODO]`, stop and ask.

| Item | Value |
|------|-------|
| Custom coolwarm center color (hex) | `[TODO: e.g. #E2E2E2]` |
| Python package manager | `[TODO: uv / pip / poetry]` |
| Frontend package manager | `[TODO: pnpm / npm / yarn]` |

---

## Phase 0: Deep Reconnaissance

Before changing anything, build a complete mental model of the codebase. Write the following artifacts:

### 0a. `/.overhaul/repo-map.md`

- Full directory tree (depth 4)
- Language/framework breakdown per component
- Build systems and toolchains (cargo, npm/pnpm, python, etc.)
- Entry points for each component (binary targets, main files, CLI commands)
- How components communicate (HTTP APIs, shared files, message passing, shared types)
- Dependency manifest and lock file locations

### 0b. `/.overhaul/architecture.md`

Write a thorough architectural description covering:

- **System overview**: What does this system do end-to-end? What is the user journey from frontend through backend to analytics?
- **Rust Backend**: What are the core computational modules? What simulation strategies are used? What data structures represent game state, strategies, probability distributions? What does the web server expose and how?
- **Frontend**: What framework? What pages/views exist? How does it communicate with the backend? What state management approach?
- **Analytics Pipeline**: What does it consume? What does it produce? What transformations and analyses does it perform? What output formats?
- **Data flow diagram**: Trace the path of a typical request — e.g., "user requests optimal strategy for a given game state" — through all three components.
- **Shared types and contracts**: Are there shared type definitions, API schemas, protobuf/OpenAPI specs? Where do the component boundaries live?

### 0c. `/.overhaul/rust-backend-deep-dive.md`

This is the most critical artifact. Map the Rust backend in detail:

- **Crate structure**: Workspace members, internal crates, dependency graph between them.
- **Hot paths**: Identify the computationally critical code paths — simulation loops, probability calculations, strategy evaluation, state space exploration. For each hot path, document:
  - Where it lives (file, function)
  - What it computes
  - What data structures it operates on
  - Any SIMD, parallelism, or cache-optimization techniques in use
  - Estimated computational complexity
- **Web server**: Framework (axum, actix, warp, etc.), route map, request/response types, middleware, serialization format.
- **API surface**: Every endpoint with its parameters, what computation it triggers, and what it returns. Flag which endpoints are latency-sensitive (gameplay) vs throughput-sensitive (batch simulation/sweeps).
- **Common simulation requests**: Document every type of simulation or sweep the system supports. What parameters does each require? What does the output look like? How are results consumed downstream by analytics and frontend?
- **Concurrency model**: How does it parallelize work? Thread pool, async, rayon, tokio? Where are the synchronization points?
- **Memory layout**: Any notable data structure choices for cache performance (SoA vs AoS, arena allocation, bitpacked state, lookup tables, etc.)

### 0d. `/.overhaul/analytics-deep-dive.md`

Map the analytics pipeline:

- What language/framework is currently used?
- Input data sources and formats
- Every analysis/transformation currently implemented — name it, describe what it does, what it takes as input, what it produces
- Output formats and destinations (charts, CSVs, dashboard data, etc.)
- Where outputs are stored (file paths, naming conventions)
- Which parts are reusable general-purpose tools vs one-off scripts
- Dependency on backend API — what does it call, with what parameters?

### 0e. `/.overhaul/frontend-deep-dive.md`

Map the frontend (this will be migrated to pure HTML + TypeScript + D3.js in Phase 6, so document what exists now as the migration baseline):

- Current framework, state management, routing
- Component hierarchy — what does each component do in plain terms?
- API integration layer — how it calls the backend, error handling, caching
- Key user flows — what does the user interact with?
- Every visualization currently rendered — what chart type, what data, what library?
- Full dependency list — what gets removed during migration vs what stays

### 0f. `/.overhaul/smells.md`

Across all components, identify:

- Files over 300 lines (frontend/analytics) or over 500 lines (Rust — higher threshold because Rust is verbose)
- Functions over 50 lines
- Duplicated logic across files or across components (e.g., game logic reimplemented in both Rust and frontend)
- Dead code, commented-out blocks, unused modules
- Inconsistent patterns (e.g., some endpoints return JSON one way, others differently)
- **Unnecessary duplication that likely resulted from insufficient overview** — this is a key target. Look for cases where similar computations, data transformations, or utility functions were written twice because the author didn't know the other one existed.
- Excessive or unnecessary dependencies — libraries pulled in for trivial functionality that could be a few lines of code
- Excessive error handling that adds bloat without handling realistic failure modes

### 0g. `/.overhaul/output-map.md`

Map where all generated artifacts currently live:

- Simulation/sweep output files — where are they stored, what naming convention, what format?
- Analytics outputs — plots, CSVs, reports — where do they end up?
- Experiment results and lab reports
- Any generated data, cached computations, or intermediate files
- Identify any outputs that are scattered or disorganized

**Do NOT make any edits in this phase.**

---

## Phase 1: Secrets & Sensitive Data

**Blocking phase** — nothing proceeds until clean.

1. Scan for hardcoded secrets: API keys, tokens, passwords, connection strings, private keys, webhook URLs, cloud credentials. **Do NOT read files into context manually — write and execute a bash script using `rg` (ripgrep) or `grep -r` to scan for common secret patterns** (e.g., `API_KEY=`, `Bearer `, `-----BEGIN`, connection strings, base64-encoded blobs). Check source files, configs, Docker files, CI/CD configs, notebooks, comments.
2. Log findings in `/.overhaul/secrets-audit.md` (file, line, type — **never log actual values**).
3. Remediate:
   - Replace hardcoded secrets with environment variable references.
   - Create `.env.example` with placeholder values and explanatory comments.
   - Ensure `.env` is in `.gitignore`.
   - If Docker is used, ensure compose files use `${VAR}` syntax.
4. Add secret-scanning pre-commit hook config (detect-secrets or gitleaks). Document in the audit file.

---

## Phase 2: Dependency Hygiene

**Philosophy: prefer fewer, well-chosen dependencies.** Avoid pulling in a library when a few lines of code would do the job. This is especially true for the frontend — there is a library for everything, but most are unnecessary. Only add a dependency when it provides substantial, non-trivial functionality that would be error-prone to reimplement. When it's a close call, skip the dependency.

For each component separately:

1. Flag unused, outdated, or vendored dependencies.
2. Flag dependencies that are overkill — libraries pulled in for trivial functionality. Recommend inlining the logic instead.
3. Write findings to `/.overhaul/dependency-audit.md`, organized by component.
4. Remove clearly unused dependencies. Replace trivial dependencies with inline code. Annotate ambiguous cases for my review.
5. Pin all versions. Ensure lock files are consistent.
6. **Rust-specific**: Check for unnecessary feature flags that bloat compile time or binary size. Check that performance-critical dependencies (rayon, simd libs, etc.) are using optimal feature configurations.

---

## Phase 2b: Wall-Clock Performance Baseline

**This phase MUST complete before any code changes begin.** It establishes the performance contract that all subsequent phases must honor.

The target machine is an **Apple M1 Max Mac**. All timings must account for natural variance.

**0. Bootstrap the justfile.** Create a `justfile` at the repo root with at least the benchmarking targets (`bench-check`, `bench-baseline`). Other targets will be added incrementally in later phases. The justfile must exist before any phase tries to call `just bench-check`.

1. **Identify every critical computational routine** in the Rust backend. These are the functions and code paths that dominate simulation/sweep wall-clock time.

2. **Write wall-clock performance tests** in `rust/benches/wall_clock/`. For each critical routine:
   - Measure wall-clock time (not just CPU time) using `criterion` or `std::time::Instant`.
   - Run enough iterations to get stable measurements (criterion handles this, but verify).
   - Record the **mean**, **standard deviation**, and **p95** for each benchmark.
   - Set the **failure threshold to `max(mean + 3σ, mean × 1.05)`** — whichever is larger. The 3σ bound catches real regressions; the 5% floor prevents false failures on benchmarks with very tight variance (e.g., a 100ns routine with 1ns σ would flake on normal OS scheduling noise without the floor).

3. **Write an end-to-end timing test** that measures the wall-clock time for a representative full simulation run (e.g., 10,000 games with a standard strategy). This catches regressions that might hide in individual micro-benchmarks.

4. **Write API latency tests** for gameplay-critical endpoints. Measure round-trip time from request to response. Set failure thresholds similarly (mean + 3σ).

5. **Record the baseline** in `/.overhaul/performance-baseline.md`:
   - Machine: Apple M1 Max
   - Date
   - Rust compiler version
   - For each benchmark: mean, σ, p95, failure threshold
   - The exact command to reproduce: `just bench-baseline`

6. **Add a `just bench-check` command** that runs all performance tests and compares against the baseline. It should print a clear PASS/FAIL for each benchmark and exit non-zero on any failure.

**From this point forward, `just bench-check` must pass after every phase. Any phase that causes a failure must be reverted and reworked.**

---

## Phase 2c: Correctness Baseline

**Establish a functional correctness contract before any refactoring begins.** If Phase 4 aggressively deduplicates game logic and introduces a bug, we need to catch it immediately — not in Phase 9.

1. Run the existing test suite for all components. Record the results in `/.overhaul/test-baseline.md`.
2. If any tests fail, document them as pre-existing failures (do not fix them in this phase).
3. Add a `just test` target to the justfile if not already present.
4. **From this point forward, `just test` must pass (modulo pre-existing failures) after every phase.** Any new test failure introduced by the overhaul must be fixed before proceeding.

---

## Phase 3: Claude Code Memory Architecture

**This is the highest-value phase of the entire overhaul.** The goal is to build a documentation system that exploits Claude Code's full memory hierarchy so that any future session can orient itself instantly — finding the right file, understanding the right abstraction, calling the right function with the right parameters.

Claude Code's memory hierarchy loads context in this order:

1. `~/.claude/CLAUDE.md` (user global — we don't touch this)
2. Root `CLAUDE.md` (loaded every session — MUST be concise)
3. `.claude/rules/*.md` (loaded every session, can be path-scoped with YAML frontmatter)
4. Child directory `CLAUDE.md` files (loaded ON DEMAND when Claude works in that subtree)
5. `@import` references (loaded when the importing file is loaded, max depth 5)

**Design principle: the root CLAUDE.md must be a lean routing document, not an encyclopedia.** Research shows Claude reliably follows ~150-200 instructions. The Claude Code system prompt already consumes ~50. That leaves ~100-150 for us. The root file should orient Claude and point it to detail — the detail lives in child CLAUDE.md files, `.claude/rules/`, and imported docs.

### 3a. Root `CLAUDE.md` — The Router (~100-150 lines max)

Create `CLAUDE.md` at the repo root. Structure it as follows:

```markdown
# Yatzy HPC System

[One-paragraph description: what this system does, who uses it, what problem it solves.]

## Components

| Component | Location | Purpose | Optimization Goal |
|-----------|----------|---------|-------------------|
| Rust Backend | `rust/` | HPC simulation engine + web server | **Performance is sacred** |
| Frontend | `frontend/` | Game UI and visualization (pure HTML + TypeScript + D3.js) | Lean, zero framework |
| Analytics | `analytics/` | Reusable analysis toolbox (Python + Polars + Seaborn) | Flexibility and modularity |

See component-specific CLAUDE.md files for detailed guidance:
- @rust/CLAUDE.md — Rust architecture, hot paths, API reference
- @frontend/CLAUDE.md — Frontend architecture and patterns
- @analytics/CLAUDE.md — Toolbox catalog and pipeline composition

## Critical Rules
- NEVER trade Rust backend performance for code aesthetics
- ALWAYS run `just bench-check` after Rust changes — wall-clock regression = failure
- Intentionally duplicated hot-path code is marked `// PERF: intentional`
- Minimize dependencies — prefer inline code over trivial libraries
- Minimize error handling — only handle errors likely to occur, keep code lean
- Secrets go through the config module, never inline
- When modifying an API endpoint, update BOTH rust/CLAUDE.md AND analytics/CLAUDE.md
- When generating output (data, plots, reports), document where it goes in the output map
- When running experiments, follow the lab report protocol (see /project:run-experiment)

## Output Map
All generated artifacts follow a consistent structure:
- `data/sweeps/` — raw simulation/sweep results
- `data/experiments/` — experiment results organized by `YYYY-MM-DD_<experiment-name>/`
- `data/cache/` — precomputed tables, cached intermediate results
- `output/plots/` — generated visualizations
- `output/reports/` — lab reports and analysis writeups
See @docs/output-map.md for the full output organization guide.

## Commands
- `just check` — lint + typecheck + test + bench-check (no regression)
- `just bench-check` — run wall-clock perf tests, compare to baseline, PASS/FAIL
- `just bench-baseline` — record new performance baseline
- `just test` — all component tests
- `just fmt` — auto-format all components
- `just dev-backend` — start Rust backend in dev mode
- `just dev-frontend` — start frontend in dev mode
- `just sweep <config>` — run parameter sweep (see @rust/CLAUDE.md)
- `just analyze <pipeline>` — run analytics pipeline (see @analytics/CLAUDE.md)

## Architecture Overview
Brief data flow: Frontend → HTTP API → Rust computation engine → JSON response
Analytics pipeline: Calls backend API for simulation/sweep data → tools process → output

For detailed architecture, see @docs/architecture.md
For data schemas, see @docs/data-formats.md
For output organization, see @docs/output-map.md
```

Note: the `@import` references tell Claude Code to pull in the referenced file when it needs deeper context, WITHOUT bloating the root file.

### 3b. `.claude/rules/` — Path-Scoped Rules

Create scoped rules that activate ONLY when Claude is editing matching files. These are loaded every session but their path-scoping means they only apply contextually.

**`.claude/rules/rust-hot-paths.md`:**
```yaml
---
paths:
  - "rust/src/prob/**/*.rs"
  - "rust/src/strategy/**/*.rs"
  - "rust/src/sim/**/*.rs"
---
```
```markdown
# Hot Path Rules — Performance Critical Code

YOU MUST follow these rules when editing hot-path code:

- Run `just bench-check` before AND after any change. Revert if any benchmark fails.
- Do NOT extract helper functions if it adds dynamic dispatch or prevents inlining.
- Do NOT replace intentionally duplicated code (marked `// PERF: intentional`).
- Do NOT add heap allocations in inner loops.
- Do NOT add error handling that isn't strictly necessary — unwrap/expect is fine for invariants that are structurally guaranteed. Avoid Result chains in hot loops.
- Prefer `#[inline]` on small functions called from hot loops.
- Use `debug_assert!` for invariants (zero cost in release builds).
- When in doubt, leave a `// PERF: <explanation>` comment rather than refactoring.
```

**`.claude/rules/rust-api.md`:**
```yaml
---
paths:
  - "rust/src/server/**/*.rs"
---
```
```markdown
# API Server Rules

- The web server layer MUST NEVER panic on user input. Handlers must catch errors and return HTTP 400/500. This is the opposite of the HPC core, where unwrap()/expect() on invariants is fine.
- Gameplay endpoints are latency-critical: no blocking, no unnecessary allocation.
- CPU-heavy computation MUST use `tokio::spawn_blocking` or a dedicated compute pool.
- Response types must be documented with their downstream consumers (frontend/analytics).
- Every endpoint parameter struct must derive Serialize + Deserialize + Debug.
- Keep error handling minimal but correct — return sensible HTTP status codes, don't over-engineer error types, but DO handle malformed input gracefully.
- When adding/changing an endpoint, update rust/CLAUDE.md API Reference section.
```

**`.claude/rules/analytics.md`:**
```yaml
---
paths:
  - "analytics/**/*.py"
---
```
```markdown
# Analytics Rules

- Stack: Python + Polars + Seaborn. Do NOT use pandas — use Polars for all dataframe operations.
- **Polars/Seaborn boundary**: Seaborn expects pandas DataFrames. Do ALL data manipulation, filtering, and aggregation in Polars. Call `.to_pandas()` strictly at the exact moment you pass finalized data into a Seaborn/Matplotlib function. Never import pandas for data processing.
- All plots use our custom coolwarm palette (see analytics/CLAUDE.md for palette definition).
- Every tool function MUST have a Google-style docstring (see analytics/CLAUDE.md for format).
- Every tool MUST declare input/output types using type hints or dataclass/pydantic models.
- Tools are composable: they take data in, return data out. No side effects except logging.
- New tools go in `analytics/tools/<category>/`. New pipelines go in `analytics/pipelines/`.
- When adding a new tool, register it in analytics/CLAUDE.md toolbox catalog.
- When a tool generates output (plots, data files), it must write to the standard output directories and the docstring must say where.
- Minimize dependencies beyond the core stack. Only add a library if it provides substantial non-trivial functionality.
```

**`.claude/rules/frontend.md`:**
```yaml
---
paths:
  - "frontend/src/**/*.{ts,js,html}"
---
```
```markdown
# Frontend Rules

- The frontend is pure HTML + TypeScript + D3.js. No frameworks (no React, Vue, Svelte, etc.).
- D3.js is the primary visualization and plotting library. When choosing a chart type, explore D3's full gallery of examples for inspiration — there are hundreds of plot types and interaction patterns.
- Game logic lives in the Rust backend. The frontend does NOT reimplement it.
- All backend calls go through the API client module — no scattered fetch() calls.
- TypeScript interfaces for all data contracts and component state.
- Minimize dependencies. D3.js is the one allowed large dependency. Beyond that, do NOT add a library for something achievable in a few lines of code.
- Keep error handling lean — a simple try/catch or error boundary is enough. Don't build elaborate error hierarchies.
- When adding a new API call, update frontend/CLAUDE.md API integration section.
```

### 3c. Child `CLAUDE.md` Files — Detailed Component Context

These are loaded on demand when Claude works in the relevant subtree. They can be longer and more detailed than the root file.

**`rust/CLAUDE.md`** (~200-400 lines):

Must contain:

- **Crate Structure**: Workspace members, dependency graph between internal crates.
- **Core Data Structures Reference**: For EVERY key struct and enum:
  - Name, location (file path), purpose, key fields with types, invariants, performance notes (cache layout, bit packing, etc.)
- **Hot Paths Table**: `| Hot Path | Location | What It Computes | Why It's Fast | What NOT To Do |`
- **API Reference**: For EVERY endpoint:
  - Route + method
  - Parameter struct with types and valid ranges
  - What computation it triggers (link to hot path if applicable)
  - Response shape
  - Latency class: `gameplay-critical` or `batch-ok`
  - Consumed by: which downstream systems use this
- **Sweep & Simulation Guide**: Step-by-step for every simulation type — what to call, required parameters with types and ranges, expected output shape, where output is saved, how analytics pipeline consumes it
- **Concurrency Model**: Tokio runtime config, blocking pool, rayon usage, synchronization points
- **Performance Testing**: How to run `just bench-check`, where baselines are stored, how to update baselines, what the failure thresholds mean
- **Function Documentation Standard** (see Section 3f below)

**`analytics/CLAUDE.md`** (~200-300 lines):

Must contain:

- **Core Stack**: Python + Polars + Seaborn. Explicitly state: do NOT use pandas. The only allowed pandas usage is `.to_pandas()` at the exact moment data is passed into a Seaborn/Matplotlib plotting function.
- **Custom Coolwarm Palette**:
  ```python
  import matplotlib.colors as mcolors
  import seaborn as sns

  # Our custom coolwarm: standard coolwarm with a distinct center color.
  # The center color is [IDENTIFY THE ACTUAL CENTER COLOR FROM THE CODEBASE
  # OR ASK ME]. This distinguishes neutral/zero values from the gradient.
  _base = sns.color_palette("coolwarm", as_cmap=True)
  # [Insert the actual palette construction code here]
  YATZY_PALETTE = ...  # The final palette object

  # ALL plotting functions must use YATZY_PALETTE as default.
  # Pass cmap=YATZY_PALETTE to seaborn/matplotlib calls.
  ```
  The center color is specified in the Pre-Flight Checklist at the top of this prompt. If it's still `[TODO]`, stop and ask — do not guess.
- **Toolbox Catalog**: Every reusable analysis function with: name, location, purpose, input signature (with types), output signature (with types), where it writes output, example usage, see-also cross-references
- **Pipeline Composition Guide**: How to chain tools into pipelines, how to define a new pipeline, example pipeline definition
- **Backend Client Reference**: Every method in the backend client module, mapped to the backend API endpoints it calls
- **Adding a New Tool — Checklist**: Step-by-step procedure including where to put the file, the required docstring format, how to test it, how to register it
- **Output Conventions**: Where each type of output goes (plots → `output/plots/`, data → `data/`, reports → `output/reports/`), naming conventions, format conventions
- **Experiment Protocol**: Reference the lab report protocol (see Section 3e, `/project:run-experiment`)
- **Function Documentation Standard** (see Section 3f below)

**`frontend/CLAUDE.md`** (~100-150 lines):

Must contain:

- **Architecture**: Pure HTML + TypeScript + D3.js. No framework. Describe the module structure, how pages are organized, how state is managed without a framework (e.g., simple event-driven or observable pattern).
- **D3 Visualization Guide**: How visualizations are structured — each chart as a self-contained module that takes a container element and data, renders with D3. List every visualization currently implemented with its location and what data it expects.
- **API Integration**: Every API call the frontend makes, mapped to backend endpoints
- **Adding a New Page — Checklist**: Step-by-step pattern
- **Adding a New Visualization — Checklist**: How to create a new D3 chart module, what interface it should expose, where to put it, how to wire it to data.
- **Dependency Philosophy**: D3.js is the one allowed large dependency. Beyond that, minimize external libraries. Before adding any dependency, consider whether a few lines of code would suffice.
- **Function Documentation Standard** (see Section 3f below)

### 3d. Shared Documentation (imported via `@`)

**`docs/architecture.md`** — Detailed system architecture, data flow diagrams, deployment topology. Imported by root CLAUDE.md.

**`docs/data-formats.md`** — Schema definitions for all data exchanged between components. Imported by both `rust/CLAUDE.md` and `analytics/CLAUDE.md`.

**`docs/output-map.md`** — Complete guide to where all generated artifacts live:

```markdown
# Output Map

All generated artifacts follow this structure. When writing code that produces output,
document the output location in the function's docstring AND keep this file up to date.

## Data Outputs
- `data/sweeps/<YYYY-MM-DD>_<name>/` — raw simulation/sweep results (JSON/Parquet)
- `data/experiments/<YYYY-MM-DD>_<experiment-name>/` — experiment data + lab report
- `data/cache/` — precomputed lookup tables, cached intermediate results (gitignored)

## Visualization Outputs
- `output/plots/<category>/` — generated plots, organized by analysis category
- `output/plots/experiments/<experiment-name>/` — plots tied to specific experiments

## Reports
- `output/reports/` — lab reports, analysis writeups

## Naming Conventions
- Dates: YYYY-MM-DD prefix for chronological sorting
- Descriptive suffixes: `2025-02-21_strategy-comparison-greedy-vs-optimal/`
- Plots: `<what-it-shows>_<parameters>.<ext>` e.g., `ev-distribution_greedy_100k.png`

## Adding New Output Types
When your code generates a new type of output:
1. Choose the appropriate directory from above (or create a new subdirectory if needed)
2. Document the output location in the producing function's docstring
3. Update this file with the new output type
4. Update analytics/CLAUDE.md if it's a new analytics output
```

### 3e. Custom Slash Commands

Create reusable workflows in `.claude/commands/`:

**`.claude/commands/run-sweep.md`:**
```markdown
Run a parameter sweep with the following configuration: $ARGUMENTS

1. Parse the configuration to identify: strategy type, iteration count, and any parameter ranges.
2. Verify the Rust backend is running (check `just dev-backend` or start it).
3. Call the sweep endpoint with all required parameters (see rust/CLAUDE.md).
4. IMPORTANT: Ensure the response includes full metadata for analytics reproducibility.
5. Save results to `data/sweeps/<YYYY-MM-DD>_<descriptive-name>/` with metadata sidecar.
6. Report summary statistics.
7. Log the output location so it's easy to find.
```

**`.claude/commands/run-analysis.md`:**
```markdown
Run the analytics pipeline: $ARGUMENTS

1. Check that the required sweep data exists in `data/sweeps/`.
2. Load the pipeline definition from `analytics/pipelines/`.
3. Execute each tool in sequence, logging inputs and outputs.
4. Save data results to `data/` and plots to `output/plots/` following output conventions.
5. All plots MUST use the YATZY_PALETTE (custom coolwarm). No exceptions.
6. Generate any visualization outputs.
7. Report where all outputs were saved.
```

**`.claude/commands/run-experiment.md`:**
```markdown
Run an experiment following the scientific method: $ARGUMENTS

This command runs a rigorous experiment and produces a lab report. Follow this
protocol EXACTLY:

## 1. Pre-Registration (BEFORE running anything)

Write the following to `data/experiments/<YYYY-MM-DD>_<experiment-name>/pre-registration.md`:

### Hypothesis
State a specific, falsifiable hypothesis. Example: "Greedy strategy achieves a higher
mean total score than random strategy over 100,000 games."

### Predictions
State concrete, quantitative predictions BEFORE seeing any data:
- What metric will you measure?
- What direction do you expect?
- What magnitude would be surprising?
- What would falsify the hypothesis?

### Method
- What simulation/sweep will you run?
- What parameters?
- How many iterations?
- What statistical tests will you use to evaluate?

## 2. Run the Experiment

Execute the simulation/sweep as specified in the method section.
Save all raw data to `data/experiments/<YYYY-MM-DD>_<experiment-name>/data/`.

## 3. Analysis

Run the pre-specified analyses on the raw data.
Save plots to `data/experiments/<YYYY-MM-DD>_<experiment-name>/plots/`.
All plots MUST use YATZY_PALETTE.

## 4. Lab Report

Write `data/experiments/<YYYY-MM-DD>_<experiment-name>/report.md`:

```
# Experiment: <name>
Date: <date>

## Hypothesis
[Copy from pre-registration]

## Predictions
[Copy from pre-registration]

## Method
[Copy from pre-registration, add any deviations noted during execution]

## Results
[Clinical reporting of results. Numbers, tables, references to plots.
DO NOT interpret or editorialize here. Just state what happened.]

## Discussion
[NOW interpret. Did the results support or falsify the hypothesis?
Were the predictions accurate? What was surprising? What are the
limitations? What follow-up experiments would be informative?]
```

## 5. Update Documentation

- Add the experiment to `data/experiments/INDEX.md` (create if it doesn't exist).
- If the experiment reveals something about the system that should be documented,
  update the relevant CLAUDE.md file.
```

**`.claude/commands/add-endpoint.md`:**
```markdown
Add a new API endpoint to the Rust backend: $ARGUMENTS

1. Read rust/CLAUDE.md to understand the existing endpoint patterns.
2. Create the handler in `rust/src/server/handlers/`.
3. Define request/response types with Serialize + Deserialize + Debug.
4. Register the route in the router.
5. Classify the endpoint's latency class (gameplay-critical or batch-ok).
6. If gameplay-critical, ensure no blocking calls in the handler.
7. Add integration test.
8. Run `just bench-check` to verify no performance regression.
9. Update rust/CLAUDE.md API Reference section with the new endpoint.
10. Update analytics/CLAUDE.md backend client section if analytics will consume it.
11. Update frontend/CLAUDE.md if the frontend will call it.
```

**`.claude/commands/add-tool.md`:**
```markdown
Add a new analytics tool: $ARGUMENTS

1. Read analytics/CLAUDE.md to understand the existing tool patterns and categories.
2. Create the tool file in `analytics/tools/<appropriate_category>/`.
3. Write the function with full Google-style docstring (see analytics/CLAUDE.md for format).
4. Use Polars for any dataframe operations (NOT pandas).
5. If the tool generates plots, use Seaborn with YATZY_PALETTE as default.
6. Document where the tool's output goes in the docstring.
7. Add type hints for all parameters and return values.
8. Add unit test in `tests/tools/<category>/`.
9. Register in `analytics/tools/<category>/__init__.py`.
10. Add entry to analytics/CLAUDE.md toolbox catalog.
11. Update docs/output-map.md if the tool creates a new type of output.
```

### 3f. Method/Function Documentation Standards

These standards must be embedded in each component's CLAUDE.md AND enforced by the corresponding `.claude/rules/` file.

**Rust functions** — doc comments with mandatory sections:

```rust
/// Brief one-line summary.
///
/// Detailed explanation of what this function computes and why.
///
/// # Arguments
/// * `state` - Current game state. Must have at least one unfilled category.
/// * `config` - Evaluation parameters. `config.depth` controls search depth.
///
/// # Returns
/// Ranked list of actions with expected values. Sorted descending by EV.
///
/// # Performance
/// O(n * k) where n = remaining categories, k = config.depth.
/// Hot path — do not add allocations. Uses precomputed lookup tables.
///
/// # Panics
/// Panics if `state.remaining_categories()` is empty.
///
/// # Consumed By
/// - API: `POST /api/strategy` (gameplay-critical latency)
/// - Internal: `sweep::run_single_game`
///
/// # Example
/// ```
/// let actions = evaluate_strategy(&game_state, &EvalConfig::default());
/// assert!(!actions.is_empty());
/// ```
```

**Python functions** — Google-style docstrings with mandatory sections:

```python
def compute_expected_value(
    sweep_result: SweepResult,
    metric: str = "total_score",
    confidence_level: float = 0.95,
) -> ExpectedValueResult:
    """Compute expected value and confidence interval for a sweep metric.

    Analyzes simulation results to produce point estimates and uncertainty
    bounds for the specified scoring metric.

    Args:
        sweep_result: Output from a backend sweep run. Must include metadata.
        metric: Score metric to analyze. One of: "total_score", "upper_bonus",
            "yahtzee_count". Defaults to "total_score".
        confidence_level: Confidence level for interval. Defaults to 0.95.

    Returns:
        ExpectedValueResult with fields:
            mean (float): Point estimate of expected value.
            std (float): Standard deviation.
            ci (tuple[float, float]): Confidence interval bounds.
            n (int): Number of samples used.

    Raises:
        ValueError: If sweep_result contains no data for the given metric.

    Output:
        No file output (returns data only).
        — OR, for tools that write files: —
        Writes plot to `output/plots/probability/<metric>_distribution.png`.

    Example:
        >>> result = compute_expected_value(sweep_data)
        >>> print(f"EV: {result.mean:.2f} +/- {result.std:.2f}")

    See Also:
        compare_strategies: For comparing EVs across strategies.
        tools.visualization.plot_distribution: For visualizing the distribution.
    """
```

**TypeScript functions** — TSDoc with mandatory sections:

```typescript
/**
 * Renders a score distribution chart into the given container.
 *
 * Uses D3.js to create an interactive histogram of game scores.
 * The container element is fully owned by this function — D3 manages
 * all DOM inside it.
 *
 * @param container - The HTML element to render into. Will be cleared first.
 * @param data - Score distribution data from the backend.
 * @param options - Optional chart configuration (dimensions, colors, etc.).
 *
 * @example
 * ```ts
 * const el = document.getElementById("score-chart")!;
 * renderScoreDistribution(el, sweepData.scores);
 * ```
 *
 * @see {@link ScoreDistributionData} for the data interface.
 * @see rust/CLAUDE.md API Reference for the backend endpoint.
 */
```

### 3g. Documentation Connection Map

Create `docs/connection-map.md` — the "index of indexes":

```markdown
# Documentation Connection Map

## File Hierarchy
CLAUDE.md (root — routing, ~100 lines, loaded every session)
├── @rust/CLAUDE.md (on-demand — Rust deep dive)
│   ├── @docs/architecture.md (shared architecture)
│   └── @docs/data-formats.md (shared schemas)
├── @frontend/CLAUDE.md (on-demand — frontend deep dive)
│   └── @docs/data-formats.md
├── @analytics/CLAUDE.md (on-demand — analytics deep dive)
│   ├── @docs/data-formats.md
│   └── @docs/architecture.md
├── @docs/architecture.md (shared — full system architecture)
└── @docs/output-map.md (shared — where all outputs live)

## Scoped Rules (.claude/rules/)
rust-hot-paths.md → activates for rust/src/{prob,strategy,sim}/**/*.rs
rust-api.md       → activates for rust/src/server/**/*.rs
analytics.md      → activates for analytics/**/*.py
frontend.md       → activates for frontend/src/**/*.{ts,js,html}

## Slash Commands (.claude/commands/)
/project:run-sweep       → execute backend parameter sweep
/project:run-analysis    → execute analytics pipeline
/project:run-experiment  → run experiment with full lab report protocol
/project:add-endpoint    → add new backend API endpoint
/project:add-tool        → add new analytics tool

## Update Protocol
When you add or modify:
- A Rust public function → update its doc comment AND rust/CLAUDE.md
- An API endpoint → update rust/CLAUDE.md AND analytics/CLAUDE.md client section
- An analytics tool → update its docstring AND analytics/CLAUDE.md toolbox catalog
- A frontend API call → update frontend/CLAUDE.md API integration section
- Code that generates output → update docstring with output location AND docs/output-map.md
- Any of the above → verify docs/connection-map.md is still accurate
```

### 3h. `CLAUDE.local.md` Template

Create a `CLAUDE.local.md.example` (gitignored template for personal overrides):

```markdown
# Personal Overrides (copy to CLAUDE.local.md)
# This file is gitignored — for your individual preferences only.

## My Environment
- Backend runs on port: 8080
- Frontend dev server: http://localhost:3000
- Python venv location: .venv/

## My Preferences
- [Add your coding style preferences here]
```

**Stop after writing all CLAUDE.md files, rules, commands, and the connection map. Ask me to review before proceeding to code changes.**

---

## Phase 3-Verify: Validate the Documentation

After I approve Phase 3, perform this validation:

Walk through these 5 tasks as a thought exercise and verify the CLAUDE.md hierarchy provides enough information to complete each WITHOUT reading source code:

1. "Add a new simulation type that evaluates risk-averse strategies"
2. "Add an analytics tool that compares two sweep results"
3. "Optimize the dice probability calculation for cache performance"
4. "Add a new D3 visualization showing strategy comparison as a parallel coordinates plot"
5. "Run an experiment: does the greedy strategy beat random over 100k games?"

For each task, trace which CLAUDE.md files would be loaded, which `.claude/rules/` would activate, and whether the documentation tells Claude exactly what to do. For task 5, verify the experiment protocol produces a proper lab report. If any task fails this test, go back and fill in the gaps before proceeding.

---

## Phase 4: Project Structure & Deduplication

### 4a. Identify and eliminate unnecessary duplication

Using the smells analysis from Phase 0, find every case of duplicated logic. Categorize each:

- **Cross-component duplication** (e.g., Yatzy game rules implemented in both Rust and JS) — document in `/.overhaul/duplication-audit.md`. Determine which is the source of truth. If the Rust backend is authoritative, the frontend should call it rather than reimplement.
- **Intra-component duplication** — similar utility functions, repeated transformation logic, copy-pasted handlers.
- **Near-duplicates** — functions that do almost the same thing with slight variations.

**Important: the goal is NOT maximum DRY-ness.** There is a tradeoff between reducing duplication and keeping code readable without jumping between files. Apply this rule of thumb:

- **Short snippets (< ~10 lines)**: Leave them duplicated if extracting them would force the reader to jump to another file to understand the logic. Inline readability > DRY purity.
- **Medium logic (10-30 lines)**: Extract only if it's duplicated 3+ times OR if the logic is complex enough that having a single source of truth prevents bugs.
- **Long/complex logic (30+ lines)**: Always extract. Multiple copies of complex logic will inevitably diverge and cause bugs.

For Rust hot paths: **if two similar functions exist and each is optimized for its specific case, KEEP THEM SEPARATE.** Document why in comments with `// PERF: intentional` and in CLAUDE.md. Do not unify them behind a generic interface that adds vtable dispatch or branch prediction overhead.

### 4b. Propose structure improvements

Write `/.overhaul/target-structure.md` with proposed changes for each component. Principles:

**Rust Backend:**

- Organize by computational concern (state representation, probability, strategy, simulation, sweep orchestration, web server)
- Keep hot-path code in dedicated modules with clear `// PERF:` markers
- Ensure the web server layer is thin — it should delegate to computation modules, not contain logic
- All simulation/sweep entry points should be clearly parameterized and documented so that analytics and frontend can invoke them correctly

**Frontend:**

- Group by feature/page, not file type
- Shared modules (API client, D3 style config, utilities) in a clear location
- Each D3 visualization as a self-contained module in `visualizations/`
- API client layer isolated

**Analytics Pipeline:**

- Split into `tools/` (reusable atomic analysis functions) and `pipelines/` (composed workflows)
- Each tool should have a clear, documented interface
- Pipeline definitions should read like recipes
- Organize output directories to mirror the tool/pipeline structure

**Stop and ask me to approve the target structure before executing moves.**

> When I confirm, proceed:

3. Move files to match the target structure.
4. Update all import paths.
5. Verify each component builds and tests pass after restructuring.
6. Run `just bench-check` and `just test` — no regressions allowed.
7. **Update all CLAUDE.md files to reflect the new structure.**

---

## Phase 5: Rust Backend — Performance-Preserving Cleanup

**Critical constraint: `just bench-check` must pass after every change in this phase. Any failure means revert immediately.**

1. **Web server optimization for gameplay latency:**
   - Audit every gameplay-critical endpoint for unnecessary allocations, serialization overhead, or blocking calls.
   - Ensure the server uses async properly — no accidental blocking of the async runtime with CPU-heavy computation.
   - If CPU-heavy computation is dispatched from async handlers, verify it's sent to a blocking thread pool (e.g., `tokio::spawn_blocking` or dedicated compute pool).
   - Audit serialization: is JSON the right format for high-frequency gameplay calls? Document if a binary format (MessagePack, bincode, protobuf) would help and by how much.
   - Audit response sizes — are we sending more data than the frontend needs?

2. **Simulation & sweep ergonomics:**
   - Ensure every simulation type and parameter sweep has a clean, well-documented entry point.
   - Parameters should be structured (typed structs with builder patterns, not loose arguments).
   - Ensure sweep results include all metadata required by the analytics pipeline (parameters used, timestamps, configuration, etc.).
   - Ensure output is saved to the standard `data/sweeps/` directory with proper naming.

3. **Code clarity without performance cost:**
   - Add extensive `// PERF:` comments explaining WHY non-obvious performance choices were made (bit tricks, cache-line alignment, lookup table strategies, unsafe blocks).
   - Improve naming where it doesn't affect inlining or codegen.
   - Add `#[inline]` annotations where appropriate and document the reasoning.
   - Remove dead code, unused feature flags, obsolete modules.
   - Add `debug_assert!` for invariants in hot paths (zero cost in release builds).
   - Add doc comments following the standard defined in Phase 3f to ALL public functions.
   - **Do NOT add excessive error handling in the HPC core.** In hot paths and simulation code, `unwrap()`/`expect()` on structurally guaranteed invariants is fine and preferred over `Result` chains that add overhead. Only handle errors that can realistically occur. **However, the web server layer (`src/server/`) is the opposite — handlers MUST NOT panic on user input.** Malformed requests must return HTTP 400/500, not crash the server.
   - **Do NOT add unnecessary dependencies.** If a utility function can be written in a few lines, write it rather than pulling in a crate.

4. **Run `just bench-check`**: Must pass. Document results in `/.overhaul/benchmark-post-cleanup.md`.

---

## Phase 6: Frontend — Strip Framework, Adopt D3.js

This phase migrates the frontend from whatever framework currently exists to **pure HTML + TypeScript + D3.js**. This is intentionally a rewrite of the frontend layer (the one exception to the "refactor, not rewrite" ground rule — approved by the owner).

### 6a. Framework removal

1. **Inventory the current frontend**: List every page, component, and interaction. Map what each component does in plain terms (not framework terms).
2. **Rewrite in pure HTML + TypeScript**: Replace framework components with vanilla TS modules. Each "page" or "view" is an HTML file (or a section of a single-page app managed with simple routing). State management uses plain TypeScript — an event emitter, observable, or simple pub/sub pattern. No framework, no virtual DOM.
3. **Remove the framework dependency** (React, Vue, Svelte, whatever was there) and all its ecosystem packages (react-dom, react-router, state management libraries, etc.). This will likely eliminate a large fraction of the dependency tree.
4. Ensure consistent API client patterns — one module for all backend calls, with type safety.
5. Ensure the frontend does NOT reimplement game logic that should come from the backend.

### 6b. D3.js as primary visualization

1. **Add D3.js** as the primary (and ideally only) large frontend dependency.
2. **Structure each visualization as a self-contained module**:
   ```typescript
   // Example: every chart module exports a function like this
   export function renderScoreDistribution(
     container: HTMLElement,
     data: ScoreDistributionData,
     options?: ChartOptions,
   ): void {
     // D3 bindspossession to the container, owns the DOM inside it
     const svg = d3.select(container).append("svg")...
   }
   ```
3. **Explore D3's full range** when choosing chart types. D3 has hundreds of examples (force-directed graphs, Sankey diagrams, treemaps, violin plots, parallel coordinates, etc.). When visualizing Yatzy data, don't default to bar charts — consider what visualization best reveals the structure of the data. Browse [D3 Gallery](https://observablehq.com/@d3/gallery) and [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html) for inspiration.
4. **Use consistent styling**: Define a shared D3 style module with the Yatzy color palette (matching the analytics YATZY_CMAP where appropriate), consistent margins, font sizes, axis formatting.

### 6c. Standard cleanup

1. Remove dead code, unused dependencies, commented-out experiments.
2. Add/improve TypeScript interfaces for all API contracts and data types.
3. Extract magic numbers into named constants.
4. Simplify control flow, improve naming.
5. Keep error handling minimal — a simple try/catch at API call boundaries is enough.
6. Add TSDoc comments following the standard defined in Phase 3f to ALL exported functions and modules.

---

## Phase 7: Analytics Pipeline — Build the Toolbox

This phase transforms scattered analytics code into a composable, well-documented toolbox built on **Python + Polars + Seaborn**.

### 7a. Establish the core stack

1. **Polars, not pandas.** Migrate all dataframe operations to Polars. Remove pandas from dependencies. **Boundary rule**: Seaborn expects pandas, so call `.to_pandas()` only at the exact moment you pass finalized data into a Seaborn/Matplotlib plotting function. Never use pandas for data processing, filtering, or aggregation.
2. **Seaborn for visualization**, with our custom coolwarm palette as the default for all plots.
3. **Define the palette** in `analytics/style.py`:
   ```python
   """Yatzy analytics visual style configuration.

   All plotting functions should import and use these defaults.
   """
   import matplotlib.pyplot as plt
   import matplotlib.colors as mcolors
   import seaborn as sns

   # Custom coolwarm: standard coolwarm with a distinct center color.
   # The center color is specified in the Pre-Flight Checklist.
   # If still [TODO], stop and ask — do not guess.
   YATZY_CMAP = ...  # LinearSegmentedColormap

   # Convenience palette for categorical data
   YATZY_CATEGORICAL = ...  # List of colors derived from the colormap

   def apply_style():
       """Apply Yatzy visual defaults to matplotlib/seaborn."""
       sns.set_theme(style="whitegrid")
       plt.rcParams.update({
           "figure.figsize": (10, 6),
           "figure.dpi": 150,
           "savefig.dpi": 150,
           "savefig.bbox": "tight",
       })
   ```
4. **Minimize dependencies beyond the core.** The stack is Python + Polars + Seaborn + matplotlib (transitive via Seaborn). Add other libraries only if they provide substantial, non-trivial functionality. Prefer writing a utility function over adding a dependency.

### 7b. Build the toolbox

1. **Inventory every analysis currently performed.** For each one, extract it into a standalone, reusable function in `tools/` with:
   - Clear function name describing what it computes
   - Typed input signature (Polars DataFrame/LazyFrame schema, parameter types)
   - Typed output signature
   - Google-style docstring following the standard in Phase 3f, including output location
   - Example usage in the docstring
   - `See Also:` cross-references to related tools

2. **Categorize tools** into logical groups:
   - `tools/probability/` — probability distributions, expected values, variance analysis
   - `tools/strategy/` — strategy comparison, optimality analysis
   - `tools/simulation/` — simulation result processing, convergence analysis
   - `tools/visualization/` — plotting functions, chart generators (ALL use YATZY_CMAP)
   - `tools/io/` — data loading, format conversion, export
   - (Adjust categories based on what actually exists)

3. **Build pipeline infrastructure:**
   - Create a lightweight pipeline runner that chains tools together
   - Each pipeline step should: accept input, produce output, log what it did, and be skippable/resumable
   - Pipeline definitions should be declarative where possible (YAML/TOML config or simple Python scripts that read like recipes)
   - Pipeline outputs should go to the standard output directories defined in `docs/output-map.md`

4. **Ensure clean backend integration:**
   - Analytics should call the backend through a well-defined client module, not scattered HTTP calls
   - The client module should mirror the API reference from CLAUDE.md
   - Sweep requests from analytics should pass all required parameters so results are complete and self-describing

5. **Update `analytics/CLAUDE.md`** with the full toolbox catalog — this is the "menu" for future Claude sessions to pick from when asked to analyze Yatzy data.

---

## Phase 8: Configuration & Environment Management

1. Each component gets a config module that:
   - Reads from environment variables with `.env` fallback for local dev
   - Validates required config at startup and fails fast with a clear message
   - Exposes typed config objects, not raw strings
   - Separates config by concern (server, computation, external APIs, feature flags)
2. Remove any scattered env var reads throughout each codebase — they should all go through the config module.
3. Update `.env.example` with complete documentation of every variable: expected type, default (if any), and what it controls.
4. For the Rust backend, ensure performance-relevant configuration (thread counts, pool sizes, cache sizes, precomputation flags) is clearly exposed and documented.
5. **Keep it simple.** Don't over-engineer the config system. A simple module that reads env vars and exposes typed values is enough. No need for elaborate config frameworks.
6. **Protect git from large generated files.** Ensure `data/sweeps/`, `data/cache/`, and `output/plots/` are in `.gitignore`. Experiment data (`data/experiments/`) and reports (`output/reports/`) should be tracked selectively — gitignore the raw data subdirectories but track the lab reports and index.

---

## Phase 9: Testing & Quality Gates

### Rust Backend:

- Performance tests already established in Phase 2b — verify they still pass.
- Unit tests for game logic, probability calculations, strategy evaluation.
- Integration tests for API endpoints — verify correct responses AND acceptable latency.
- Property-based tests where applicable (probabilities sum to 1, strategy evaluation is deterministic, etc.).

### Frontend:

- Unit tests for API client module and data transformation logic.
- Integration tests verifying D3 visualizations render without errors given known data.
- End-to-end tests for key user flows.

### Analytics:

- Unit tests for every tool in the toolbox — known inputs produce known outputs.
- Integration test: run a small end-to-end pipeline and verify output.
- Verify all plots use YATZY_CMAP by default.

### All components — Task runner:

Add a `justfile` (or `Makefile`) at the repo root with these standard targets:

```
lint            # lint all components
typecheck       # typecheck all components
test            # run all tests
bench-check     # run wall-clock perf tests, compare to baseline, PASS/FAIL
bench-baseline  # record new performance baseline (M1 Max)
fmt             # auto-format all components
check           # lint + typecheck + test + bench-check (fail on any failure)
dev-backend     # start Rust backend in dev mode
dev-frontend    # start frontend in dev mode
sweep           # run parameter sweep (with args)
analyze         # run analytics pipeline (with args)
build           # production build for all components
```

Configure linter and formatter for each component if not already present.

---

## Phase 10: Final Documentation Pass

1. **Update all CLAUDE.md files** with everything learned during the overhaul. Walk through each file and ensure accuracy against the actual codebase as it now exists. This is the most important deliverable.

2. **Update README.md**:
   - What the project does (1-2 paragraphs)
   - Quickstart (clone, install deps, configure, run)
   - Development workflow (using the justfile targets from Phase 9)
   - Architecture overview (link to docs/architecture.md)
   - Configuration reference (link to .env.example)

3. **Verify all doc comments/docstrings**: Every public function in all three components must have documentation following the standards from Phase 3f. Output-producing functions must document where their output goes.

4. **Update `docs/output-map.md`**: Ensure it accurately reflects all output locations after the overhaul.

5. **Write `CONTRIBUTING.md`** with coding conventions established in this overhaul, including:
   - The function documentation standards for each language
   - The CLAUDE.md update protocol (what to update when you change what)
   - Performance rules for Rust hot paths
   - How to add new tools to the analytics toolbox
   - The experiment/lab report protocol
   - Dependency philosophy (minimize, prefer inline)
   - Error handling philosophy (minimize, keep code lean)

6. **Run the Phase 3-Verify validation again**: Walk through the same 5 tasks and verify the final CLAUDE.md hierarchy has enough information to guide each one without exploration. Fix any gaps.

7. **Run `just bench-check` and `just test`** one final time. Confirm all performance tests still pass against the original baseline from Phase 2b, and all correctness tests pass per Phase 2c.

8. **Verify CLAUDE.md conciseness**: Ensure the root CLAUDE.md is still under ~150 lines. If it has grown, move detail into child files or `@imports`. The root file is a router, not an encyclopedia.

---

## Ground Rules (ALL phases)

- **Never commit secrets.** If found during any phase, handle per Phase 1 immediately.
- **Atomic commits.** At the end of every successful phase, run `git add .` and `git commit -m 'overhaul: complete Phase X'`. Never move to the next phase with uncommitted changes. If a change must be reverted, `git stash` or `git checkout` cleanly.
- **Rust performance is non-negotiable.** `just bench-check` must pass after every phase. Wall-clock regression on M1 Max = failure, full stop.
- **Correctness is non-negotiable.** `just test` must pass (modulo pre-existing failures documented in Phase 2c) after every phase.
- **Do not DRY-ify Rust hot paths at the cost of indirection.** Duplicated-but-fast beats abstracted-but-slow. Mark intentional duplication with `// PERF: intentionally duplicated — see CLAUDE.md` comments.
- **DRY is not a religion.** Short, self-contained snippets (< ~10 lines) can stay duplicated if extracting them would force jumping between files. Extract complex logic (30+ lines); leave simple repetition alone.
- **Minimize dependencies.** Prefer a few lines of code over a library. Especially in the frontend. Only add a dependency when it provides substantial, non-trivial functionality.
- **Minimize error handling — with a boundary.** In the HPC core and simulation code, `unwrap()`/`expect()` on guaranteed invariants is preferred. Don't build elaborate error hierarchies. **But the web server layer must never panic on user input** — handlers return HTTP errors, not panics. In the frontend, a simple try/catch at API boundaries is enough.
- **Analytics stack is Python + Polars + Seaborn.** No pandas except `.to_pandas()` at the Seaborn boundary. All plots use YATZY_CMAP (custom coolwarm).
- **Experiments follow the lab report protocol.** Falsifiable hypothesis, pre-registered predictions, clinical results, separate discussion. Always. No exceptions.
- **Track where outputs go.** Every function that generates files must document the output location in its docstring. Keep `docs/output-map.md` current.
- **Atomic phases.** Each phase leaves the project buildable and runnable.
- **Document everything.** The primary output of this overhaul is not cleaner code — it's a codebase that Claude can navigate instantly. Every decision, every "why", every non-obvious choice gets documented.
- **Preserve behavior.** This is a refactor, not a rewrite. Flag suspected bugs in `/.overhaul/potential-bugs.md` but don't fix them unless asked. **Exception: Phase 6 is an approved rewrite of the frontend from its current framework to pure HTML + TypeScript + D3.js.**
- **Frontend is pure HTML + TypeScript + D3.js.** No frameworks. D3 is the primary visualization library — explore its full gallery for chart type inspiration. Each visualization is a self-contained module that owns its DOM container.
- **When in doubt, document rather than change.** A well-documented quirk is better than a well-intentioned refactor that breaks something.
- **Update CLAUDE.md continuously.** Don't save it all for the end. Every phase should leave CLAUDE.md files more complete than it found them.
- **Keep the root CLAUDE.md lean.** Route to detail, don't contain it. Use `@imports` and child CLAUDE.md files for depth.
- **Stop between phases.** At the end of each phase, STOP. Print a summary of what was done, commit, and prompt me to `/clear` context and give permission to begin the next phase. Do NOT proceed to the next phase autonomously.
