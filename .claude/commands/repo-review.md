# Repo Review: Yatzy Project

A systematic audit skill for the yatzy polyglot monorepo. Designed to catch the kinds of decay that no single linter or test suite will find: stale files left behind after migrations, endpoints serving content nobody visits, convention drift between sub-projects, and the slow accumulation of things that once mattered but no longer do.

## Philosophy

Repos decay in predictable ways. Code gets moved but the old copy lingers. Routes get renamed but the old endpoint keeps serving. A sub-project adopts a new convention but the others don't follow. Dependencies get added for a feature that was abandoned. None of these individually break anything. Together they create a codebase where nobody is quite sure what's alive and what's dead.

This skill is not a linter. It is an investigative audit. It produces a findings report with evidence and recommendations, not auto-fixes.

## When to run

- Periodically (monthly or quarterly) as hygiene
- After a significant restructuring or migration
- Before a major release or demo
- When onboarding someone new and you realize you can't explain the repo layout cleanly
- When something feels off but you can't point at what

## Audit procedure

Work through these phases in order. For each phase, report findings with:
- **What** you found (specific files, paths, endpoints)
- **Why** it matters (serving stale content, confusing future contributors, wasting build time)
- **Severity**: CRITICAL (actively serving wrong content or security risk), MODERATE (dead weight, confusion risk), LOW (cosmetic, minor inconsistency)
- **Recommended action**: delete, migrate, update, or document-as-intentional

---

### Phase 1: Ghost files and orphaned paths

The most common form of decay. Something gets moved or replaced, but the original stays.

**What to check:**

1. **Stale web assets after route migrations.** Look for HTML, CSS, JS files at paths that were superseded by newer routes. The live routes are `/yatzy/` (treatise), `/yatzy/play/` (game UI), `/yatzy/profile/` (profiler), and `/yatzy/probabilities/` (probabilities tab), plus `/yatzy/shared/` (shared nav, dice, and math modules) and `/yatzy/data/` (chart JSON). For every current route, check whether an older path variant still has files. Note: a `/blog/yatzy/` path was retired long ago and `blog/` is gone, so that specific migration is settled; the remaining ghost is the completed C-to-Rust backend migration (see item 4).

2. **Duplicate implementations.** Search for files with highly similar names or content across different directories. The biggest real case here is the C-to-Rust backend migration: `legacy/` holds the old C implementation and `solver/` is the current Rust one, both described as producing bit-identical output. `legacy/` is documented as intentionally kept for reference (`legacy/README.md`), so confirm that framing still holds rather than assuming it is dead. Otherwise look for two copies of the same module across sub-projects (e.g. a math helper duplicated in `shared/` and inlined in a chart, or a `utils.ts` copied between UIs).

3. **Dead imports and unused modules.** For each sub-project, trace the import/dependency graph from the entry points. Flag any source file that is not reachable from any entry point, test file, or build target.

4. **Abandoned feature branches merged as directories.** Sometimes experimental code gets committed as a directory (`/experimental/`, `/old/`, `/backup/`, `/tmp/`, `/legacy/`) and never cleaned up. This repo has a `legacy/` directory (the old C backend). It is currently intentional-for-reference per its README; the audit's job is to confirm that is still the intent and that nothing in the live build depends on it, not to assume it is either dead or sacred.

5. **Generated files committed to source.** Build artifacts, compiled outputs, `.pyc` files, `node_modules` fragments, `target/` contents, or lockfiles for tools not actually used.

**How to investigate:**

```bash
# Find HTML files that might be stale served content
find . -name "*.html" -not -path "*/node_modules/*" -not -path "*/target/*" | sort

# Find files not modified in a long time relative to the repo
git log --all --diff-filter=M --name-only --pretty=format: --since="6 months ago" | sort -u > /tmp/recently_modified.txt
find . -name "*.rs" -o -name "*.ts" -o -name "*.tsx" -o -name "*.py" | sort > /tmp/all_source.txt
comm -23 /tmp/all_source.txt /tmp/recently_modified.txt

# Find potential duplicates by filename
find . -type f -name "*.ts" -o -name "*.rs" -o -name "*.py" | xargs -I{} basename {} | sort | uniq -d

# Find directories that look like abandoned experiments
find . -maxdepth 3 -type d \( -name "old" -o -name "backup" -o -name "tmp" -o -name "deprecated" -o -name "experimental" -o -name "legacy" -o -name "v1" -o -name "v2" \) -not -path "*/node_modules/*"
```

Combine mechanical scanning with reading. The scripts find candidates; you read the code to determine if they're actually dead.

---

### Phase 2: Endpoint and routing audit

Web servers happily serve stale content forever. This phase specifically checks what is being served versus what should be.

**What to check:**

1. **Server configuration vs. actual content.** Read the web server config (`deploy/nginx.conf` for production, `frontend/vite.config.ts` `serveDir` plugins for dev, and the axum router in `solver/`) and map every configured route to the files or handlers it resolves to. The two must agree: a UI added to `vite.config.ts` but not to `nginx.conf` (or vice versa) works locally but 404s in production, or the reverse. Current routes: `/yatzy/` (treatise, volume-mounted), `/yatzy/play/`, `/yatzy/profile/`, `/yatzy/probabilities/` (baked into the frontend image under `apps/`), `/yatzy/shared/` and `/yatzy/data/` (served from the treatise volume root), and `/yatzy/api/` (proxied to the solver). Flag routes pointing at content that has not been updated alongside the rest of the project.

2. **Static file directories.** Identify every directory served as static content: `treatise/`, `frontend/dist/`, `profiler/`, `probabilities/`, and `shared/`. Walk each and check whether every file is referenced by current application code (index.html script/link tags, chart modules, `shared/` imports). Also confirm the three places a new static UI must be wired stay in sync: `shared/nav.js` (the tab), `frontend/vite.config.ts` (dev serving), and `deploy/nginx.conf` + `deploy/deploy.sh` (production).

3. **API endpoints with no consumers.** For every API endpoint defined in the Rust solver (the axum backend under `solver/`), search the consumers for references: `frontend/src/api.ts` is the main client, and `analytics/` may call it too. Note the profiler and probabilities tabs are fully client-side (they read pre-computed JSON, not the API), so absence of a reference there is expected, not a dead endpoint. An endpoint with zero consumers anywhere is either used externally (document it) or dead (remove it).

4. **Redirect chains.** Look for redirects that point to other redirects, or redirects to paths that no longer exist.

5. **CORS / security headers on dead endpoints.** Stale endpoints that still accept requests may have outdated security configurations.

**How to investigate:**

```bash
# Extract route definitions from the Rust solver (axum backend)
grep -rn "route\|get(\|post(\|put(\|delete(\|\.at(\|\.route(" solver/src/ --include="*.rs"

# Extract fetch/API calls from the frontend (and client-side data fetches in the UIs)
grep -rn "fetch(\|axios\.\|api\.\|/api/\|/yatzy/data/" frontend/src/ profiler/js/ probabilities/js/ --include="*.ts" --include="*.js"

# Map dev vs prod routing (must agree)
grep -n "serveDir\|proxy" frontend/vite.config.ts
grep -nE "location" deploy/nginx.conf
```

---

### Phase 3: Cross-project coherence

The yatzy repo is polyglot. The surfaces are: the Rust solver (`solver/`), the TypeScript + Vite game UI (`frontend/`), the Python analytics (`analytics/`), and three vanilla-JS static UIs with no build step (`treatise/`, `profiler/`, `probabilities/`) plus a `shared/` directory of plain ES-module JS (nav, dice renderer, and the `path-prob.js` / `score-prob.js` math modules) imported by several of them. Each surface has its own idioms, but certain things should stay aligned.

**What to check:**

1. **Shared data contracts.** The Rust solver defines types that the TypeScript frontend consumes (API response shapes, game state). Verify they match `frontend/src/api.ts`. Flag any fields present in one but not the other, or type mismatches.

2. **Shared-module drift.** The `shared/` modules are single sources of truth used by more than one UI, so a change on one side that the other does not expect is a real bug. In particular: `shared/path-prob.js` and `shared/score-prob.js` are consumed by BOTH the probabilities tab (`probabilities/js/`) and the treatise charts (`treatise/js/charts/path-probability.js`, `risk-theta.js`); `shared/dice.js` / `dice.css` render dice everywhere; `shared/nav.js` is the nav for every UI. Confirm each shared export still matches every caller, and that the probabilities tab's `kde_curves.json` fetch still matches the file the treatise generates. When a UI is added, `shared/nav.js`'s `PAGES` and `detectActive()` must include it.

3. **Naming conventions.** Check whether equivalent concepts use the same names across sub-projects. If the solver calls it `game_session` and the frontend calls it `match`, that's drift. Compile a glossary of key domain terms and check consistency.

4. **Configuration drift.** Compare linter and formatting configs across sub-projects (there is no CI to compare; see Phase 4). Are they using compatible settings where they should be? Note: some drift is expected, since Rust, TypeScript, Python, and plain JS have different norms. Flag only where conventions should agree but diverge.

5. **Dependency version alignment.** For shared concerns (serialization formats, HTTP libraries, test frameworks), check that sub-projects use compatible versions. A Rust solver serializing with serde while the Python analytics expects a different JSON shape is a real bug.

6. **Documentation coherence.** The repo carries a lot of prose: the root and per-sub-project `CLAUDE.md` files, the `theory/` directory (which has its own `.claude/rules/theory.md` conventions and README index), the `docs/` directory, and the treatise itself. Check that the treatise accurately describes the current architecture, that the `CLAUDE.md` files agree with each other about interfaces and boundaries, and that any place claiming to describe the directory structure (root `CLAUDE.md` "Data Layout" / "Components" tables, `theory/README.md`) still matches reality. New top-level dirs to be aware of: `theory/`, `docs/`, `research/`, `configs/`.

---

### Phase 4: Dependency and build hygiene

**What to check:**

1. **Unused dependencies.** For each sub-project's dependency manifest (Cargo.toml, package.json, pyproject.toml / requirements.txt), check whether each dependency is actually imported somewhere in the source code.

2. **Pinning and lockfile consistency.** Are lockfiles committed? Are dependency versions pinned appropriately? Is anyone using `*` or `latest`?

3. **Build targets that don't build.** Try building each sub-project independently. Flag any that fail or produce warnings. A sub-project that hasn't been built in months may have silently broken.

4. **CI/CD coverage.** This project deliberately has NO CI (there is no `.github/workflows`; testing is manual via `cargo test` and `just audit` on the dev machine). Do NOT flag the absence of CI as a finding. Instead, since there is no automated gate, put extra weight on the manual equivalents: does `just check` / `just test-all` still run and pass, do the test counts referenced in the `CLAUDE.md` files still match, and is there test code referencing modules that no longer exist?

5. **Docker and deployment artifacts.** If there are Dockerfiles, docker-compose files, or deployment configs, check they reference current paths, current build commands, and current environment variables.

**How to investigate:**

```bash
# Rust: find unused dependencies (Rust lives in solver/, not backend/)
grep -oP 'name = "\K[^"]+' solver/Cargo.toml | while read dep; do
  grep -rq "$dep" solver/src/ || echo "Possibly unused: $dep"
done

# TypeScript: find unused dependencies
npx depcheck frontend/ 2>/dev/null || echo "Install depcheck for automated analysis"

# Python: find unused dependencies
pip install pipreqs 2>/dev/null
pipreqs analytics/ --print 2>/dev/null

# Static JS UIs (treatise, profiler, probabilities) have no build/manifest;
# check their imports resolve against shared/ and each other instead
grep -rn "import .* from '/yatzy/shared/" treatise/js probabilities/js profiler/js 2>/dev/null
```

---

### Phase 5: History and intent archaeology

Sometimes the only way to know if something is dead is to look at when and why it was last touched.

**What to check:**

1. **Files untouched since a major refactor.** Identify the last big structural change (from git log). Find files that predate it and were never updated to match the new structure.

2. **TODOs and FIXMEs with no associated work.** Grep for TODO, FIXME, HACK, TEMP, XXX. For each one, check if it references an issue or has a date. Flag any older than 6 months with no associated issue.

3. **Commented-out code.** Large blocks of commented-out code are almost always dead. Flag any block longer than 10 lines.

4. **Git blame anomalies.** Files where the most recent commit to every line is the same (bulk reformatting or migration) may be hiding the fact that the file hasn't been meaningfully touched since before the migration.

**How to investigate:**

```bash
# Find TODOs with age
grep -rn "TODO\|FIXME\|HACK\|XXX\|TEMP" --include="*.rs" --include="*.ts" --include="*.tsx" --include="*.py" | while read line; do
  file=$(echo "$line" | cut -d: -f1)
  lineno=$(echo "$line" | cut -d: -f2)
  last_touched=$(git log -1 --format="%ai" -L "$lineno,$lineno:$file" 2>/dev/null | head -1)
  echo "$line  [last touched: $last_touched]"
done

# Find large commented-out blocks (rough heuristic)
grep -rn "^[[:space:]]*//" --include="*.rs" --include="*.ts" | awk -F: '{print $1}' | uniq -c | sort -rn | head -20

# Files with oldest last-modified dates
git ls-files | while read f; do
  echo "$(git log -1 --format='%ai' -- "$f") $f"
done | sort | head -30
```

---

## Output format

Produce a structured report in Markdown with:

```markdown
# Repo Review: [date]

## Summary
[2-3 sentence overview: overall health, number of findings by severity, biggest concern]

## Findings

### CRITICAL
#### [Finding title]
- **Location:** [file paths or endpoints]
- **Evidence:** [what you found and how]
- **Impact:** [what could go wrong]
- **Recommendation:** [specific action]

### MODERATE
...

### LOW
...

## Cross-project coherence status
[Brief table or summary of alignment between Rust solver, TypeScript frontend, and Python analytics]

## Recommendations
[Prioritized list of actions, grouped by effort level: quick wins, medium effort, larger refactors]
```

Save the report to `repo-review-[YYYY-MM-DD].md` in the repo root.

---

## Important caveats

- **Not everything old is dead.** Some files are intentionally stable. A utility function untouched for a year might be perfect, not abandoned. Use judgment.
- **Ask before deleting.** This skill produces a report, not a cleanup script. The human decides what to act on.
- **Check what's actually being served.** For web content findings, verify by actually requesting the URL if possible, not just by looking at the filesystem.
- **Respect the treatise.** The treatise is a first-class artifact describing the project's architecture and design. Check it for accuracy but treat it as authoritative intent, not just documentation to be auto-updated.
