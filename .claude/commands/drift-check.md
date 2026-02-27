# Structural Drift Check

Audit the codebase for structural drift from project conventions. New code and files are expected and welcome — this check verifies they follow the rules, not that nothing changed.

## 1. Documentation Currency

For each CLAUDE.md file (root, solver/, frontend/, analytics/, theory/README.md):

- Pick 5 random entries and verify they match actual code. Flag any that are stale.
- Check for new public functions/endpoints/tools that are NOT documented.
- Verify line count of root CLAUDE.md is under ~150 lines.

**Escalate if:** Any CLAUDE.md entry is stale, or any new public API is undocumented.

## 2. File Placement

Scan for files that are in the wrong place:

- Markdown files in repo root or unexpected locations that should be in `theory/`
- Generated output (plots, CSVs, parquet, binaries) outside `output/` or `data/`
- Test files outside the test directories
- Config files scattered instead of in the config module
- Any `.jsx` or `.tsx` files anywhere (framework was removed)

**Escalate if:** Any file is misplaced.

## 3. Theory Directory Health

- Every file in `theory/` subdirectories is listed in its subdirectory README
- No file exceeds ~500 lines (`wc -l`)
- No orphaned files (in a subdirectory but not in its README)
- `theory/README.md` key findings are still accurate
- New theory files use lowercase-kebab-case and describe content, not author

**Escalate if:** Any README is out of date, any file over limit, any orphan found.

## 4. Dependency Discipline

- Frontend `package.json`: no dependencies beyond D3.js and dev tooling
- Python: no `import pandas` (only `.to_pandas()` at Seaborn boundary)
- Rust `Cargo.toml`: any crate without a `# Why this dependency` comment
- Check for new dependencies added since last git tag — flag each for review

**Escalate if:** Any undocumented dependency or banned import found.

## 5. Output Directory Hygiene

- Output directories in `outputs/` and `data/` match the Data Layout in root `CLAUDE.md`
- No new output directories that aren't documented there
- `.gitignore` covers `data/`, `outputs/`, and cache directories

**Escalate if:** Data Layout section is stale or new output dirs are undocumented.

## 6. Convention Compliance (spot check)

Sample 3 recently modified files per component (`git log --diff-filter=M -10`):

- Rust: public functions have doc comments, hot paths aren't over-abstracted, web handlers return errors (no panics on user input)
- Frontend: uses D3 container pattern for visualizations, API calls through client module, no framework imports
- Analytics: uses Polars (not pandas), plots use YATZY_CMAP, functions have Google-style docstrings
- All: output-producing functions document where output goes

**Escalate if:** Any sampled file violates conventions.

## Reporting

Print a summary as you go. For each section, report CLEAN or list specific violations.

**If everything is clean:** Say so. Done.

**If violations found:** Collect all violations into a numbered list, grouped by severity:

- **STRUCTURAL** — file in wrong place, missing from README, framework code detected
- **DOCUMENTATION** — stale CLAUDE.md, undocumented function, output map drift
- **CONVENTION** — style violations, banned imports, missing comments

Present the list and ask: "Want me to fix these? I'll take them one at a time and show you each change before committing."

Do NOT fix anything without explicit approval. New code is welcome — the point is that it follows the rules.
