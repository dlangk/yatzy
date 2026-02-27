# Documentation Quality Check

Audit code comments and project documentation for completeness, consistency, and readability. The goal: a human should be able to read any hot path from entry point to leaf function guided entirely by comments, never needing to puzzle out what a block does or why.

## 1. Hot Path Narrative Audit

This is the most important check. For each hot path listed in `solver/CLAUDE.md`:

1. Start at the entry point function.
2. Read through the call chain, function by function, following the execution path.
3. At each function, check:
   - **Module-level comment** exists explaining what this file is responsible for
   - **Function doc comment** explains what it does, not how (the "what" and "why", not line-by-line "how")
   - **Section comments** break the function body into logical steps. A reader should be able to skim just the comments and understand the algorithm's structure.
   - **Non-obvious lines** have inline comments. If a line requires domain knowledge (Yatzy rules, NEON intrinsics, CSR format details, probability math) or makes a surprising choice, it must have a comment.
   - **Performance comments** explain why a particular approach was chosen over the obvious one. Use PERF prefix for performance-motivated decisions. Use SAFETY prefix for unsafe blocks.
   - **No comment parrots the code.** A comment like "increment counter" above counter += 1 is worse than no comment. Comments explain intent, not syntax.

4. Write a reading log: "I can follow the path from X to Y. Gap at Z — the transition from A to B is unclear because [reason]."

**Escalate if:** Any point in a hot path where a reader would have to stop and reverse-engineer what's happening.

## 2. Function Documentation Coverage

### Rust
- Every pub fn has a doc comment (///).
- Doc comments for solver/simulation functions include: what it computes, what the inputs represent in game terms, and what the output means.
- Unsafe blocks have SAFETY comments explaining the invariant that makes them sound.

### TypeScript
- Every exported function has a TSDoc comment.
- D3 visualization modules document: what data they expect, what they render, and what container interface they need.

### Python
- Every function in analytics/src/ has a Google-style docstring with Args, Returns, and a one-line description.
- Docstrings for output-producing functions state where the output goes.

Sample 10 functions per component. Report coverage as X/10.

**Escalate if:** Coverage below 8/10 in any component.

## 3. Comment Style Consistency

Scan for inconsistent comment styles within each component:

### Rust
- Section headers use // --- Section Name --- or blank-line separation (pick whichever the codebase already uses more — flag the minority style)
- No block comments in function bodies — use line comments instead
- Doc comments use triple-slash (///), not bang-comments except at module top

### TypeScript
- TSDoc uses block doc comments, not line comments
- Section comments within functions use //

### Python
- Docstrings use triple-quoted strings
- Inline comments use #

**Escalate if:** More than 5 instances of inconsistent style in any component.

## 4. CLAUDE.md and README Consistency

- Every CLAUDE.md file's "Architecture" section matches the actual module structure (spot-check 5 entries)
- API reference entries in solver/CLAUDE.md match actual handler signatures
- CLI catalog in analytics/CLAUDE.md matches actual CLI commands
- theory/ subdirectory READMEs match actual files and line counts
- Cross-references between docs (See Also links) point to files that exist

**Escalate if:** Any stale reference found.

## 5. Naming Clarity

Sample 10 recently modified functions per component. For each, check:

- Function name describes what it does (verb + noun), not how
- Parameter names are meaningful (no single letters except i, j, k in tight loops, or conventional math like theta, sigma)
- Return types are named types, not raw tuples or anonymous objects (for anything with more than 2 fields)
- Constants have descriptive names (BONUS_THRESHOLD not BT)

**Escalate if:** More than 3 unclear names per component.

## 6. Change-Relative Comment Pollution

Scan for comments that describe what changed rather than what the code does now. These rot instantly. Use Grep to find patterns like: "Fixed", "Previously", "Changed from", "Refactored", "Now handles", "Workaround for", "Updated to", "Added support for", issue/PR numbers as sole explanation.

For each match, determine if it is a genuine change-relative comment or a false positive (e.g. "fixed" used as an adjective like "fixed size"). Only flag genuine change-relative comments.

Count total matches per component.

**Escalate if:** Any genuine change-relative comment found. Present matches grouped by file with proposed rewrites: either rewrite to explain current behavior, or delete if the code is self-explanatory.

## 7. Dead Documentation

Check for documentation that describes things that no longer exist:

- Comments referencing removed functions, old module names, or deleted files
- TODO/FIXME/HACK comments older than 60 days (git log the file to estimate)
- Commented-out code blocks longer than 5 lines (should be deleted, not commented)

**Escalate if:** Any dead documentation found.

## Reporting

For each section, report CLEAN or list specific issues.

**If everything is clean:** Say so. Done.

**If issues found:** Group by type:

- **READABILITY GAP** — hot path has unclear transitions, missing section comments, non-obvious code without explanation
- **MISSING DOCS** — undocumented public function, missing docstring, no module comment
- **STALE COMMENTS** — change-relative comments ("Fixed...", "Previously...", "Refactored...") that describe diffs instead of current code
- **INCONSISTENCY** — mixed comment styles, stale references, naming issues
- **DEAD DOCS** — stale comments, old TODOs, commented-out code

For readability gaps, include the specific file, line range, and what's missing. These are the highest priority — the hot path narrative must be seamless.

Present the list and ask: "Want me to fix these? I'll start with hot path readability gaps, then work through the rest."

Do NOT fix anything without explicit approval.
