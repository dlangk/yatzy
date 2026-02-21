---
paths:
  - "solver/src/widget_solver.rs"
  - "solver/src/batched_solver.rs"
  - "solver/src/simd.rs"
  - "solver/src/state_computation.rs"
  - "solver/src/simulation/lockstep.rs"
  - "solver/src/simulation/fast_prng.rs"
  - "solver/src/density/**/*.rs"
---

# Hot Path Rules — Performance Critical Code

These files contain the computational hot paths that have been heavily optimized with NEON SIMD intrinsics, cache-friendly memory layouts, and branchless algorithms. Changes here directly affect wall-clock performance.

YOU MUST follow these rules when editing hot-path code:

- Run `just bench-check` before AND after any change. Revert if any benchmark regresses.
- Do NOT extract helper functions if it adds dynamic dispatch or prevents inlining.
- Do NOT replace intentionally duplicated code (marked `// PERF: intentional`).
- Do NOT add heap allocations in inner loops.
- Do NOT change STATE_STRIDE (128) or state_index() layout without understanding the topological padding scheme.
- Prefer `#[inline]` or `#[inline(always)]` on small functions called from hot loops.
- Use `debug_assert!` for invariants (zero cost in release builds).
- All computation uses f32. Do NOT introduce f64 in hot paths.
- When in doubt, leave a `// PERF: <explanation>` comment rather than refactoring.

## Key Performance Numbers (M1 Max, 8 threads)

- Precompute θ=0: ~504ms (backward induction over 1.43M states)
- Lockstep simulation: ~593ms/100K games
- API evaluate: 2-9μs per call
