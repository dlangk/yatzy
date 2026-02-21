# Lab Reports

Performance experiments with measured results on Apple M1 Max hardware.

| File | Lines | Description |
|------|------:|-------------|
| `batched-soa-solver.md` | 260 | Batched struct-of-arrays solver design and benchmarks |
| `cache-hierarchy-targeting.md` | 308 | L1/L2/SLC targeting: state layout, working set, false sharing |
| `density-condensation.md` | 354 | Forward density evolution: prob-array propagation, 126× oracle speedup |
| `hardware-and-hot-path.md` | 375 | Full optimization history, Firestorm microarchitecture reference, gather bottleneck |
| `neon-intrinsics.md` | 244 | NEON SIMD kernels: FMA, argmax, fast exp, measured speedups |
| `optimization-log.md` | 288 | Chronological optimization log with before/after timings |
| `oracle-policy.md` | 302 | Precomputed argmax oracle: build, simulate, density at 5.6M games/s |
| `rosetta-distillation.md` | 200 | Knowledge distillation from solver to compact policy |
| `scaling-laws.md` | 146 | How solver performance scales with state space and θ grid size |
| `semantic-reroll-actions.md` | 148 | Semantic reroll action encoding and its effect on surrogate training |
| `strategy-guide-prompt.md` | 166 | LLM-generated strategy guide evaluation against solver |
