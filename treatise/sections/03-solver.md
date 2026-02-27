:::section{#solver}

## The Solver

Knowing the mathematics is not enough. The state space has 1.43 million positions,
each requiring evaluation of hundreds of dice-keep-category combinations. A naive
implementation in C took minutes. The production solver in Rust, after a long
sequence of architectural and micro-optimisations, completes the same computation
in 1.1 seconds --a factor of roughly 1,000.

### The Widget Solver

The core of the engine is the
::concept[widget solver]{widget-solver}, a
self-contained unit that evaluates the
::concept[Bellman equation]{bellman-equation}
for a single state. Given a state (upper score, scored categories) and the
precomputed strategy table for all future states, it computes the expected value
of optimal play by iterating over dice outcomes, keep choices, and category
assignments. The widget is called once per reachable state during backward
induction, and its performance dominates total solve time.

:::html
<div class="chart-container" id="chart-widget-interactive">
  <div class="chart-controls">
    <select id="widget-scenario-select" class="chart-select">
      <option value="0">Scenario: Keep triple or chase straight?</option>
    </select>
    <button class="chart-btn" id="widget-solver-btn">Let Solver Play</button>
    <button class="chart-btn" id="widget-reset-btn">Reset</button>
  </div>
  <div id="widget-flow"></div>
  <div id="widget-fan-panel" class="math"></div>
</div>
:::

### Memory Layout and Cache Efficiency

Modern processors do not access memory uniformly. They rely on hierarchical
caches (L1, L2, L3) that reward sequential, predictable access patterns and
punish random lookups. The solver's memory layout is designed around this
reality. Strategy tables use a stride of 128 instead of the natural 64, padding
the upper-score dimension to a power of two. This wastes half the storage (16 MB
instead of 8 MB per table) but eliminates a conditional branch on every state
access --a trade that pays for itself many times over in the inner loop.

:::html
<div class="chart-container" id="chart-backward-wave">
  <div class="chart-controls">
    <button class="chart-btn" id="wave-play-btn">&#9654; Play</button>
    <button class="chart-btn" id="wave-reset-btn">Reset</button>
    <input type="range" id="wave-scrub" class="chart-slider" min="0" max="15" value="15">
    <span class="scenario-label" id="wave-label">Layer 15 — terminal states</span>
  </div>
  <div id="chart-backward-wave-svg"></div>
</div>
:::

### Parallelism and SIMD

The topological structure of the state space --the fact that layer <var>k</var>
depends only on layer <var>k</var>+1 --enables embarrassingly parallel
evaluation within each layer. The solver uses Rayon to distribute states across
all available cores, achieving near-linear scaling on 8-core Apple Silicon. Within
each core, ARM NEON SIMD instructions process four floating-point values
simultaneously: fused multiply-add for expected values, parallel max for the
Bellman optimisation, and a custom fast-exp polynomial for the risk-sensitive
log-sum-exp variant.

### The Optimisation Timeline

The path from the first working prototype to the current solver spans dozens of
targeted optimisations. The largest single gains came from eliminating memory
allocations in the inner loop, switching from f64 to f32 arithmetic, and
introducing NEON intrinsics for the keep-evaluation kernel. Each change was
benchmarked in isolation using a strict regression test: if the benchmark
slowed down, the change was reverted.

:::html
<div class="chart-container" id="chart-optimization-timeline">
  <div id="chart-optimization-timeline-svg"></div>
</div>
:::

:::insight
**Key insight:** The f32-vs-f64 decision was not obvious. Across all
1.43 million reachable states, only 22 show a different optimal action under f32
than f64 --and the expected-value difference at the initial state is less
than 0.001 points. The 2× bandwidth savings and native NEON f32 throughput
made the switch overwhelmingly worthwhile.
:::

:::math

The widget solver evaluates the Bellman equation in three phases. Phase 1
computes the immediate score for each (dice-outcome, category) pair. Phase 2
evaluates the best keep at each reroll stage using ping-pong buffers. Phase 3
takes the expectation over all 252 dice outcomes, weighted by their multinomial
multiplicities.

:::equation
<var>V</var>(<var>u</var>, <var>S</var>) =
&sum;<sub><var>d</var></sub> <var>w</var>(<var>d</var>) &middot;
max<sub><var>a</var></sub> <var>Q</var>(<var>u</var>, <var>S</var>, <var>d</var>, <var>a</var>)
&nbsp;/&nbsp; 7776
:::

The risk-sensitive variant replaces the linear expectation with an exponential
utility transform parameterised by &theta;. For |&theta;| &le; 0.15, a direct
utility-domain formulation avoids logarithms entirely and runs at the same speed
as the EV solver. For larger |&theta;|, a numerically stable log-sum-exp
formulation is used, at roughly 2.5× the cost.

:::

:::code

The NEON kernel for keep evaluation processes 4 state values simultaneously.
The critical function `neon_fma_max_f32x4` performs a fused
multiply-add followed by a lane-wise maximum, avoiding a round-trip to
scalar registers.

```rust
// simd.rs — NEON FMA+max kernel
#[inline(always)]
pub unsafe fn neon_fma_max_f32x4(
    acc: float32x4_t,
    a: float32x4_t,
    b: float32x4_t,
    prob: float32x4_t,
) -> float32x4_t {
    let product = vmulq_f32(a, prob);
    let sum = vaddq_f32(b, product);
    vmaxq_f32(acc, sum)
}
```

Parallelism across states uses Rayon's `par_iter` over the scored-mask
dimension. Each thread processes all 64 upper-score values for a given mask,
ensuring contiguous memory access within the 128-stride layout. The thread count
is pinned to 8 on Apple Silicon (matching the performance core count).

```rust
// state_computation.rs
scored_masks.par_iter().for_each(|&scored| {
    let mut buffers = BatchedBuffers::new();
    for upper in 0..64 {
        let si = state_index(upper, scored);
        if !reachable[si] { continue; }
        let ev = solve_widget(&ctx, &buffers, upper, scored);
        state_values[si] = ev;
    }
});
```

:::

:::
