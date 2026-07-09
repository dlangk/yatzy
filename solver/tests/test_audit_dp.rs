//! Hot-path audit tests: full-DP invariants that need no data files.
//!
//! Motivated by the 2026-07 fast-exp bug (theory/lab-reports/fast-exp-lse-bias.md):
//! a silent numerical error that survived because no test compared independent
//! computations of the same quantity. Every test here is designed to fail under
//! a systematic bias, not just an exact-identity break.
//!
//! Fast tests run on every `cargo test`. Heavy variants are `#[ignore]`d and
//! run via `just audit` (cargo test -- --include-ignored).

use yatzy::constants::*;
use yatzy::phase0_tables;
use yatzy::state_computation::compute_all_state_values_nocache;
use yatzy::types::YatzyContext;

const ALL_SCORED: usize = (1 << CATEGORY_COUNT) - 1;

/// Solve the full DP fresh (no cache, no save), optionally poisoning every
/// non-terminal slot with NaN first.
fn solve_fresh(theta: f32, max_policy: bool, poison: bool) -> Box<YatzyContext> {
    let mut ctx = YatzyContext::new_boxed();
    ctx.theta = theta;
    ctx.max_policy = max_policy;
    phase0_tables::precompute_lookup_tables(&mut ctx); // includes terminal init

    if poison {
        let sv = ctx.state_values.as_mut_slice();
        for scored in 0..(1usize << CATEGORY_COUNT) {
            if scored == ALL_SCORED {
                continue; // keep the terminal init from phase 0
            }
            for slot in 0..STATE_STRIDE {
                sv[scored * STATE_STRIDE + slot] = f32::NAN;
            }
        }
    }

    compute_all_state_values_nocache(&mut ctx);
    ctx
}

/// Game-start value: state_index(0, 0).
fn start_value(ctx: &YatzyContext) -> f32 {
    ctx.state_values.as_slice()[state_index(0, 0)]
}

// ── T1: NaN canary ─────────────────────────────────────────────────────────
//
// Poison everything, solve, then require:
//  - every REACHABLE (mask, up) slot is finite (no NaN leaked in), and
//  - every UNREACHABLE (mask, up) slot is still NaN (the solver never wrote
//    it — if it did, it may also have READ pruned states elsewhere).
// A reachability-too-tight bug, a padding misread, or an uninitialized read
// all surface as NaN in a reachable slot, because NEON min/max propagate NaN.

fn assert_canary(ctx: &YatzyContext, mode: &str) {
    let sv = ctx.state_values.as_slice();
    let mut reachable_checked = 0u64;
    let mut unreachable_checked = 0u64;
    for scored in 0..(1usize << CATEGORY_COUNT) {
        let upper_mask = scored & 0x3F;
        let is_terminal = scored == ALL_SCORED;
        for up in 0..=63usize {
            let v = sv[state_index(up, scored)];
            if ctx.reachable[upper_mask][up] || is_terminal {
                assert!(
                    v.is_finite(),
                    "[{mode}] reachable state (scored={scored:#x}, up={up}) is {v} — \
                     the solver read a poisoned (pruned/uninitialized) slot somewhere"
                );
                reachable_checked += 1;
            } else if !is_terminal {
                assert!(
                    v.is_nan(),
                    "[{mode}] unreachable state (scored={scored:#x}, up={up}) was \
                     written (= {v}) — writes outside the reachable set"
                );
                unreachable_checked += 1;
            }
        }
        // Topological padding: only masks where up=63 is REACHABLE have a
        // guarantee — their padding must equal slot 63 bitwise (it is read by
        // predecessors whose up+score lands past 63). For masks where 63 is
        // unreachable, results[63] is computed from pruned successors, so its
        // padding holds garbage (NaN under poisoning) BY CONSTRUCTION — and is
        // provably never read: any read would require a reachable predecessor
        // reaching up ≥ 64, which makes 63 reachable for this mask. If that
        // proof were wrong, the poison would propagate into a reachable slot
        // and the assertions above would catch it.
        if ctx.reachable[upper_mask][63] || is_terminal {
            let v63 = sv[state_index(63, scored)];
            for pad in 64..STATE_STRIDE {
                let v = sv[scored * STATE_STRIDE + pad];
                assert!(
                    v.to_bits() == v63.to_bits(),
                    "[{mode}] padding slot (scored={scored:#x}, pad={pad}) = {v} \
                     != slot-63 value {v63}"
                );
            }
        }
    }
    // Sanity on the census itself (≈1.43M reachable, rest unreachable).
    assert!(reachable_checked > 1_000_000, "[{mode}] census too small");
    assert!(unreachable_checked > 100_000, "[{mode}] census too small");
}

#[test]
fn test_nan_canary_ev() {
    let ctx = solve_fresh(0.0, false, true);
    assert_canary(&ctx, "EV");
    // And the answer must still be right after poisoning.
    assert!((start_value(&ctx) - 248.44).abs() < 0.01);
}

#[test]
#[ignore] // audit tier: ~2s for the three risk modes together
fn test_nan_canary_risk_modes() {
    let ctx = solve_fresh(0.12, false, true); // utility domain
    assert_canary(&ctx, "utility");
    let ctx = solve_fresh(0.20, false, true); // log/LSE domain
    assert_canary(&ctx, "LSE");
    let ctx = solve_fresh(-0.20, false, true); // LSE, minimize
    assert_canary(&ctx, "LSE-min");
    let ctx = solve_fresh(0.0, true, true); // max-policy
    assert_canary(&ctx, "max");
}

// ── T2: θ-boundary continuity + known answers ──────────────────────────────
//
// The utility (|θ| ≤ UTILITY_THETA_LIMIT) and LSE (|θ| > limit) solvers are
// two independent implementations of the same math. CE(θ) = L0/θ is smooth in
// θ, so values straddling the boundary must be close; a systematic bias in
// either implementation shows up as a jump. (The 2026-07 fast-exp bug was a
// ~26-point CE jump at exactly this seam.)

fn ce(theta: f32) -> f64 {
    let ctx = solve_fresh(theta, false, false);
    start_value(&ctx) as f64 / theta as f64
}

#[test]
fn test_theta_boundary_continuity() {
    let lim = UTILITY_THETA_LIMIT;
    let ce_utility = ce(lim - 0.001); // utility path
    let ce_lse = ce(lim + 0.001); // LSE path
    let jump = ce_lse - ce_utility;
    // Expected slope dCE/dθ ≈ 160 per unit θ here → ~0.32 over 0.002.
    assert!(
        (0.0..1.5).contains(&jump),
        "CE jump across the utility/LSE boundary: {ce_utility:.3} -> {ce_lse:.3} \
         (jump {jump:+.3}); the two solver implementations disagree"
    );

    // Same check on the negative side (minimize mode).
    let ce_utility_neg = ce(-(lim - 0.001));
    let ce_lse_neg = ce(-(lim + 0.001));
    let jump_neg = ce_utility_neg - ce_lse_neg; // CE nondecreasing in θ
    assert!(
        (0.0..1.5).contains(&jump_neg),
        "negative-side CE jump: {ce_lse_neg:.3} -> {ce_utility_neg:.3} (jump {jump_neg:+.3})"
    );
}

#[test]
fn test_ev_start_value_pin() {
    // Exact θ=0 expected value, independently replicated in the literature.
    let ctx = solve_fresh(0.0, false, false);
    let v = start_value(&ctx) as f64;
    assert!(
        (v - 248.4400677).abs() < 1e-3,
        "EV(start) = {v}, expected 248.4400677"
    );
}

// ── T13: value monotonicity in upper score (direction-matrixed) ────────────
//
// For a fixed category mask, a higher banked upper score weakly dominates
// (only the bonus indicator depends on it), so along reachable ups:
//   EV / max / θ>0 stored values: nondecreasing, increment ≤ bonus effect;
//   θ<0 STORED values (L = ln E[e^{θX}], θ<0): NONINCREASING (CE still
//   nondecreasing). A naive "always increasing" assertion is wrong — the
//   direction matrix is the invariant.

fn assert_upper_monotone(ctx: &YatzyContext, mode: &str, nonincreasing: bool, max_step: f64) {
    let sv = ctx.state_values.as_slice();
    for scored in 0..(1usize << CATEGORY_COUNT) {
        let upper_mask = scored & 0x3F;
        let ups: Vec<usize> = (0..=63usize)
            .filter(|&up| ctx.reachable[upper_mask][up] || scored == ALL_SCORED)
            .collect();
        for w in ups.windows(2) {
            let a = sv[state_index(w[0], scored)] as f64;
            let b = sv[state_index(w[1], scored)] as f64;
            let (lo, hi) = if nonincreasing { (b, a) } else { (a, b) };
            assert!(
                hi >= lo - 1e-3,
                "[{mode}] mask {scored:#x}: V(up={}) = {a} vs V(up={}) = {b} \
                 violates {} monotonicity",
                w[0],
                w[1],
                if nonincreasing {
                    "nonincreasing"
                } else {
                    "nondecreasing"
                }
            );
            assert!(
                (hi - lo) <= max_step * (w[1] - w[0]) as f64 + 1e-3,
                "[{mode}] mask {scored:#x}: step V(up={})={a} -> V(up={})={b} \
                 exceeds the bonus-only bound {max_step}/unit",
                w[0],
                w[1]
            );
        }
    }
}

#[test]
fn test_upper_score_monotonicity_ev() {
    let ctx = solve_fresh(0.0, false, false);
    // Only the 50-point bonus depends on up; one unit of up can change the
    // bonus probability by at most 1 → step ≤ 50.
    assert_upper_monotone(&ctx, "EV", false, 50.0);
}

#[test]
#[ignore] // audit tier: ~2s
fn test_upper_score_monotonicity_all_modes() {
    let ctx = solve_fresh(0.12, false, false); // utility → stored log
    assert_upper_monotone(&ctx, "utility", false, 0.12 * 50.0);
    let ctx = solve_fresh(0.20, false, false); // LSE
    assert_upper_monotone(&ctx, "LSE", false, 0.20 * 50.0);
    let ctx = solve_fresh(-0.20, false, false); // LSE minimize: stored L decreasing
    assert_upper_monotone(&ctx, "LSE-min", true, 0.20 * 50.0);
    let ctx = solve_fresh(0.0, true, false); // max policy
    assert_upper_monotone(&ctx, "max", false, 50.0);
}

// ── T8: DP determinism across thread counts ────────────────────────────────
//
// Each scored mask is solved by one thread with a deterministic inner order,
// so the output must be bit-identical regardless of rayon scheduling. A future
// reduction-order change would silently break table/oracle reproducibility.

#[test]
fn test_dp_deterministic_across_thread_counts() {
    let solve_with_threads = |n: usize| -> Vec<u32> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("pool");
        pool.install(|| {
            let ctx = solve_fresh(0.07, false, false); // utility path
            ctx.state_values.as_slice()[..NUM_STATES]
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
    };
    let single = solve_with_threads(1);
    let multi = solve_with_threads(8);
    assert_eq!(
        single, multi,
        "DP output differs between 1 and 8 threads — scheduling-dependent \
         floating-point order"
    );
}

// ── T3: independent f64 mini-solver differential ───────────────────────────
//
// A naive, definition-based solver sharing ZERO code with production: f64
// arithmetic, reroll transitions by exhaustive 6^n enumeration (no CSR), an
// explicit min(up+scr, 63) successor (no topological padding), scalar
// probability sums (no SIMD), and MGF-domain risk values (no LSE trick, no
// fast exp). Any systematic bias in scoring, keep-table, multinomial
// probabilities, the successor cap, padding reads, or mode dispatch shows up
// as a differential at the compared levels.

mod mini {
    use std::collections::HashMap;
    use yatzy::constants::*;
    use yatzy::game_mechanics::calculate_category_score;

    #[derive(Clone, Copy, PartialEq)]
    pub enum Mode {
        Ev,
        Mgf(f64), // θ ≠ 0: values are E[e^{θ·(remaining+bonus)}]
        Max,
    }

    /// All 252 sorted 5-dice multisets, in lexicographic order (independent
    /// enumeration; only the SET matters, order is internal to this module).
    pub fn all_multisets() -> Vec<[i32; 5]> {
        let mut v = Vec::with_capacity(252);
        for a in 1..=6 {
            for b in a..=6 {
                for c in b..=6 {
                    for d in c..=6 {
                        for e in d..=6 {
                            v.push([a, b, c, d, e]);
                        }
                    }
                }
            }
        }
        v
    }

    fn ds_of(sorted: &[i32; 5], sets: &[[i32; 5]]) -> usize {
        sets.iter().position(|s| s == sorted).expect("multiset")
    }

    /// P(initial roll = multiset), by exhaustive 6^5 enumeration.
    pub fn roll5_probs(sets: &[[i32; 5]]) -> Vec<f64> {
        let mut p = vec![0.0f64; 252];
        for code in 0..7776usize {
            let mut c = code;
            let mut dice = [0i32; 5];
            for d in dice.iter_mut() {
                *d = (c % 6) as i32 + 1;
                c /= 6;
            }
            dice.sort_unstable();
            p[ds_of(&dice, sets)] += 1.0 / 7776.0;
        }
        p
    }

    /// Transition lists: for every (dice set, reroll mask 1..=31), the
    /// (successor ds, probability) pairs by exhaustive 6^n enumeration.
    pub fn transitions(sets: &[[i32; 5]]) -> Vec<Vec<Vec<(usize, f64)>>> {
        let mut out = vec![vec![Vec::new(); 32]; 252];
        for (ds, dice) in sets.iter().enumerate() {
            for mask in 1..32usize {
                let kept: Vec<i32> = (0..5)
                    .filter(|i| mask & (1 << i) == 0)
                    .map(|i| dice[i])
                    .collect();
                let n = 5 - kept.len();
                let total = 6usize.pow(n as u32);
                let mut acc: HashMap<usize, f64> = HashMap::new();
                for code in 0..total {
                    let mut c = code;
                    let mut full = [0i32; 5];
                    full[..kept.len()].copy_from_slice(&kept);
                    for slot in full.iter_mut().skip(kept.len()) {
                        *slot = (c % 6) as i32 + 1;
                        c /= 6;
                    }
                    full.sort_unstable();
                    *acc.entry(ds_of(&full, sets)).or_insert(0.0) += 1.0 / total as f64;
                }
                out[ds][mask] = acc.into_iter().collect();
            }
        }
        out
    }

    pub struct MiniSolver {
        pub sets: Vec<[i32; 5]>,
        pub p5: Vec<f64>,
        pub trans: Vec<Vec<Vec<(usize, f64)>>>,
        /// values[(scored, up)] in the mode's own domain
        pub values: HashMap<(usize, usize), f64>,
        pub mode: Mode,
    }

    impl MiniSolver {
        pub fn new(mode: Mode) -> Self {
            let sets = all_multisets();
            let p5 = roll5_probs(&sets);
            let trans = transitions(&sets);
            let mut values = HashMap::new();
            // Terminal level: bonus in this mode's domain.
            let all = (1usize << CATEGORY_COUNT) - 1;
            for up in 0..=63usize {
                let bonus = if up >= 63 { 50.0f64 } else { 0.0 };
                let v = match mode {
                    Mode::Ev | Mode::Max => bonus,
                    Mode::Mgf(theta) => (theta * bonus).exp(),
                };
                values.insert((all, up), v);
            }
            MiniSolver {
                sets,
                p5,
                trans,
                values,
                mode,
            }
        }

        fn minimize(&self) -> bool {
            matches!(self.mode, Mode::Mgf(t) if t < 0.0)
        }

        fn better(&self, a: f64, b: f64) -> f64 {
            if self.minimize() {
                a.min(b)
            } else {
                a.max(b)
            }
        }

        /// Value of scoring category c with dice, from state (scored, up).
        fn score_value(&self, scored: usize, up: usize, dice: &[i32; 5], c: usize) -> f64 {
            let scr = calculate_category_score(dice, c) as f64;
            let new_scored = scored | (1 << c);
            let new_up = if c < 6 {
                (up + scr as usize).min(63)
            } else {
                up
            };
            let succ = self.values[&(new_scored, new_up)];
            match self.mode {
                Mode::Ev | Mode::Max => scr + succ,
                Mode::Mgf(theta) => (theta * scr).exp() * succ,
            }
        }

        /// Chance node: expectation (or max) over a transition list.
        fn chance(&self, list: &[(usize, f64)], v: &[f64; 252]) -> f64 {
            match self.mode {
                Mode::Max => list
                    .iter()
                    .map(|&(ds, _)| v[ds])
                    .fold(f64::NEG_INFINITY, f64::max),
                _ => list.iter().map(|&(ds, p)| p * v[ds]).sum(),
            }
        }

        /// Solve one widget: value of state (scored, up) before the initial
        /// roll, given all successor-level values are present.
        pub fn widget(&self, scored: usize, up: usize) -> f64 {
            // V0: best category choice per dice set.
            let mut v0 = [0.0f64; 252];
            for (ds, dice) in self.sets.iter().enumerate() {
                let mut best = if self.minimize() {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                };
                for c in 0..CATEGORY_COUNT {
                    if scored & (1 << c) == 0 {
                        best = self.better(best, self.score_value(scored, up, dice, c));
                    }
                }
                v0[ds] = best;
            }
            // V1, V2: reroll decisions (keep-all = previous value).
            let mut prev = v0;
            for _ in 0..2 {
                let mut cur = [0.0f64; 252];
                for ds in 0..252 {
                    let mut best = prev[ds]; // mask 0: keep all
                    for mask in 1..32usize {
                        best = self.better(best, self.chance(&self.trans[ds][mask], &prev));
                    }
                    cur[ds] = best;
                }
                prev = cur;
            }
            // Group 1: expectation (or max) over the initial roll.
            match self.mode {
                Mode::Max => prev.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                _ => (0..252).map(|ds| self.p5[ds] * prev[ds]).sum(),
            }
        }

        /// Solve and store every state at the given level (masks with
        /// `level` categories scored), for the given reachable ups.
        pub fn solve_level(&mut self, level: u32, reachable: &dyn Fn(usize, usize) -> bool) {
            let masks: Vec<usize> = (0..(1usize << CATEGORY_COUNT))
                .filter(|m| m.count_ones() == level)
                .collect();
            for scored in masks {
                for up in 0..=63usize {
                    if reachable(scored, up) {
                        let v = self.widget(scored, up);
                        self.values.insert((scored, up), v);
                    }
                }
            }
        }
    }
}

/// Compare the production table against the mini-solver at the given levels.
/// `theta == 0.0 && !max` = EV; `theta != 0` = risk (utility or LSE depending
/// on |θ| vs UTILITY_THETA_LIMIT — both are compared in log domain).
fn run_mini_differential(theta: f32, max_policy: bool, levels: &[u32], samples_per_mask: usize) {
    let ctx = solve_fresh(theta, max_policy, false);
    let sv = ctx.state_values.as_slice();

    let mode = if max_policy {
        mini::Mode::Max
    } else if theta == 0.0 {
        mini::Mode::Ev
    } else {
        mini::Mode::Mgf(theta as f64)
    };
    let mut solver = mini::MiniSolver::new(mode);

    // The mini-solver needs successor levels fully solved. Solve from the
    // deepest requested level up to 14, then compare at the requested levels.
    let deepest = *levels.iter().min().unwrap();
    let reach = |scored: usize, up: usize| ctx.reachable[scored & 0x3F][up];
    for level in (deepest..=14).rev() {
        solver.solve_level(level, &reach);
    }

    let mut compared = 0usize;
    for &level in levels {
        let masks: Vec<usize> = (0..(1usize << CATEGORY_COUNT))
            .filter(|m| m.count_ones() == level)
            .collect();
        for scored in masks {
            let ups: Vec<usize> = (0..=63usize).filter(|&up| reach(scored, up)).collect();
            // Sample evenly (always include first and last = boundary ups).
            let take: Vec<usize> = if ups.len() <= samples_per_mask {
                ups
            } else {
                let mut t: Vec<usize> = (0..samples_per_mask)
                    .map(|i| ups[i * (ups.len() - 1) / (samples_per_mask - 1)])
                    .collect();
                t.dedup();
                t
            };
            for up in take {
                let naive = solver.values[&(scored, up)];
                let prod = sv[state_index(up, scored)] as f64;
                // Production risk tables store L = ln(MGF); compare in log.
                let (naive_cmp, what) = match mode {
                    mini::Mode::Mgf(_) => (naive.ln(), "log-domain"),
                    _ => (naive, "value"),
                };
                let tol = match mode {
                    mini::Mode::Ev => 5e-4,
                    mini::Mode::Max => 5e-4,
                    // fast-exp ≤2e-5/node × 3 chance nodes + f32 rounding
                    mini::Mode::Mgf(_) => 2e-3,
                };
                assert!(
                    (prod - naive_cmp).abs() < tol,
                    "θ={theta} max={max_policy} state (scored={scored:#x}, up={up}) \
                     level {level}: production {what} {prod} vs independent {naive_cmp} \
                     (diff {:.2e})",
                    (prod - naive_cmp).abs()
                );
                compared += 1;
            }
        }
    }
    assert!(compared > 20, "differential compared too few states");
}

#[test]
fn test_mini_solver_differential_ev_level14() {
    // 15 masks × ≤4 sampled ups (incl. boundary), terminal successors only.
    run_mini_differential(0.0, false, &[14], 4);
}

#[test]
#[ignore] // audit tier: ~1-2 min (levels 13-14, all four modes)
fn test_mini_solver_differential_all_modes() {
    run_mini_differential(0.0, false, &[13, 14], 6);
    run_mini_differential(0.12, false, &[13, 14], 6); // utility path
    run_mini_differential(0.20, false, &[13, 14], 6); // LSE path
    run_mini_differential(-0.20, false, &[13, 14], 6); // LSE minimize
    run_mini_differential(0.0, true, &[13, 14], 6); // max policy
}

// ── T11: ln-MGF adjudicator (exact PMF vs stored L₀) ───────────────────────
//
// The forward density evolution produces the exact score PMF of the policy in
// a strategy table. ln Σ p(x)·e^{θx} computed from that PMF in f64 is ground
// truth the solver cannot argue with: the stored L₀ must match. This is the
// exact check that adjudicated the fast-exp bug (stored 45.58 vs true 49.48
// at θ=0.16). Data-gated: needs outputs/density/*.json + strategy tables.

fn repo_file(rel: &str) -> Option<String> {
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    let _ = std::env::set_current_dir(&base_path);
    for candidate in [rel.to_string(), format!("../{rel}")] {
        if std::path::Path::new(&candidate).exists() {
            return Some(candidate);
        }
    }
    if std::env::var("YATZY_REQUIRE_DATA").is_ok() {
        panic!("YATZY_REQUIRE_DATA is set but {rel} is missing");
    }
    eprintln!("SKIPPED (no data): {rel} not found");
    None
}

#[test]
#[ignore] // audit tier: data-gated, <1s
fn test_stored_l0_matches_exact_pmf_ln_mgf() {
    // Spans both solver regimes, both signs, and the seam.
    let thetas: [f32; 8] = [0.05, 0.10, 0.15, 0.16, 0.20, 0.30, -0.10, -0.20];
    let mut checked = 0;
    for &theta in &thetas {
        let density_rel = format!("outputs/density/density_{theta}.json");
        let table_rel = format!("data/strategy_tables/all_states_theta_{theta:.3}.bin");
        let (Some(density_path), Some(table_path)) =
            (repo_file(&density_rel), repo_file(&table_rel))
        else {
            continue;
        };

        // Stored L0: first f32 after the 16-byte header (state_index(0,0)=0).
        let bytes = std::fs::read(&table_path).expect("read table");
        let l0 = f32::from_le_bytes(bytes[16..20].try_into().unwrap()) as f64;

        // Ground truth from the exact PMF, f64 log-sum-exp.
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&density_path).expect("read density"))
                .expect("parse density JSON");
        let pmf = doc["pmf"].as_array().expect("pmf array");
        let t = theta as f64;
        let m = pmf
            .iter()
            .filter(|p| p[1].as_f64().unwrap() > 0.0)
            .map(|p| t * p[0].as_f64().unwrap())
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = pmf
            .iter()
            .filter(|p| p[1].as_f64().unwrap() > 0.0)
            .map(|p| p[1].as_f64().unwrap() * (t * p[0].as_f64().unwrap() - m).exp())
            .sum();
        let ln_mgf = m + sum.ln();

        // Tolerance: fast-exp ≤2e-5/node over a full game (~1e-3) + f32
        // keep-table drift in the density (~1e-6) + f32 storage rounding.
        assert!(
            (l0 - ln_mgf).abs() < 5e-3,
            "θ={theta}: stored L0 = {l0:.4} but exact-PMF ln MGF = {ln_mgf:.4} \
             (diff {:.2e}) — the solver misvalues its own policy",
            (l0 - ln_mgf).abs()
        );
        checked += 1;
    }
    if checked == 0 {
        eprintln!("SKIPPED: no theta had both density and table on disk");
    } else {
        println!("ln-MGF adjudicator: {checked} thetas verified");
    }
}

#[test]
#[ignore] // audit tier: ~5s (10 full solves)
fn test_ce_monotone_over_theta_grid() {
    // CE(θ) of the θ-optimal policy is nondecreasing in θ (for fixed X this
    // is standard; optimality preserves it). Crossing θ=0 it passes through
    // the EV. A biased solver on either side breaks the chain.
    let lim = UTILITY_THETA_LIMIT;
    let grid: Vec<f32> = vec![
        -1.0,
        -0.3,
        -(lim + 0.001),
        -(lim - 0.001),
        -0.05,
        0.05,
        lim - 0.001,
        lim + 0.001,
        0.3,
        1.0,
    ];
    let mut prev = f64::NEG_INFINITY;
    let mut prev_theta = f64::NEG_INFINITY;
    for &t in &grid {
        let c = ce(t);
        assert!(
            c >= prev - 0.05,
            "CE(θ) not monotone: CE({prev_theta}) = {prev:.3} > CE({t}) = {c:.3}"
        );
        // θ=0 midpoint: negative-side CE below EV, positive-side above.
        if t < 0.0 {
            assert!(c < 248.45, "CE({t}) = {c:.3} exceeds the θ=0 EV");
        } else {
            assert!(c > 248.43, "CE({t}) = {c:.3} below the θ=0 EV");
        }
        prev = c;
        prev_theta = t as f64;
    }
}
