//! Statistical equivalence of the two Monte Carlo engines.
//!
//! The vertical engine (per-game, SmallRng) and the lockstep engine
//! (horizontal, SplitMix64) play the same deterministic EV-optimal policy;
//! only dice sampling differs. Both sample means must therefore agree with
//! the precomputed game-start EV within Monte Carlo error.
//!
//! Data-gated: skips when no strategy table is on disk (like test_precomputed).

use yatzy::constants::state_index;
use yatzy::phase0_tables;
use yatzy::simulation::lockstep::simulate_batch_lockstep;
use yatzy::simulation::simulate_batch;
use yatzy::storage::{file_exists, load_all_state_values};
use yatzy::types::YatzyContext;

const NUM_GAMES: usize = 100_000;

fn setup() -> Option<Box<YatzyContext>> {
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    let _ = std::env::set_current_dir(&base_path);

    let state_file = yatzy::storage::state_file_path(0.0);
    let path = if file_exists(&state_file) {
        state_file
    } else {
        let parent = format!("../{}", state_file);
        if !file_exists(&parent) {
            // Skip-visibility contract: `just audit` sets YATZY_REQUIRE_DATA=1
            // so a missing table fails loudly instead of skipping green.
            if std::env::var("YATZY_REQUIRE_DATA").is_ok() {
                panic!(
                    "YATZY_REQUIRE_DATA is set but {state_file} is missing — \
                     run `just precompute` first"
                );
            }
            eprintln!("SKIPPED (no data): {state_file} not found");
            return None;
        }
        parent
    };

    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    if !load_all_state_values(&mut ctx, &path) {
        return None;
    }
    Some(ctx)
}

fn assert_mean_matches_ev(engine: &str, mean: f64, std_dev: f64, ev: f64) {
    let se = std_dev / (NUM_GAMES as f64).sqrt();
    let z = (mean - ev) / se;
    assert!(
        z.abs() < 4.0,
        "{engine}: mean {mean:.2} deviates from EV {ev:.2} by {z:.1} standard errors"
    );
    assert!(
        (30.0..50.0).contains(&std_dev),
        "{engine}: std dev {std_dev:.1} outside plausible range"
    );
}

#[test]
fn test_lockstep_mean_matches_ev() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };
    let ev = ctx.state_values.as_slice()[state_index(0, 0)] as f64;
    let result = simulate_batch_lockstep(&ctx, NUM_GAMES, 42);
    assert_mean_matches_ev("lockstep", result.mean, result.std_dev, ev);
}

#[test]
fn test_vertical_mean_matches_ev() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };
    let ev = ctx.state_values.as_slice()[state_index(0, 0)] as f64;
    let result = simulate_batch(&ctx, NUM_GAMES, 42);
    assert_mean_matches_ev("vertical", result.mean, result.std_dev, ev);
}

#[test]
fn test_engines_agree_with_each_other() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };
    let lockstep = simulate_batch_lockstep(&ctx, NUM_GAMES, 42);
    let vertical = simulate_batch(&ctx, NUM_GAMES, 42);

    // Two-sample z-test on means: SE = σ·√(2/N)
    let pooled_std = (lockstep.std_dev + vertical.std_dev) / 2.0;
    let se = pooled_std * (2.0 / NUM_GAMES as f64).sqrt();
    let z = (lockstep.mean - vertical.mean) / se;
    assert!(
        z.abs() < 4.0,
        "engines disagree: lockstep {:.2} vs vertical {:.2} (z = {z:.1})",
        lockstep.mean,
        vertical.mean
    );

    let std_ratio = lockstep.std_dev / vertical.std_dev;
    assert!(
        (0.97..1.03).contains(&std_ratio),
        "std dev mismatch: lockstep {:.2} vs vertical {:.2}",
        lockstep.std_dev,
        vertical.std_dev
    );
}

/// T15 (audit tier): the 1M-game variant. SE shrinks √10× vs the 100k test,
/// so the mean cross-check tightens from ~0.2% to ~0.06% sensitivity —
/// enough to catch sub-point systematic bias between the engines.
#[test]
#[ignore]
fn test_engines_agree_1m_games() {
    const N: usize = 1_000_000;
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };
    let ev = ctx.state_values.as_slice()[state_index(0, 0)] as f64;
    let lockstep = simulate_batch_lockstep(&ctx, N, 42);
    let vertical = simulate_batch(&ctx, N, 42);

    for (name, r) in [("lockstep", &lockstep), ("vertical", &vertical)] {
        let se = r.std_dev / (N as f64).sqrt();
        let z = (r.mean - ev) / se;
        assert!(
            z.abs() < 4.5,
            "{name} (1M): mean {:.3} deviates from EV {ev:.3} by {z:.1} SE",
            r.mean
        );
    }
    let pooled_std = (lockstep.std_dev + vertical.std_dev) / 2.0;
    let se = pooled_std * (2.0 / N as f64).sqrt();
    let z = (lockstep.mean - vertical.mean) / se;
    assert!(
        z.abs() < 4.5,
        "engines disagree at 1M games: lockstep {:.3} vs vertical {:.3} (z = {z:.1})",
        lockstep.mean,
        vertical.mean
    );
}
