//! Export state-space EV heatmap data for the blog.
//!
//! Loads `all_states.bin`, computes the average EV for each
//! (num_scored_categories, upper_score) pair across all reachable category masks.
//! Outputs `profiler/data/state_heatmap.json`.

use yatzy::constants::*;
use yatzy::phase0_tables;
use yatzy::storage::load_all_state_values;
use yatzy::types::YatzyContext;

fn main() {
    let _base_path = yatzy::env_config::init_base_path();

    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    if !load_all_state_values(&mut ctx, "data/strategy_tables/all_states.bin") {
        eprintln!("Failed to load state values. Run yatzy-precompute first.");
        std::process::exit(1);
    }

    let sv = ctx.state_values.as_slice();
    let total_masks = 1u32 << CATEGORY_COUNT; // 32768

    // For each (num_scored, upper_score), accumulate EV sum and count
    let mut sum = vec![vec![0.0f64; 64]; 16]; // [num_scored][upper]
    let mut count = vec![vec![0u32; 64]; 16];

    for scored in 0..total_masks as usize {
        let num_scored = (scored as u32).count_ones() as usize;
        for upper in 0..64usize {
            let idx = state_index(upper, scored);
            let ev = sv[idx];
            // Skip unreachable states (EV == 0 for empty terminal states is valid,
            // but states with all scored + upper < 63 have EV 0 which is fine)
            if ev != 0.0 || (num_scored == CATEGORY_COUNT && upper >= 63) {
                sum[num_scored][upper] += ev as f64;
                count[num_scored][upper] += 1;
            }
        }
    }

    // Build JSON: array of {num_scored, upper_score, ev, count}
    let mut rows = Vec::new();
    for ns in 0..=CATEGORY_COUNT {
        for upper in 0..64usize {
            let c = count[ns][upper];
            if c > 0 {
                let avg_ev = sum[ns][upper] / c as f64;
                rows.push(serde_json::json!({
                    "num_scored": ns,
                    "upper_score": upper,
                    "ev": (avg_ev * 10.0).round() / 10.0,
                    "count": c,
                }));
            }
        }
    }

    let output = serde_json::json!({ "states": rows });
    let json = serde_json::to_string(&output).unwrap();

    let out_path = "profiler/data/state_heatmap.json";
    std::fs::create_dir_all("profiler/data").unwrap();
    std::fs::write(out_path, json).unwrap();
    println!("Wrote {} rows to {}", rows.len(), out_path);
}
