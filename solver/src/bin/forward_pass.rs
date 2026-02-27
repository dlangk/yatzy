//! yatzy-forward-pass: Exact state visitation probabilities under optimal play.
//!
//! Produces three D3.js-ready JSON files for visualization:
//! - graph_race_to_63.json — upper-score Sankey
//! - graph_category_sankey.json — category selection probabilities per turn
//! - graph_ev_funnel.json — EV distribution per turn

use std::fs;
use std::time::Instant;

use yatzy::constants::CATEGORY_COUNT;
use yatzy::forward_pass::forward_pass;
use yatzy::phase0_tables;
use yatzy::storage::{load_all_state_values, load_oracle, state_file_path, ORACLE_FILE_PATH};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut output_dir = "outputs".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                i += 1;
                output_dir = args[i].clone();
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let _base = yatzy::env_config::init_base_path();
    let _threads = yatzy::env_config::init_rayon_threads();

    println!("=== yatzy-forward-pass ===");
    println!("Output: {}", output_dir);

    // Build phase0 tables
    let mut ctx = yatzy::types::YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    // Load θ=0 strategy table (needed for EV Funnel binning)
    let state_file = state_file_path(0.0);
    if !load_all_state_values(&mut ctx, &state_file) {
        eprintln!("Failed to load strategy table: {}", state_file);
        eprintln!("Run `just precompute` first.");
        std::process::exit(1);
    }

    // Load oracle
    let t_orc = Instant::now();
    let oracle = match load_oracle(ORACLE_FILE_PATH) {
        Some(o) => {
            println!("Oracle loaded in {:.2}s", t_orc.elapsed().as_secs_f64());
            o
        }
        None => {
            eprintln!("Failed to load oracle. Run `yatzy-precompute --oracle` first.");
            std::process::exit(1);
        }
    };

    // Get state values slice for EV lookups
    let sv = ctx.state_values.as_slice();

    // Run forward pass
    let t0 = Instant::now();
    let result = forward_pass(&ctx, &oracle, sv);
    let elapsed = t0.elapsed().as_secs_f64();
    println!("Forward pass completed in {:.1}s", elapsed);

    // Create output directory
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Write Race to 63
    let path = format!("{}/graph_race_to_63.json", output_dir);
    let json = serde_json::to_string_pretty(&result.race_to_63).expect("JSON serialization failed");
    fs::write(&path, json).expect("Failed to write");
    println!(
        "Wrote {} ({} nodes, {} links)",
        path,
        result.race_to_63.nodes.len(),
        result.race_to_63.links.len()
    );

    // Write Category Sankey
    let path = format!("{}/graph_category_sankey.json", output_dir);
    let json =
        serde_json::to_string_pretty(&result.category_sankey).expect("JSON serialization failed");
    fs::write(&path, json).expect("Failed to write");
    println!("Wrote {} ({} entries)", path, result.category_sankey.len());

    // Write EV Funnel
    let path = format!("{}/graph_ev_funnel.json", output_dir);
    let json = serde_json::to_string_pretty(&result.ev_funnel).expect("JSON serialization failed");
    fs::write(&path, json).expect("Failed to write");
    println!("Wrote {} ({} entries)", path, result.ev_funnel.len());

    // Verification: check probability conservation
    let final_turn = CATEGORY_COUNT as u8;
    let final_mass: f64 = result
        .ev_funnel
        .iter()
        .filter(|e| e.turn == final_turn)
        .map(|e| e.mass)
        .sum();
    println!(
        "\nVerification: final turn mass = {:.10} (should be 1.0)",
        final_mass
    );

    for turn in 0..CATEGORY_COUNT as u8 {
        let turn_mass: f64 = result
            .category_sankey
            .iter()
            .filter(|e| e.turn == turn)
            .map(|e| e.mass)
            .sum();
        if (turn_mass - 1.0).abs() > 1e-6 {
            eprintln!(
                "WARNING: Category mass at turn {} = {:.10} (expected 1.0)",
                turn, turn_mass
            );
        }
    }

    println!("Done.");
}

fn print_usage() {
    println!(
        "yatzy-forward-pass: Exact state visitation probabilities under optimal play.

USAGE:
    yatzy-forward-pass [OPTIONS]

OPTIONS:
    --output <DIR>    Output directory [default: outputs]
    -h, --help        Print this help

REQUIRES:
    - Strategy table: data/strategy_tables/all_states.bin (run `just precompute`)
    - Oracle: data/strategy_tables/oracle.bin (run `yatzy-precompute --oracle`)

OUTPUT:
    outputs/graph_race_to_63.json        Upper-score Sankey
    outputs/graph_category_sankey.json   Category selection per turn
    outputs/graph_ev_funnel.json         EV distribution per turn"
    );
}
