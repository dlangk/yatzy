//! Theta sweep infrastructure: inventory scanning, grid resolution, and strategy table management.

use std::path::Path;
use std::time::Instant;

use crate::constants::NUM_STATES;
use crate::phase0_tables;
use crate::state_computation::compute_all_state_values;
use crate::storage::state_file_path;
use crate::types::{StateValues, YatzyContext};

use super::raw_storage::{ScoresHeader, SCORES_MAGIC, SCORES_VERSION};

/// Inventory entry: metadata from an existing scores.bin file.
pub struct InventoryEntry {
    pub theta: f32,
    pub num_games: u32,
    pub seed: u64,
    pub path: String,
}

/// Format theta to directory name matching Python's fmt_theta_dir.
/// 0.0 → "0", 0.01 → "0.01", -3.0 → "-3", 1.5 → "1.5"
pub fn format_theta_dir(theta: f32) -> String {
    if theta == 0.0 {
        return "0".to_string();
    }
    // Use {:g}-like formatting: strip trailing zeros
    let s = format!("{}", theta);
    s
}

/// Full path to scores.bin for a given theta.
pub fn theta_scores_path(theta: f32) -> String {
    format!(
        "data/simulations/theta/theta_{}/scores.bin",
        format_theta_dir(theta)
    )
}

/// Scan theta directories, read 32-byte headers from scores.bin files.
pub fn scan_inventory() -> Vec<InventoryEntry> {
    let base = Path::new("data/simulations/theta");
    if !base.is_dir() {
        return Vec::new();
    }

    let mut entries = Vec::new();
    let mut dirs: Vec<_> = std::fs::read_dir(base)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().map(|ft| ft.is_dir()).unwrap_or(false)
                && e.file_name().to_string_lossy().starts_with("theta_")
        })
        .collect();
    dirs.sort_by_key(|e| e.file_name().to_string_lossy().to_string());

    for entry in dirs {
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let theta_str = &dir_name["theta_".len()..];
        let theta: f32 = match theta_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        let scores_path = entry.path().join("scores.bin");
        if !scores_path.exists() {
            continue;
        }

        if let Some(header) = read_scores_header(scores_path.to_str().unwrap()) {
            entries.push(InventoryEntry {
                theta,
                num_games: header.num_games,
                seed: header.seed,
                path: scores_path.to_string_lossy().to_string(),
            });
        }
    }

    entries
}

/// Read just the 32-byte header from a scores.bin file.
fn read_scores_header(path: &str) -> Option<ScoresHeader> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; std::mem::size_of::<ScoresHeader>()];
    f.read_exact(&mut buf).ok()?;
    let header = unsafe { *(buf.as_ptr() as *const ScoresHeader) };
    if header.magic != SCORES_MAGIC || header.version != SCORES_VERSION {
        return None;
    }
    Some(header)
}

/// Ensure a θ strategy table exists, precomputing if necessary.
/// Returns true on success.
pub fn ensure_strategy_table(ctx: &mut Box<YatzyContext>, theta: f32) -> bool {
    let file = state_file_path(theta);
    if Path::new(&file).exists() {
        return true;
    }

    println!("    Precomputing strategy table for θ={:.4}...", theta);
    let t0 = Instant::now();
    ctx.theta = theta;

    // Reset state values to owned for computation
    ctx.state_values = StateValues::Owned(vec![0.0f32; NUM_STATES]);

    // Re-initialize terminal states for this theta
    phase0_tables::initialize_final_states(ctx);

    // Run DP
    compute_all_state_values(ctx);

    println!(
        "    Precomputed θ={:.4} in {:.1}s",
        theta,
        t0.elapsed().as_secs_f64()
    );
    true
}

/// Parse a named grid. Returns None if grid name is unknown.
pub fn resolve_grid(grid_name: &str) -> Option<Vec<f32>> {
    match grid_name {
        "all" => Some(vec![
            -3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.3, -0.2, -0.15, -0.1, -0.07, -0.05, -0.04,
            -0.03, -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05,
            0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
        ]),
        "dense" => Some(vec![
            -0.1, -0.07, -0.05, -0.04, -0.03, -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01,
            0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1,
        ]),
        "sparse" => Some(vec![
            -3.0, -2.0, -1.5, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5,
            2.0, 3.0,
        ]),
        "frontier" => {
            // 89 thetas: dense near origin, coarser in active zone, sparse shoulders
            // Core: [-0.05, +0.05] at 0.002 step = 51 values
            // Active: [-0.20, -0.06] and [+0.06, +0.20] at 0.01 step = 30 values
            // Shoulder: ±0.30, ±0.50, ±0.75, ±1.00 = 8 values
            let mut v = Vec::with_capacity(89);
            // Shoulder (negative)
            for &t in &[-1.0, -0.75, -0.5, -0.3] {
                v.push(t);
            }
            // Active (negative): -0.20 to -0.06
            for i in (-20..=-6).step_by(1) {
                let t = i as f32 / 100.0;
                v.push((t * 10000.0).round() / 10000.0);
            }
            // Core: -0.050 to +0.050 at 0.002
            for i in -25..=25 {
                let t = i as f32 * 0.002;
                v.push((t * 10000.0).round() / 10000.0);
            }
            // Active (positive): 0.06 to 0.20
            for i in (6..=20).step_by(1) {
                let t = i as f32 / 100.0;
                v.push((t * 10000.0).round() / 10000.0);
            }
            // Shoulder (positive)
            for &t in &[0.3, 0.5, 0.75, 1.0] {
                v.push(t);
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.dedup();
            Some(v)
        }
        _ => None,
    }
}

/// Generate an arithmetic range of theta values with proper float rounding.
pub fn range_grid(lo: f32, hi: f32, step: f32) -> Vec<f32> {
    let mut values = Vec::new();
    let n_steps = ((hi - lo) / step).round() as i32;
    for i in 0..=n_steps {
        let v = lo + i as f32 * step;
        // Round to avoid float drift (e.g. 0.1 + 0.1 + 0.1 != 0.3)
        let rounded = (v * 10000.0).round() / 10000.0;
        values.push(rounded);
    }
    values
}

/// Check if two f32 theta values are "equal" for inventory matching purposes.
/// Uses a small epsilon to handle float formatting round-trips.
pub fn theta_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}
