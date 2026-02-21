//! File I/O helpers for scenario pipeline.

use crate::storage::load_state_values_standalone;
use crate::types::StateValues;

/// Scan data/strategy_tables/ for all .bin files and parse θ from filenames.
pub fn discover_theta_files() -> Vec<(f32, String)> {
    let dir = "data/strategy_tables";
    let mut entries: Vec<(f32, String)> = Vec::new();

    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            eprintln!("Cannot read {}: {}", dir, e);
            return entries;
        }
    };

    for entry in read_dir.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".bin") {
            continue;
        }
        if name == "all_states.bin" {
            entries.push((0.0, format!("{}/{}", dir, name)));
        } else if let Some(rest) = name.strip_prefix("all_states_theta_") {
            if let Some(theta_str) = rest.strip_suffix(".bin") {
                if let Ok(theta) = theta_str.parse::<f32>() {
                    entries.push((theta, format!("{}/{}", dir, name)));
                }
            }
        }
    }

    entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    entries
}

/// Loaded θ table.
pub struct ThetaEntry {
    pub theta: f32,
    pub sv: StateValues,
}

/// Load theta entries from discovered files, optionally filtering by range.
pub fn load_theta_entries(
    theta_files: &[(f32, String)],
    theta_min: Option<f32>,
    theta_max: Option<f32>,
) -> Vec<ThetaEntry> {
    let filtered: Vec<&(f32, String)> = theta_files
        .iter()
        .filter(|(theta, _)| {
            if let Some(tmin) = theta_min {
                if *theta < tmin - 1e-6 {
                    return false;
                }
            }
            if let Some(tmax) = theta_max {
                if *theta > tmax + 1e-6 {
                    return false;
                }
            }
            true
        })
        .collect();

    let mut entries: Vec<ThetaEntry> = Vec::new();
    for (theta, path) in &filtered {
        match load_state_values_standalone(path) {
            Some(sv) => {
                entries.push(ThetaEntry { theta: *theta, sv });
            }
            None => {
                eprintln!("  WARNING: Failed to load {}", path);
            }
        }
    }
    entries
}
