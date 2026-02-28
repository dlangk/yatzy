//! Binary I/O for the E_table state values.
//!
//! Format: 16-byte header + float32[4,194,304] in `state_index(m, C)` order
//! (v6 layout: `C * STATE_STRIDE + m`, STATE_STRIDE=128 with topological padding).
//! Total file size: 16,777,232 bytes (~16 MB).
//!
//! Loading uses zero-copy memory mapping via `memmap2` for <1ms startup.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use memmap2::Mmap;

use crate::constants::*;
use crate::types::{PercentileEntry, PolicyOracle, StateValues, YatzyContext, ORACLE_ENTRIES};
use std::collections::HashMap;

/// Binary file header: magic + version + state count + theta_bits.
///
/// Magic bytes "STZY" (0x59545A53) identify the file format.
/// V4: reserved field is 0. V5: reserved field stores θ as f32 bits.
#[repr(C)]
struct StateFileHeader {
    magic: u32,
    version: u32,
    total_states: u32,
    /// V4: 0. V5: f32::to_bits(theta).
    theta_bits: u32,
}

/// Return the state file path for a given θ value.
/// θ=0.0 uses `data/strategy_tables/all_states.bin`.
/// Other θ values use `data/strategy_tables/all_states_theta_{theta:.3}.bin`.
pub fn state_file_path(theta: f32) -> String {
    if theta == 0.0 {
        "data/strategy_tables/all_states.bin".to_string()
    } else {
        format!("data/strategy_tables/all_states_theta_{:.3}.bin", theta)
    }
}

/// Check if a file exists on disk.
pub fn file_exists(filename: &str) -> bool {
    Path::new(filename).exists()
}

/// Load state values via zero-copy mmap. Returns true on success.
pub fn load_all_state_values(ctx: &mut YatzyContext, filename: &str) -> bool {
    let start_time = Instant::now();
    println!("Loading state values from {}...", filename);

    let file = match File::open(filename) {
        Ok(f) => f,
        Err(_) => {
            println!("File not found: {}", filename);
            return false;
        }
    };

    let metadata = match file.metadata() {
        Ok(m) => m,
        Err(_) => return false,
    };

    let expected_size =
        std::mem::size_of::<StateFileHeader>() + NUM_STATES * std::mem::size_of::<f32>();
    if metadata.len() as usize != expected_size {
        println!(
            "File size mismatch: expected {}, got {}",
            expected_size,
            metadata.len()
        );
        return false;
    }

    let mmap = match unsafe { Mmap::map(&file) } {
        Ok(m) => m,
        Err(e) => {
            println!("Failed to memory map file: {}", e);
            return false;
        }
    };

    // Validate header (accept v4 or v5)
    let header_ptr = mmap.as_ptr() as *const StateFileHeader;
    let header = unsafe { &*header_ptr };
    if header.magic != STATE_FILE_MAGIC
        || (header.version != STATE_FILE_VERSION && header.version != STATE_FILE_VERSION_V5)
    {
        println!(
            "Invalid file format (magic=0x{:08x} version={})",
            header.magic, header.version
        );
        return false;
    }

    let elapsed = start_time.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Loaded {} states via zero-copy mmap in {:.2} ms",
        header.total_states, elapsed
    );

    ctx.state_values = StateValues::Mmap { mmap };
    true
}

/// Load state values via zero-copy mmap, returning `StateValues` directly.
///
/// Unlike `load_all_state_values`, this does not require a `YatzyContext`.
/// Used by adaptive policies to load multiple θ tables independently.
pub fn load_state_values_standalone(filename: &str) -> Option<StateValues> {
    let file = File::open(filename).ok()?;
    let metadata = file.metadata().ok()?;

    let expected_size =
        std::mem::size_of::<StateFileHeader>() + NUM_STATES * std::mem::size_of::<f32>();
    if metadata.len() as usize != expected_size {
        eprintln!(
            "File size mismatch for {}: expected {}, got {}",
            filename,
            expected_size,
            metadata.len()
        );
        return None;
    }

    let mmap = unsafe { Mmap::map(&file) }.ok()?;

    let header_ptr = mmap.as_ptr() as *const StateFileHeader;
    let header = unsafe { &*header_ptr };
    if header.magic != STATE_FILE_MAGIC
        || (header.version != STATE_FILE_VERSION && header.version != STATE_FILE_VERSION_V5)
    {
        eprintln!(
            "Invalid file format for {}: magic=0x{:08x} version={}",
            filename, header.magic, header.version
        );
        return None;
    }

    Some(StateValues::Mmap { mmap })
}

/// Save state values as a flat dump in STATE_INDEX order.
pub fn save_all_state_values(ctx: &YatzyContext, filename: &str) {
    let start_time = Instant::now();
    println!("Saving state values to {}...", filename);

    // Ensure parent directory exists
    if let Some(parent) = Path::new(filename).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let mut f = match File::create(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error creating file: {}", e);
            return;
        }
    };

    let header = StateFileHeader {
        magic: STATE_FILE_MAGIC,
        version: if ctx.theta == 0.0 {
            STATE_FILE_VERSION
        } else {
            STATE_FILE_VERSION_V5
        },
        total_states: NUM_STATES as u32,
        theta_bits: ctx.theta.to_bits(),
    };

    // Write header
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const StateFileHeader as *const u8,
            std::mem::size_of::<StateFileHeader>(),
        )
    };
    f.write_all(header_bytes).unwrap();

    // Write state values
    let state_slice = ctx.state_values.as_slice();
    let data_bytes = unsafe {
        std::slice::from_raw_parts(
            state_slice.as_ptr() as *const u8,
            NUM_STATES * std::mem::size_of::<f32>(),
        )
    };
    f.write_all(data_bytes).unwrap();

    let elapsed = start_time.elapsed().as_secs_f64() * 1000.0;
    println!("Saved {} states in {:.2} ms", NUM_STATES, elapsed);
}

/// Oracle file magic number: "ORCL" in hex.
const ORACLE_MAGIC: u32 = 0x4C43524F;
/// Oracle file version.
const ORACLE_VERSION: u32 = 1;

/// Binary file header for oracle.
#[repr(C)]
struct OracleFileHeader {
    magic: u32,
    version: u32,
    entries_per_array: u64,
}

/// Default oracle file path.
pub const ORACLE_FILE_PATH: &str = "data/strategy_tables/oracle.bin";

/// Save policy oracle to disk.
///
/// Format: 24-byte header + 3 × ORACLE_ENTRIES bytes (~3.17 GB).
pub fn save_oracle(oracle: &PolicyOracle, filename: &str) {
    let start_time = Instant::now();
    println!("Saving oracle to {}...", filename);

    if let Some(parent) = Path::new(filename).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let mut f = match File::create(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error creating oracle file: {}", e);
            return;
        }
    };

    let header = OracleFileHeader {
        magic: ORACLE_MAGIC,
        version: ORACLE_VERSION,
        entries_per_array: ORACLE_ENTRIES as u64,
    };

    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const OracleFileHeader as *const u8,
            std::mem::size_of::<OracleFileHeader>(),
        )
    };
    f.write_all(header_bytes).unwrap();
    f.write_all(oracle.cat()).unwrap();
    f.write_all(oracle.keep1()).unwrap();
    f.write_all(oracle.keep2()).unwrap();

    let elapsed = start_time.elapsed().as_secs_f64();
    let size_gb = (std::mem::size_of::<OracleFileHeader>() + 3 * ORACLE_ENTRIES) as f64
        / 1024.0
        / 1024.0
        / 1024.0;
    println!("Saved oracle ({:.2} GB) in {:.2}s", size_gb, elapsed);
}

/// Load policy oracle via mmap. Returns None if file doesn't exist or is invalid.
pub fn load_oracle(filename: &str) -> Option<PolicyOracle> {
    let start_time = Instant::now();
    println!("Loading oracle from {}...", filename);

    let file = match File::open(filename) {
        Ok(f) => f,
        Err(_) => {
            println!("Oracle file not found: {}", filename);
            return None;
        }
    };

    let metadata = file.metadata().ok()?;
    let header_size = std::mem::size_of::<OracleFileHeader>();
    let expected_size = header_size + 3 * ORACLE_ENTRIES;

    if metadata.len() as usize != expected_size {
        println!(
            "Oracle file size mismatch: expected {}, got {}",
            expected_size,
            metadata.len()
        );
        return None;
    }

    let mmap = unsafe { Mmap::map(&file) }.ok()?;

    let header = unsafe { &*(mmap.as_ptr() as *const OracleFileHeader) };
    if header.magic != ORACLE_MAGIC || header.version != ORACLE_VERSION {
        println!(
            "Invalid oracle format (magic=0x{:08x} version={})",
            header.magic, header.version
        );
        return None;
    }

    // Zero-copy mmap: slices are computed on access via cat()/keep1()/keep2()
    let data_start = header_size;

    let oracle = PolicyOracle::Mmap { mmap, data_start };

    let elapsed = start_time.elapsed().as_secs_f64();
    println!("Loaded oracle (zero-copy mmap) in {:.2}s", elapsed);

    Some(oracle)
}

// ── Percentile table I/O ────────────────────────────────────────────

/// Default percentile table file path.
pub const PERCENTILE_FILE_PATH: &str = "data/strategy_tables/percentiles.bin";

/// Magic number for percentile table files: "PCTI".
const PERCENTILE_MAGIC: u32 = 0x50435449;
/// Percentile file version.
const PERCENTILE_VERSION: u32 = 1;

/// Binary header for percentile table.
#[repr(C)]
struct PercentileFileHeader {
    magic: u32,
    version: u32,
    num_entries: u32,
    num_games: u32,
}

/// Per-entry record in the binary file: state_index + 9 × i16 percentiles + mean_x100.
#[repr(C, packed)]
struct PercentileRecord {
    state_index: u32,
    percentiles: [i16; 9],
    mean_x100: i32,
}

/// Save percentile table to binary file.
pub fn save_percentile_table(entries: &[(u32, PercentileEntry)], num_games: u32, filename: &str) {
    println!(
        "Saving {} percentile entries to {}...",
        entries.len(),
        filename
    );

    if let Some(parent) = Path::new(filename).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let mut f = match File::create(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error creating percentile file: {}", e);
            return;
        }
    };

    let header = PercentileFileHeader {
        magic: PERCENTILE_MAGIC,
        version: PERCENTILE_VERSION,
        num_entries: entries.len() as u32,
        num_games,
    };

    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const PercentileFileHeader as *const u8,
            std::mem::size_of::<PercentileFileHeader>(),
        )
    };
    f.write_all(header_bytes).unwrap();

    for (si, entry) in entries {
        let mut pcts = [0i16; 9];
        for (i, &p) in entry.percentiles.iter().enumerate() {
            pcts[i] = p as i16;
        }
        let record = PercentileRecord {
            state_index: *si,
            percentiles: pcts,
            mean_x100: (entry.mean * 100.0).round() as i32,
        };
        let record_bytes = unsafe {
            std::slice::from_raw_parts(
                &record as *const PercentileRecord as *const u8,
                std::mem::size_of::<PercentileRecord>(),
            )
        };
        f.write_all(record_bytes).unwrap();
    }

    let file_size = std::mem::size_of::<PercentileFileHeader>()
        + entries.len() * std::mem::size_of::<PercentileRecord>();
    println!(
        "Saved {} entries ({:.1} KB)",
        entries.len(),
        file_size as f64 / 1024.0,
    );
}

/// Load percentile table from binary file. Returns None if file not found.
pub fn load_percentile_table(filename: &str) -> Option<HashMap<u32, PercentileEntry>> {
    let start_time = Instant::now();
    println!("Loading percentile table from {}...", filename);

    let data = match fs::read(filename) {
        Ok(d) => d,
        Err(_) => {
            println!("Percentile table not found: {}", filename);
            return None;
        }
    };

    let header_size = std::mem::size_of::<PercentileFileHeader>();
    if data.len() < header_size {
        println!("Percentile file too small");
        return None;
    }

    let header = unsafe { &*(data.as_ptr() as *const PercentileFileHeader) };
    if header.magic != PERCENTILE_MAGIC || header.version != PERCENTILE_VERSION {
        println!(
            "Invalid percentile format (magic=0x{:08x} version={})",
            header.magic, header.version
        );
        return None;
    }

    let record_size = std::mem::size_of::<PercentileRecord>();
    let expected_size = header_size + header.num_entries as usize * record_size;
    if data.len() != expected_size {
        println!(
            "Percentile file size mismatch: expected {}, got {}",
            expected_size,
            data.len()
        );
        return None;
    }

    let mut map = HashMap::with_capacity(header.num_entries as usize);
    let base_ptr = unsafe { data.as_ptr().add(header_size) };

    for i in 0..header.num_entries as usize {
        let record_ptr = unsafe { base_ptr.add(i * record_size) as *const PercentileRecord };
        // Read packed fields via read_unaligned to avoid UB
        let si = unsafe { std::ptr::addr_of!((*record_ptr).state_index).read_unaligned() };
        let pcts_raw = unsafe { std::ptr::addr_of!((*record_ptr).percentiles).read_unaligned() };
        let mean_x100 = unsafe { std::ptr::addr_of!((*record_ptr).mean_x100).read_unaligned() };

        let mean = mean_x100 as f64 / 100.0;
        let mut pcts = [0i32; 9];
        for (j, &p) in pcts_raw.iter().enumerate() {
            pcts[j] = p as i32;
        }
        map.insert(
            si,
            PercentileEntry {
                mean,
                std_dev: 0.0, // Not stored in compact format
                percentiles: pcts,
            },
        );
    }

    let elapsed = start_time.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Loaded {} percentile entries in {:.2} ms",
        map.len(),
        elapsed
    );

    Some(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0_tables;

    #[test]
    fn test_file_exists() {
        assert!(file_exists("/tmp"));
        assert!(!file_exists("/tmp/nonexistent_yatzy_test_xyz"));
    }

    #[test]
    fn test_v4_round_trip() {
        let test_file = "/tmp/yatzy_test_all_states_v4_rust.bin";

        let mut ctx1 = YatzyContext::new_boxed();
        phase0_tables::precompute_lookup_tables(&mut ctx1);

        // Set some known values
        let all_scored = (1 << CATEGORY_COUNT) - 1;
        {
            let sv = ctx1.state_values.as_mut_slice();
            for up in 0..=63usize {
                sv[state_index(up, all_scored)] = if up >= 63 { 50.0 } else { 0.0 };
            }
            sv[state_index(0, 0)] = 123.5;
            sv[state_index(63, 1)] = 789.0;
        }

        save_all_state_values(&ctx1, test_file);
        assert!(file_exists(test_file));

        let mut ctx2 = YatzyContext::new_boxed();
        assert!(load_all_state_values(&mut ctx2, test_file));

        // Compare all states
        let sv1 = ctx1.state_values.as_slice();
        let sv2 = ctx2.state_values.as_slice();
        for i in 0..NUM_STATES {
            assert!(
                (sv1[i] - sv2[i]).abs() < 1e-6,
                "V4 mismatch at index {}: {} != {}",
                i,
                sv1[i],
                sv2[i]
            );
        }

        // Spot-check
        assert!((sv2[state_index(0, 0)] - 123.5).abs() < 1e-6);
        assert!((sv2[state_index(63, 1)] - 789.0).abs() < 1e-6);
        assert!((sv2[state_index(63, all_scored)] - 50.0).abs() < 1e-9);

        // Cleanup
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_load_nonexistent() {
        let mut ctx = YatzyContext::new_boxed();
        assert!(!load_all_state_values(
            &mut ctx,
            "/tmp/nonexistent_yatzy_v3_rust.bin"
        ));
    }
}
