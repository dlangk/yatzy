//! Binary I/O for the E_table state values (Storage v3 format).
//!
//! Format: 16-byte header + float32[2,097,152] in `state_index(m, C)` order.
//! Total file size: 8,388,624 bytes (~8 MB).
//!
//! Loading uses zero-copy memory mapping via `memmap2` for <1ms startup.
//! The binary format is compatible between the C and Rust backends.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use memmap2::Mmap;

use crate::constants::*;
use crate::types::{StateValues, YatzyContext};

/// Storage v3 file header: magic + version + state count + reserved.
///
/// Magic bytes "STZY" (0x59545A53) identify the file format.
/// Binary layout matches the C backend for cross-compatibility.
#[repr(C)]
struct StateFileHeader {
    magic: u32,
    version: u32,
    total_states: u32,
    reserved: u32,
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

    // Validate header
    let header_ptr = mmap.as_ptr() as *const StateFileHeader;
    let header = unsafe { &*header_ptr };
    if header.magic != STATE_FILE_MAGIC || header.version != STATE_FILE_VERSION {
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
        version: STATE_FILE_VERSION,
        total_states: NUM_STATES as u32,
        reserved: 0,
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
    fn test_v3_round_trip() {
        let test_file = "/tmp/yatzy_test_all_states_v3_rust.bin";

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
                "V3 mismatch at index {}: {} != {}",
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
