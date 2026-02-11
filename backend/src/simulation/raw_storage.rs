//! Binary I/O for raw simulation data.
//!
//! Format: 32-byte header + GameRecord[N] in packed layout.
//! Loading uses zero-copy mmap for instant reaggregation.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use memmap2::Mmap;

use super::engine::GameRecord;

/// Magic number: "RTSZ" in little-endian.
pub const RAW_SIM_MAGIC: u32 = 0x59545352;
pub const RAW_SIM_VERSION: u32 = 1;

/// Binary file header (32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RawSimulationHeader {
    pub magic: u32,          // 4
    pub version: u32,        // 4
    pub num_games: u32,      // 4
    pub expected_value: f32, // 4
    pub seed: u64,           // 8
    pub _reserved: [u8; 8],  // 8
}

const _: () = assert!(std::mem::size_of::<RawSimulationHeader>() == 32);

/// Save raw simulation data to a binary file.
pub fn save_raw_simulation(records: &[GameRecord], seed: u64, ev: f32, path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let mut f = File::create(path).expect("Failed to create raw simulation file");

    let header = RawSimulationHeader {
        magic: RAW_SIM_MAGIC,
        version: RAW_SIM_VERSION,
        num_games: records.len() as u32,
        expected_value: ev,
        seed,
        _reserved: [0u8; 8],
    };

    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const RawSimulationHeader as *const u8,
            std::mem::size_of::<RawSimulationHeader>(),
        )
    };
    f.write_all(header_bytes).unwrap();

    let data_bytes = unsafe {
        std::slice::from_raw_parts(
            records.as_ptr() as *const u8,
            std::mem::size_of_val(records),
        )
    };
    f.write_all(data_bytes).unwrap();
}

/// Loaded raw simulation: owns the mmap and provides access to header + records.
pub struct RawSimulation {
    _mmap: Mmap,
    header: RawSimulationHeader,
    records_ptr: *const GameRecord,
    records_len: usize,
}

// Safety: the mmap is immutable and the GameRecord slice is derived from it.
unsafe impl Send for RawSimulation {}
unsafe impl Sync for RawSimulation {}

impl RawSimulation {
    pub fn header(&self) -> &RawSimulationHeader {
        &self.header
    }

    pub fn records(&self) -> &[GameRecord] {
        unsafe { std::slice::from_raw_parts(self.records_ptr, self.records_len) }
    }
}

/// Load raw simulation data via mmap. Returns None on error.
pub fn load_raw_simulation(path: &str) -> Option<RawSimulation> {
    let file = File::open(path).ok()?;
    let metadata = file.metadata().ok()?;
    let file_size = metadata.len() as usize;

    let header_size = std::mem::size_of::<RawSimulationHeader>();
    if file_size < header_size {
        return None;
    }

    let mmap = unsafe { Mmap::map(&file).ok()? };

    let header_ptr = mmap.as_ptr() as *const RawSimulationHeader;
    let header = unsafe { *header_ptr };

    if header.magic != RAW_SIM_MAGIC || header.version != RAW_SIM_VERSION {
        return None;
    }

    let record_size = std::mem::size_of::<GameRecord>();
    let data_size = file_size - header_size;
    let num_records = data_size / record_size;

    if num_records != header.num_games as usize {
        return None;
    }

    let records_ptr = unsafe { mmap.as_ptr().add(header_size) as *const GameRecord };

    Some(RawSimulation {
        _mmap: mmap,
        header,
        records_ptr,
        records_len: num_records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::engine::TurnRecord;

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<RawSimulationHeader>(), 32);
    }

    #[test]
    fn test_round_trip() {
        let test_path = "/tmp/yatzy_test_raw_sim.bin";

        let mut records = Vec::new();
        for i in 0..10u16 {
            let mut rec = GameRecord::default();
            rec.total_score = 200 + i;
            rec.upper_total = 40;
            rec.got_bonus = 0;
            rec.turns[0] = TurnRecord {
                dice_initial: [1, 2, 3, 4, 5],
                mask1: 7,
                dice_after_reroll1: [1, 3, 4, 5, 6],
                mask2: 0,
                dice_final: [1, 3, 4, 5, 6],
                category: i as u8,
                score: 15,
            };
            records.push(rec);
        }

        save_raw_simulation(&records, 42, 245.87, test_path);

        let loaded = load_raw_simulation(test_path).expect("Failed to load");
        assert_eq!(loaded.header().num_games, 10);
        assert_eq!(loaded.header().seed, 42);
        assert!((loaded.header().expected_value - 245.87).abs() < 0.01);

        let loaded_records = loaded.records();
        assert_eq!(loaded_records.len(), 10);

        for (i, rec) in loaded_records.iter().enumerate() {
            let total = rec.total_score;
            let upper = rec.upper_total;
            let turn0 = rec.turns[0];
            assert_eq!(total, 200 + i as u16);
            assert_eq!(upper, 40);
            assert_eq!(turn0.dice_initial, [1, 2, 3, 4, 5]);
            assert_eq!(turn0.mask1, 7);
            assert_eq!(turn0.category, i as u8);
        }

        let _ = std::fs::remove_file(test_path);
    }

    #[test]
    fn test_load_nonexistent() {
        assert!(load_raw_simulation("/tmp/nonexistent_yatzy_raw.bin").is_none());
    }
}
