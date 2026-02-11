//! Game simulation and statistics.
//!
//! - [`engine`]: Core simulation (play N games with optimal strategy)
//! - [`statistics`]: Aggregate statistics from recorded games
//! - [`raw_storage`]: Binary I/O for raw simulation data

pub mod engine;
pub mod raw_storage;
pub mod statistics;

// Re-export commonly used items
pub use engine::{
    simulate_batch, simulate_batch_with_recording, simulate_game, GameRecord, SimulationResult,
    TurnRecord,
};
pub use raw_storage::{load_raw_simulation, save_raw_simulation};
pub use statistics::{aggregate_statistics, save_statistics, GameStatistics};
