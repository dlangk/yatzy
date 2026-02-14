//! Game simulation and statistics.
//!
//! - [`engine`]: Core simulation (play N games with optimal strategy)
//! - [`adaptive`]: Adaptive θ policies (switch tables per-turn based on game state)
//! - [`statistics`]: Aggregate statistics from recorded games
//! - [`raw_storage`]: Binary I/O for raw simulation data

pub mod adaptive;
pub mod engine;
pub mod raw_storage;
pub mod statistics;

// Re-export commonly used items
pub use adaptive::{
    make_policy, policy_thetas, simulate_batch_adaptive, simulate_batch_adaptive_with_recording,
    ThetaTable, POLICY_CONFIGS,
};
pub use engine::{
    simulate_batch, simulate_batch_summaries, simulate_batch_with_recording, simulate_game,
    GameRecord, GameSummary, SimulationResult, TurnRecord, TurnSummary,
};
pub use raw_storage::{
    load_raw_simulation, load_scores, save_raw_simulation, save_scores, ScoresHeader, SCORES_MAGIC,
    SCORES_VERSION,
};
pub use statistics::{aggregate_statistics, save_statistics, GameStatistics};
