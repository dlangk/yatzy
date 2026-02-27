//! Game simulation and statistics.
//!
//! - [`engine`]: Core simulation (play N games with optimal strategy)
//! - [`adaptive`]: Adaptive θ policies (switch tables per-turn based on game state)
//! - [`statistics`]: Aggregate statistics from recorded games
//! - [`raw_storage`]: Binary I/O for raw simulation data

pub mod engine;

#[cfg(feature = "full")]
pub mod adaptive;
#[cfg(feature = "full")]
pub mod fast_prng;
#[cfg(feature = "full")]
pub mod heuristic;
#[cfg(feature = "full")]
pub mod lockstep;
#[cfg(feature = "full")]
pub mod multiplayer;
#[cfg(feature = "full")]
pub mod radix_sort;
#[cfg(feature = "full")]
pub mod raw_storage;
#[cfg(feature = "full")]
pub mod statistics;
#[cfg(feature = "full")]
pub mod strategy;
#[cfg(feature = "full")]
pub mod sweep;

// Re-export commonly used items
pub use engine::{
    simulate_forecast, simulate_game, simulate_game_from_state, ForecastResult, ForecastTurn,
    GameRecord, GameSummary, TurnRecord, TurnSummary,
};

#[cfg(feature = "full")]
pub use engine::{
    simulate_batch, simulate_batch_summaries, simulate_batch_with_recording, SimulationResult,
};

#[cfg(feature = "full")]
pub use adaptive::{
    make_policy, policy_thetas, simulate_batch_adaptive, simulate_batch_adaptive_with_recording,
    ThetaTable, POLICY_CONFIGS,
};
#[cfg(feature = "full")]
pub use multiplayer::{
    aggregate_from_records, simulate_multiplayer_with_recording, MultiplayerGameRecord,
};
#[cfg(feature = "full")]
pub use raw_storage::{
    load_raw_simulation, load_scores, save_multiplayer_recording, save_raw_simulation, save_scores,
    MultiplayerHeader, ScoresHeader, MULTIPLAYER_MAGIC, SCORES_MAGIC, SCORES_VERSION,
};
#[cfg(feature = "full")]
pub use statistics::{aggregate_statistics, save_statistics, GameStatistics};
