//! Unified scenario generation pipeline.
//!
//! Replaces duplicated code across `difficult_scenarios.rs`, `scenario_sensitivity.rs`,
//! and `profiling/scenarios.rs` with a single configurable two-stage pipeline.
//!
//! - **Stage 1** (`collect`): Simulate games, collect decision points, validate, classify
//! - **Stage 2** (`select`): Difficulty-based or diagnostic-based scenario selection
//! - **Enrichment** (`enrich`): θ-sensitivity analysis or Q-grid computation

pub mod actions;
pub mod classify;
pub mod collect;
pub mod enrich;
pub mod io;
pub mod select;
pub mod types;
