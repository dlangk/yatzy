//! Exact density evolution — computes mathematically perfect score distributions.
//!
//! Instead of Monte Carlo simulation (sampling N games), this module pushes
//! `P(state, accumulated_score)` forward through all 15 turns using the optimal
//! policy derived from the precomputed strategy table.
//!
//! The result is an exact PMF with zero variance, making it superior to MC
//! for sweep statistics (mean, percentiles, CVaR, etc.).

pub mod forward;
pub mod transitions;
