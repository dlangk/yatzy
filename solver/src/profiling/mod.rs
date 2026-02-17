//! Cognitive profiling: scenario generation and Q-value grids.
//!
//! Generates 20 diagnostic scenarios via diversity-constrained stratified
//! sampling across game phases, decision types, and board tensions.
//! Pre-computes Q-value grids for client-side MLE estimation.

pub mod qvalues;
pub mod scenarios;
