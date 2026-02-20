//! Core types for the unified scenario pipeline.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Decision type ──

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DecisionType {
    Reroll1,
    Reroll2,
    Category,
}

impl DecisionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DecisionType::Reroll1 => "reroll1",
            DecisionType::Reroll2 => "reroll2",
            DecisionType::Category => "category",
        }
    }

    pub fn phase_label(&self) -> &'static str {
        match self {
            DecisionType::Reroll1 => "1st reroll",
            DecisionType::Reroll2 => "2nd reroll",
            DecisionType::Category => "category choice",
        }
    }
}

// ── Game phase ──

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GamePhase {
    Early,
    Mid,
    Late,
}

impl GamePhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            GamePhase::Early => "early",
            GamePhase::Mid => "mid",
            GamePhase::Late => "late",
        }
    }
}

// ── Board tension ──

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BoardTension {
    BonusChase,
    PatternHunt,
    Cleanup,
    Open,
}

impl BoardTension {
    pub fn as_str(&self) -> &'static str {
        match self {
            BoardTension::BonusChase => "bonus_chase",
            BoardTension::PatternHunt => "pattern_hunt",
            BoardTension::Cleanup => "cleanup",
            BoardTension::Open => "open",
        }
    }
}

// ── Diagnostic quadrant ──

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quadrant {
    Theta,
    Gamma,
    Depth,
    Beta,
}

impl Quadrant {
    pub fn as_str(&self) -> &'static str {
        match self {
            Quadrant::Theta => "theta",
            Quadrant::Gamma => "gamma",
            Quadrant::Depth => "depth",
            Quadrant::Beta => "beta",
        }
    }
}

// ── Semantic bucket ──

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SemanticBucket {
    pub phase: GamePhase,
    pub dtype: DecisionType,
    pub tension: BoardTension,
}

// ── Diagnostic scores ──

#[derive(Clone, Debug)]
pub struct DiagnosticScores {
    pub s_theta: f32,
    pub s_gamma: f32,
    pub s_d: f32,
    pub s_beta: f32,
}

impl DiagnosticScores {
    pub fn max_score(&self) -> f32 {
        self.s_theta
            .max(self.s_gamma)
            .max(self.s_d)
            .max(self.s_beta)
    }

    pub fn total_score(&self) -> f32 {
        self.s_theta + self.s_gamma + self.s_d + self.s_beta
    }
}

// ── Raw decision (collected during simulation) ──

#[derive(Clone)]
pub struct RawDecision {
    pub upper_score: i32,
    pub scored: i32,
    pub dice: [i32; 5],
    pub turn: usize,
    pub decision_type: DecisionType,
    pub category_scores: [i32; 15],
}

pub type DecisionKey = (i32, i32, [i32; 5], DecisionType);

pub fn decision_key(d: &RawDecision) -> DecisionKey {
    (d.upper_score, d.scored, d.dice, d.decision_type)
}

// ── Action info ──

#[derive(Clone, Serialize, Deserialize)]
pub struct ActionInfo {
    pub id: i32,
    pub label: String,
    pub ev: f32,
}

// ── Candidate (post-analysis) ──

pub struct Candidate {
    pub decision: RawDecision,
    pub visit_count: usize,
    pub actions: Vec<ActionInfo>,
    pub ev_gap: f32,
    pub difficulty_score: f64,
    pub game_phase: GamePhase,
    pub board_tension: BoardTension,
    pub diagnostic_scores: Option<DiagnosticScores>,
    pub quadrant: Option<Quadrant>,
    pub fingerprint: String,
    pub top_action_labels: Vec<String>,
}

// ── Scored candidate (for profiling assembly) ──

pub struct ScoredCandidate {
    pub decision: RawDecision,
    pub visit_count: usize,
    pub bucket: SemanticBucket,
    pub scores: DiagnosticScores,
    pub ev_gap: f32,
    pub fingerprint: String,
    pub top_action_labels: Vec<String>,
}

// ── Profiling scenario ──

pub struct ProfilingScenario {
    pub id: usize,
    pub upper_score: i32,
    pub scored_categories: i32,
    pub dice: [i32; 5],
    pub turn: usize,
    pub decision_type: DecisionType,
    pub quadrant: Quadrant,
    pub visit_count: usize,
    pub ev_gap: f32,
    pub optimal_action_id: i32,
    pub actions: Vec<ActionInfo>,
    pub description: String,
}

// ── Theta sensitivity result ──

#[derive(Clone, Serialize)]
pub struct ThetaResult {
    pub theta: f32,
    pub action: String,
    pub action_id: i32,
    pub value: f32,
    pub runner_up: String,
    pub runner_up_id: i32,
    pub runner_up_value: f32,
    pub gap: f32,
}

// ── Q-grid for profiling ──

pub type QGrid = HashMap<String, Vec<f32>>;
