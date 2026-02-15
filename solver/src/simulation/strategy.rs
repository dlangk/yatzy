//! Strategy types for multiplayer simulation.
//!
//! Defines the observable game state ([`PlayerState`], [`GameView`]) and the
//! [`Strategy`] abstraction that maps CLI specs like `"ev"`, `"theta:0.05"`,
//! or `"mp:trailing"` to concrete table-lookup + policy combinations.

use std::path::Path;

use crate::constants::state_index;
use crate::storage::{load_state_values_standalone, state_file_path};
use crate::types::{StateValues, YatzyContext};

use super::adaptive::{make_policy, policy_thetas, AdaptivePolicy, ThetaTable, TurnConfig};

// ── Observable state ──────────────────────────────────────────────────────

/// Observable state of one player (public information).
#[derive(Clone, Debug, Default)]
pub struct PlayerState {
    pub upper_score: i32,
    pub scored_categories: i32,
    pub total_score: i32,
}

/// The full game context visible to a strategy when making decisions.
pub struct GameView<'a> {
    pub my_index: usize,
    pub players: &'a [PlayerState],
    pub turn: usize,
}

impl GameView<'_> {
    /// This player's state.
    #[inline]
    pub fn me(&self) -> &PlayerState {
        &self.players[self.my_index]
    }

    /// Iterator over all opponents' states.
    pub fn opponents(&self) -> impl Iterator<Item = &PlayerState> {
        let idx = self.my_index;
        self.players
            .iter()
            .enumerate()
            .filter(move |(i, _)| *i != idx)
            .map(|(_, p)| p)
    }

    /// The highest total_score among all players.
    pub fn leading_score(&self) -> i32 {
        self.players
            .iter()
            .map(|p| p.total_score)
            .max()
            .unwrap_or(0)
    }

    /// How many points this player trails the leader by (0 if leading).
    pub fn trailing_by(&self) -> i32 {
        let leader = self.leading_score();
        (leader - self.me().total_score).max(0)
    }
}

// ── Multiplayer policy trait ──────────────────────────────────────────────

/// An opponent-aware policy that can see all players' states.
pub trait MultiplayerPolicy: Send + Sync {
    fn name(&self) -> &str;
    fn select_theta_index(&self, view: &GameView, tables: &[ThetaTable]) -> usize;
}

// ── TrailingRisk policy ───────────────────────────────────────────────────

/// When trailing the leader by >= threshold, switch to risk-seeking θ.
/// When leading or close, play EV-optimal (θ=0).
pub struct TrailingRisk {
    pub ev_idx: usize,
    pub high_idx: usize,
    pub threshold: i32,
}

impl MultiplayerPolicy for TrailingRisk {
    fn name(&self) -> &str {
        "trailing-risk"
    }

    fn select_theta_index(&self, view: &GameView, _tables: &[ThetaTable]) -> usize {
        if view.trailing_by() >= self.threshold {
            self.high_idx
        } else {
            self.ev_idx
        }
    }
}

// ── UnderdogPolicy ───────────────────────────────────────────────────────

/// Available θ values for the underdog ramp (must have precomputed tables).
const UNDERDOG_THETAS: &[f32] = &[0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];

/// Opponent-aware adaptive strategy with continuous θ ramp.
///
/// Uses **expected final scores** (current total + EV remaining from state-value
/// table) rather than raw totals, so the deficit accounts for upper-bonus
/// progress and which categories remain.
///
/// θ ramps linearly from 0 to θ_max as the EV deficit grows:
///   `desired_θ = θ_max × clamp(ev_deficit / scale, 0, 1)`
///
/// Then snaps to the nearest precomputed table. When leading or even,
/// ev_deficit ≤ 0 → θ = 0 (pure EV play). When the player catches up,
/// ev_deficit shrinks → θ decreases automatically.
pub struct UnderdogPolicy {
    pub ev_idx: usize,  // index of θ=0 table (for EV lookups)
    pub theta_max: f32, // maximum θ when deeply trailing
    pub scale: f32,     // EV deficit at which θ reaches θ_max
}

impl UnderdogPolicy {
    /// Compute the EV deficit: best opponent expected final − my expected final.
    fn ev_deficit(view: &GameView, ev_sv: &[f32]) -> f32 {
        let my = view.me();
        let my_ev_remaining =
            ev_sv[state_index(my.upper_score as usize, my.scored_categories as usize)];
        let my_expected_final = my.total_score as f32 + my_ev_remaining;

        let best_opp_expected = view
            .opponents()
            .map(|p| {
                let ev_rem =
                    ev_sv[state_index(p.upper_score as usize, p.scored_categories as usize)];
                p.total_score as f32 + ev_rem
            })
            .fold(f32::NEG_INFINITY, f32::max);

        best_opp_expected - my_expected_final
    }
}

impl MultiplayerPolicy for UnderdogPolicy {
    fn name(&self) -> &str {
        "underdog"
    }

    fn select_theta_index(&self, view: &GameView, tables: &[ThetaTable]) -> usize {
        if view.turn <= 1 {
            return self.ev_idx; // too early, no signal
        }
        let ev_sv = tables[self.ev_idx].sv.as_slice();
        let deficit = Self::ev_deficit(view, ev_sv);

        // Linear ramp: 0 when even/leading, θ_max when deficit >= scale
        let desired_theta = self.theta_max * (deficit / self.scale).clamp(0.0, 1.0);

        // Snap to nearest precomputed table
        tables
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (a.theta - desired_theta)
                    .abs()
                    .partial_cmp(&(b.theta - desired_theta).abs())
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(self.ev_idx)
    }
}

// ── Strategy ──────────────────────────────────────────────────────────────

/// Strategy kind — determines which state-value table to use.
pub enum StrategyKind {
    /// Fixed θ (θ=0 is standard EV-optimal).
    FixedTheta {
        theta: f32,
        sv: StateValues,
        minimize: bool,
    },
    /// Existing adaptive policy (ignores opponents).
    Adaptive {
        tables: Vec<ThetaTable>,
        policy: Box<dyn AdaptivePolicy>,
    },
    /// Multiplayer-aware adaptive (can see GameView).
    MultiplayerAdaptive {
        tables: Vec<ThetaTable>,
        policy: Box<dyn MultiplayerPolicy>,
    },
    /// Heuristic (human-like) strategy — no state-value tables needed.
    Heuristic,
}

pub struct Strategy {
    pub name: String,
    pub kind: StrategyKind,
}

impl Strategy {
    /// Parse a strategy from a CLI spec string.
    ///
    /// Supported specs:
    /// - `"ev"` — EV-optimal (θ=0)
    /// - `"theta:0.05"` — fixed θ
    /// - `"human"` — heuristic (human-like) strategy, no tables needed
    /// - `"adaptive:bonus"` / `"adaptive:phase"` / `"adaptive:combined"` — existing adaptive policies
    /// - `"mp:trailing"` — multiplayer-aware trailing-risk policy
    /// - `"mp:trailing:20"` — trailing-risk with custom threshold (default 15)
    pub fn from_spec(spec: &str, base_path: &Path, _ctx: &YatzyContext) -> Result<Self, String> {
        if spec == "human" {
            return Ok(Strategy {
                name: "human".to_string(),
                kind: StrategyKind::Heuristic,
            });
        }

        if spec == "ev" {
            let path = base_path.join(state_file_path(0.0));
            let sv = load_state_values_standalone(path.to_str().unwrap())
                .ok_or_else(|| format!("Failed to load θ=0 table from {}", path.display()))?;
            return Ok(Strategy {
                name: "ev".to_string(),
                kind: StrategyKind::FixedTheta {
                    theta: 0.0,
                    sv,
                    minimize: false,
                },
            });
        }

        if let Some(theta_str) = spec.strip_prefix("theta:") {
            let theta: f32 = theta_str
                .parse()
                .map_err(|_| format!("Invalid theta value: {}", theta_str))?;
            let path = base_path.join(state_file_path(theta));
            let sv = load_state_values_standalone(path.to_str().unwrap()).ok_or_else(|| {
                format!("Failed to load θ={} table from {}", theta, path.display())
            })?;
            return Ok(Strategy {
                name: format!("theta:{}", theta_str),
                kind: StrategyKind::FixedTheta {
                    theta,
                    sv,
                    minimize: theta < 0.0,
                },
            });
        }

        if let Some(policy_name) = spec.strip_prefix("adaptive:") {
            let full_name = match policy_name {
                "bonus" => "bonus-adaptive",
                "phase" => "phase-based",
                "combined" => "combined",
                "upper-deficit" => "upper-deficit",
                "ev" => "always-ev",
                other => other,
            };
            let thetas = policy_thetas(full_name)
                .ok_or_else(|| format!("Unknown adaptive policy: {}", full_name))?;
            let tables = load_theta_tables(thetas, base_path)?;
            let policy = make_policy(full_name, &tables)
                .ok_or_else(|| format!("Failed to create policy: {}", full_name))?;
            return Ok(Strategy {
                name: format!("adaptive:{}", policy_name),
                kind: StrategyKind::Adaptive { tables, policy },
            });
        }

        if let Some(mp_spec) = spec.strip_prefix("mp:") {
            let parts: Vec<&str> = mp_spec.split(':').collect();
            match parts[0] {
                "trailing" => {
                    let threshold: i32 = if parts.len() > 1 {
                        parts[1]
                            .parse()
                            .map_err(|_| format!("Invalid threshold: {}", parts[1]))?
                    } else {
                        15
                    };
                    let thetas: &[f32] = &[0.0, 0.08];
                    let tables = load_theta_tables(thetas, base_path)?;
                    let ev_idx = tables
                        .iter()
                        .position(|t| t.theta == 0.0)
                        .expect("θ=0 table missing");
                    let high_idx = tables
                        .iter()
                        .position(|t| (t.theta - 0.08).abs() < 1e-6)
                        .expect("θ=0.08 table missing");
                    let policy = Box::new(TrailingRisk {
                        ev_idx,
                        high_idx,
                        threshold,
                    });
                    return Ok(Strategy {
                        name: if threshold == 15 {
                            "mp:trailing".to_string()
                        } else {
                            format!("mp:trailing:{}", threshold)
                        },
                        kind: StrategyKind::MultiplayerAdaptive { tables, policy },
                    });
                }
                "underdog" => {
                    // mp:underdog[:theta_max[:scale]]
                    let theta_max: f32 = if parts.len() > 1 {
                        parts[1]
                            .parse()
                            .map_err(|_| format!("Invalid theta_max: {}", parts[1]))?
                    } else {
                        0.05
                    };
                    let scale: f32 = if parts.len() > 2 {
                        parts[2]
                            .parse()
                            .map_err(|_| format!("Invalid scale: {}", parts[2]))?
                    } else {
                        50.0
                    };

                    // Load all available θ tables from 0 to theta_max
                    let thetas: Vec<f32> = UNDERDOG_THETAS
                        .iter()
                        .copied()
                        .filter(|&t| t <= theta_max + 1e-6)
                        .collect();
                    let tables = load_theta_tables(&thetas, base_path)?;

                    let ev_idx = tables
                        .iter()
                        .position(|t| t.theta == 0.0)
                        .expect("θ=0 table missing");

                    let policy = Box::new(UnderdogPolicy {
                        ev_idx,
                        theta_max,
                        scale,
                    });

                    let name = if parts.len() == 1 {
                        "mp:underdog".to_string()
                    } else if parts.len() == 2 {
                        format!("mp:underdog:{}", parts[1])
                    } else {
                        format!("mp:underdog:{}:{}", parts[1], parts[2])
                    };

                    return Ok(Strategy {
                        name,
                        kind: StrategyKind::MultiplayerAdaptive { tables, policy },
                    });
                }
                other => return Err(format!("Unknown multiplayer policy: {}", other)),
            }
        }

        Err(format!(
            "Unknown strategy spec: '{}'. Expected: ev, human, theta:<f>, adaptive:<name>, mp:trailing[:<threshold>], mp:underdog[:<theta_max>[:<scale>]]",
            spec
        ))
    }

    /// Returns true if this strategy is heuristic (no state-value tables).
    pub fn is_heuristic(&self) -> bool {
        matches!(self.kind, StrategyKind::Heuristic)
    }

    /// Resolve the turn configuration for this strategy given the game view.
    ///
    /// Panics if called on a `Heuristic` strategy — the caller should check
    /// `is_heuristic()` first and use the heuristic code path.
    pub fn resolve_turn<'a>(&'a self, view: &GameView) -> TurnConfig<'a> {
        match &self.kind {
            StrategyKind::FixedTheta {
                theta,
                sv,
                minimize,
            } => TurnConfig {
                theta: *theta,
                sv: sv.as_slice(),
                minimize: *minimize,
            },
            StrategyKind::Adaptive { tables, policy } => {
                let me = view.me();
                let ti = policy.select_theta_index(me.upper_score, me.scored_categories, view.turn);
                let table = &tables[ti];
                TurnConfig {
                    theta: table.theta,
                    sv: table.sv.as_slice(),
                    minimize: table.minimize,
                }
            }
            StrategyKind::MultiplayerAdaptive { tables, policy } => {
                let ti = policy.select_theta_index(view, tables);
                let table = &tables[ti];
                TurnConfig {
                    theta: table.theta,
                    sv: table.sv.as_slice(),
                    minimize: table.minimize,
                }
            }
            StrategyKind::Heuristic => {
                panic!("resolve_turn() called on Heuristic strategy — use is_heuristic() check")
            }
        }
    }
}

/// Load state-value tables for a set of θ values.
fn load_theta_tables(thetas: &[f32], base_path: &Path) -> Result<Vec<ThetaTable>, String> {
    let mut tables = Vec::with_capacity(thetas.len());
    for &theta in thetas {
        let path = base_path.join(state_file_path(theta));
        let sv = load_state_values_standalone(path.to_str().unwrap())
            .ok_or_else(|| format!("Failed to load θ={} table from {}", theta, path.display()))?;
        tables.push(ThetaTable {
            theta,
            sv,
            minimize: theta < 0.0,
        });
    }
    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_player_state_default() {
        let ps = PlayerState::default();
        assert_eq!(ps.upper_score, 0);
        assert_eq!(ps.scored_categories, 0);
        assert_eq!(ps.total_score, 0);
    }

    #[test]
    fn test_game_view_trailing_by() {
        let players = vec![
            PlayerState {
                total_score: 100,
                ..Default::default()
            },
            PlayerState {
                total_score: 80,
                ..Default::default()
            },
            PlayerState {
                total_score: 120,
                ..Default::default()
            },
        ];
        let view = GameView {
            my_index: 1,
            players: &players,
            turn: 5,
        };
        assert_eq!(view.trailing_by(), 40); // 120 - 80
        assert_eq!(view.leading_score(), 120);

        let view_leader = GameView {
            my_index: 2,
            players: &players,
            turn: 5,
        };
        assert_eq!(view_leader.trailing_by(), 0);
    }

    #[test]
    fn test_game_view_opponents() {
        let players = vec![
            PlayerState {
                total_score: 10,
                ..Default::default()
            },
            PlayerState {
                total_score: 20,
                ..Default::default()
            },
            PlayerState {
                total_score: 30,
                ..Default::default()
            },
        ];
        let view = GameView {
            my_index: 1,
            players: &players,
            turn: 0,
        };
        let opp_scores: Vec<i32> = view.opponents().map(|p| p.total_score).collect();
        assert_eq!(opp_scores, vec![10, 30]);
    }

    /// Build fake tables with θ = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05] and
    /// uniform EV remaining for all states.
    fn fake_ramp_tables(ev_remaining: f32) -> Vec<ThetaTable> {
        use crate::constants::NUM_STATES;
        [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
            .iter()
            .map(|&theta| ThetaTable {
                theta,
                sv: StateValues::Owned(vec![ev_remaining; NUM_STATES]),
                minimize: false,
            })
            .collect()
    }

    #[test]
    fn test_underdog_continuous_ramp() {
        let tables = fake_ramp_tables(100.0);
        let policy = UnderdogPolicy {
            ev_idx: 0, // θ=0 is at index 0
            theta_max: 0.05,
            scale: 50.0,
        };

        // Turn 0 → always ev (too early)
        let players = vec![
            PlayerState {
                total_score: 100,
                ..Default::default()
            },
            PlayerState {
                total_score: 50,
                ..Default::default()
            },
        ];
        let view = GameView {
            my_index: 1,
            players: &players,
            turn: 0,
        };
        assert_eq!(tables[policy.select_theta_index(&view, &tables)].theta, 0.0);

        // Leading → deficit ≤ 0 → θ = 0
        let players_leading = vec![
            PlayerState {
                total_score: 80,
                ..Default::default()
            },
            PlayerState {
                total_score: 120,
                ..Default::default()
            },
        ];
        let view_leading = GameView {
            my_index: 1,
            players: &players_leading,
            turn: 7,
        };
        assert_eq!(
            tables[policy.select_theta_index(&view_leading, &tables)].theta,
            0.0
        );

        // Trailing by 25 pts (same EV remaining): desired_θ = 0.05 * 25/50 = 0.025 → snap to 0.02 or 0.03
        let players_mid = vec![
            PlayerState {
                total_score: 125,
                ..Default::default()
            },
            PlayerState {
                total_score: 100,
                ..Default::default()
            },
        ];
        let view_mid = GameView {
            my_index: 1,
            players: &players_mid,
            turn: 7,
        };
        let theta_mid = tables[policy.select_theta_index(&view_mid, &tables)].theta;
        assert!(
            (theta_mid - 0.02).abs() < 0.011 || (theta_mid - 0.03).abs() < 0.011,
            "Expected θ near 0.025, got {}",
            theta_mid
        );

        // Trailing by 60 pts: desired_θ = 0.05 * 60/50 = 0.06 → clamped to 0.05 (θ_max)
        let players_badly = vec![
            PlayerState {
                total_score: 160,
                ..Default::default()
            },
            PlayerState {
                total_score: 100,
                ..Default::default()
            },
        ];
        let view_badly = GameView {
            my_index: 1,
            players: &players_badly,
            turn: 7,
        };
        assert_eq!(
            tables[policy.select_theta_index(&view_badly, &tables)].theta,
            0.05
        );

        // Catches up: deficit shrinks → θ decreases
        let players_close = vec![
            PlayerState {
                total_score: 105,
                ..Default::default()
            },
            PlayerState {
                total_score: 100,
                ..Default::default()
            },
        ];
        let view_close = GameView {
            my_index: 1,
            players: &players_close,
            turn: 7,
        };
        let theta_close = tables[policy.select_theta_index(&view_close, &tables)].theta;
        assert!(
            theta_close <= 0.01,
            "Expected θ near 0 for small deficit, got {}",
            theta_close
        );
    }

    #[test]
    fn test_underdog_ev_aware() {
        // Test that EV-remaining shifts the deficit.
        // Player 0: state (0,0) → sv[0]=100
        // Player 1: state (5,1) → sv[69]=50
        // Same total but different EV remaining → deficit = 50
        use crate::constants::NUM_STATES;
        let mut sv_data = vec![0.0f32; NUM_STATES];
        sv_data[0] = 100.0;
        sv_data[69] = 50.0;

        let tables: Vec<ThetaTable> = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
            .iter()
            .map(|&theta| ThetaTable {
                theta,
                sv: StateValues::Owned(sv_data.clone()),
                minimize: false,
            })
            .collect();

        let policy = UnderdogPolicy {
            ev_idx: 0,
            theta_max: 0.05,
            scale: 50.0,
        };

        // Same score but player 1 has worse EV remaining (50 vs 100)
        // → deficit = (100+100) - (100+50) = 50 → desired_θ = 0.05 * 50/50 = 0.05
        let players = vec![
            PlayerState {
                upper_score: 0,
                scored_categories: 0,
                total_score: 100,
            },
            PlayerState {
                upper_score: 5,
                scored_categories: 1,
                total_score: 100,
            },
        ];
        let view = GameView {
            my_index: 1,
            players: &players,
            turn: 10,
        };
        assert_eq!(
            tables[policy.select_theta_index(&view, &tables)].theta,
            0.05
        ); // max θ despite equal scores

        // Flip: player 1 has better EV remaining → deficit = -50 → θ = 0
        let players_flipped = vec![
            PlayerState {
                upper_score: 5,
                scored_categories: 1,
                total_score: 100,
            },
            PlayerState {
                upper_score: 0,
                scored_categories: 0,
                total_score: 100,
            },
        ];
        let view_flipped = GameView {
            my_index: 1,
            players: &players_flipped,
            turn: 10,
        };
        assert_eq!(
            tables[policy.select_theta_index(&view_flipped, &tables)].theta,
            0.0
        ); // θ=0 despite equal scores
    }

    #[test]
    fn test_trailing_risk_policy() {
        let policy = TrailingRisk {
            ev_idx: 0,
            high_idx: 1,
            threshold: 15,
        };

        // Not trailing enough → ev
        let players = vec![
            PlayerState {
                total_score: 100,
                ..Default::default()
            },
            PlayerState {
                total_score: 90,
                ..Default::default()
            },
        ];
        let view = GameView {
            my_index: 1,
            players: &players,
            turn: 5,
        };
        assert_eq!(policy.select_theta_index(&view, &[]), 0);

        // Trailing by 20 >= 15 → high
        let players2 = vec![
            PlayerState {
                total_score: 120,
                ..Default::default()
            },
            PlayerState {
                total_score: 100,
                ..Default::default()
            },
        ];
        let view2 = GameView {
            my_index: 1,
            players: &players2,
            turn: 5,
        };
        assert_eq!(policy.select_theta_index(&view2, &[]), 1);
    }
}
