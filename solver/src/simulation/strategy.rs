//! Strategy types for multiplayer simulation.
//!
//! Defines the observable game state ([`PlayerState`], [`GameView`]) and the
//! [`Strategy`] abstraction that maps CLI specs like `"ev"`, `"theta:0.05"`,
//! or `"mp:trailing"` to concrete table-lookup + policy combinations.

use std::path::Path;

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
    /// - `"adaptive:bonus"` / `"adaptive:phase"` / `"adaptive:combined"` — existing adaptive policies
    /// - `"mp:trailing"` — multiplayer-aware trailing-risk policy
    /// - `"mp:trailing:20"` — trailing-risk with custom threshold (default 15)
    pub fn from_spec(spec: &str, base_path: &Path, _ctx: &YatzyContext) -> Result<Self, String> {
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
                other => return Err(format!("Unknown multiplayer policy: {}", other)),
            }
        }

        Err(format!(
            "Unknown strategy spec: '{}'. Expected: ev, theta:<f>, adaptive:<name>, mp:trailing[:<threshold>]",
            spec
        ))
    }

    /// Resolve the turn configuration for this strategy given the game view.
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
