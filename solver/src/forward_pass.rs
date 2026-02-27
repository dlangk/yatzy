//! Exact forward pass — state visitation probabilities under optimal play (θ=0).
//!
//! Propagates a single f64 per state (visitation probability) through the game DAG
//! using the PolicyOracle for O(1) decision lookups. Zero Monte Carlo, zero variance.
//!
//! Produces three D3.js-ready JSON structures:
//! - **Race to 63**: node-link graph of upper-score transitions per turn
//! - **Category Sankey**: (turn, category) → probability mass
//! - **EV Funnel**: (turn, ev_bin) → probability mass

use std::collections::HashMap;

use rayon::prelude::*;
use serde::Serialize;

use crate::constants::*;
use crate::density::transitions::compute_transitions_oracle;
use crate::types::{PolicyOracle, YatzyContext};

// ── Result types ────────────────────────────────────────────────────────────

/// Complete forward pass result with three aggregation structures.
pub struct ForwardPassResult {
    pub race_to_63: RaceTo63,
    pub category_sankey: Vec<CategorySankeyEntry>,
    pub ev_funnel: Vec<EvFunnelEntry>,
}

/// Node-link graph for Race to 63 Sankey visualization.
#[derive(Serialize)]
pub struct RaceTo63 {
    pub nodes: Vec<RaceTo63Node>,
    pub links: Vec<RaceTo63Link>,
}

#[derive(Serialize)]
pub struct RaceTo63Node {
    pub id: String,
    pub turn: u8,
    pub upper_score: u8,
    pub mass: f64,
}

#[derive(Serialize)]
pub struct RaceTo63Link {
    pub source: String,
    pub target: String,
    pub mass: f64,
}

/// Category Sankey: probability of choosing each category at each turn.
#[derive(Serialize)]
pub struct CategorySankeyEntry {
    pub turn: u8,
    pub category: &'static str,
    pub category_index: u8,
    pub mass: f64,
}

/// EV Funnel: probability mass at each EV bin per turn.
#[derive(Serialize)]
pub struct EvFunnelEntry {
    pub turn: u8,
    pub ev_bin: i32,
    pub mass: f64,
}

// ── Core algorithm ──────────────────────────────────────────────────────────

/// Run exact forward pass with oracle transitions.
///
/// Propagates visitation probabilities from the start state (upper=0, scored=0)
/// through all 15 turns under optimal play, collecting three aggregation structures.
pub fn forward_pass(
    ctx: &YatzyContext,
    oracle: &PolicyOracle,
    state_values: &[f32],
) -> ForwardPassResult {
    let mut p_table = vec![0.0f64; NUM_STATES];
    p_table[state_index(0, 0)] = 1.0;

    // Aggregation accumulators
    // Race to 63: (turn, src_up, dst_up) → mass
    let mut race_links: HashMap<(u8, u8, u8), f64> = HashMap::new();
    // Node masses: (turn, up) → mass
    let mut race_nodes: HashMap<(u8, u8), f64> = HashMap::new();
    race_nodes.insert((0, 0), 1.0);

    // Category Sankey: (turn, cat) → mass
    let mut cat_mass: HashMap<(u8, u8), f64> = HashMap::new();

    // EV Funnel: (turn, ev_bin) → mass
    let mut ev_mass: HashMap<(u8, i32), f64> = HashMap::new();

    for num_scored in 0..CATEGORY_COUNT as u32 {
        let turn = num_scored as u8;
        println!("  Turn {}/15 (popcount={})", turn, num_scored);

        // Collect active states: all (si, scored, up) with p > 0 and correct popcount
        let mut active: Vec<(usize, i32, i32)> = Vec::new();
        for scored_mask in 0..32768u32 {
            if scored_mask.count_ones() != num_scored {
                continue;
            }
            let scored = scored_mask as i32;
            for up in 0..64i32 {
                let si = state_index(up as usize, scored as usize);
                if p_table[si] > 0.0 {
                    active.push((si, scored, up));
                }
            }
        }

        println!("    {} active states", active.len());

        // Aggregate EV Funnel: bin state_values[si] for each active state
        for &(si, _scored, _up) in &active {
            let ev = state_values[si] as f64;
            let ev_bin = ev.round() as i32;
            *ev_mass.entry((turn, ev_bin)).or_insert(0.0) += p_table[si];
        }

        // Compute transitions in parallel
        let transitions: Vec<(usize, Vec<crate::density::transitions::StateTransition>)> = active
            .par_iter()
            .map(|&(si, scored, up)| {
                let trans = compute_transitions_oracle(ctx, oracle, up, scored);
                (si, trans)
            })
            .collect();

        // Apply transitions sequentially (multiple sources can write to same dest)
        for (si, trans) in &transitions {
            let src_prob = p_table[*si];
            let src_scored = *si / STATE_STRIDE;
            let src_up = *si % STATE_STRIDE;

            for t in trans {
                let mass = src_prob * t.prob;
                let dst_si = t.next_state as usize;
                p_table[dst_si] += mass;

                // Race to 63: track upper score transitions
                let dst_up = (dst_si % STATE_STRIDE) as u8;
                let src_up_u8 = src_up as u8;
                *race_links.entry((turn, src_up_u8, dst_up)).or_insert(0.0) += mass;
                *race_nodes.entry((turn + 1, dst_up)).or_insert(0.0) += mass;

                // Category Sankey: derive category from scored bitmask change
                let dst_scored = dst_si / STATE_STRIDE;
                let diff = (dst_scored as i32) ^ (src_scored as i32);
                let cat = diff.trailing_zeros() as u8;
                *cat_mass.entry((turn, cat)).or_insert(0.0) += mass;
            }
        }
    }

    // Aggregate EV Funnel for final turn (turn 15 = all categories scored)
    {
        let turn = CATEGORY_COUNT as u8;
        let all_scored = (1u32 << CATEGORY_COUNT) - 1;
        for up in 0..64i32 {
            let si = state_index(up as usize, all_scored as usize);
            if p_table[si] > 0.0 {
                let ev = state_values[si] as f64;
                let ev_bin = ev.round() as i32;
                *ev_mass.entry((turn, ev_bin)).or_insert(0.0) += p_table[si];
            }
        }
    }

    // Build Race to 63 output
    let mut nodes: Vec<RaceTo63Node> = race_nodes
        .into_iter()
        .filter(|&(_, mass)| mass > 1e-10)
        .map(|((turn, up), mass)| RaceTo63Node {
            id: format!("{}-{}", turn, up),
            turn,
            upper_score: up,
            mass,
        })
        .collect();
    nodes.sort_by(|a, b| a.turn.cmp(&b.turn).then(a.upper_score.cmp(&b.upper_score)));

    let mut links: Vec<RaceTo63Link> = race_links
        .into_iter()
        .filter(|&(_, mass)| mass > 1e-10)
        .map(|((turn, src, dst), mass)| RaceTo63Link {
            source: format!("{}-{}", turn, src),
            target: format!("{}-{}", turn + 1, dst),
            mass,
        })
        .collect();
    links.sort_by(|a, b| a.source.cmp(&b.source).then(a.target.cmp(&b.target)));

    let race_to_63 = RaceTo63 { nodes, links };

    // Build Category Sankey output
    let mut category_sankey: Vec<CategorySankeyEntry> = cat_mass
        .into_iter()
        .filter(|&(_, mass)| mass > 1e-10)
        .map(|((turn, cat), mass)| CategorySankeyEntry {
            turn,
            category: CATEGORY_NAMES[cat as usize],
            category_index: cat,
            mass,
        })
        .collect();
    category_sankey.sort_by(|a, b| {
        a.turn
            .cmp(&b.turn)
            .then(a.category_index.cmp(&b.category_index))
    });

    // Build EV Funnel output
    let mut ev_funnel: Vec<EvFunnelEntry> = ev_mass
        .into_iter()
        .filter(|&(_, mass)| mass > 1e-10)
        .map(|((turn, ev_bin), mass)| EvFunnelEntry { turn, ev_bin, mass })
        .collect();
    ev_funnel.sort_by(|a, b| a.turn.cmp(&b.turn).then(a.ev_bin.cmp(&b.ev_bin)));

    ForwardPassResult {
        race_to_63,
        category_sankey,
        ev_funnel,
    }
}
