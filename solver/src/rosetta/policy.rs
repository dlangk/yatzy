//! Rule-based policy evaluation via backward induction.
//!
//! Loads a skill ladder (JSON rule set) and evaluates it exactly by running
//! backward induction with the rule policy instead of argmax.

use crate::constants::*;
use crate::dice_mechanics::find_dice_set_index;
use crate::game_mechanics::update_upper_score;
use crate::rosetta::dsl::{compute_features, SemanticFeatures};
use crate::types::YatzyContext;

/// A single condition in a rule: feature_name op threshold.
#[derive(Debug, Clone)]
pub struct Condition {
    pub feature: String,
    pub op: CompareOp,
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A single rule in the decision list.
#[derive(Debug, Clone)]
pub struct Rule {
    pub conditions: Vec<Condition>,
    pub action: usize,
    pub decision_type: String,
    pub coverage: usize,
    pub mean_regret: f64,
}

/// The complete skill ladder: ordered rule lists per decision type.
#[derive(Debug, Clone)]
pub struct SkillLadder {
    pub category_rules: Vec<Rule>,
    pub reroll1_rules: Vec<Rule>,
    pub reroll2_rules: Vec<Rule>,
    pub default_category: usize,
    pub default_reroll1: usize,
    pub default_reroll2: usize,
}

impl SkillLadder {
    /// Parse from JSON value.
    pub fn from_json(json: &serde_json::Value) -> Result<Self, String> {
        let parse_rules = |key: &str| -> Result<Vec<Rule>, String> {
            let arr = json
                .get(key)
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("Missing or invalid '{}'", key))?;
            let mut rules = Vec::with_capacity(arr.len());
            for item in arr {
                let conditions = item
                    .get("conditions")
                    .and_then(|v| v.as_array())
                    .ok_or("Missing conditions")?
                    .iter()
                    .map(|c| {
                        Ok(Condition {
                            feature: c["feature"]
                                .as_str()
                                .ok_or("Missing feature")?
                                .to_string(),
                            op: match c["op"].as_str().ok_or("Missing op")? {
                                "==" | "eq" => CompareOp::Eq,
                                "!=" | "neq" => CompareOp::Neq,
                                "<" | "lt" => CompareOp::Lt,
                                "<=" | "le" => CompareOp::Le,
                                ">" | "gt" => CompareOp::Gt,
                                ">=" | "ge" => CompareOp::Ge,
                                other => return Err(format!("Unknown op: {}", other)),
                            },
                            threshold: c["threshold"]
                                .as_f64()
                                .ok_or("Missing threshold")?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let action = item["action"]
                    .as_u64()
                    .ok_or("Missing action")? as usize;
                rules.push(Rule {
                    conditions,
                    action,
                    decision_type: item
                        .get("decision_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    coverage: item.get("coverage").and_then(|v| v.as_u64()).unwrap_or(0)
                        as usize,
                    mean_regret: item
                        .get("mean_regret")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0),
                });
            }
            Ok(rules)
        };

        let get_default = |key: &str| -> usize {
            json.get("meta")
                .and_then(|m| m.get(key))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize
        };

        Ok(SkillLadder {
            category_rules: parse_rules("category_rules")?,
            reroll1_rules: parse_rules("reroll1_rules")?,
            reroll2_rules: parse_rules("reroll2_rules")?,
            default_category: get_default("default_category"),
            default_reroll1: get_default("default_reroll1"),
            default_reroll2: get_default("default_reroll2"),
        })
    }
}

/// Evaluate a condition against semantic features.
fn eval_condition(cond: &Condition, features: &SemanticFeatures) -> bool {
    let val = get_feature_value(&cond.feature, features);
    match cond.op {
        CompareOp::Eq => (val - cond.threshold).abs() < 1e-6,
        CompareOp::Neq => (val - cond.threshold).abs() >= 1e-6,
        CompareOp::Lt => val < cond.threshold,
        CompareOp::Le => val <= cond.threshold + 1e-9,
        CompareOp::Gt => val > cond.threshold,
        CompareOp::Ge => val >= cond.threshold - 1e-9,
    }
}

/// Get a named feature value from SemanticFeatures.
fn get_feature_value(name: &str, f: &SemanticFeatures) -> f64 {
    match name {
        "turn" => f.turn as f64,
        "categories_left" => f.categories_left as f64,
        "rerolls_remaining" => f.rerolls_remaining as f64,
        "upper_score" => f.upper_score as f64,
        "bonus_secured" => if f.bonus_secured { 1.0 } else { 0.0 },
        "bonus_pace" => f.bonus_pace as f64,
        "upper_cats_left" => f.upper_cats_left as f64,
        "max_count" => f.max_count as f64,
        "num_distinct" => f.num_distinct as f64,
        "dice_sum" => f.dice_sum as f64,
        "has_pair" => if f.has_pair { 1.0 } else { 0.0 },
        "has_two_pair" => if f.has_two_pair { 1.0 } else { 0.0 },
        "has_three_of_kind" => if f.has_three_of_kind { 1.0 } else { 0.0 },
        "has_four_of_kind" => if f.has_four_of_kind { 1.0 } else { 0.0 },
        "has_full_house" => if f.has_full_house { 1.0 } else { 0.0 },
        "has_small_straight" => if f.has_small_straight { 1.0 } else { 0.0 },
        "has_large_straight" => if f.has_large_straight { 1.0 } else { 0.0 },
        "has_yatzy" => if f.has_yatzy { 1.0 } else { 0.0 },
        "zeros_available" => f.zeros_available as f64,
        "best_available_score" => f.best_available_score as f64,
        // Face counts
        "face_count_1" => f.face_counts[0] as f64,
        "face_count_2" => f.face_counts[1] as f64,
        "face_count_3" => f.face_counts[2] as f64,
        "face_count_4" => f.face_counts[3] as f64,
        "face_count_5" => f.face_counts[4] as f64,
        "face_count_6" => f.face_counts[5] as f64,
        // Category availability (raw booleans, not normalized)
        s if s.starts_with("cat_avail_") => {
            let idx = cat_name_to_index(s.strip_prefix("cat_avail_").unwrap());
            if f.cat_available[idx] { 1.0 } else { 0.0 }
        }
        // Category scores (raw, not normalized)
        s if s.starts_with("cat_score_") => {
            let idx = cat_name_to_index(s.strip_prefix("cat_score_").unwrap());
            f.cat_scores[idx] as f64
        }
        _ => 0.0,
    }
}

fn cat_name_to_index(name: &str) -> usize {
    match name {
        "ones" => 0,
        "twos" => 1,
        "threes" => 2,
        "fours" => 3,
        "fives" => 4,
        "sixes" => 5,
        "one_pair" => 6,
        "two_pairs" => 7,
        "three_of_kind" | "three_of_a_kind" => 8,
        "four_of_kind" | "four_of_a_kind" => 9,
        "small_straight" => 10,
        "large_straight" => 11,
        "full_house" => 12,
        "chance" => 13,
        "yatzy" => 14,
        _ => 0,
    }
}

/// Apply a rule list to features, returning the chosen action or default.
fn apply_rules(rules: &[Rule], features: &SemanticFeatures, default: usize) -> usize {
    for rule in rules {
        if rule.conditions.iter().all(|c| eval_condition(c, features)) {
            return rule.action;
        }
    }
    default
}

/// Pick category using rule-based policy. Returns (category, score).
/// Falls back to the highest-scoring available category if the rule picks an unavailable one.
pub fn pick_category_by_rules(
    ctx: &YatzyContext,
    ladder: &SkillLadder,
    up_score: i32,
    scored: i32,
    dice: &[i32; 5],
    turn: usize,
) -> (usize, i32) {
    let features = compute_features(turn, up_score, scored, dice, 0);
    let chosen = apply_rules(&ladder.category_rules, &features, ladder.default_category);

    // Validate: category must be available
    if !is_category_scored(scored, chosen) {
        let ds_index = find_dice_set_index(ctx, dice);
        return (chosen, ctx.precomputed_scores[ds_index][chosen]);
    }

    // Fallback: pick first available category (this is the "else" branch)
    let ds_index = find_dice_set_index(ctx, dice);
    let mut best_cat = 0;
    let mut best_score = i32::MIN;
    for c in 0..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            if scr > best_score {
                best_score = scr;
                best_cat = c;
            }
        }
    }
    (best_cat, best_score)
}

/// Pick reroll mask using rule-based policy.
pub fn pick_reroll_by_rules(
    ladder: &SkillLadder,
    up_score: i32,
    scored: i32,
    dice: &[i32; 5],
    turn: usize,
    rerolls_remaining: u8,
) -> i32 {
    let features = compute_features(turn, up_score, scored, dice, rerolls_remaining);
    let (rules, default) = if rerolls_remaining == 2 {
        (&ladder.reroll1_rules, ladder.default_reroll1)
    } else {
        (&ladder.reroll2_rules, ladder.default_reroll2)
    };
    apply_rules(rules, &features, default) as i32
}

/// Simulate a single game using the rule-based policy. Returns total score.
pub fn simulate_game_with_rules(
    ctx: &YatzyContext,
    ladder: &SkillLadder,
    rng: &mut impl rand::Rng,
) -> i32 {
    use crate::dice_mechanics::sort_dice_set;

    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;

    for turn in 0..CATEGORY_COUNT {
        // Roll initial dice
        let mut dice = [0i32; 5];
        for d in &mut dice {
            *d = rng.random_range(1..=6);
        }
        sort_dice_set(&mut dice);

        // Reroll 1
        let mask1 = pick_reroll_by_rules(ladder, up_score, scored, &dice, turn, 2);
        if mask1 != 0 {
            for i in 0..5 {
                if mask1 & (1 << i) != 0 {
                    dice[i] = rng.random_range(1..=6);
                }
            }
            sort_dice_set(&mut dice);
        }

        // Reroll 2
        let mask2 = pick_reroll_by_rules(ladder, up_score, scored, &dice, turn, 1);
        if mask2 != 0 {
            for i in 0..5 {
                if mask2 & (1 << i) != 0 {
                    dice[i] = rng.random_range(1..=6);
                }
            }
            sort_dice_set(&mut dice);
        }

        // Pick category
        let (cat, scr) = pick_category_by_rules(ctx, ladder, up_score, scored, &dice, turn);
        total_score += scr;
        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
    }

    // Upper bonus
    if up_score >= 63 {
        total_score += 50;
    }

    total_score
}

/// Simulate using rule-based category selection but optimal reroll policy.
/// This isolates the quality of category rules from reroll rules.
pub fn simulate_game_category_rules_only(
    ctx: &YatzyContext,
    ladder: &SkillLadder,
    rng: &mut impl rand::Rng,
) -> i32 {
    use crate::dice_mechanics::sort_dice_set;
    use crate::widget_solver::{choose_best_reroll_mask, compute_max_ev_for_n_rerolls};

    let sv = ctx.state_values.as_slice();
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        // Roll initial dice
        let mut dice = [0i32; 5];
        for d in &mut dice {
            *d = rng.random_range(1..=6);
        }
        sort_dice_set(&mut dice);

        // Compute optimal reroll tables for this state
        // Group 6: best category EV for each dice set
        for ds_i in 0..252 {
            let mut best_val = f32::NEG_INFINITY;
            for c in 0..6 {
                if !is_category_scored(scored, c) {
                    let scr = ctx.precomputed_scores[ds_i][c];
                    let new_up = update_upper_score(up_score, c, scr);
                    let new_scored = scored | (1 << c);
                    let val = scr as f32
                        + unsafe {
                            *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                        };
                    if val > best_val {
                        best_val = val;
                    }
                }
            }
            for c in 6..CATEGORY_COUNT {
                if !is_category_scored(scored, c) {
                    let scr = ctx.precomputed_scores[ds_i][c];
                    let new_scored = scored | (1 << c);
                    let val = scr as f32
                        + unsafe {
                            *sv.get_unchecked(state_index(up_score as usize, new_scored as usize))
                        };
                    if val > best_val {
                        best_val = val;
                    }
                }
            }
            e_ds_0[ds_i] = best_val;
        }
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        // Optimal reroll 1
        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            for i in 0..5 {
                if mask1 & (1 << i) != 0 {
                    dice[i] = rng.random_range(1..=6);
                }
            }
            sort_dice_set(&mut dice);
        }

        // Optimal reroll 2
        let mask2 = choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            for i in 0..5 {
                if mask2 & (1 << i) != 0 {
                    dice[i] = rng.random_range(1..=6);
                }
            }
            sort_dice_set(&mut dice);
        }

        // Rule-based category selection
        let (cat, scr) = pick_category_by_rules(ctx, ladder, up_score, scored, &dice, turn);
        total_score += scr;
        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
    }

    // Upper bonus
    if up_score >= 63 {
        total_score += 50;
    }

    total_score
}
