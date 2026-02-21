//! Rule-based policy evaluation via backward induction.
//!
//! Loads a skill ladder (JSON rule set) and evaluates it exactly by running
//! backward induction with the rule policy instead of argmax.

use crate::constants::*;
use crate::dice_mechanics::find_dice_set_index;
use crate::game_mechanics::update_upper_score;
use crate::rosetta::dsl::{compute_features, SemanticFeatures};
use crate::types::YatzyContext;

/// Semantic reroll actions — position-independent intents.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SemanticRerollAction {
    RerollAll,
    KeepFace1,
    KeepFace2,
    KeepFace3,
    KeepFace4,
    KeepFace5,
    KeepFace6,
    KeepPair,
    KeepTwoPairs,
    KeepTriple,
    KeepQuad,
    KeepTriplePlusHighest,
    KeepPairPlusKicker,
    KeepStraightDraw,
    KeepAll,
}

impl SemanticRerollAction {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Reroll_All" => Some(Self::RerollAll),
            "Keep_Face_1" => Some(Self::KeepFace1),
            "Keep_Face_2" => Some(Self::KeepFace2),
            "Keep_Face_3" => Some(Self::KeepFace3),
            "Keep_Face_4" => Some(Self::KeepFace4),
            "Keep_Face_5" => Some(Self::KeepFace5),
            "Keep_Face_6" => Some(Self::KeepFace6),
            "Keep_Pair" => Some(Self::KeepPair),
            "Keep_Two_Pairs" => Some(Self::KeepTwoPairs),
            "Keep_Triple" => Some(Self::KeepTriple),
            "Keep_Quad" => Some(Self::KeepQuad),
            "Keep_Triple_Plus_Highest" => Some(Self::KeepTriplePlusHighest),
            "Keep_Pair_Plus_Kicker" => Some(Self::KeepPairPlusKicker),
            "Keep_Straight_Draw" => Some(Self::KeepStraightDraw),
            "Keep_All" => Some(Self::KeepAll),
            _ => None,
        }
    }
}

/// Action types in the rule system.
#[derive(Debug, Clone)]
pub enum RuleAction {
    CategoryIndex(usize),
    SemanticReroll(SemanticRerollAction),
    BitmaskReroll(usize),
}

/// Compile a semantic reroll action to a bitmask for specific sorted dice.
///
/// Convention: bit i set = REROLL position i. Dice are sorted ascending.
/// Invalid actions fall back to 31 (Reroll All).
pub fn compile_to_mask(dice: &[i32; 5], action: SemanticRerollAction) -> i32 {
    match action {
        SemanticRerollAction::RerollAll => 31,
        SemanticRerollAction::KeepAll => 0,
        SemanticRerollAction::KeepFace1 => keep_face(dice, 1),
        SemanticRerollAction::KeepFace2 => keep_face(dice, 2),
        SemanticRerollAction::KeepFace3 => keep_face(dice, 3),
        SemanticRerollAction::KeepFace4 => keep_face(dice, 4),
        SemanticRerollAction::KeepFace5 => keep_face(dice, 5),
        SemanticRerollAction::KeepFace6 => keep_face(dice, 6),
        SemanticRerollAction::KeepPair => keep_pair(dice),
        SemanticRerollAction::KeepTwoPairs => keep_two_pairs(dice),
        SemanticRerollAction::KeepTriple => keep_triple(dice),
        SemanticRerollAction::KeepQuad => keep_quad(dice),
        SemanticRerollAction::KeepTriplePlusHighest => keep_triple_plus_highest(dice),
        SemanticRerollAction::KeepPairPlusKicker => keep_pair_plus_kicker(dice),
        SemanticRerollAction::KeepStraightDraw => keep_straight_draw(dice),
    }
}

fn face_counts(dice: &[i32; 5]) -> [u8; 6] {
    let mut counts = [0u8; 6];
    for &d in dice {
        counts[(d - 1) as usize] += 1;
    }
    counts
}

fn keep_face(dice: &[i32; 5], face: i32) -> i32 {
    let mut mask = 31i32;
    let mut found = false;
    for i in 0..5 {
        if dice[i] == face {
            mask &= !(1 << i);
            found = true;
        }
    }
    if found {
        mask
    } else {
        31
    }
}

/// Keep 2 of highest face with count≥2.
fn keep_pair(dice: &[i32; 5]) -> i32 {
    let counts = face_counts(dice);
    // Find highest face with count≥2
    let mut target = -1i32;
    for f in (0..6).rev() {
        if counts[f] >= 2 {
            target = f as i32 + 1;
            break;
        }
    }
    if target < 0 {
        return 31;
    }
    // Keep 2 rightmost positions with that face
    let mut mask = 31i32;
    let mut kept = 0;
    for i in (0..5).rev() {
        if dice[i] == target && kept < 2 {
            mask &= !(1 << i);
            kept += 1;
        }
    }
    mask
}

/// Keep two highest pair faces (2 of each).
fn keep_two_pairs(dice: &[i32; 5]) -> i32 {
    let counts = face_counts(dice);
    let mut pair_faces: Vec<i32> = Vec::new();
    for f in (0..6).rev() {
        if counts[f] >= 2 {
            pair_faces.push(f as i32 + 1);
        }
    }
    if pair_faces.len() < 2 {
        return 31;
    }
    let keep_faces = [pair_faces[0], pair_faces[1]];
    let mut mask = 31i32;
    for &face in &keep_faces {
        let mut kept = 0;
        for i in (0..5).rev() {
            if dice[i] == face && kept < 2 {
                mask &= !(1 << i);
                kept += 1;
            }
        }
    }
    mask
}

/// Keep 3 of highest face with count≥3.
fn keep_triple(dice: &[i32; 5]) -> i32 {
    let counts = face_counts(dice);
    let mut target = -1i32;
    for f in (0..6).rev() {
        if counts[f] >= 3 {
            target = f as i32 + 1;
            break;
        }
    }
    if target < 0 {
        return 31;
    }
    let mut mask = 31i32;
    let mut kept = 0;
    for i in (0..5).rev() {
        if dice[i] == target && kept < 3 {
            mask &= !(1 << i);
            kept += 1;
        }
    }
    mask
}

/// Keep 4 of highest face with count≥4.
fn keep_quad(dice: &[i32; 5]) -> i32 {
    let counts = face_counts(dice);
    let mut target = -1i32;
    for f in (0..6).rev() {
        if counts[f] >= 4 {
            target = f as i32 + 1;
            break;
        }
    }
    if target < 0 {
        return 31;
    }
    let mut mask = 31i32;
    let mut kept = 0;
    for i in (0..5).rev() {
        if dice[i] == target && kept < 4 {
            mask &= !(1 << i);
            kept += 1;
        }
    }
    mask
}

/// Keep triple + highest remaining die.
fn keep_triple_plus_highest(dice: &[i32; 5]) -> i32 {
    let counts = face_counts(dice);
    let mut target = -1i32;
    for f in (0..6).rev() {
        if counts[f] >= 3 {
            target = f as i32 + 1;
            break;
        }
    }
    if target < 0 {
        return 31;
    }
    let mut mask = 31i32;
    let mut kept = 0;
    let mut triple_positions = [false; 5];
    for i in (0..5).rev() {
        if dice[i] == target && kept < 3 {
            mask &= !(1 << i);
            triple_positions[i] = true;
            kept += 1;
        }
    }
    // Find highest remaining position (rightmost non-triple)
    for i in (0..5).rev() {
        if !triple_positions[i] {
            mask &= !(1 << i);
            break;
        }
    }
    if kept < 3 {
        return 31;
    }
    mask
}

/// Keep pair + highest remaining die.
fn keep_pair_plus_kicker(dice: &[i32; 5]) -> i32 {
    let counts = face_counts(dice);
    let mut target = -1i32;
    for f in (0..6).rev() {
        if counts[f] >= 2 {
            target = f as i32 + 1;
            break;
        }
    }
    if target < 0 {
        return 31;
    }
    let mut mask = 31i32;
    let mut kept = 0;
    let mut pair_positions = [false; 5];
    for i in (0..5).rev() {
        if dice[i] == target && kept < 2 {
            mask &= !(1 << i);
            pair_positions[i] = true;
            kept += 1;
        }
    }
    // Highest remaining
    for i in (0..5).rev() {
        if !pair_positions[i] {
            mask &= !(1 << i);
            break;
        }
    }
    mask
}

/// Keep longest consecutive run of unique faces (1 die per face).
fn keep_straight_draw(dice: &[i32; 5]) -> i32 {
    // Collect unique faces sorted
    let mut unique = Vec::with_capacity(5);
    for &d in dice {
        if unique.last() != Some(&d) {
            unique.push(d);
        }
    }
    if unique.len() < 2 {
        return 31;
    }
    // Find longest consecutive run
    let mut best_start = 0;
    let mut best_len = 1;
    let mut run_start = 0;
    let mut run_len = 1;
    for i in 1..unique.len() {
        if unique[i] == unique[i - 1] + 1 {
            run_len += 1;
        } else {
            if run_len > best_len {
                best_len = run_len;
                best_start = run_start;
            }
            run_start = i;
            run_len = 1;
        }
    }
    if run_len > best_len {
        best_len = run_len;
        best_start = run_start;
    }
    if best_len < 2 {
        return 31;
    }
    let run_faces: Vec<i32> = unique[best_start..best_start + best_len].to_vec();
    // Keep 1 die per face in the run
    let mut mask = 31i32;
    let mut used = [false; 7]; // face 1..6
    for i in 0..5 {
        let f = dice[i] as usize;
        if run_faces.contains(&dice[i]) && !used[f] {
            mask &= !(1 << i);
            used[f] = true;
        }
    }
    mask
}

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
    pub action: RuleAction,
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
    pub default_category: RuleAction,
    pub default_reroll1: RuleAction,
    pub default_reroll2: RuleAction,
}

impl SkillLadder {
    /// Parse from JSON value.
    pub fn from_json(json: &serde_json::Value) -> Result<Self, String> {
        let parse_rules = |key: &str| -> Result<Vec<Rule>, String> {
            let is_reroll = key.starts_with("reroll");
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
                            feature: c["feature"].as_str().ok_or("Missing feature")?.to_string(),
                            op: match c["op"].as_str().ok_or("Missing op")? {
                                "==" | "eq" => CompareOp::Eq,
                                "!=" | "neq" => CompareOp::Neq,
                                "<" | "lt" => CompareOp::Lt,
                                "<=" | "le" => CompareOp::Le,
                                ">" | "gt" => CompareOp::Gt,
                                ">=" | "ge" => CompareOp::Ge,
                                other => return Err(format!("Unknown op: {}", other)),
                            },
                            threshold: c["threshold"].as_f64().ok_or("Missing threshold")?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let action = parse_action(&item["action"], is_reroll)?;
                rules.push(Rule {
                    conditions,
                    action,
                    decision_type: item
                        .get("decision_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    coverage: item.get("coverage").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                    mean_regret: item
                        .get("mean_regret")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0),
                });
            }
            Ok(rules)
        };

        let get_default_action = |key: &str, is_reroll: bool| -> RuleAction {
            let meta_val = json.get("meta").and_then(|m| m.get(key));
            match meta_val {
                Some(v) if v.is_string() => {
                    let s = v.as_str().unwrap();
                    if let Some(sem) = SemanticRerollAction::parse(s) {
                        RuleAction::SemanticReroll(sem)
                    } else if is_reroll {
                        RuleAction::BitmaskReroll(0)
                    } else {
                        RuleAction::CategoryIndex(0)
                    }
                }
                Some(v) if v.is_u64() => {
                    let idx = v.as_u64().unwrap() as usize;
                    if is_reroll {
                        RuleAction::BitmaskReroll(idx)
                    } else {
                        RuleAction::CategoryIndex(idx)
                    }
                }
                _ => {
                    if is_reroll {
                        RuleAction::BitmaskReroll(0)
                    } else {
                        RuleAction::CategoryIndex(0)
                    }
                }
            }
        };

        Ok(SkillLadder {
            category_rules: parse_rules("category_rules")?,
            reroll1_rules: parse_rules("reroll1_rules")?,
            reroll2_rules: parse_rules("reroll2_rules")?,
            default_category: get_default_action("default_category", false),
            default_reroll1: get_default_action("default_reroll1", true),
            default_reroll2: get_default_action("default_reroll2", true),
        })
    }
}

/// Parse an action field from JSON — string (semantic) or integer (category/bitmask).
fn parse_action(val: &serde_json::Value, is_reroll: bool) -> Result<RuleAction, String> {
    if let Some(s) = val.as_str() {
        if let Some(sem) = SemanticRerollAction::parse(s) {
            return Ok(RuleAction::SemanticReroll(sem));
        }
        return Err(format!("Unknown semantic action: {}", s));
    }
    if let Some(n) = val.as_u64() {
        let idx = n as usize;
        if is_reroll {
            Ok(RuleAction::BitmaskReroll(idx))
        } else {
            Ok(RuleAction::CategoryIndex(idx))
        }
    } else {
        Err("Action must be string or integer".to_string())
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
        "bonus_secured" => {
            if f.bonus_secured {
                1.0
            } else {
                0.0
            }
        }
        "bonus_pace" => f.bonus_pace as f64,
        "upper_cats_left" => f.upper_cats_left as f64,
        "max_count" => f.max_count as f64,
        "num_distinct" => f.num_distinct as f64,
        "dice_sum" => f.dice_sum as f64,
        "has_pair" => {
            if f.has_pair {
                1.0
            } else {
                0.0
            }
        }
        "has_two_pair" => {
            if f.has_two_pair {
                1.0
            } else {
                0.0
            }
        }
        "has_three_of_kind" => {
            if f.has_three_of_kind {
                1.0
            } else {
                0.0
            }
        }
        "has_four_of_kind" => {
            if f.has_four_of_kind {
                1.0
            } else {
                0.0
            }
        }
        "has_full_house" => {
            if f.has_full_house {
                1.0
            } else {
                0.0
            }
        }
        "has_small_straight" => {
            if f.has_small_straight {
                1.0
            } else {
                0.0
            }
        }
        "has_large_straight" => {
            if f.has_large_straight {
                1.0
            } else {
                0.0
            }
        }
        "has_yatzy" => {
            if f.has_yatzy {
                1.0
            } else {
                0.0
            }
        }
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
            if f.cat_available[idx] {
                1.0
            } else {
                0.0
            }
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
fn apply_rules(rules: &[Rule], features: &SemanticFeatures, default: &RuleAction) -> RuleAction {
    for rule in rules {
        if rule.conditions.iter().all(|c| eval_condition(c, features)) {
            return rule.action.clone();
        }
    }
    default.clone()
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
    let action = apply_rules(&ladder.category_rules, &features, &ladder.default_category);

    let chosen = match action {
        RuleAction::CategoryIndex(idx) => idx,
        _ => 0, // fallback
    };

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
        (&ladder.reroll1_rules, &ladder.default_reroll1)
    } else {
        (&ladder.reroll2_rules, &ladder.default_reroll2)
    };
    let action = apply_rules(rules, &features, default);
    match action {
        RuleAction::SemanticReroll(sem) => compile_to_mask(dice, sem),
        RuleAction::BitmaskReroll(m) => m as i32,
        RuleAction::CategoryIndex(_) => 31, // shouldn't happen — reroll all
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reroll_all() {
        assert_eq!(
            compile_to_mask(&[1, 2, 3, 4, 5], SemanticRerollAction::RerollAll),
            31
        );
    }

    #[test]
    fn test_keep_all() {
        assert_eq!(
            compile_to_mask(&[1, 2, 3, 4, 5], SemanticRerollAction::KeepAll),
            0
        );
    }

    #[test]
    fn test_keep_face() {
        // dice=[1,1,3,5,5], Keep_Face_5 → keep positions 3,4 → mask=0b00111=7
        assert_eq!(
            compile_to_mask(&[1, 1, 3, 5, 5], SemanticRerollAction::KeepFace5),
            0b00111
        );
        // Keep_Face_1 → keep positions 0,1 → mask=0b11100=28
        assert_eq!(
            compile_to_mask(&[1, 1, 3, 5, 5], SemanticRerollAction::KeepFace1),
            0b11100
        );
        // Keep_Face_4 on dice with no 4s → reroll all
        assert_eq!(
            compile_to_mask(&[1, 1, 3, 5, 5], SemanticRerollAction::KeepFace4),
            31
        );
    }

    #[test]
    fn test_keep_pair() {
        // dice=[1,1,3,5,5] → highest pair=5s at positions 3,4 → mask=0b00111=7
        assert_eq!(
            compile_to_mask(&[1, 1, 3, 5, 5], SemanticRerollAction::KeepPair),
            0b00111
        );
        // dice=[2,3,4,5,6] → no pair → 31
        assert_eq!(
            compile_to_mask(&[2, 3, 4, 5, 6], SemanticRerollAction::KeepPair),
            31
        );
    }

    #[test]
    fn test_keep_two_pairs() {
        // dice=[1,1,3,5,5] → pairs are 5,1 → keep 5s at 3,4 and 1s at 0,1 → mask=0b00100=4
        assert_eq!(
            compile_to_mask(&[1, 1, 3, 5, 5], SemanticRerollAction::KeepTwoPairs),
            0b00100
        );
        // dice=[1,2,3,4,5] → only 0 pairs → 31
        assert_eq!(
            compile_to_mask(&[1, 2, 3, 4, 5], SemanticRerollAction::KeepTwoPairs),
            31
        );
    }

    #[test]
    fn test_keep_triple() {
        // dice=[2,3,3,3,6] → triple of 3s at positions 1,2,3 → mask=0b10001=17
        assert_eq!(
            compile_to_mask(&[2, 3, 3, 3, 6], SemanticRerollAction::KeepTriple),
            0b10001
        );
        // dice=[1,2,3,4,5] → no triple → 31
        assert_eq!(
            compile_to_mask(&[1, 2, 3, 4, 5], SemanticRerollAction::KeepTriple),
            31
        );
    }

    #[test]
    fn test_keep_quad() {
        // dice=[3,3,3,3,6] → quad of 3s at 0,1,2,3 → mask=0b10000=16
        assert_eq!(
            compile_to_mask(&[3, 3, 3, 3, 6], SemanticRerollAction::KeepQuad),
            0b10000
        );
        // dice=[2,3,3,3,6] → no quad → 31
        assert_eq!(
            compile_to_mask(&[2, 3, 3, 3, 6], SemanticRerollAction::KeepQuad),
            31
        );
    }

    #[test]
    fn test_keep_triple_plus_highest() {
        // dice=[2,3,3,3,6] → triple 3s at 1,2,3 + highest remaining is 6 at pos 4
        // → keep 1,2,3,4 → mask=0b00001=1
        assert_eq!(
            compile_to_mask(
                &[2, 3, 3, 3, 6],
                SemanticRerollAction::KeepTriplePlusHighest
            ),
            0b00001
        );
    }

    #[test]
    fn test_keep_pair_plus_kicker() {
        // dice=[1,1,3,5,5] → pair 5s at 3,4 + highest remaining = 3 at pos 2
        // → keep 2,3,4 → mask=0b00011=3
        assert_eq!(
            compile_to_mask(&[1, 1, 3, 5, 5], SemanticRerollAction::KeepPairPlusKicker),
            0b00011
        );
    }

    #[test]
    fn test_keep_straight_draw() {
        // dice=[1,2,3,5,6] → longest run is [1,2,3] (len 3)
        // → keep 1 die per face: pos 0(1), 1(2), 2(3) → mask=0b11000=24
        assert_eq!(
            compile_to_mask(&[1, 2, 3, 5, 6], SemanticRerollAction::KeepStraightDraw),
            0b11000
        );
        // dice=[2,3,4,5,6] → run is [2,3,4,5,6] (len 5) → keep all → mask=0
        assert_eq!(
            compile_to_mask(&[2, 3, 4, 5, 6], SemanticRerollAction::KeepStraightDraw),
            0
        );
    }

    #[test]
    fn test_semantic_from_str() {
        assert_eq!(
            SemanticRerollAction::parse("Keep_Triple"),
            Some(SemanticRerollAction::KeepTriple)
        );
        assert_eq!(
            SemanticRerollAction::parse("Reroll_All"),
            Some(SemanticRerollAction::RerollAll)
        );
        assert_eq!(SemanticRerollAction::parse("Unknown"), None);
    }
}
