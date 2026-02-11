//! Statistics aggregation from simulation records.
//!
//! Computes comprehensive game statistics from raw `GameRecord` data:
//! total score distribution, per-category breakdowns, upper section analysis,
//! and reroll usage patterns.

use serde::Serialize;
use std::collections::BTreeMap;

use crate::constants::{CATEGORY_COUNT, CATEGORY_NAMES};

use super::engine::GameRecord;

/// Maximum score per category (Ones..Sixes, OnePair..Yatzy).
const CATEGORY_MAX_SCORES: [u8; CATEGORY_COUNT] = [
    5, 10, 15, 20, 25, 30, // Ones–Sixes
    12, 22, 18, 24, 15, 20, 28, 30, 50, // One Pair–Yatzy
];

// ── Top-level statistics ────────────────────────────────────────────

#[derive(Serialize)]
pub struct GameStatistics {
    pub num_games: u64,
    pub seed: u64,
    pub expected_value: f64,
    pub total_score: ScoreDistribution,
    pub categories: Vec<CategoryStatistics>,
    pub upper_section: UpperSectionStatistics,
    pub rerolls: RerollStatistics,
}

// ── Total score distribution ────────────────────────────────────────

#[derive(Serialize)]
pub struct ScoreDistribution {
    pub mean: f64,
    pub std_dev: f64,
    pub min: u16,
    pub max: u16,
    pub median: u16,
    pub percentiles: Percentiles,
    /// 5-point bins from 80 to 380 (60 bins).
    pub histogram: Vec<HistogramBin>,
    /// P(score >= X) for each threshold.
    pub cumulative: Vec<CumulativeEntry>,
    /// Named thresholds: P(score >= 200), etc.
    pub thresholds: Vec<ThresholdEntry>,
}

#[derive(Serialize)]
pub struct Percentiles {
    pub p5: u16,
    pub p10: u16,
    pub p25: u16,
    pub p50: u16,
    pub p75: u16,
    pub p90: u16,
    pub p95: u16,
    pub p99: u16,
}

#[derive(Serialize)]
pub struct HistogramBin {
    pub lower: u16,
    pub upper: u16,
    pub count: u32,
}

#[derive(Serialize)]
pub struct CumulativeEntry {
    pub score: u16,
    pub probability: f64,
}

#[derive(Serialize)]
pub struct ThresholdEntry {
    pub score: u16,
    pub probability: f64,
}

// ── Per-category statistics ─────────────────────────────────────────

#[derive(Serialize)]
pub struct CategoryStatistics {
    pub id: usize,
    pub name: String,
    pub max_score: u8,
    pub mean_score: f64,
    pub std_dev: f64,
    pub zero_rate: f64,
    pub max_rate: f64,
    /// Sparse score distribution: only scores that actually occur.
    pub score_distribution: BTreeMap<u8, f64>,
    /// Mean turn number (0-indexed) when this category is scored.
    pub mean_turn_scored: f64,
    /// Distribution of turn numbers when scored.
    pub turn_distribution: Vec<f64>,
}

// ── Upper section statistics ────────────────────────────────────────

#[derive(Serialize)]
pub struct UpperSectionStatistics {
    pub mean_upper_total: f64,
    pub bonus_rate: f64,
    /// Distribution of upper totals [0..=63].
    pub upper_total_distribution: Vec<UpperTotalEntry>,
    pub per_category: Vec<UpperCategoryEntry>,
}

#[derive(Serialize)]
pub struct UpperTotalEntry {
    pub upper_total: u8,
    pub probability: f64,
}

#[derive(Serialize)]
pub struct UpperCategoryEntry {
    pub id: usize,
    pub name: String,
    pub mean_score: f64,
    pub par: u8,
    pub above_par_rate: f64,
}

// ── Reroll statistics ───────────────────────────────────────────────

#[derive(Serialize)]
pub struct RerollStatistics {
    /// Mean number of rerolls per game (0-30).
    pub mean_per_game: f64,
    /// Distribution of rerolls per game.
    pub distribution: Vec<RerollDistEntry>,
}

#[derive(Serialize)]
pub struct RerollDistEntry {
    pub rerolls: u8,
    pub probability: f64,
}

// ── Aggregation ─────────────────────────────────────────────────────

/// Aggregate statistics from a slice of GameRecords.
pub fn aggregate_statistics(
    records: &[GameRecord],
    expected_value: f64,
    seed: u64,
) -> GameStatistics {
    let n = records.len() as f64;
    let num_games = records.len() as u64;

    // ── Total scores ────────────────────────────────────────────
    let mut sorted_scores: Vec<u16> = records.iter().map(|r| r.total_score).collect();
    sorted_scores.sort_unstable();

    let sum: f64 = sorted_scores.iter().map(|&s| s as f64).sum();
    let mean = sum / n;
    let variance: f64 = sorted_scores
        .iter()
        .map(|&s| (s as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let std_dev = variance.sqrt();

    let percentile = |p: f64| -> u16 {
        let idx = ((p / 100.0) * (sorted_scores.len() - 1) as f64).round() as usize;
        sorted_scores[idx.min(sorted_scores.len() - 1)]
    };

    let percentiles = Percentiles {
        p5: percentile(5.0),
        p10: percentile(10.0),
        p25: percentile(25.0),
        p50: percentile(50.0),
        p75: percentile(75.0),
        p90: percentile(90.0),
        p95: percentile(95.0),
        p99: percentile(99.0),
    };

    // Histogram: 5-point bins from 80 to 380
    let hist_min = 80u16;
    let hist_max = 380u16;
    let bin_width = 5u16;
    let num_bins = ((hist_max - hist_min) / bin_width) as usize;
    let mut hist_counts = vec![0u32; num_bins];
    for &s in &sorted_scores {
        if s >= hist_min && s < hist_max {
            let bin = ((s - hist_min) / bin_width) as usize;
            if bin < num_bins {
                hist_counts[bin] += 1;
            }
        }
    }
    let histogram: Vec<HistogramBin> = (0..num_bins)
        .map(|i| HistogramBin {
            lower: hist_min + (i as u16) * bin_width,
            upper: hist_min + (i as u16 + 1) * bin_width,
            count: hist_counts[i],
        })
        .collect();

    // Cumulative P(score >= X) at 10-point intervals
    let cumulative: Vec<CumulativeEntry> = (80..=380)
        .step_by(10)
        .map(|threshold| {
            let count = sorted_scores.iter().filter(|&&s| s >= threshold).count();
            CumulativeEntry {
                score: threshold,
                probability: count as f64 / n,
            }
        })
        .collect();

    // Named thresholds
    let thresholds: Vec<ThresholdEntry> = [200, 250, 300, 350]
        .iter()
        .map(|&threshold| {
            let count = sorted_scores.iter().filter(|&&s| s >= threshold).count();
            ThresholdEntry {
                score: threshold,
                probability: count as f64 / n,
            }
        })
        .collect();

    let total_score = ScoreDistribution {
        mean,
        std_dev,
        min: sorted_scores[0],
        max: *sorted_scores.last().unwrap(),
        median: percentile(50.0),
        percentiles,
        histogram,
        cumulative,
        thresholds,
    };

    // ── Per-category statistics ─────────────────────────────────
    let mut cat_scores: Vec<Vec<u8>> = (0..CATEGORY_COUNT)
        .map(|_| Vec::with_capacity(records.len()))
        .collect();
    let mut cat_turns: Vec<Vec<u8>> = (0..CATEGORY_COUNT)
        .map(|_| Vec::with_capacity(records.len()))
        .collect();

    for record in records {
        for (turn_idx, turn) in record.turns.iter().enumerate() {
            let cat = turn.category as usize;
            cat_scores[cat].push(turn.score);
            cat_turns[cat].push(turn_idx as u8);
        }
    }

    let categories: Vec<CategoryStatistics> = (0..CATEGORY_COUNT)
        .map(|c| {
            let scores = &cat_scores[c];
            let turns = &cat_turns[c];
            let max_score = CATEGORY_MAX_SCORES[c];

            let sum: f64 = scores.iter().map(|&s| s as f64).sum();
            let cat_mean = sum / scores.len().max(1) as f64;
            let cat_var: f64 = scores
                .iter()
                .map(|&s| (s as f64 - cat_mean).powi(2))
                .sum::<f64>()
                / scores.len().max(1) as f64;

            let zero_count = scores.iter().filter(|&&s| s == 0).count();
            let max_count = scores.iter().filter(|&&s| s == max_score).count();

            // Score distribution (sparse)
            let mut score_counts: BTreeMap<u8, u32> = BTreeMap::new();
            for &s in scores {
                *score_counts.entry(s).or_insert(0) += 1;
            }
            let score_distribution: BTreeMap<u8, f64> = score_counts
                .into_iter()
                .map(|(s, c)| (s, c as f64 / n))
                .collect();

            // Turn distribution
            let turn_sum: f64 = turns.iter().map(|&t| t as f64).sum();
            let mean_turn = turn_sum / turns.len().max(1) as f64;
            let mut turn_counts = [0u32; CATEGORY_COUNT];
            for &t in turns {
                turn_counts[t as usize] += 1;
            }
            let turn_distribution: Vec<f64> = turn_counts.iter().map(|&c| c as f64 / n).collect();

            CategoryStatistics {
                id: c,
                name: CATEGORY_NAMES[c].to_string(),
                max_score,
                mean_score: cat_mean,
                std_dev: cat_var.sqrt(),
                zero_rate: zero_count as f64 / n,
                max_rate: max_count as f64 / n,
                score_distribution,
                mean_turn_scored: mean_turn,
                turn_distribution,
            }
        })
        .collect();

    // ── Upper section statistics ────────────────────────────────
    let upper_sum: f64 = records.iter().map(|r| r.upper_total as f64).sum();
    let mean_upper = upper_sum / n;

    let bonus_count = records.iter().filter(|r| r.got_bonus != 0).count();
    let bonus_rate = bonus_count as f64 / n;

    // Upper total distribution
    let mut upper_dist = vec![0u32; 64];
    for record in records {
        upper_dist[record.upper_total as usize] += 1;
    }
    let upper_total_distribution: Vec<UpperTotalEntry> = upper_dist
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(u, &c)| UpperTotalEntry {
            upper_total: u as u8,
            probability: c as f64 / n,
        })
        .collect();

    // Par values: 3 of each face (3*1=3, 3*2=6, ..., 3*6=18)
    let par_values: [u8; 6] = [3, 6, 9, 12, 15, 18];

    let per_category: Vec<UpperCategoryEntry> = (0..6)
        .map(|c| {
            let scores = &cat_scores[c];
            let sum: f64 = scores.iter().map(|&s| s as f64).sum();
            let cat_mean = sum / scores.len().max(1) as f64;
            let par = par_values[c];
            let above_par = scores.iter().filter(|&&s| s > par).count();

            UpperCategoryEntry {
                id: c,
                name: CATEGORY_NAMES[c].to_string(),
                mean_score: cat_mean,
                par,
                above_par_rate: above_par as f64 / n,
            }
        })
        .collect();

    let upper_section = UpperSectionStatistics {
        mean_upper_total: mean_upper,
        bonus_rate,
        upper_total_distribution,
        per_category,
    };

    // ── Reroll statistics ───────────────────────────────────────
    let mut reroll_counts: Vec<u8> = Vec::with_capacity(records.len());
    for record in records {
        let mut count = 0u8;
        for turn in &record.turns {
            if turn.mask1 != 0 {
                count += 1;
            }
            if turn.mask2 != 0 {
                count += 1;
            }
        }
        reroll_counts.push(count);
    }

    let reroll_sum: f64 = reroll_counts.iter().map(|&c| c as f64).sum();
    let mean_rerolls = reroll_sum / n;

    let max_rerolls = *reroll_counts.iter().max().unwrap_or(&0) as usize;
    let mut reroll_dist = vec![0u32; max_rerolls + 1];
    for &c in &reroll_counts {
        reroll_dist[c as usize] += 1;
    }
    let distribution: Vec<RerollDistEntry> = reroll_dist
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(r, &c)| RerollDistEntry {
            rerolls: r as u8,
            probability: c as f64 / n,
        })
        .collect();

    let rerolls = RerollStatistics {
        mean_per_game: mean_rerolls,
        distribution,
    };

    GameStatistics {
        num_games,
        seed,
        expected_value,
        total_score,
        categories,
        upper_section,
        rerolls,
    }
}

/// Save aggregated statistics as JSON.
pub fn save_statistics(stats: &GameStatistics, path: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(stats).expect("Failed to serialize statistics");
    std::fs::write(path, json).expect("Failed to write statistics file");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::engine::{GameRecord, TurnRecord};

    fn make_test_records(n: usize) -> Vec<GameRecord> {
        (0..n)
            .map(|i| {
                let mut rec = GameRecord::default();
                let mut score_total = 0u16;
                let mut upper = 0u8;
                for t in 0..15 {
                    let cat = t as u8;
                    let scr = if t < 6 {
                        // Upper: score face * 3 (par)
                        ((t + 1) * 3) as u8
                    } else {
                        // Lower: some variation
                        10 + (i % 5) as u8 + (t as u8 % 3)
                    };
                    rec.turns[t] = TurnRecord {
                        dice_initial: [1, 2, 3, 4, 5],
                        mask1: if t % 2 == 0 { 7 } else { 0 },
                        dice_after_reroll1: [1, 2, 3, 4, 5],
                        mask2: if t % 3 == 0 { 3 } else { 0 },
                        dice_final: [1, 2, 3, 4, 5],
                        category: cat,
                        score: scr,
                    };
                    score_total += scr as u16;
                    if cat < 6 {
                        upper += scr;
                    }
                }
                let upper_capped = upper.min(63);
                let got_bonus = upper_capped >= 63;
                if got_bonus {
                    score_total += 50;
                }
                rec.total_score = score_total;
                rec.upper_total = upper_capped;
                rec.got_bonus = got_bonus as u8;
                rec
            })
            .collect()
    }

    #[test]
    fn test_aggregate_basic() {
        let records = make_test_records(100);
        let stats = aggregate_statistics(&records, 245.87, 42);

        assert_eq!(stats.num_games, 100);
        assert_eq!(stats.seed, 42);
        assert_eq!(stats.categories.len(), CATEGORY_COUNT);
        assert!(stats.total_score.mean > 0.0);
        assert!(stats.total_score.min <= stats.total_score.max);
        assert!(stats.total_score.std_dev >= 0.0);
    }

    #[test]
    fn test_aggregate_percentiles_ordered() {
        let records = make_test_records(1000);
        let stats = aggregate_statistics(&records, 245.87, 42);
        let p = &stats.total_score.percentiles;
        assert!(p.p5 <= p.p10);
        assert!(p.p10 <= p.p25);
        assert!(p.p25 <= p.p50);
        assert!(p.p50 <= p.p75);
        assert!(p.p75 <= p.p90);
        assert!(p.p90 <= p.p95);
        assert!(p.p95 <= p.p99);
    }

    #[test]
    fn test_aggregate_categories_complete() {
        let records = make_test_records(100);
        let stats = aggregate_statistics(&records, 245.87, 42);

        for (i, cat) in stats.categories.iter().enumerate() {
            assert_eq!(cat.id, i);
            assert_eq!(cat.name, CATEGORY_NAMES[i]);
            assert!(cat.mean_score >= 0.0);
            assert!(cat.zero_rate >= 0.0 && cat.zero_rate <= 1.0);
            assert!(cat.max_rate >= 0.0 && cat.max_rate <= 1.0);
            assert_eq!(cat.turn_distribution.len(), CATEGORY_COUNT);
        }
    }

    #[test]
    fn test_save_load_json() {
        let records = make_test_records(50);
        let stats = aggregate_statistics(&records, 245.87, 42);
        let path = "/tmp/yatzy_test_stats.json";
        save_statistics(&stats, path);

        let content = std::fs::read_to_string(path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["num_games"], 50);
        assert_eq!(
            parsed["categories"].as_array().unwrap().len(),
            CATEGORY_COUNT
        );

        let _ = std::fs::remove_file(path);
    }
}
