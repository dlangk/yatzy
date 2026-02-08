/*
 * game_mechanics.h — Scoring rules (Phase 0)
 *
 * Declares the score function s(S, r, c) and upper-score successor m'.
 * See: theory/optimal_yahtzee_pseudocode.md, "Notation" table.
 */
#pragma once

/* Compute s(S, r, c): score for placing dice[] in the given category.
 * Implements Scandinavian Yatzy's 15-category scoring rules. */
int CalculateCategoryScore(const int dice[5], int category);

/* Compute successor upper score: m' = min(m + u(S,r,c), 63).
 * Only upper categories (0–5) contribute; others leave m unchanged. */
int UpdateUpperScore(int upper_score, int category, int score);
