/*
 * game_mechanics.c — Phase 0: Scoring rules
 *
 * Implements s(S, r, c), the score function for Scandinavian Yatzy's
 * 15 categories, and the upper-score successor m' = min(m + u(S,r,c), 63).
 *
 * See: theory/optimal_yahtzee_pseudocode.md, "Notation" table.
 */
#include "context.h"
#include "dice_mechanics.h"

/* Compute s(S, r, c): the score for placing dice[5] in the given category.
 * Categories 0–5 are upper section (Ones–Sixes), 6–14 are lower section. */
int CalculateCategoryScore(const int dice[5], const int category) {
    int face_count[7];
    CountFaces(dice, face_count);
    int sum_all = 0;
    for (int i = 0; i < 5; i++) sum_all += dice[i];

    switch (category) {
        case CATEGORY_ONES:
        case CATEGORY_TWOS:
        case CATEGORY_THREES:
        case CATEGORY_FOURS:
        case CATEGORY_FIVES:
        case CATEGORY_SIXES: {
            const int face = category + 1;
            return face_count[face] * face;
        }
        case CATEGORY_ONE_PAIR:
            for (int f = 6; f >= 1; f--) if (face_count[f] >= 2) return 2 * f;
            return 0;
        case CATEGORY_TWO_PAIRS: {
            int pairs[2], pcount = 0;
            for (int f = 6; f >= 1; f--) {
                if (face_count[f] >= 2) {
                    pairs[pcount++] = f;
                    if (pcount == 2) break;
                }
            }
            if (pcount == 2) return 2 * pairs[0] + 2 * pairs[1];
            return 0;
        }
        case CATEGORY_THREE_OF_A_KIND: return NOfAKindScore(face_count, 3);
        case CATEGORY_FOUR_OF_A_KIND: return NOfAKindScore(face_count, 4);
        case CATEGORY_SMALL_STRAIGHT:
            if (face_count[1] == 1 && face_count[2] == 1
                && face_count[3] == 1 && face_count[4] == 1 && face_count[5] == 1)
                return 15;
            return 0;
        case CATEGORY_LARGE_STRAIGHT:
            if (face_count[2] == 1 && face_count[3] == 1
                && face_count[4] == 1 && face_count[5] == 1 && face_count[6] == 1)
                return 20;
            return 0;
        case CATEGORY_FULL_HOUSE: {
            int three_face = 0, pair_face = 0;
            for (int f = 1; f <= 6; f++) {
                if (face_count[f] == 3) three_face = f;
                else if (face_count[f] == 2) pair_face = f;
            }
            if (three_face && pair_face) return sum_all;
            return 0;
        }
        case CATEGORY_CHANCE:
            return sum_all;
        case CATEGORY_YATZY:
            for (int f = 1; f <= 6; f++) {
                if (face_count[f] == 5) return 50;
            }
            return 0;
        default: ;
    }
    return 0;
}

/* Compute successor upper score: m' = min(m + u(S,r,c), 63).
 * Upper categories (0–5) add their score to m; others leave m unchanged.
 * The cap at 63 reflects the upper bonus threshold. */
int UpdateUpperScore(const int upper_score, const int category, const int score) {
    if (category < 6) {
        const int new_upper_score = upper_score + score;
        return (new_upper_score > 63) ? 63 : new_upper_score;
    }
    return upper_score;
}
