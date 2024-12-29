#include <stdio.h>
#include <math.h>
#include <string.h>
#include <json-c/json.h>
#include <precompute_scores.h>
#include <yatzy.h>

// --------------------------- Category Mapping ---------------------------

int categoryToIndex(const char *category_name) {
    if (strcasecmp(category_name, "ONES") == 0) return CATEGORY_ONES;
    else if (strcasecmp(category_name, "TWOS") == 0) return CATEGORY_TWOS;
    else if (strcasecmp(category_name, "THREES") == 0) return CATEGORY_THREES;
    else if (strcasecmp(category_name, "FOURS") == 0) return CATEGORY_FOURS;
    else if (strcasecmp(category_name, "FIVES") == 0) return CATEGORY_FIVES;
    else if (strcasecmp(category_name, "SIXES") == 0) return CATEGORY_SIXES;
    else if (strcasecmp(category_name, "ONE_PAIR") == 0) return CATEGORY_ONE_PAIR;
    else if (strcasecmp(category_name, "TWO_PAIRS") == 0) return CATEGORY_TWO_PAIRS;
    else if (strcasecmp(category_name, "THREE_OF_A_KIND") == 0) return CATEGORY_THREE_OF_A_KIND;
    else if (strcasecmp(category_name, "FOUR_OF_A_KIND") == 0) return CATEGORY_FOUR_OF_A_KIND;
    else if (strcasecmp(category_name, "SMALL_STRAIGHT") == 0) return CATEGORY_SMALL_STRAIGHT;
    else if (strcasecmp(category_name, "LARGE_STRAIGHT") == 0) return CATEGORY_LARGE_STRAIGHT;
    else if (strcasecmp(category_name, "FULL_HOUSE") == 0) return CATEGORY_FULL_HOUSE;
    else if (strcasecmp(category_name, "CHANCE") == 0) return CATEGORY_CHANCE;
    else if (strcasecmp(category_name, "YATZY") == 0) return CATEGORY_YATZY;
    return -1;
}

// --------------------------- Utility Functions ---------------------------

int UpdateUpperScore(int upper_score, int category, int score) {
    if (category < 6) {
        int new_up = upper_score + score;
        return (new_up > 63) ? 63 : new_up;
    }
    return upper_score;
}

// SortDiceSet function
void SortDiceSet(int arr[5]) {
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 5; j++) {
            if (arr[j] < arr[i]) {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
}

// CountFaces function
void CountFaces(const int dice[5], int face_count[7]) {
    for (int i = 1; i <= 6; i++) face_count[i] = 0;
    for (int i = 0; i < 5; i++) face_count[dice[i]]++;
}

// NOfAKindScore function
int NOfAKindScore(const int face_count[7], int n) {
    for (int face = 6; face >= 1; face--) {
        if (face_count[face] >= n) return face * n;
    }
    return 0;
}

// --------------------------- Precomputation Functions ---------------------------

void PrecomputeScoredCategoryCounts(YatzyContext *ctx) {
    for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
        int count = 0;
        int temp = scored;
        for (int i = 0; i < CATEGORY_COUNT; i++) {
            count += (temp & 1);
            temp >>= 1;
        }
        ctx->scored_category_count_cache[scored] = count;
    }
}

void BuildAllDiceCombinations(YatzyContext *ctx) {
    ctx->num_combinations = 0;
    for (int a = 1; a <= 6; a++) {
        for (int b = a; b <= 6; b++) {
            for (int c = b; c <= 6; c++) {
                for (int d = c; d <= 6; d++) {
                    for (int e = d; e <= 6; e++) {
                        int idx = ctx->num_combinations++;
                        ctx->all_dice_sets[idx][0] = a;
                        ctx->all_dice_sets[idx][1] = b;
                        ctx->all_dice_sets[idx][2] = c;
                        ctx->all_dice_sets[idx][3] = d;
                        ctx->all_dice_sets[idx][4] = e;
                        ctx->index_lookup[a - 1][b - 1][c - 1][d - 1][e - 1] = idx;
                    }
                }
            }
        }
    }
}

// --------------------------- Dice Rolling Functions ---------------------------

void RollDice(int dice[5]) {
    for (int i = 0; i < 5; i++) {
        dice[i] = (rand() % 6) + 1;
    }
}

void RerollDice(int dice[5], int mask) {
    for (int i = 0; i < 5; i++) {
        if (mask & (1 << i)) {
            dice[i] = (rand() % 6) + 1;
        }
    }
}

// --------------------------- Scoring Functions ---------------------------

int FindDiceSetIndex(const YatzyContext *ctx, const int dice[5]) {
    return ctx->index_lookup[dice[0] - 1][dice[1] - 1][dice[2] - 1][dice[3] - 1][dice[4] - 1];
}

int CalculateCategoryScore(const int dice[5], int category) {
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
            int face = category + 1;
            return face_count[face] * face;
        }
        case CATEGORY_ONE_PAIR:
            return NOfAKindScore(face_count, 2);
        case CATEGORY_TWO_PAIRS: {
            int pairs[2], pcount = 0;
            for (int f = 6; f >= 1; f--) {
                if (face_count[f] >= 2) {
                    pairs[pcount++] = f;
                    if (pcount == 2) break;
                }
            }
            return (pcount == 2) ? 2 * pairs[0] + 2 * pairs[1] : 0;
        }
        case CATEGORY_THREE_OF_A_KIND:
            return NOfAKindScore(face_count, 3);
        case CATEGORY_FOUR_OF_A_KIND:
            return NOfAKindScore(face_count, 4);
        case CATEGORY_SMALL_STRAIGHT:
            return (face_count[1] && face_count[2] && face_count[3] && face_count[4] && face_count[5]) ? 15 : 0;
        case CATEGORY_LARGE_STRAIGHT:
            return (face_count[2] && face_count[3] && face_count[4] && face_count[5] && face_count[6]) ? 20 : 0;
        case CATEGORY_FULL_HOUSE: {
            int three_face = 0, pair_face = 0;
            for (int f = 1; f <= 6; f++) {
                if (face_count[f] == 3) three_face = f;
                else if (face_count[f] == 2) pair_face = f;
            }
            return (three_face && pair_face) ? sum_all : 0;
        }
        case CATEGORY_CHANCE:
            return sum_all;
        case CATEGORY_YATZY:
            return (NOfAKindScore(face_count, 5) == 50) ? 50 : 0;
        default:
            return 0;
    }
}

// --------------------------- Context Management ---------------------------

YatzyContext *CreateYatzyContext() {
    YatzyContext *ctx = (YatzyContext *)malloc(sizeof(YatzyContext));

    const char *names[] = {
        "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
        "One Pair", "Two Pairs", "Three of a Kind", "Four of a Kind",
        "Small Straight", "Large Straight", "Full House", "Chance", "Yatzy"
    };
    for (int i = 0; i < CATEGORY_COUNT; i++) {
        ctx->category_names[i] = strdup(names[i]);
    }

    ctx->state_values = (double *)malloc(NUM_STATES * sizeof(double));
    for (int i = 0; i < NUM_STATES; i++) {
        ctx->state_values[i] = 0.0;
    }

    PrecomputeScoredCategoryCounts(ctx);
    BuildAllDiceCombinations(ctx);
    PrecomputeCategoryScores(ctx);
    PrecomputeRerollTransitionProbabilities(ctx);
    PrecomputeDiceSetProbabilities(ctx);
    InitializeFinalStates(ctx);

    return ctx;
}

void FreeYatzyContext(YatzyContext *ctx) {
    for (int i = 0; i < CATEGORY_COUNT; i++) {
        free(ctx->category_names[i]);
    }
    free(ctx->state_values);
    free(ctx);
}

// --------------------------- Decision-Making and Evaluation ---------------------------

int ChooseBestCategoryNoRerolls(const YatzyContext *ctx, int upper_score, int scored_categories,
                                const int dice[5], double *best_ev) {
    int ds_index = FindDiceSetIndex(ctx, dice);
    double best_val = -INFINITY;
    int best_category = -1;

    for (int c = 0; c < CATEGORY_COUNT; c++) {
        if (IS_CATEGORY_SCORED(scored_categories, c)) continue;

        int scr = ctx->precomputed_scores[ds_index][c];
        if (scr == 0) continue;

        int new_up = UpdateUpperScore(upper_score, c, scr);
        int new_scored = scored_categories | (1 << c);
        double val = scr + GetStateValue(ctx, new_up, new_scored);

        if (val > best_val) {
            best_val = val;
            best_category = c;
        }
    }

    *best_ev = best_val;
    return best_category;
}

int ChooseBestRerollMask(const YatzyContext *ctx, const double E_ds_for_masks[252],
                         const int dice[5], double *best_ev) {
    int sorted_dice[5];
    memcpy(sorted_dice, dice, 5 * sizeof(int));
    SortDiceSet(sorted_dice);

    int ds_index = FindDiceSetIndex(ctx, sorted_dice);

    double best_val = -INFINITY;
    int best_mask = 0;

    for (int mask = 0; mask < 32; mask++) {
        double ev = ComputeExpectedValueForRerollMask(ctx, ds_index, E_ds_for_masks, mask);
        if (ev > best_val) {
            best_val = ev;
            best_mask = mask;
        }
    }

    *best_ev = best_val;
    return best_mask;
}

void ComputeBestRerollStrategy(const YatzyContext *ctx, int upper_score, int scored_categories,
                               const int dice[5], int rerolls_remaining, int *best_mask, double *best_ev) {
    double E_ds_0[252];
    ComputeEDs0ForState(ctx, upper_score, scored_categories, E_ds_0);

    if (rerolls_remaining == 1) {
        *best_mask = ChooseBestRerollMask(ctx, E_ds_0, dice, best_ev);
        return;
    }

    double E_ds_1[252];
    int dummy_mask[252];
    ComputeExpectedValuesForNRerolls(ctx, 1, E_ds_0, E_ds_1, dummy_mask);
    *best_mask = ChooseBestRerollMask(ctx, E_ds_1, dice, best_ev);
}

double SuggestOptimalAction(const YatzyContext *ctx, int upper_score, int scored_categories,
                            const int dice[5], int rerolls_remaining) {
    printf("Original dice: ");
    for (int i = 0; i < 5; i++) printf("%d ", dice[i]);
    printf("\n");

    int sorted_dice[5];
    memcpy(sorted_dice, dice, 5 * sizeof(int));
    SortDiceSet(sorted_dice);

    if (rerolls_remaining == 0) {
        double ev;
        int best_category = ChooseBestCategoryNoRerolls(ctx, upper_score, scored_categories, sorted_dice, &ev);
        printf("No rerolls left. Best category: %d (Expected Value: %f)\n", best_category, ev);
        return ev;
    } else {
        int best_mask;
        double ev;
        ComputeBestRerollStrategy(ctx, upper_score, scored_categories, sorted_dice, rerolls_remaining, &best_mask, &ev);
        printf("%d rerolls left. Best mask: %d (Expected Value: %f)\n", rerolls_remaining, best_mask, ev);
        return ev;
    }
}

double EvaluateChosenCategory(const YatzyContext *ctx, int upper_score, int scored_categories,
                              const int dice[5], int chosen_category) {
    if (IS_CATEGORY_SCORED(scored_categories, chosen_category)) return -INFINITY;

    int ds_index = FindDiceSetIndex(ctx, dice);
    int scr = ctx->precomputed_scores[ds_index][chosen_category];
    if (scr == 0) return -INFINITY;

    int new_up = UpdateUpperScore(upper_score, chosen_category, scr);
    int new_scored = scored_categories | (1 << chosen_category);
    double future_val = GetStateValue(ctx, new_up, new_scored);
    return scr + future_val;
}

double EvaluateChosenRerollMask(const YatzyContext *ctx, int upper_score, int scored_categories,
                                 const int dice[5], int chosen_mask, int rerolls_remaining) {
    double E_ds_0[252];
    ComputeEDs0ForState(ctx, upper_score, scored_categories, E_ds_0);

    double E_ds_1[252];
    if (rerolls_remaining == 2) {
        int dummy_mask[252];
        ComputeExpectedValuesForNRerolls(ctx, 1, E_ds_0, E_ds_1, dummy_mask);
    }

    const double *E_ds_for_masks = (rerolls_remaining == 1) ? E_ds_0 : E_ds_1;

    int sorted_dice[5];
    memcpy(sorted_dice, dice, 5 * sizeof(int));
    SortDiceSet(sorted_dice);

    int ds_index = FindDiceSetIndex(ctx, sorted_dice);
    return ComputeExpectedValueForRerollMask(ctx, ds_index, E_ds_for_masks, chosen_mask);
}

double EvaluateAction(const YatzyContext *ctx, int upper_score, int scored_categories,
                      const int dice[5], int action, int rerolls_remaining) {
    if (rerolls_remaining == 0) {
        return EvaluateChosenCategory(ctx, upper_score, scored_categories, dice, action);
    } else {
        return EvaluateChosenRerollMask(ctx, upper_score, scored_categories, dice, action, rerolls_remaining);
    }
}

// --------------------------- Simulation and Utility Functions ---------------------------

void WriteResultsToCSV(const char *filename, const int *scores, int num_scores) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Could not open file '%s' for writing.\n", filename);
        return;
    }

    fprintf(file, "Score\n");
    for (int i = 0; i < num_scores; i++) {
        fprintf(file, "%d\n", scores[i]);
    }
    fclose(file);
    printf("Results successfully written to %s\n", filename);
}

int SimulateSingleGame(YatzyContext *ctx) {
    int upper_score = 0;
    int scored_categories = 0;
    int total_score = 0;

    for (int turn = 0; turn < 15; turn++) {
        int dice[5];
        RollDice(dice);

        int rerolls_remaining = 2;
        while (rerolls_remaining > 0) {
            SortDiceSet(dice);
            int best_mask;
            double best_ev;
            ComputeBestRerollStrategy(ctx, upper_score, scored_categories, dice, rerolls_remaining, &best_mask, &best_ev);

            RerollDice(dice, best_mask);
            SortDiceSet(dice);
            rerolls_remaining--;
        }

        double best_ev;
        int best_category = ChooseBestCategoryNoRerolls(ctx, upper_score, scored_categories, dice, &best_ev);

        int ds_index = FindDiceSetIndex(ctx, dice);
        int score = ctx->precomputed_scores[ds_index][best_category];
        total_score += score;
        upper_score = UpdateUpperScore(upper_score, best_category, score);
        scored_categories |= (1 << best_category);
    }

    if (upper_score >= 63) total_score += 50; // Bonus for upper section
    return total_score;
}

int MaskFromBinaryString(const char *action_str) {
    int mask = 0;
    for (int i = 0; i < 5; i++) {
        if (action_str[i] == '1') {
            mask |= (1 << i);
        }
    }
    return mask;
}