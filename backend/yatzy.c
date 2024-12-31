#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include <limits.h>
#include <microhttpd.h>
#include <json-c/json.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CATEGORY_COUNT 15

#define CATEGORY_ONES 0
#define CATEGORY_TWOS 1
#define CATEGORY_THREES 2
#define CATEGORY_FOURS 3
#define CATEGORY_FIVES 4
#define CATEGORY_SIXES 5
#define CATEGORY_ONE_PAIR 6
#define CATEGORY_TWO_PAIRS 7
#define CATEGORY_THREE_OF_A_KIND 8
#define CATEGORY_FOUR_OF_A_KIND 9
#define CATEGORY_SMALL_STRAIGHT 10
#define CATEGORY_LARGE_STRAIGHT 11
#define CATEGORY_FULL_HOUSE 12
#define CATEGORY_CHANCE 13
#define CATEGORY_YATZY 14

#define NUM_STATES (64*(1<<15))

#define IS_CATEGORY_SCORED(scored,cat) (((scored)&(1<<(cat)))!=0)
#define SET_CATEGORY(scored,cat) ((scored)|=(1<<(cat)))
#define CLEAR_CATEGORY(scored,cat) ((scored)&=~(1<<(cat)))
#define STATE_INDEX(upper_score, scored_categories) ((upper_score)*(1<<15)+(scored_categories))

#define ALLOWED_METHODS "GET, POST, OPTIONS"
#define ALLOWED_HEADERS "Content-Type"

typedef struct {
    int all_dice_sets[252][5];
    double transition_table[252][32][252];
    int num_combinations;

    int index_lookup[6][6][6][6][6];

    int precomputed_scores[252][CATEGORY_COUNT];
    double dice_set_probabilities[252];

    int factorial[6 + 1];

    double *state_values;

    const char *category_names[CATEGORY_COUNT];

    int scored_category_count_cache[1 << CATEGORY_COUNT];
} YatzyContext;

typedef struct {
    int upper_score;
    int scored_categories;
} YatzyState;

struct RequestContext {
    char *post_data;
    size_t post_size;
};

// Add a helper function to set CORS headers on every response
static void AddCORSHeaders(struct MHD_Response *response) {
    MHD_add_response_header(response, "Access-Control-Allow-Origin", "*");
    MHD_add_response_header(response, "Access-Control-Allow-Methods", ALLOWED_METHODS);
    MHD_add_response_header(response, "Access-Control-Allow-Headers", ALLOWED_HEADERS);
}

#define MAX_LINE_LENGTH 1024
#define MAX_FIELDS 100

void parse_csv_line(char *line, char *fields[], int *num_fields) {
    *num_fields = 0;
    char *token = strtok(line, ",");

    while (token != NULL && *num_fields < MAX_FIELDS) {
        // Remove leading/trailing whitespace
        while (*token == ' ') token++;
        char *end = token + strlen(token) - 1;
        while (end > token && *end == ' ') end--;
        *(end + 1) = '\0';

        // Remove quotes if present
        if (*token == '"' && *end == '"') {
            token++;
            *end = '\0';
        }

        fields[*num_fields] = strdup(token);
        (*num_fields)++;
        token = strtok(NULL, ",");
    }
}

int read_csv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s\n", filename);
        return 1;
    }

    char line[MAX_LINE_LENGTH];
    char *fields[MAX_FIELDS];
    int num_fields;
    int line_number = 0;

    // Read each line
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        // Remove newline character
        line[strcspn(line, "\n")] = 0;

        parse_csv_line(line, fields, &num_fields);

        // Print the fields (you can modify this part based on your needs)
        printf("Line %d:\n", line_number++);
        for (int i = 0; i < num_fields; i++) {
            printf("Field %d: %s\n", i, fields[i]);
            free(fields[i]); // Don't forget to free the memory
        }
        printf("\n");
    }

    fclose(file);
    return 0;
}


// --------------------------- Utility Functions ---------------------------
static inline void SortDiceSet(int arr[5]) {
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

static inline void CountFaces(const int dice[5],
                              int face_count[7]) {
    for (int i = 1; i <= 6; i++) face_count[i] = 0;
    for (int i = 0; i < 5; i++) face_count[dice[i]]++;
}

static inline int NOfAKindScore(const int face_count[7], int n) {
    for (int face = 6; face >= 1; face--) {
        if (face_count[face] >= n) return face * n;
    }
    return 0;
}

static int UpdateUpperScore(int upper_score, int category, int score) {
    if (category < 6) {
        int new_up = upper_score + score;
        return (new_up > 63) ? 63 : new_up;
    }
    return upper_score;
}

static void PrecomputeScoredCategoryCounts(YatzyContext *ctx) {
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

static void BuildAllDiceCombinations(YatzyContext *ctx) {
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

static inline int FindDiceSetIndex(const YatzyContext *ctx,
                                   const int dice[5]) {
    return ctx->index_lookup[dice[0] - 1][dice[1] - 1][dice[2] - 1][dice[3] - 1][dice[4] - 1];
}

static int CalculateCategoryScore(const int dice[5],
                                  int category) {
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
            if (face_count[1] == 1 && face_count[2] == 1 && face_count[3] == 1 && face_count[4] == 1 && face_count[5] ==
                1)
                return 15;
            return 0;
        case CATEGORY_LARGE_STRAIGHT:
            if (face_count[2] == 1 && face_count[3] == 1 && face_count[4] == 1 && face_count[5] == 1 && face_count[6] ==
                1)
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
    }
    return 0;
}

static void PrecomputeCategoryScores(YatzyContext *ctx) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < 252; i++) {
        int *dice = ctx->all_dice_sets[i];
        for (int cat = 0; cat < CATEGORY_COUNT; cat++) {
            ctx->precomputed_scores[i][cat] = CalculateCategoryScore(dice, cat);
        }
    }
}

static void PrecomputeRerollTransitionProbabilities(YatzyContext *ctx) {
#pragma omp parallel for schedule(dynamic)
    for (int ds1_i = 0; ds1_i < 252; ds1_i++) {
        int *ds1 = ctx->all_dice_sets[ds1_i];
        for (int reroll_mask = 0; reroll_mask < 32; reroll_mask++) {
            int locked_indices[5];
            int count_locked = 0, count_reroll = 0;
            for (int i = 0; i < 5; i++) {
                if (reroll_mask & (1 << i)) count_reroll++;
                else locked_indices[count_locked++] = i;
            }

            if (count_reroll == 0) {
                for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                    ctx->transition_table[ds1_i][reroll_mask][ds2_i] = (ds1_i == ds2_i) ? 1.0 : 0.0;
                }
                continue;
            }

            int total_outcomes = 1;
            for (int p = 0; p < count_reroll; p++) total_outcomes *= 6;
            double counts[252];
            for (int k = 0; k < 252; k++) counts[k] = 0.0;

            for (int outcome_id = 0; outcome_id < total_outcomes; outcome_id++) {
                int temp = outcome_id;
                int rerolled_values[5];
                for (int p = 0; p < count_reroll; p++) {
                    rerolled_values[p] = (temp % 6) + 1;
                    temp /= 6;
                }

                int final_dice[5];
                for (int idx = 0; idx < count_locked; idx++) final_dice[idx] = ds1[locked_indices[idx]];
                for (int idx = 0; idx < count_reroll; idx++) final_dice[count_locked + idx] = rerolled_values[idx];

                SortDiceSet(final_dice);
                int ds2_index = ctx->index_lookup[final_dice[0] - 1][final_dice[1] - 1][final_dice[2] - 1][
                    final_dice[3] - 1][final_dice[4] - 1];
                counts[ds2_index] += 1.0;
            }

            double inv_total = 1.0 / total_outcomes;
            for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                ctx->transition_table[ds1_i][reroll_mask][ds2_i] = counts[ds2_i] * inv_total;
            }
        }
    }
}

static void PrecomputeFactorials(YatzyContext *ctx) {
    ctx->factorial[0] = 1;
    for (int i = 1; i <= 5; i++) ctx->factorial[i] = ctx->factorial[i - 1] * i;
}

static double ComputeProbabilityOfDiceSetOnce(const YatzyContext *ctx,
                                              const int dice[5]) {
    int face_count[7];
    for (int i = 1; i <= 6; i++) face_count[i] = 0;
    for (int i = 0; i < 5; i++) face_count[dice[i]]++;

    int numerator = ctx->factorial[5];
    int denominator = 1;
    for (int f = 1; f <= 6; f++) {
        if (face_count[f] > 1) {
            int fc = face_count[f];
            int sub_fact = 1;
            for (int x = 2; x <= fc; x++) sub_fact *= x;
            denominator *= sub_fact;
        }
    }

    double permutations = (double) numerator / (double) denominator;
    double total_outcomes = pow(6, 5);
    return permutations / total_outcomes;
}

static void PrecomputeDiceSetProbabilities(YatzyContext *ctx) {
#pragma omp parallel for schedule(static)
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        ctx->dice_set_probabilities[ds_i] = ComputeProbabilityOfDiceSetOnce(ctx, ctx->all_dice_sets[ds_i]);
    }
}

static void InitializeFinalStates(YatzyContext *ctx) {
    int all_scored_mask = (1 << CATEGORY_COUNT) - 1;
    for (int up = 0; up <= 63; up++) {
        double final_val = (up >= 63) ? 50.0 : 0.0;
        ctx->state_values[STATE_INDEX(up, all_scored_mask)] = final_val;
    }
}

static double GetStateValue(const YatzyContext *ctx,
                            int up,
                            int scored) {
    return ctx->state_values[STATE_INDEX(up, scored)];
}

static double ComputeBestScoringValueForDiceSet(const YatzyContext *ctx,
                                                const YatzyState *S,
                                                const int dice[5]) {
    int ds_index = FindDiceSetIndex(ctx, dice);
    double best_val = -INFINITY;
    int scored = S->scored_categories;
    int up_score = S->upper_score;

    for (int c = 0; c < CATEGORY_COUNT; c++) {
        if (!IS_CATEGORY_SCORED(scored, c)) {
            int scr = ctx->precomputed_scores[ds_index][c];
            int new_up = UpdateUpperScore(up_score, c, scr);
            int new_scored = (scored | (1 << c));
            double val = scr + GetStateValue(ctx, new_up, new_scored);
            if (val > best_val) best_val = val;
        }
    }

    return best_val;
}

static void ComputeExpectedValuesForNRerolls(
    const YatzyContext *ctx,
    int n,
    double E_ds_prev[252],
    double E_ds_current[252],
    int best_mask_for_n[252]
) {
#pragma omp parallel for schedule(static)
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        double best_val = -INFINITY;
        int best_mask = 0;
        const double (*row)[252] = ctx->transition_table[ds_i];
        for (int mask = 0; mask < 32; mask++) {
            const double *probs = row[mask];
            double ev = 0.0;
            for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                ev += probs[ds2_i] * E_ds_prev[ds2_i];
            }
            if (ev > best_val) {
                best_val = ev;
                best_mask = mask;
            }
        }
        E_ds_current[ds_i] = best_val;
        best_mask_for_n[ds_i] = best_mask;
    }
}

static double ComputeExpectedStateValue(const YatzyContext *ctx,
                                        const YatzyState *S) {
    double E_ds_0[252], E_ds_1[252], E_ds_2[252];
    int best_mask_1[252], best_mask_2[252];

#pragma omp parallel for schedule(static)
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_ds_0[ds_i] = ComputeBestScoringValueForDiceSet(ctx, S, ctx->all_dice_sets[ds_i]);
    }

    ComputeExpectedValuesForNRerolls(ctx, 1, E_ds_0, E_ds_1, best_mask_1);
    ComputeExpectedValuesForNRerolls(ctx, 2, E_ds_1, E_ds_2, best_mask_2);

    double E_S = 0.0;
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_S += ctx->dice_set_probabilities[ds_i] * E_ds_2[ds_i];
    }

    return E_S;
}

static YatzyContext *CreateYatzyContext() {
    YatzyContext *ctx = (YatzyContext *) malloc(sizeof(YatzyContext));

    ctx->category_names[0] = "Ones";
    ctx->category_names[1] = "Twos";
    ctx->category_names[2] = "Threes";
    ctx->category_names[3] = "Fours";
    ctx->category_names[4] = "Fives";
    ctx->category_names[5] = "Sixes";
    ctx->category_names[6] = "One Pair";
    ctx->category_names[7] = "Two Pairs";
    ctx->category_names[8] = "Three of a Kind";
    ctx->category_names[9] = "Four of a Kind";
    ctx->category_names[10] = "Small Straight";
    ctx->category_names[11] = "Large Straight";
    ctx->category_names[12] = "Full House";
    ctx->category_names[13] = "Chance";
    ctx->category_names[14] = "Yatzy";

    ctx->state_values = (double *) malloc(NUM_STATES * sizeof(double));
    for (int i = 0; i < NUM_STATES; i++) ctx->state_values[i] = 0.0;

    PrecomputeFactorials(ctx);
    BuildAllDiceCombinations(ctx);
    PrecomputeCategoryScores(ctx);
    PrecomputeRerollTransitionProbabilities(ctx);
    PrecomputeDiceSetProbabilities(ctx);
    PrecomputeScoredCategoryCounts(ctx);
    InitializeFinalStates(ctx);

    return ctx;
}

static int FileExists(const char *filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

void SaveStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename) {
    printf("Saving state values to file: %s for scored_count = %d\n", filename, scored_count);

    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file for writing: %s\n", filename);
        return;
    }

    // Step 1: Count the number of states
    int count = 0;
    for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
        if (ctx->scored_category_count_cache[scored] == scored_count) {
            for (int up = 0; up <= 63; up++) {
                count++;
            }
        }
    }
    printf("Total states to save for scored_count = %d: %d\n", scored_count, count);

    // Write the count to the file
    fwrite(&count, sizeof(int), 1, f);
    printf("Wrote state count (%d) to file: %s\n", count, filename);

    // Step 2: Write state values
    for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
        if (ctx->scored_category_count_cache[scored] == scored_count) {
            for (int up = 0; up <= 63; up++) {
                double val = ctx->state_values[STATE_INDEX(up, scored)];
                fwrite(&up, sizeof(int), 1, f);
                fwrite(&scored, sizeof(int), 1, f);
                fwrite(&val, sizeof(double), 1, f);
                printf("Saved state_values[%d] = %f (up = %d, scored = %04x) to file: %s\n",
                       STATE_INDEX(up, scored), val, up, scored, filename);
            }
        }
    }

    fclose(f);
    printf("Successfully saved state values for scored_count = %d to file: %s\n", scored_count, filename);
}

int LoadStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename) {
    printf("Attempting to load state values from file: %s for scored_count = %d\n", filename, scored_count);

    if (!FileExists(filename)) {
        printf("File does not exist: %s\n", filename);
        return 0; // File not found
    }

    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Warning: Could not open file for reading: %s\n", filename);
        return 0;
    }

    // Step 1: Read the count of states
    int count;
    if (fread(&count, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Warning: Could not read count from file %s\n", filename);
        fclose(f);
        return 0;
    }
    printf("File %s contains %d states for scored_count = %d\n", filename, count, scored_count);

    // Step 2: Load state values
    for (int i = 0; i < count; i++) {
        int up, scored;
        double val;

        // Read the state values
        if (fread(&up, sizeof(int), 1, f) != 1 ||
            fread(&scored, sizeof(int), 1, f) != 1 ||
            fread(&val, sizeof(double), 1, f) != 1) {
            fprintf(stderr, "Warning: Unexpected end of file while reading %s (state %d)\n", filename, i);
            fclose(f);
            return 0;
        }

        // Calculate the index and store the value
        int index = STATE_INDEX(up, scored);
        if (index < 0 || index >= NUM_STATES) {
            fprintf(stderr, "Error: Invalid index %d for up = %d, scored = %04x in file %s\n", index, up, scored, filename);
            fclose(f);
            return 0;
        }

        ctx->state_values[index] = val;
    }

    fclose(f);
    printf("--> Successfully loaded state values for scored_count = %d from file: %s\n", scored_count, filename);
    return 1;
}

typedef struct {
    double ev; // The expected value if the resulting dice set is ds2_i
    double probability; // Probability that ds2_i occurs given ds_index + reroll mask
    int ds2_index; // (Optional) the index of ds2_i, in case you need it
} EVProbabilityPair;

void ComputeDistributionForRerollMask(const YatzyContext *ctx,
                                      int ds_index,
                                      const double E_ds_for_masks[252],
                                      int mask,
                                      EVProbabilityPair out_distribution[252]) {
    // Probability array for going from ds_index -> ds2_i under 'mask'
    const double *prob_row = ctx->transition_table[ds_index][mask];

    // Fill out the distribution
    for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
        out_distribution[ds2_i].ev = E_ds_for_masks[ds2_i];
        out_distribution[ds2_i].probability = prob_row[ds2_i];
        out_distribution[ds2_i].ds2_index = ds2_i;
    }
}

static void ComputeAllStateValues(YatzyContext *ctx) {
    for (int scored_count = 15; scored_count >= 0; scored_count--) {
        if (scored_count == 15) {
            char filename[64];
            snprintf(filename, sizeof(filename), "states_%d.bin", scored_count);

            if (!LoadStateValuesForCount(ctx, scored_count, filename)) {
                SaveStateValuesForCount(ctx, scored_count, filename);
            }
            continue;
        }
        char filename[64];
        snprintf(filename, sizeof(filename), "states_%d.bin", scored_count);

        if (LoadStateValuesForCount(ctx, scored_count, filename)) {
            continue;
        }

        int needed_count = 0;
        for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
            if (ctx->scored_category_count_cache[scored] == scored_count) {
                needed_count += 64;
            }
        }

        int (*states)[2] = malloc(sizeof(int) * 2 * needed_count);
        int idx = 0;
        for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
            if (ctx->scored_category_count_cache[scored] == scored_count) {
                for (int up = 0; up <= 63; up++) {
                    states[idx][0] = up;
                    states[idx][1] = scored;
                    idx++;
                }
            }
        }

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < idx; i++) {
            int up = states[i][0];
            int scored = states[i][1];
            YatzyState S = {up, scored};
            double val = ComputeExpectedStateValue(ctx, &S);
            ctx->state_values[STATE_INDEX(up, scored)] = val;
        }

        free(states);
        SaveStateValuesForCount(ctx, scored_count, filename);
    }
}

static void FreeYatzyContext(YatzyContext *ctx) {
    free(ctx->state_values);
    free(ctx);
}

static int ChooseBestCategoryNoRerolls(const YatzyContext *ctx,
                                       int upper_score,
                                       int scored_categories,
                                       const int dice[5],
                                       double *best_ev) {
    int ds_index = FindDiceSetIndex(ctx, dice);
    double best_val = -INFINITY;
    int best_category = -1;

    for (int c = 0; c < CATEGORY_COUNT; c++) {
        if (IS_CATEGORY_SCORED(scored_categories, c)) {
            continue;
        }

        int scr = ctx->precomputed_scores[ds_index][c];

        // If this category yields no points, it's effectively useless now
        // Treat it as -INFINITY so we never pick it.
        if (scr == 0) {
            continue; // or explicitly val = -INFINITY
        }

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

static void ComputeEDs0ForState(const YatzyContext *ctx,
                                int upper_score,
                                int scored_categories,
                                double E_ds_0[252]) {
    YatzyState state = {upper_score, scored_categories};
#pragma omp parallel for
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_ds_0[ds_i] = ComputeBestScoringValueForDiceSet(ctx, &state, ctx->all_dice_sets[ds_i]);
    }
}

static double ComputeExpectedValueForRerollMask(const YatzyContext *ctx,
                                                int ds_index,
                                                const double E_ds_for_masks[252],
                                                int mask) {
    const double *probs = ctx->transition_table[ds_index][mask];
    double ev = 0.0;
    for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
        ev += probs[ds2_i] * E_ds_for_masks[ds2_i];
    }
    return ev;
}

static int ChooseBestRerollMask(const YatzyContext *ctx,
                                const double E_ds_for_masks[252],
                                const int dice[5],
                                double *best_ev) {
    int sorted_dice[5] = {dice[0], dice[1], dice[2], dice[3], dice[4]};
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

static void ComputeBestRerollStrategy(const YatzyContext *ctx,
                                      int upper_score,
                                      int scored_categories,
                                      const int dice[5],
                                      int rerolls_remaining,
                                      int *best_mask,
                                      double *best_ev) {
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

double SuggestOptimalAction(const YatzyContext *ctx,
                            int upper_score,
                            int scored_categories,
                            const int dice[5],
                            int rerolls_remaining) {
    printf("Original dice: ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", dice[i]);
    }
    printf("\n");

    int sorted_dice[5];
    for (int i = 0; i < 5; i++) {
        sorted_dice[i] = dice[i];
    }
    SortDiceSet(sorted_dice);

    printf("Sorted dice:   ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", sorted_dice[i]);
    }
    printf("\n");

    if (rerolls_remaining == 0) {
        double ev;
        int best_category = ChooseBestCategoryNoRerolls(ctx, upper_score, scored_categories, sorted_dice, &ev);
        printf("No rerolls left.\n");
        if (best_category >= 0) {
            printf("Best category to pick: %s (Category #%d)\n", ctx->category_names[best_category], best_category);
            printf("Expected value after choosing this category: %f\n", ev);
            return ev;
        } else {
            printf("No valid category found.\n");
            return 0.0;
        }
    } else {
        int best_mask;
        double ev;
        ComputeBestRerollStrategy(ctx, upper_score, scored_categories, sorted_dice, rerolls_remaining, &best_mask, &ev);
        printf("%d rerolls left.\n", rerolls_remaining);
        printf("Best mask to reroll : %d (binary: ", best_mask);
        for (int i = 0; i < 5; i++) {
            printf("%d", (best_mask & (1 << i)) ? 1 : 0);
        }
        printf(")\n");
        printf("Expected mask value : %f\n", ev);
        return ev;
    }
}

static int categoryToIndex(const char *category_name) {
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

static double EvaluateChosenCategory(const YatzyContext *ctx,
                                     int upper_score,
                                     int scored_categories,
                                     const int dice[5],
                                     int chosen_category) {
    if (IS_CATEGORY_SCORED(scored_categories, chosen_category)) {
        return -INFINITY;
    }

    int ds_index = FindDiceSetIndex(ctx, dice);
    int scr = ctx->precomputed_scores[ds_index][chosen_category];
    if (scr == 0) return -INFINITY;
    int new_up = UpdateUpperScore(upper_score, chosen_category, scr);
    int new_scored = scored_categories | (1 << chosen_category);
    double future_val = GetStateValue(ctx, new_up, new_scored);
    return scr + future_val;
}

static double EvaluateChosenRerollMask(const YatzyContext *ctx,
                                       int upper_score,
                                       int scored_categories,
                                       const int dice[5],
                                       int chosen_mask,
                                       int rerolls_remaining) {
    double E_ds_0[252];
    ComputeEDs0ForState(ctx, upper_score, scored_categories, E_ds_0);

    double E_ds_1[252];
    int dummy_mask[252];
    if (rerolls_remaining == 2) {
        ComputeExpectedValuesForNRerolls(ctx, 1, E_ds_0, E_ds_1, dummy_mask);
    }

    const double *E_ds_for_masks = (rerolls_remaining == 1) ? E_ds_0 : E_ds_1;

    int sorted_dice[5];
    for (int i = 0; i < 5; i++) sorted_dice[i] = dice[i];
    SortDiceSet(sorted_dice);
    int ds_index = FindDiceSetIndex(ctx, sorted_dice);

    double ev = ComputeExpectedValueForRerollMask(ctx, ds_index, E_ds_for_masks, chosen_mask);
    return ev;
}

double EvaluateAction(const YatzyContext *ctx,
                      int upper_score,
                      int scored_categories,
                      const int dice[5],
                      int action,
                      int rerolls_remaining) {
    if (rerolls_remaining == 0) {
        return EvaluateChosenCategory(ctx, upper_score, scored_categories, dice, action);
    } else {
        return EvaluateChosenRerollMask(ctx, upper_score, scored_categories, dice, action, rerolls_remaining);
    }
}

static void WriteResultsToCSV(const char *filename, const int *scores, int num_scores) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Could not open file '%s' for writing.\n", filename);
        return;
    }

    fprintf(f, "Score\n");
    for (int i = 0; i < num_scores; i++) {
        fprintf(f, "%d\n", scores[i]);
    }
    fclose(f);
    printf("Results successfully written to %s\n", filename);
}

int SimulateSingleGame(YatzyContext *ctx) {
    int upper_score = 0;
    int scored_categories = 0;
    int total_score = 0;
    int actual_upper_score = 0;

    for (int turn = 0; turn < 15; turn++) {
        int dice[5];
        RollDice(dice);

        int rerolls_remaining = 2;
        while (rerolls_remaining > 0) {
            SortDiceSet(dice);
            int best_mask;
            double best_ev;
            ComputeBestRerollStrategy(ctx,
                                      upper_score,
                                      scored_categories,
                                      dice,
                                      rerolls_remaining,
                                      &best_mask,
                                      &best_ev);

            RerollDice(dice, best_mask);
            SortDiceSet(dice);
            rerolls_remaining--;
        }

        SortDiceSet(dice);
        double best_ev;
        int best_category = ChooseBestCategoryNoRerolls(ctx, upper_score, scored_categories, dice, &best_ev);

        int ds_index = FindDiceSetIndex(ctx, dice);
        int score = ctx->precomputed_scores[ds_index][best_category];
        total_score += score;
        upper_score = UpdateUpperScore(upper_score, best_category, score);
        if (best_category < 6) actual_upper_score += score;
        scored_categories |= (1 << best_category);
    }

    if (upper_score >= 63) total_score += 50;
    return total_score;
}

static int MaskFromBinaryString(const char *action_str) {
    int mask = 0;
    for (int i = 0; i < 5; i++) {
        if (action_str[i] == '1') {
            mask |= (1 << i);
        }
    }
    return mask;
}

// Endpoint handlers (copied from previous integrated code, but now they receive a ctx pointer):

static void handle_get_score_histogram(YatzyContext *ctx, struct MHD_Connection *connection) {
    FILE *file = fopen("score_histogram.csv", "r");
    if (file == NULL) {
        const char *err = "{\"error\":\"Could not open score histogram file\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(
            strlen(err),
            (void *) err,
            MHD_RESPMEM_PERSISTENT
        );
        AddCORSHeaders(resp);
        MHD_queue_response(connection, MHD_HTTP_NOT_FOUND, resp);
        MHD_destroy_response(resp);
        return;
    }

    // Create bins for the histogram (using same parameters as frontend)
    const int min_ev = 100;
    const int max_ev = 380;
    const int bin_count = 56;
    const double bin_width = (max_ev - min_ev) / (double)bin_count;
    int bins[56] = {0};  // Initialize all bins to 0

    char line[MAX_LINE_LENGTH];
    // Skip header if present
    if (fgets(line, MAX_LINE_LENGTH, file)) {
        // Skip header
    }

    // Read and bin the data
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        line[strcspn(line, "\n")] = 0;
        int score = atoi(line);
        if (score >= min_ev && score <= max_ev) {
            int bin_index = (int)((score - min_ev) / bin_width);
            if (bin_index >= 0 && bin_index < bin_count) {
                bins[bin_index]++;
            }
        }
    }

    // Create JSON response
    struct json_object *resp = json_object_new_object();
    struct json_object *bins_arr = json_object_new_array();

    // Add bin data
    for (int i = 0; i < bin_count; i++) {
        json_object_array_add(bins_arr, json_object_new_int(bins[i]));
    }

    // Add metadata
    json_object_object_add(resp, "bins", bins_arr);
    json_object_object_add(resp, "min_ev", json_object_new_int(min_ev));
    json_object_object_add(resp, "max_ev", json_object_new_int(max_ev));
    json_object_object_add(resp, "bin_count", json_object_new_int(bin_count));

    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(
        strlen(response_str),
        (void *)response_str,
        MHD_RESPMEM_MUST_COPY
    );
    AddCORSHeaders(mhd_response);

    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
    fclose(file);
}

// GET /state_value?upper_score=X&scored_categories=Y
static void handle_get_state_value(YatzyContext *ctx, struct MHD_Connection *connection) {
    const char *up_str = MHD_lookup_connection_value(connection, MHD_GET_ARGUMENT_KIND, "upper_score");
    const char *sc_str = MHD_lookup_connection_value(connection, MHD_GET_ARGUMENT_KIND, "scored_categories");
    if (!up_str || !sc_str) {
        const char *err = "{\"error\":\"Missing upper_score or scored_categories query params\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection,MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }

    int upper_score = atoi(up_str);
    int scored_categories = atoi(sc_str);

    double val = GetStateValue(ctx, upper_score, scored_categories);

    struct json_object *resp = json_object_new_object();
    json_object_object_add(resp, "upper_score", json_object_new_int(upper_score));
    json_object_object_add(resp, "scored_categories", json_object_new_int(scored_categories));
    json_object_object_add(resp, "expected_final_score", json_object_new_double(val));

    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str), (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    MHD_queue_response(connection,MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}

static void handle_evaluate_category_score(YatzyContext *ctx, struct MHD_Connection *connection,
                                           struct json_object *parsed) {
    struct json_object *dice_arr = json_object_object_get(parsed, "dice");
    struct json_object *cat_obj = json_object_object_get(parsed, "category_id");
    if (!dice_arr || !cat_obj) {
        const char *err = "{\"error\":\"Missing dice or category_id\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }
    int dice[5];
    for (int i = 0; i < 5; i++) {
        struct json_object *d = json_object_array_get_idx(dice_arr, i);
        dice[i] = json_object_get_int(d);
    }
    int cat_id = json_object_get_int(cat_obj);
    SortDiceSet(dice);
    int score = ctx->precomputed_scores[FindDiceSetIndex(ctx, dice)][cat_id];

    struct json_object *resp = json_object_new_object();
    json_object_object_add(resp, "category_id", json_object_new_int(cat_id));
    json_object_object_add(resp, "category_name", json_object_new_string(ctx->category_names[cat_id]));
    json_object_object_add(resp, "score", json_object_new_int(score));

    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str),
                                                                        (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}

static void handle_available_categories(YatzyContext *ctx, struct MHD_Connection *connection,
                                        struct json_object *parsed) {
    struct json_object *upper_obj = json_object_object_get(parsed, "upper_score");
    struct json_object *scored_obj = json_object_object_get(parsed, "scored_categories");
    struct json_object *dice_arr = json_object_object_get(parsed, "dice");
    struct json_object *rerolls_obj = json_object_object_get(parsed, "rerolls_remaining");

    if (!upper_obj || !scored_obj || !dice_arr || !rerolls_obj) {
        const char *err = "{\"error\":\"Missing required fields\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        AddCORSHeaders(resp);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }

    int upper_score = json_object_get_int(upper_obj);
    int scored_categories = json_object_get_int(scored_obj);
    // rerolls not needed for scoring, but we parse it for consistency
    int rerolls = json_object_get_int(rerolls_obj);

    int dice[5];
    for (int i = 0; i < 5; i++) {
        struct json_object *d = json_object_array_get_idx(dice_arr, i);
        if (!d) {
            const char *err = "{\"error\":\"Invalid dice array\"}";
            struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err,
                                                                        MHD_RESPMEM_PERSISTENT);
            AddCORSHeaders(resp);
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
            MHD_destroy_response(resp);
            return;
        }
        dice[i] = json_object_get_int(d);
    }

    SortDiceSet(dice);
    int ds_index = FindDiceSetIndex(ctx, dice);

    // Construct response array
    struct json_object *resp = json_object_new_object();
    struct json_object *arr = json_object_new_array();

    for (int c = 0; c < CATEGORY_COUNT; c++) {
        int scr = ctx->precomputed_scores[ds_index][c];
        // valid if not scored and scr>0
        int is_already_scored = IS_CATEGORY_SCORED(scored_categories, c);
        bool valid = (!is_already_scored && scr > 0);

        struct json_object *cat_obj = json_object_new_object();
        json_object_object_add(cat_obj, "id", json_object_new_int(c));
        json_object_object_add(cat_obj, "name", json_object_new_string(ctx->category_names[c]));
        json_object_object_add(cat_obj, "score", json_object_new_int(scr));
        json_object_object_add(cat_obj, "valid", json_object_new_boolean(valid));
        json_object_array_add(arr, cat_obj);
    }

    json_object_object_add(resp, "categories", arr);

    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str), (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    AddCORSHeaders(mhd_response);
    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}

static void handle_evaluate_all_categories(YatzyContext *ctx, struct MHD_Connection *connection,
                                           struct json_object *parsed) {
    struct json_object *upper_obj = json_object_object_get(parsed, "upper_score");
    struct json_object *scored_obj = json_object_object_get(parsed, "scored_categories");
    struct json_object *dice_arr = json_object_object_get(parsed, "dice");
    struct json_object *rerolls_obj = json_object_object_get(parsed, "rerolls_remaining");

    if (!upper_obj || !scored_obj || !dice_arr || !rerolls_obj) {
        const char *err = "{\"error\":\"Missing required fields\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }

    int upper_score = json_object_get_int(upper_obj);
    int scored_categories = json_object_get_int(scored_obj);
    int rerolls = json_object_get_int(rerolls_obj);

    if (rerolls != 0) {
        const char *err = "{\"error\":\"rerolls_remaining must be 0 for this endpoint\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }

    int dice[5];
    for (int i = 0; i < 5; i++) {
        struct json_object *d = json_object_array_get_idx(dice_arr, i);
        if (!d) {
            const char *err = "{\"error\":\"Invalid dice array\"}";
            struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err,
                                                                        MHD_RESPMEM_PERSISTENT);
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
            MHD_destroy_response(resp);
            return;
        }
        dice[i] = json_object_get_int(d);
    }
    SortDiceSet(dice);

    int ds_index = FindDiceSetIndex(ctx, dice);
    struct json_object *resp = json_object_new_object();
    struct json_object *arr = json_object_new_array();

    // For each unscored category, compute the EV if chosen now
    for (int c = 0; c < CATEGORY_COUNT; c++) {
        if (!IS_CATEGORY_SCORED(scored_categories, c)) {
            int scr = ctx->precomputed_scores[ds_index][c];
            int new_up = UpdateUpperScore(upper_score, c, scr);
            int new_scored = scored_categories | (1 << c);
            double val = scr + GetStateValue(ctx, new_up, new_scored);

            struct json_object *cat_obj = json_object_new_object();
            json_object_object_add(cat_obj, "id", json_object_new_int(c));
            json_object_object_add(cat_obj, "name", json_object_new_string(ctx->category_names[c]));
            json_object_object_add(cat_obj, "score", json_object_new_int(scr));
            json_object_object_add(cat_obj, "expected_value_if_chosen", json_object_new_double(val));
            json_object_array_add(arr, cat_obj);
        }
    }

    json_object_object_add(resp, "categories", arr);
    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str), (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}

static void handle_evaluate_actions(YatzyContext *ctx, struct MHD_Connection *connection, struct json_object *parsed) {
    struct json_object *upper_obj = json_object_object_get(parsed, "upper_score");
    struct json_object *scored_obj = json_object_object_get(parsed, "scored_categories");
    struct json_object *dice_arr = json_object_object_get(parsed, "dice");
    struct json_object *rerolls_obj = json_object_object_get(parsed, "rerolls_remaining");

    if (!upper_obj || !scored_obj || !dice_arr || !rerolls_obj) {
        const char *err = "{\"error\":\"Missing required fields\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }

    int upper_score = json_object_get_int(upper_obj);
    int scored_categories = json_object_get_int(scored_obj);
    int rerolls = json_object_get_int(rerolls_obj);

    if (rerolls <= 0) {
        const char *err = "{\"error\":\"rerolls_remaining must be > 0\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }

    int dice[5];
    for (int i = 0; i < 5; i++) {
        struct json_object *d = json_object_array_get_idx(dice_arr, i);
        if (!d) {
            const char *err = "{\"error\":\"Invalid dice array\"}";
            struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err,
                                                                        MHD_RESPMEM_PERSISTENT);
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
            MHD_destroy_response(resp);
            return;
        }
        dice[i] = json_object_get_int(d);
    }
    SortDiceSet(dice);

    // Compute E_ds_0 for no rerolls scenario
    double E_ds_0[252]; {
        YatzyState state = {upper_score, scored_categories};
#pragma omp parallel for
        for (int ds_i = 0; ds_i < 252; ds_i++) {
            double best_val = -INFINITY;
            for (int c = 0; c < CATEGORY_COUNT; c++) {
                if (!IS_CATEGORY_SCORED(scored_categories, c)) {
                    int scr = ctx->precomputed_scores[ds_i][c];
                    int new_up = UpdateUpperScore(upper_score, c, scr);
                    int new_scored = scored_categories | (1 << c);
                    double val = scr + GetStateValue(ctx, new_up, new_scored);
                    if (val > best_val) best_val = val;
                }
            }
            E_ds_0[ds_i] = best_val;
        }
    }

    // If rerolls=2, we need E_ds_1 for one reroll scenario
    double E_ds_1[252];
    if (rerolls == 2) {
#pragma omp parallel for
        for (int ds_i = 0; ds_i < 252; ds_i++) {
            double best_val = -INFINITY;
            const double (*row)[252] = ctx->transition_table[ds_i];
            for (int mask = 0; mask < 32; mask++) {
                double ev = 0.0;
                for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                    ev += row[mask][ds2_i] * E_ds_0[ds2_i];
                }
                if (ev > best_val) best_val = ev;
            }
            E_ds_1[ds_i] = best_val;
        }
    }

    const double *E_ds_for_masks = (rerolls == 1) ? E_ds_0 : E_ds_1;

    int sorted[5];
    for (int i = 0; i < 5; i++) sorted[i] = dice[i];
    SortDiceSet(sorted);
    int ds_index = FindDiceSetIndex(ctx, sorted);

    struct json_object *resp = json_object_new_object();
    struct json_object *arr = json_object_new_array();
    for (int mask = 0; mask < 32; mask++) {
        EVProbabilityPair distribution[252];
        ComputeDistributionForRerollMask(ctx, ds_index, E_ds_for_masks, mask, distribution);

        double ev = 0.0;
        struct json_object *dist_arr = json_object_new_array();
        for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
            if (distribution[ds2_i].probability > 0.0) {
                ev += distribution[ds2_i].ev * distribution[ds2_i].probability;

                struct json_object *entry = json_object_new_object();
                json_object_object_add(entry, "ds2_index", json_object_new_int(distribution[ds2_i].ds2_index));
                json_object_object_add(entry, "ev", json_object_new_double(distribution[ds2_i].ev));
                json_object_object_add(entry, "probability", json_object_new_double(distribution[ds2_i].probability));
                json_object_array_add(dist_arr, entry);
            }
        }

        struct json_object *o = json_object_new_object();
        json_object_object_add(o, "mask", json_object_new_int(mask));
        char bin_str[6] = {0};
        for (int i = 0; i < 5; i++) bin_str[i] = (mask & (1 << i)) ? '1' : '0';
        json_object_object_add(o, "binary", json_object_new_string(bin_str));
        json_object_object_add(o, "expected_value", json_object_new_double(ev));
        json_object_object_add(o, "distribution", dist_arr);

        json_object_array_add(arr, o);
    }
    json_object_object_add(resp, "actions", arr);

    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str), (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    AddCORSHeaders(mhd_response);

    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}

static void handle_suggest_optimal_action(YatzyContext *ctx, struct MHD_Connection *connection,
                                          struct json_object *parsed) {
    struct json_object *upper_score_obj = json_object_object_get(parsed, "upper_score");
    struct json_object *scored_categories_obj = json_object_object_get(parsed, "scored_categories");
    struct json_object *dice_arr = json_object_object_get(parsed, "dice");
    struct json_object *rerolls_obj = json_object_object_get(parsed, "rerolls_remaining");

    if (!upper_score_obj || !scored_categories_obj || !dice_arr || !rerolls_obj) {
        const char *error_msg = "{\"error\":\"Missing required fields\"}";
        struct MHD_Response *error_response = MHD_create_response_from_buffer(strlen(error_msg),
                                                                              (void *) error_msg,
                                                                              MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
        MHD_destroy_response(error_response);
        return;
    }

    int upper_score = json_object_get_int(upper_score_obj);
    int scored_categories = json_object_get_int(scored_categories_obj);
    int rerolls = json_object_get_int(rerolls_obj);

    int dice[5];
    for (int i = 0; i < 5; i++) {
        struct json_object *die_val = json_object_array_get_idx(dice_arr, i);
        if (!die_val) {
            const char *error_msg = "{\"error\":\"Invalid dice array\"}";
            struct MHD_Response *error_response = MHD_create_response_from_buffer(strlen(error_msg),
                (void *) error_msg,
                MHD_RESPMEM_PERSISTENT);
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
            MHD_destroy_response(error_response);
            return;
        }
        dice[i] = json_object_get_int(die_val);
    }
    SortDiceSet(dice);


    // Create a new response object
    struct json_object *resp = json_object_new_object();

    // Initialize response keys
    json_object_object_add(resp, "dice_set", NULL);

    // Conver the dice set into a string
    char dice_str[8] = {0};
    for (int i = 0; i < 5; i++) {
        dice_str[i] = dice[i] + '0';
    }
    json_object_object_add(resp, "dice_set", json_object_new_string(dice_str));

    json_object_object_add(resp, "best_category", NULL);
    json_object_object_add(resp, "best_reroll", NULL);
    json_object_object_add(resp, "expected_value", NULL);

    if (rerolls > 0) {
        int best_mask;
        double ev;
        ComputeBestRerollStrategy(ctx, upper_score, scored_categories, dice, rerolls, &best_mask, &ev);

        // Create a "best_reroll" object
        struct json_object *reroll_obj = json_object_new_object();
        json_object_object_add(reroll_obj, "id", json_object_new_int(best_mask));

        // Binary string for clarity
        char bin_str[6] = {0};
        for (int i = 0; i < 5; i++) {
            bin_str[i] = (best_mask & (1 << i)) ? '1' : '0';
        }
        json_object_object_add(reroll_obj, "mask_binary", json_object_new_string(bin_str));
        json_object_object_add(resp, "best_reroll", reroll_obj);
        json_object_object_add(resp, "expected_value", json_object_new_double(ev));

        // Do NOT call json_object_put(reroll_obj) after adding it to `resp`
    }

    if (rerolls == 0) {
        double best_ev;
        int best_category = ChooseBestCategoryNoRerolls(ctx, upper_score, scored_categories, dice, &best_ev);

        if (best_category >= 0) {
            struct json_object *category_obj = json_object_new_object();
            json_object_object_add(category_obj, "id", json_object_new_int(best_category));
            json_object_object_add(category_obj, "name", json_object_new_string(ctx->category_names[best_category]));
            json_object_object_add(resp, "best_category", category_obj);
            json_object_object_add(resp, "expected_value", json_object_new_double(best_ev));
        }
    }

    // Send the response
    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str), (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    // Add CORS headers here
    AddCORSHeaders(mhd_response);

    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}

static void handle_evaluate_user_action(YatzyContext *ctx, struct MHD_Connection *connection,
                                        struct json_object *parsed) {
    struct json_object *upper_score_obj = json_object_object_get(parsed, "upper_score");
    struct json_object *scored_categories_obj = json_object_object_get(parsed, "scored_categories");
    struct json_object *dice_arr = json_object_object_get(parsed, "dice");
    struct json_object *rerolls_obj = json_object_object_get(parsed, "rerolls_remaining");
    struct json_object *user_action_obj = json_object_object_get(parsed, "user_action");

    if (!upper_score_obj || !scored_categories_obj || !dice_arr || !rerolls_obj || !user_action_obj) {
        const char *error_msg = "{\"error\":\"Missing required fields\"}";
        struct MHD_Response *error_response = MHD_create_response_from_buffer(
            strlen(error_msg), (void *) error_msg, MHD_RESPMEM_PERSISTENT);
        AddCORSHeaders(error_response); // Add CORS headers
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
        MHD_destroy_response(error_response);
        return;
    }

    int upper_score = json_object_get_int(upper_score_obj);
    int scored_categories = json_object_get_int(scored_categories_obj);
    int rerolls = json_object_get_int(rerolls_obj);

    int dice[5];
    for (int i = 0; i < 5; i++) {
        struct json_object *die_val = json_object_array_get_idx(dice_arr, i);
        if (!die_val) {
            const char *error_msg = "{\"error\":\"Invalid dice array\"}";
            struct MHD_Response *error_response = MHD_create_response_from_buffer(
                strlen(error_msg), (void *) error_msg, MHD_RESPMEM_PERSISTENT);
            AddCORSHeaders(error_response); // Add CORS headers
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
            MHD_destroy_response(error_response);
            return;
        }
        dice[i] = json_object_get_int(die_val);
    }
    SortDiceSet(dice);

    struct json_object *resp = json_object_new_object();
    json_object_object_add(resp, "expected_value", NULL);
    json_object_object_add(resp, "best_category", NULL);
    json_object_object_add(resp, "best_reroll", NULL);

    if (rerolls > 0) {
        struct json_object *best_reroll_obj = json_object_object_get(user_action_obj, "best_reroll");
        if (!best_reroll_obj) {
            const char *error_msg = "{\"error\":\"User action missing best_reroll field\"}";
            struct MHD_Response *error_response = MHD_create_response_from_buffer(
                strlen(error_msg), (void *) error_msg, MHD_RESPMEM_PERSISTENT);
            AddCORSHeaders(error_response); // Add CORS headers
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
            MHD_destroy_response(error_response);
            json_object_put(resp); // Free resp
            return;
        }

        struct json_object *mask_id_obj = json_object_object_get(best_reroll_obj, "id");
        if (!mask_id_obj) {
            const char *error_msg = "{\"error\":\"User action missing id in best_reroll\"}";
            struct MHD_Response *error_response = MHD_create_response_from_buffer(
                strlen(error_msg), (void *) error_msg, MHD_RESPMEM_PERSISTENT);
            AddCORSHeaders(error_response); // Add CORS headers
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
            MHD_destroy_response(error_response);
            json_object_put(resp);
            return;
        }

        int mask = json_object_get_int(mask_id_obj);
        double ev = EvaluateChosenRerollMask(ctx, upper_score, scored_categories, dice, mask, rerolls);

        json_object_object_add(resp, "expected_value", json_object_new_double(ev));

        // Increment ref count before adding to resp
        json_object_get(best_reroll_obj);
        json_object_object_add(resp, "best_reroll", best_reroll_obj);
    } else {
        struct json_object *best_category_obj = json_object_object_get(user_action_obj, "best_category");
        if (!best_category_obj) {
            const char *error_msg = "{\"error\":\"User action missing best_category field\"}";
            struct MHD_Response *error_response = MHD_create_response_from_buffer(
                strlen(error_msg), (void *) error_msg, MHD_RESPMEM_PERSISTENT);
            AddCORSHeaders(error_response); // Add CORS headers
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
            MHD_destroy_response(error_response);
            json_object_put(resp);
            return;
        }

        struct json_object *category_id_obj = json_object_object_get(best_category_obj, "id");
        if (!category_id_obj) {
            const char *error_msg = "{\"error\":\"User action missing id in best_category\"}";
            struct MHD_Response *error_response = MHD_create_response_from_buffer(
                strlen(error_msg), (void *) error_msg, MHD_RESPMEM_PERSISTENT);
            AddCORSHeaders(error_response); // Add CORS headers
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
            MHD_destroy_response(error_response);
            json_object_put(resp);
            return;
        }

        int category_id = json_object_get_int(category_id_obj);
        double ev = EvaluateChosenCategory(ctx, upper_score, scored_categories, dice, category_id);

        json_object_object_add(resp, "expected_value", json_object_new_double(ev));

        // Increment ref count before adding to resp
        json_object_get(best_category_obj);
        json_object_object_add(resp, "best_category", best_category_obj);
    }

    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str), (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    AddCORSHeaders(mhd_response); // Add CORS headers to the successful response
    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}


// Utility function to send an error response with CORS headers
static enum MHD_Result respond_with_error(struct MHD_Connection *connection, int status_code,
                                          const char *error_message, struct RequestContext *req_ctx) {
    struct MHD_Response *response = MHD_create_response_from_buffer(strlen(error_message),
                                                                    (void *) error_message,
                                                                    MHD_RESPMEM_PERSISTENT);
    AddCORSHeaders(response);
    MHD_queue_response(connection, status_code, response);
    MHD_destroy_response(response);

    if (req_ctx) {
        free(req_ctx->post_data);
        free(req_ctx);
    }
    return MHD_YES;
}

// Main request handler:
static enum MHD_Result answer_to_connection(void *cls, struct MHD_Connection *connection,
                                            const char *url, const char *method,
                                            const char *version, const char *upload_data,
                                            size_t *upload_data_size, void **con_cls) {
    YatzyContext *ctx = (YatzyContext *) cls; // Server context

    // Initialize request context if not already done
    if (*con_cls == NULL) {
        struct RequestContext *req_ctx = (struct RequestContext *) calloc(1, sizeof(struct RequestContext));
        req_ctx->post_data = NULL;
        req_ctx->post_size = 0;
        *con_cls = req_ctx;
        return MHD_YES;
    }

    struct RequestContext *req_ctx = (struct RequestContext *) (*con_cls);

    // Handle OPTIONS preflight requests
    if (strcmp(method, "OPTIONS") == 0) {
        struct MHD_Response *options_response = MHD_create_response_from_buffer(0, "", MHD_RESPMEM_PERSISTENT);
        AddCORSHeaders(options_response);
        MHD_add_response_header(options_response, "Access-Control-Max-Age", "86400");
        MHD_queue_response(connection, MHD_HTTP_OK, options_response);
        MHD_destroy_response(options_response);

        free(req_ctx->post_data);
        free(req_ctx);
        *con_cls = NULL;
        return MHD_YES;
    }

    // Handle POST requests
    if (strcmp(method, "POST") == 0) {
        // Collect upload data
        if (*upload_data_size != 0) {
            size_t new_size = req_ctx->post_size + *upload_data_size;
            req_ctx->post_data = (char *) realloc(req_ctx->post_data, new_size + 1);
            memcpy(req_ctx->post_data + req_ctx->post_size, upload_data, *upload_data_size);
            req_ctx->post_size = new_size;
            req_ctx->post_data[req_ctx->post_size] = '\0';
            *upload_data_size = 0;
            return MHD_YES;
        }

        // Process full POST data
        if (!req_ctx->post_data) {
            return respond_with_error(connection, MHD_HTTP_BAD_REQUEST, "No data received", req_ctx);
        }

        struct json_object *parsed = json_tokener_parse(req_ctx->post_data);
        if (!parsed) {
            return respond_with_error(connection, MHD_HTTP_BAD_REQUEST, "Invalid JSON", req_ctx);
        }

        // Route the POST request
        if (strcmp(url, "/evaluate_category_score") == 0) {
            handle_evaluate_category_score(ctx, connection, parsed);
        } else if (strcmp(url, "/available_categories") == 0) {
            handle_available_categories(ctx, connection, parsed);
        } else if (strcmp(url, "/evaluate_all_categories") == 0) {
            handle_evaluate_all_categories(ctx, connection, parsed);
        } else if (strcmp(url, "/evaluate_actions") == 0) {
            handle_evaluate_actions(ctx, connection, parsed);
        } else if (strcmp(url, "/suggest_optimal_action") == 0) {
            handle_suggest_optimal_action(ctx, connection, parsed);
        } else if (strcmp(url, "/evaluate_user_action") == 0) {
            handle_evaluate_user_action(ctx, connection, parsed);
        } else {
            json_object_put(parsed);
            return respond_with_error(connection, MHD_HTTP_NOT_FOUND, "Unknown endpoint", req_ctx);
        }

        json_object_put(parsed);
        free(req_ctx->post_data);
        free(req_ctx);
        *con_cls = NULL;
        return MHD_YES;
    }

    // Handle GET requests
    if (strcmp(method, "GET") == 0) {
        if (strncmp(url, "/state_value", 12) == 0) {
            handle_get_state_value(ctx, connection);
        } else if (strcmp(url, "/score_histogram") == 0) {
            handle_get_score_histogram(ctx, connection);
        } else {
            return respond_with_error(connection, MHD_HTTP_NOT_FOUND, "Unknown endpoint", req_ctx);
        }

        free(req_ctx->post_data);
        free(req_ctx);
        *con_cls = NULL;
        return MHD_YES;
    }

    // Unsupported methods
    return respond_with_error(connection, MHD_HTTP_METHOD_NOT_ALLOWED, "Only POST and GET supported", req_ctx);
}

volatile bool running = true;

void handle_signal(int signal) {
    running = false;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Get the base path from the environment variable
    const char *base_path = getenv("YATZY_BASE_PATH");
    if (!base_path) {
        base_path = "."; // Default to current directory if not set
    }

    // Change to the specified base path
    if (chdir(base_path) != 0) {
        perror("chdir failed");
        exit(EXIT_FAILURE);
    }

    // Log the working directory
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Working directory changed to: %s\n", cwd);
    } else {
        perror("getcwd failed");
    }

    printf("Starting yatzy API server...\n");

    printf("Initializing context...\n");
    YatzyContext *ctx = CreateYatzyContext();
    printf("Context created!\n");

    printf("Precomputing all state values...\n");
    ComputeAllStateValues(ctx);
    printf("State values computed!\n");

    struct MHD_Daemon *daemon = MHD_start_daemon(MHD_USE_INTERNAL_POLLING_THREAD,
                                                 9000,
                                                 NULL, NULL,
                                                 &answer_to_connection, ctx,
                                                 MHD_OPTION_END);

    if (NULL == daemon) {
        FreeYatzyContext(ctx);
        return 1;
    }

    signal(SIGINT, handle_signal); // Handle Ctrl+C (SIGINT) to stop the server

    printf("Server is running. Press Ctrl+C to stop.\n");
    while (running) {
        sleep(1); // Sleep to avoid busy-waiting
    }

    printf("\nStopping server...\n");
    MHD_stop_daemon(daemon);
    FreeYatzyContext(ctx);

    return 0;
}