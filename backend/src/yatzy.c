#include <computations.h>
#include <math.h>
#include <stdio.h>
#include <signal.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <json-c/json.h>

#include <context.h>
#include <dice_mechanics.h>
#include <webserver.h>

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


static void ComputeAllStateValues(YatzyContext *ctx) {
    for (int scored_count = 15; scored_count >= 0; scored_count--) {
        char filename[64];
        snprintf(filename, sizeof(filename), "data/states_%d.bin", scored_count);

        if (scored_count == 15) {
            if (!LoadStateValuesForCount(ctx, scored_count, filename)) {
                SaveStateValuesForCount(ctx, scored_count, filename);
            }
            continue;
        }

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

static double SuggestOptimalAction(const YatzyContext *ctx,
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

    // Start the webserver
    int result = start_webserver(ctx);

    FreeYatzyContext(ctx);

    return 0;
}