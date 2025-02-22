#include <computations.h>
#include <context.h>
#include <dice_mechanics.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_CATEGORIES 15
#define NUM_GAMES 1000
#define NUM_DICE 5  // Assuming 5 dice per roll

// A bitmask with all categories set.
// For example, if NUM_CATEGORIES is 15, then FULL_SCORECARD == (1 << 15) - 1.
#define FULL_SCORECARD ((1 << NUM_CATEGORIES) - 1)

// Helper: Validate that the returned dice set index and category are within expected bounds.
// Adjust MAX_DS_INDEX according to your actual precomputed_scores array size.
#define MAX_DS_INDEX 1024  // Example maximum; adjust accordingly.

// Fallback: Select the first available (unscored) category.
int SelectFallbackCategory(int scored_categories) {
    for (int cat = 0; cat < NUM_CATEGORIES; cat++) {
        if (!(scored_categories & (1 << cat))) {
            return cat;
        }
    }
    return -1; // No category available (should not happen)
}

int SimulateGame(YatzyContext *ctx, int category_scores[NUM_CATEGORIES]) {
    if (!ctx) {
        fprintf(stderr, "SimulateGame error: ctx is NULL.\n");
        return -1;
    }
    if (!ctx->precomputed_scores) {
        fprintf(stderr, "SimulateGame error: ctx->precomputed_scores is NULL.\n");
        return -1;
    }

    int upper_score = 0;
    int scored_categories = 0;
    int total_score = 0;
    int actual_upper_score = 0;

    // Initialize category scores to indicate they are not scored.
    for (int i = 0; i < NUM_CATEGORIES; i++) {
        category_scores[i] = -1;
    }

    // Loop over each turn (max NUM_CATEGORIES turns).
    for (int turn = 0; turn < NUM_CATEGORIES; turn++) {
        // If all categories have been scored, exit the simulation early.
        if (scored_categories == FULL_SCORECARD) {
            break;
        }

        int dice[NUM_DICE] = {0}; // Initialize dice array.
        RollDice(dice);

        int rerolls_remaining = 2;
        while (rerolls_remaining > 0) {
            SortDiceSet(dice);

            int best_mask = 0;
            double best_ev = 0.0;
            ComputeBestRerollStrategy(ctx,
                                      upper_score,
                                      scored_categories,
                                      dice,
                                      rerolls_remaining,
                                      &best_mask,
                                      &best_ev);

            RerollDice(dice, best_mask);
            rerolls_remaining--;
        }

        SortDiceSet(dice);

        double best_ev = 0.0;
        int best_category = ChooseBestCategoryNoRerolls(ctx, upper_score, scored_categories, dice, &best_ev);

        // If best_category is -1 or out of bounds, choose a fallback.
        if (best_category < 0 || best_category >= NUM_CATEGORIES) {
            fprintf(stderr, "Warning: ChooseBestCategoryNoRerolls returned %d. Falling back.\n", best_category);
            best_category = SelectFallbackCategory(scored_categories);
            if (best_category < 0 || best_category >= NUM_CATEGORIES) {
                fprintf(stderr, "Error: No valid category available as fallback.\n");
                return -1;
            }
        }

        int ds_index = FindDiceSetIndex(ctx, dice);
        if (ds_index < 0 || ds_index >= MAX_DS_INDEX) {
            fprintf(stderr, "Error: ds_index (%d) out of bounds.\n", ds_index);
            return -1;
        }

        int score = ctx->precomputed_scores[ds_index][best_category];
        total_score += score;
        upper_score = UpdateUpperScore(upper_score, best_category, score);
        if (best_category < 6)
            actual_upper_score += score;

        // Mark the category as scored.
        scored_categories |= (1 << best_category);

        // Record the score for this category.
        category_scores[best_category] = score;
    }
    // Add bonus if upper_score is high enough.
    if (upper_score >= 63)
        total_score += 50;
    return total_score;
}

int main() {
    // Initialize Yatzy context.
    YatzyContext *ctx = CreateYatzyContext();
    if (!ctx) {
        fprintf(stderr, "Error: CreateYatzyContext() returned NULL.\n");
        return 1;
    }

    // Allocate a 2D array for the per-category score distribution.
    int **categoryDistribution = malloc(NUM_CATEGORIES * sizeof(int *));
    if (!categoryDistribution) {
        fprintf(stderr, "Failed to allocate memory for category distribution.\n");
        free(ctx);
        return 1;
    }
    for (int cat = 0; cat < NUM_CATEGORIES; cat++) {
        categoryDistribution[cat] = malloc(NUM_GAMES * sizeof(int));
        if (!categoryDistribution[cat]) {
            fprintf(stderr, "Failed to allocate memory for category %d.\n", cat);
            for (int j = 0; j < cat; j++) {
                free(categoryDistribution[j]);
            }
            free(categoryDistribution);
            free(ctx);
            return 1;
        }
    }

    // Array to store the total score for each game.
    int totalScores[NUM_GAMES] = {0};

    // Run the simulation for NUM_GAMES games.
    for (int game = 0; game < NUM_GAMES; game++) {
        int category_scores[NUM_CATEGORIES] = {0};
        int total_score = SimulateGame(ctx, category_scores);
        if (total_score < 0) {
            // If SimulateGame returns an error code.
            fprintf(stderr, "Simulation error in game %d\n", game + 1);
            continue; // Skip this game.
        }
        totalScores[game] = total_score;

        // Record the scores for each category for this game.
        for (int cat = 0; cat < NUM_CATEGORIES; cat++) {
            categoryDistribution[cat][game] = category_scores[cat];
        }

        // Optionally, print the total score for this game.
        printf("Game %d: Total Score: %d\n", game + 1, total_score);
    }

    // Write CSV file for total scores.
    FILE *f_total = fopen("total_scores.csv", "w");
    if (!f_total) {
        fprintf(stderr, "Error: Unable to open total_scores.csv for writing.\n");
        for (int cat = 0; cat < NUM_CATEGORIES; cat++) {
            free(categoryDistribution[cat]);
        }
        free(categoryDistribution);
        free(ctx);
        return 1;
    }
    fprintf(f_total, "Game,TotalScore\n");
    for (int game = 0; game < NUM_GAMES; game++) {
        fprintf(f_total, "%d,%d\n", game + 1, totalScores[game]);
    }
    fclose(f_total);
    printf("Total scores written to total_scores.csv\n");

    // Write CSV files for each category.
    for (int cat = 0; cat < NUM_CATEGORIES; cat++) {
        char filename[64];
        snprintf(filename, sizeof(filename), "category_%d.csv", cat);
        FILE *f_cat = fopen(filename, "w");
        if (!f_cat) {
            fprintf(stderr, "Error: Unable to open %s for writing.\n", filename);
            continue;
        }
        fprintf(f_cat, "Game,Score\n");
        for (int game = 0; game < NUM_GAMES; game++) {
            fprintf(f_cat, "%d,%d\n", game + 1, categoryDistribution[cat][game]);
        }
        fclose(f_cat);
        printf("Category %d scores written to %s\n", cat, filename);
    }

    // Free allocated memory.
    for (int cat = 0; cat < NUM_CATEGORIES; cat++) {
        free(categoryDistribution[cat]);
    }
    free(categoryDistribution);
    free(ctx);

    return 0;
}
