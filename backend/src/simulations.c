#include <dice_mechanics.h>
#include <context.h>
#include <computations.h>

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