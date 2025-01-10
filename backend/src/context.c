#include <stdlib.h>

#include "context.h"
#include <computations.h>

double GetStateValue(const YatzyContext *ctx, int up, int scored) {
    return ctx->state_values[STATE_INDEX(up, scored)];
}

YatzyContext *CreateYatzyContext() {
    YatzyContext *ctx = malloc(sizeof(YatzyContext));

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

void FreeYatzyContext(YatzyContext *ctx) {
    free(ctx->state_values);
    free(ctx);
}