#ifndef CONTEXT_H
#define CONTEXT_H

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

// Defines the total number of states by:
// 1. Calculating 1 << 15, which represents 2^15 (shifting 1 left by 15 bits).
// 2. Multiplying the result (32,768) by 64 to get the total number of states.
#define NUM_STATES (64*(1<<15))

// Checks if category 'cat' is scored by:
// 1. Creating a bitmask with a 1 at position 'cat' using (1 << cat).
// 2. Performing a bitwise AND between the bitmask and 'scored'.
// 3. Checking if the result is non-zero, which indicates the category is scored.
#define IS_CATEGORY_SCORED(scored,cat) ((scored & (1 << cat)) != 0)

#define SET_CATEGORY(scored,cat) ((scored)|=(1<<(cat)))
#define CLEAR_CATEGORY(scored,cat) ((scored)&=~(1<<(cat)))
#define STATE_INDEX(upper_score, scored_categories) ((upper_score)*(1<<15)+(scored_categories))

typedef struct {
    const char *category_names[CATEGORY_COUNT];

    int all_dice_sets[252][5];
    int num_combinations;
    int index_lookup[6][6][6][6][6];
    int precomputed_scores[252][CATEGORY_COUNT];
    int scored_category_count_cache[1 << CATEGORY_COUNT];
    int factorial[6 + 1];

    double *state_values;
    double dice_set_probabilities[252];
    double transition_table[252][32][252];
} YatzyContext;

typedef struct {
    int upper_score;
    int scored_categories;
} YatzyState;

double GetStateValue(const YatzyContext *ctx, int up, int scored);

YatzyContext *CreateYatzyContext();

void FreeYatzyContext(YatzyContext *ctx);

#endif // CONTEXT_H
