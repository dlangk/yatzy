/*
 * context.h — Core data structures for the Yatzy DP solver
 *
 * Defines YatzyContext (all precomputed tables and the DP result table)
 * and YatzyState (a turn-start state S = (m, C)).
 *
 * Field names reference the pseudocode notation from
 * theory/optimal_yahtzee_pseudocode.md.
 */
#ifndef CONTEXT_H
#define CONTEXT_H

/* Number of scoring categories in Scandinavian Yatzy (Ones through Yatzy). */
#define CATEGORY_COUNT 15

/* Category indices — used as bit positions in the scored_categories bitmask C. */
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

/*
 * Total number of game states: 64 possible upper-section scores m ∈ [0,63]
 * times 2^15 scored-category bitmasks C = 2,097,152 states.
 */
#define NUM_STATES (64 * (1 << 15))

/* Test whether category 'cat' has been scored (bit 'cat' is set in C). */
#define IS_CATEGORY_SCORED(scored, cat) ((scored & (1 << cat)) != 0)

/* Set/clear the bit for category 'cat' in the scored bitmask C. */
#define SET_CATEGORY(scored, cat) ((scored) |= (1 << (cat)))
#define CLEAR_CATEGORY(scored, cat) ((scored) &= ~(1 << (cat)))

/*
 * Map state S = (m, C) to a flat array index in E_table.
 * Layout: m * 2^15 + C.
 */
#define STATE_INDEX(upper_score, scored_categories) \
    ((upper_score) * (1 << 15) + (scored_categories))

/*
 * Sparse CSR representation of the P(r' → r) transition table.
 *
 * The dense table (252 × 32 × 252 doubles = 16.3 MB) is ~79% zeros.
 * This sparse format stores only non-zero entries (~424K) in ~5 MB,
 * fitting in L2 cache on Apple Silicon (12 MB shared among P-cores).
 *
 * Row index for (ds, mask) = ds * 32 + mask.  Total rows = 8064.
 * row_offsets[r] .. row_offsets[r+1] gives the range in values/col_indices.
 */
typedef struct {
    double *values;          /* probability values, flat CSR array */
    int *col_indices;        /* target dice set index for each non-zero entry */
    int row_offsets[252 * 32 + 1]; /* CSR row pointers */
    int total_nonzero;
} SparseTransitionTable;

typedef struct {
    /* Human-readable names for each of the 15 categories. */
    const char *category_names[CATEGORY_COUNT];

    /* R_{5,6}: all 252 distinct sorted 5-dice multisets. */
    int all_dice_sets[252][5];
    /* |R_{5,6}| (always 252 = C(10,5)). */
    int num_combinations;
    /* Reverse lookup: sorted dice values → index in R_{5,6}. */
    int index_lookup[6][6][6][6][6];
    /* precomputed_scores[r][c] = s(S, r, c) for dice set r and category c. */
    int precomputed_scores[252][CATEGORY_COUNT];
    /* Popcount cache: scored_category_count_cache[C] = |C|. */
    int scored_category_count_cache[1 << CATEGORY_COUNT];
    /* factorial[n] for n ∈ 0..6, used for multinomial P(⊥ → r) calculations. */
    int factorial[6 + 1];

    /* E_table[S]: DP result — expected game value for state S = (m, C). */
    double *state_values;
    /* P(⊥ → r): probability of rolling each r ∈ R_{5,6} from 5 fresh dice. */
    double dice_set_probabilities[252];
    /*
     * P(r' → r) sparse transition table in CSR format.
     * Replaces the dense transition_table[252][32][252].
     */
    SparseTransitionTable sparse_transitions;
} YatzyContext;

typedef struct {
    int upper_score;          /* m: upper-section total, capped at 63. */
    int scored_categories;    /* C: bitmask of which categories have been scored. */
} YatzyState;

/* Look up E_table[S] for a given state (m, C). */
double GetStateValue(const YatzyContext *ctx, int up, int scored);

/*
 * Allocate and zero-initialize a YatzyContext. Sets up category names and
 * allocates the state_values (E_table) array. Returns NULL on failure.
 * Call PrecomputeLookupTables() after this to populate Phase 0 tables.
 */
YatzyContext *CreateYatzyContext();

/*
 * Phase 0 orchestrator: build all lookup tables in dependency order.
 * Must be called after CreateYatzyContext() and before ComputeAllStateValues().
 */
void PrecomputeLookupTables(YatzyContext *ctx);

/* Free all memory owned by the context. Safe to call with NULL. */
void FreeYatzyContext(YatzyContext *ctx);

#endif // CONTEXT_H
