#include <stdlib.h>
#include <math.h>

#include <context.h>

void RollDice(int dice[5]) {
    for (int i = 0; i < 5; i++) {
        dice[i] = rand() % 6 + 1;
    }
}

void RerollDice(int dice[5], const int mask) {
    for (int i = 0; i < 5; i++) {
        if (mask & (1 << i)) {
            dice[i] = rand() % 6 + 1;
        }
    }
}

void SortDiceSet(int arr[5]) {
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 5; j++) {
            if (arr[j] < arr[i]) {
                const int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
}

/**
 * @brief Finds the index of a specific sorted dice set in the context's index lookup table.
 *
 * This function retrieves the precomputed index for a sorted set of dice values (`dice[5]`)
 * by accessing the `index_lookup` table within the `YatzyContext`. The table maps all possible
 * sorted dice combinations to unique indices, which can be used for efficient state lookup
 * or probability computation in the game logic.
 *
 * The steps are:
 * 1. The input `dice` array is expected to be sorted in ascending order (values between 1-6).
 * 2. Each dice value is adjusted by subtracting 1 to align with 0-based indexing.
 * 3. The function performs a lookup in the 5-dimensional `index_lookup` table using the adjusted
 *    dice values as indices.
 * 4. Returns the precomputed index associated with the given dice combination.
 *
 * @param ctx Pointer to the YatzyContext containing the precomputed `index_lookup` table.
 * @param dice An array of 5 integers representing a sorted dice set (values 1-6).
 * @return The precomputed index corresponding to the given sorted dice set.
 */
int FindDiceSetIndex(const YatzyContext *ctx, const int dice[5]) {
    return ctx->index_lookup[dice[0] - 1][dice[1] - 1][dice[2] - 1][dice[3] - 1][dice[4] - 1];
}

/**
 * @brief Computes the probability of a specific sorted dice set occurring in a single roll.
 *
 * This function calculates the probability of rolling a specific sorted dice set (`dice[5]`)
 * during a single roll of five dice. The calculation considers the number of permutations
 * that can produce the same sorted set, accounting for duplicate dice values, and divides
 * by the total number of possible outcomes (6^5).
 *
 * The steps are:
 * 1. Count the occurrences of each dice face in the input dice set.
 * 2. Compute the numerator as the factorial of the total dice count (5!).
 * 3. Compute the denominator by calculating the product of factorials of duplicate dice counts.
 * 4. Divide the numerator by the denominator to determine the number of unique permutations.
 * 5. Normalize by the total number of outcomes (6^5) to obtain the final probability.
 *
 * @param ctx Pointer to the YatzyContext containing precomputed factorial values.
 * @param dice An array of 5 integers representing a sorted dice set.
 * @return The probability of rolling the specified dice set in a single roll.
 */
double ComputeProbabilityOfDiceSet(const YatzyContext *ctx, const int dice[5]) {
    int face_count[7];
    for (int i = 1; i <= 6; i++) face_count[i] = 0;
    for (int i = 0; i < 5; i++) face_count[dice[i]]++;

    const int numerator = ctx->factorial[5];
    int denominator = 1;
    for (int f = 1; f <= 6; f++) {
        if (face_count[f] > 1) {
            const int fc = face_count[f];
            int sub_fact = 1;
            for (int x = 2; x <= fc; x++) sub_fact *= x;
            denominator *= sub_fact;
        }
    }

    const double permutations = (double) numerator / (double) denominator;
    const double total_outcomes = pow(6, 5);
    return permutations / total_outcomes;
}

/**
 * @brief Counts the occurrences of each dice face in a given dice set.
 *
 * This function iterates over a set of 5 dice and counts the number of times each face
 * (1 through 6) appears, storing the results in the `face_count` array.
 *
 * The `face_count` array must have at least 7 elements (index 0 is unused), and it will
 * be initialized to zeros for all faces before counting.
 *
 * @param dice An array of 5 integers representing the dice values.
 * @param face_count An array of 7 integers where the counts for each face (1-6) will be stored.
 *                   Index 0 is unused.
 */
void CountFaces(const int dice[5], int face_count[7]) {
    for (int i = 1; i <= 6; i++) face_count[i] = 0;
    for (int i = 0; i < 5; i++) face_count[dice[i]]++;
}

/**
 * @brief Calculates the score for an "N-of-a-Kind" category in Yatzy.
 *
 * This function computes the score for an "N-of-a-Kind" category by checking if
 * any dice face appears at least `n` times in the given face count. If such a
 * face exists, the score is calculated as `face * n` (the value of the face
 * multiplied by the count of `n` dice). If no face meets the condition, the
 * function returns a score of 0.
 *
 * The function prioritizes higher face values by iterating from 6 down to 1.
 *
 * @param face_count An array of integers where `face_count[i]` represents the
 *                   number of dice showing the face value `i`.
 * @param n The required number of identical dice to qualify for the category.
 * @return The score for the "N-of-a-Kind" category, or 0 if no valid combination exists.
 */
int NOfAKindScore(const int face_count[7], int n) {
    for (int face = 6; face >= 1; face--) {
        if (face_count[face] >= n) return face * n;
    }
    return 0;
}
