#include <context.h>
#include <stdlib.h>
#include <math.h>

void CountFaces(const int dice[5], int face_count[7]) {
    for (int i = 1; i <= 6; i++) face_count[i] = 0;
    for (int i = 0; i < 5; i++) face_count[dice[i]]++;
}

int NOfAKindScore(const int face_count[7], int n) {
    for (int face = 6; face >= 1; face--) {
        if (face_count[face] >= n) return face * n;
    }
    return 0;
}

double ComputeProbabilityOfDiceSetOnce(const YatzyContext *ctx, const int dice[5]) {
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

int FindDiceSetIndex(const YatzyContext *ctx, const int dice[5]) {
    return ctx->index_lookup[dice[0] - 1][dice[1] - 1][dice[2] - 1][dice[3] - 1][dice[4] - 1];
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