#pragma once
#include "context.h"

void CountFaces(const int dice[5], int face_count[7]);
int NOfAKindScore(const int face_count[7], int n);
double ComputeProbabilityOfDiceSet(const YatzyContext *ctx, const int dice[5]);
void SortDiceSet(int arr[5]);
int FindDiceSetIndex(const YatzyContext *ctx, const int dice[5]);
void RollDice(int dice[5]);
void RerollDice(int dice[5], int mask);