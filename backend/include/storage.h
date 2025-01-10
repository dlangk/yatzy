#ifndef CSV_AND_STATE_H
#define CSV_AND_STATE_H

#include "context.h"

#define MAX_FIELDS 100         // Define the maximum number of fields per line
#define MAX_LINE_LENGTH 1024   // Define the maximum length of a line

// Function declarations for CSV parsing
void parse_csv_line(char *line, char *fields[], int *num_fields);
int read_csv(const char *filename);

// Utility functions
void SetWorkingDirectory(const char *base_path);
int FileExists(const char *filename);

// Functions for saving and loading state values
void SaveStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename);
int LoadStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename);

#endif // CSV_AND_STATE_H