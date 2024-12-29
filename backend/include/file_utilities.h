#ifndef FILE_UTILITIES_H
#define FILE_UTILITIES_H

#include "yatzy.h"

// Maximum number of fields in a CSV row
#define MAX_FIELDS 128
#define MAX_LINE_LENGTH 1024

// CSV Parsing Functions
void parse_csv_line(char *line, char *fields[], int *num_fields);
int read_csv(const char *filename);

// File Utilities
int FileExists(const char *filename);

// State Value File Handling
void SaveStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename);
int LoadStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename);

#endif // FILE_UTILITIES_H