#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <file_utilities.h>
#include <sys/stat.h>

void parse_csv_line(char *line, char *fields[], int *num_fields) {
    *num_fields = 0;
    char *token = strtok(line, ",");

    while (token != NULL && *num_fields < MAX_FIELDS) {
        // Remove leading/trailing whitespace
        while (*token == ' ') token++;
        char *end = token + strlen(token) - 1;
        while (end > token && *end == ' ') end--;
        *(end + 1) = '\0';

        // Remove quotes if present
        if (*token == '"' && *end == '"') {
            token++;
            *end = '\0';
        }

        fields[*num_fields] = strdup(token);
        (*num_fields)++;
        token = strtok(NULL, ",");
    }
}

int read_csv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s\n", filename);
        return 1;
    }

    char line[MAX_LINE_LENGTH];
    char *fields[MAX_FIELDS];
    int num_fields;
    int line_number = 0;

    // Read each line
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        // Remove newline character
        line[strcspn(line, "\n")] = 0;

        parse_csv_line(line, fields, &num_fields);

        // Print the fields (you can modify this part based on your needs)
        printf("Line %d:\n", line_number++);
        for (int i = 0; i < num_fields; i++) {
            printf("Field %d: %s\n", i, fields[i]);
            free(fields[i]); // Don't forget to free the memory
        }
        printf("\n");
    }

    fclose(file);
    return 0;
}

int FileExists(const char *filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

void SaveStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file for writing: %s\n", filename);
        return;
    }

    int count = 0;
    for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
        if (ctx->scored_category_count_cache[scored] == scored_count) {
            for (int up = 0; up <= 63; up++) {
                count++;
            }
        }
    }

    fwrite(&count, sizeof(int), 1, f);

    for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
        if (ctx->scored_category_count_cache[scored] == scored_count) {
            for (int up = 0; up <= 63; up++) {
                double val = ctx->state_values[STATE_INDEX(up, scored)];
                fwrite(&up, sizeof(int), 1, f);
                fwrite(&scored, sizeof(int), 1, f);
                fwrite(&val, sizeof(double), 1, f);
            }
        }
    }

    fclose(f);
}

int LoadStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename) {
    if (!FileExists(filename)) {
        return 0; // File not found
    }

    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Warning: Could not open file for reading: %s\n", filename);
        return 0;
    }

    int count;
    if (fread(&count, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Warning: Could not read count from %s\n", filename);
        fclose(f);
        return 0;
    }

    for (int i = 0; i < count; i++) {
        int up, scored;
        double val;
        if (fread(&up, sizeof(int), 1, f) != 1 ||
            fread(&scored, sizeof(int), 1, f) != 1 ||
            fread(&val, sizeof(double), 1, f) != 1) {
            fprintf(stderr, "Warning: Unexpected end of file reading %s\n", filename);
            fclose(f);
            return 0;
        }

        ctx->state_values[STATE_INDEX(up, scored)] = val;
    }

    fclose(f);
    return 1;
}