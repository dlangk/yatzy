#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syslimits.h>

#include <webserver.h>

void parse_csv_line(char *line, char *fields[], int *num_fields) {
    *num_fields = 0;
    char *token = strtok(line, ",");

    while (token != NULL && *num_fields < MAX_FIELDS) {
        while (*token == ' ') token++;
        char *end = token + strlen(token) - 1;
        while (end > token && *end == ' ') end--;
        *(end + 1) = '\0';
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

    while (fgets(line, MAX_LINE_LENGTH, file)) {
        line[strcspn(line, "\n")] = 0;
        parse_csv_line(line, fields, &num_fields);
        printf("Line %d:\n", line_number++);
        for (int i = 0; i < num_fields; i++) {
            printf("Field %d: %s\n", i, fields[i]);
            free(fields[i]);
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
    printf("Saving state values to file: %s for scored_count = %d\n", filename, scored_count);

    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file for writing: %s\n", filename);
        return;
    }

    // Step 1: Count the number of states
    int count = 0;
    for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
        if (ctx->scored_category_count_cache[scored] == scored_count) {
            for (int up = 0; up <= 63; up++) {
                count++;
            }
        }
    }
    printf("Total states to save for scored_count = %d: %d\n", scored_count, count);

    // Write the count to the file
    fwrite(&count, sizeof(int), 1, f);
    printf("Wrote state count (%d) to file: %s\n", count, filename);

    // Step 2: Write state values
    for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
        if (ctx->scored_category_count_cache[scored] == scored_count) {
            for (int up = 0; up <= 63; up++) {
                double val = ctx->state_values[STATE_INDEX(up, scored)];
                fwrite(&up, sizeof(int), 1, f);
                fwrite(&scored, sizeof(int), 1, f);
                fwrite(&val, sizeof(double), 1, f);
                printf("Saved state_values[%d] = %f (up = %d, scored = %04x) to file: %s\n",
                       STATE_INDEX(up, scored), val, up, scored, filename);
            }
        }
    }

    fclose(f);
    printf("Successfully saved state values for scored_count = %d to file: %s\n", scored_count, filename);
}

int LoadStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename) {
    printf("Attempting to load state values from file: %s for scored_count = %d\n", filename, scored_count);

    if (!FileExists(filename)) {
        printf("File does not exist: %s\n", filename);
        return 0; // File not found
    }

    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Warning: Could not open file for reading: %s\n", filename);
        return 0;
    }

    // Step 1: Read the count of states
    int count;
    if (fread(&count, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Warning: Could not read count from file %s\n", filename);
        fclose(f);
        return 0;
    }
    printf("File %s contains %d states for scored_count = %d\n", filename, count, scored_count);

    // Step 2: Load state values
    for (int i = 0; i < count; i++) {
        int up, scored;
        double val;

        // Read the state values
        if (fread(&up, sizeof(int), 1, f) != 1 ||
            fread(&scored, sizeof(int), 1, f) != 1 ||
            fread(&val, sizeof(double), 1, f) != 1) {
            fprintf(stderr, "Warning: Unexpected end of file while reading %s (state %d)\n", filename, i);
            fclose(f);
            return 0;
        }

        // Calculate the index and store the value
        int index = STATE_INDEX(up, scored);
        if (index < 0 || index >= NUM_STATES) {
            fprintf(stderr, "Error: Invalid index %d for up = %d, scored = %04x in file %s\n", index, up, scored,
                    filename);
            fclose(f);
            return 0;
        }

        ctx->state_values[index] = val;
    }

    fclose(f);
    printf("--> Successfully loaded state values for scored_count = %d from file: %s\n", scored_count, filename);
    return 1;
}

void SetWorkingDirectory(const char *base_path) {
    if (!base_path) {
        base_path = ".";
    }

    // Change to the specified base path
    if (chdir(base_path) != 0) {
        perror("chdir failed");
        exit(EXIT_FAILURE);
    }

    // Log the working directory
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Working directory changed to: %s\n", cwd);
    } else {
        perror("getcwd failed");
    }
}