/*
 * storage.c — Binary file I/O for precomputed state values
 *
 * Provides per-level and consolidated save/load for the DP result table,
 * including memory-mapped variants for better performance.
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include "storage.h"
#include "timing.h"

/* ── File utilities ──────────────────────────────────────────────────── */

int FileExists(const char *filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

/* ── Per-level I/O ───────────────────────────────────────────────────── */

void SaveStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename) {
    printf("Saving state values to file: %s for scored_count = %d\n", filename, scored_count);

    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file for writing: %s\n", filename);
        return;
    }

    /* Count the number of states */
    int count = 0;
    for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
        if (ctx->scored_category_count_cache[scored] == scored_count) {
            for (int up = 0; up <= 63; up++) {
                count++;
            }
        }
    }
    printf("Total states to save for scored_count = %d: %d\n", scored_count, count);

    fwrite(&count, sizeof(int), 1, f);

    /* Write state values */
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
    printf("Successfully saved state values for scored_count = %d to file: %s\n", scored_count, filename);
}

int LoadStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename) {
    printf("Attempting to load state values from file: %s for scored_count = %d\n", filename, scored_count);

    if (!FileExists(filename)) {
        printf("File does not exist: %s\n", filename);
        return 0;
    }

    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Warning: Could not open file for reading: %s\n", filename);
        return 0;
    }

    int count;
    if (fread(&count, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Warning: Could not read count from file %s\n", filename);
        fclose(f);
        return 0;
    }
    printf("File %s contains %d states for scored_count = %d\n", filename, count, scored_count);

    for (int i = 0; i < count; i++) {
        int up, scored;
        double val;

        if (fread(&up, sizeof(int), 1, f) != 1 ||
            fread(&scored, sizeof(int), 1, f) != 1 ||
            fread(&val, sizeof(double), 1, f) != 1) {
            fprintf(stderr, "Warning: Unexpected end of file while reading %s (state %d)\n", filename, i);
            fclose(f);
            return 0;
        }

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

/* ── Consolidated file I/O ───────────────────────────────────────────── */

int LoadAllStateValues(YatzyContext *ctx, const char *filename) {
    double start_time = timer_now();
    printf("Loading all state values from %s...\n", filename);

    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("File not found: %s\n", filename);
        return 0;
    }

    StateFileHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    if (header.magic != STATE_FILE_MAGIC || header.version != STATE_FILE_VERSION) {
        printf("Invalid file format\n");
        fclose(f);
        return 0;
    }

    printf("Loading %u total states...\n", header.total_states);

    int loaded = 0;
    for (int scored_count = 0; scored_count <= 15; scored_count++) {
        for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
            if (__builtin_popcount(scored) == scored_count) {
                for (int up = 0; up <= 63; up++) {
                    double val;
                    if (fread(&val, sizeof(double), 1, f) != 1) {
                        printf("Error reading state values\n");
                        fclose(f);
                        return 0;
                    }
                    ctx->state_values[STATE_INDEX(up, scored)] = val;
                    loaded++;
                }
            }
        }

        if (loaded % 10000 == 0) {
            printf("\rLoaded %d/%u states (%.1f%%)...", loaded, header.total_states,
                   (double)loaded / header.total_states * 100.0);
            fflush(stdout);
        }
    }

    fclose(f);
    double elapsed = timer_now() - start_time;
    printf("\nSuccessfully loaded %d states in %.2f seconds (%.0f states/sec)\n",
           loaded, elapsed, loaded / elapsed);
    return 1;
}

void SaveAllStateValues(YatzyContext *ctx, const char *filename) {
    double start_time = timer_now();
    printf("Saving all state values to %s...\n", filename);

    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file for writing: %s\n", filename);
        return;
    }

    StateFileHeader header;
    header.magic = STATE_FILE_MAGIC;
    header.version = STATE_FILE_VERSION;
    header.total_states = 0;

    uint32_t offset = 0;
    for (int scored_count = 0; scored_count <= 15; scored_count++) {
        header.level_offsets[scored_count] = offset;
        int count = 0;
        for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
            if (__builtin_popcount(scored) == scored_count) {
                count += 64;
            }
        }
        offset += count;
        header.total_states += count;
    }

    fwrite(&header, sizeof(header), 1, f);

    int saved = 0;
    for (int scored_count = 0; scored_count <= 15; scored_count++) {
        for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
            if (__builtin_popcount(scored) == scored_count) {
                for (int up = 0; up <= 63; up++) {
                    double val = ctx->state_values[STATE_INDEX(up, scored)];
                    fwrite(&val, sizeof(double), 1, f);
                    saved++;
                }
            }
        }

        if (saved % 10000 == 0) {
            printf("\rSaved %d/%u states (%.1f%%)...", saved, header.total_states,
                   (double)saved / header.total_states * 100.0);
            fflush(stdout);
        }
    }

    fclose(f);
    double elapsed = timer_now() - start_time;
    printf("\nSuccessfully saved %d states in %.2f seconds (%.0f states/sec)\n",
           saved, elapsed, saved / elapsed);
}

/* ── Memory-mapped I/O ───────────────────────────────────────────────── */

int LoadAllStateValuesMmap(YatzyContext *ctx, const char *filename) {
    double start_time = timer_now();
    printf("Loading state values using memory mapping from %s...\n", filename);

    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        printf("File not found: %s\n", filename);
        return 0;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return 0;
    }

    void *mapped = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        printf("Failed to memory map file: %s\n", strerror(errno));
        close(fd);
        return 0;
    }

    madvise(mapped, st.st_size, MADV_SEQUENTIAL);

    StateFileHeader *header = (StateFileHeader *)mapped;
    if (header->magic != STATE_FILE_MAGIC || header->version != STATE_FILE_VERSION) {
        printf("Invalid file format\n");
        munmap(mapped, st.st_size);
        close(fd);
        return 0;
    }

    printf("Loading %u total states via mmap...\n", header->total_states);

    double *data = (double *)((char *)mapped + sizeof(StateFileHeader));
    int loaded = 0;

    for (int scored_count = 0; scored_count <= 15; scored_count++) {
        for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
            if (__builtin_popcount(scored) == scored_count) {
                for (int up = 0; up <= 63; up++) {
                    ctx->state_values[STATE_INDEX(up, scored)] = data[loaded];
                    loaded++;
                }
            }
        }
    }

    munmap(mapped, st.st_size);
    close(fd);

    double elapsed = timer_now() - start_time;
    printf("Successfully loaded %d states in %.2f seconds (%.0f states/sec)\n",
           loaded, elapsed, loaded / elapsed);
    return 1;
}

void SaveAllStateValuesMmap(YatzyContext *ctx, const char *filename) {
    double start_time = timer_now();
    printf("Saving state values using memory mapping to %s...\n", filename);

    size_t file_size = sizeof(StateFileHeader) + NUM_STATES * sizeof(double);

    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        fprintf(stderr, "Error creating file: %s\n", strerror(errno));
        return;
    }

    if (ftruncate(fd, file_size) < 0) {
        fprintf(stderr, "Error sizing file: %s\n", strerror(errno));
        close(fd);
        return;
    }

    void *mapped = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "Failed to memory map file: %s\n", strerror(errno));
        close(fd);
        return;
    }

    StateFileHeader *header = (StateFileHeader *)mapped;
    header->magic = STATE_FILE_MAGIC;
    header->version = STATE_FILE_VERSION;
    header->total_states = 0;

    uint32_t offset = 0;
    for (int scored_count = 0; scored_count <= 15; scored_count++) {
        header->level_offsets[scored_count] = offset;
        int count = 0;
        for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
            if (__builtin_popcount(scored) == scored_count) {
                count += 64;
            }
        }
        offset += count;
        header->total_states += count;
    }

    double *data = (double *)((char *)mapped + sizeof(StateFileHeader));
    int saved = 0;

    for (int scored_count = 0; scored_count <= 15; scored_count++) {
        for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
            if (__builtin_popcount(scored) == scored_count) {
                for (int up = 0; up <= 63; up++) {
                    data[saved] = ctx->state_values[STATE_INDEX(up, scored)];
                    saved++;
                }
            }
        }
    }

    msync(mapped, file_size, MS_SYNC);
    munmap(mapped, file_size);
    close(fd);

    double elapsed = timer_now() - start_time;
    printf("Successfully saved %d states in %.2f seconds (%.0f states/sec)\n",
           saved, elapsed, saved / elapsed);
}
