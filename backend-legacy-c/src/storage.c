/*
 * storage.c — Binary file I/O for precomputed state values
 *
 * Flat v3 format: 16-byte header + NUM_STATES floats in STATE_INDEX order.
 * Loading uses zero-copy mmap (no iteration, no copying).
 */
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

/* ── Zero-copy mmap load ────────────────────────────────────────────── */

int LoadAllStateValues(YatzyContext *ctx, const char *filename) {
    double start_time = timer_now();
    printf("Loading state values from %s...\n", filename);

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

    size_t expected_size = sizeof(StateFileHeader) + (size_t)NUM_STATES * sizeof(float);
    if ((size_t)st.st_size != expected_size) {
        printf("File size mismatch: expected %zu, got %lld\n", expected_size, (long long)st.st_size);
        close(fd);
        return 0;
    }

    void *mapped = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        printf("Failed to memory map file: %s\n", strerror(errno));
        return 0;
    }

    StateFileHeader *header = (StateFileHeader *)mapped;
    if (header->magic != STATE_FILE_MAGIC || header->version != STATE_FILE_VERSION) {
        printf("Invalid file format (magic=0x%08x version=%u)\n",
               header->magic, header->version);
        munmap(mapped, st.st_size);
        return 0;
    }

    madvise(mapped, st.st_size, MADV_RANDOM);

    /* Zero-copy: point state_values directly into mmap'd region */
    free(ctx->state_values);
    ctx->state_values = (float *)((char *)mapped + sizeof(StateFileHeader));
    ctx->mmap_base = mapped;
    ctx->mmap_size = st.st_size;

    double elapsed = timer_now() - start_time;
    printf("Loaded %u states via zero-copy mmap in %.2f ms\n",
           header->total_states, elapsed * 1000.0);
    return 1;
}

/* ── Flat dump save ──────────────────────────────────────────────────── */

void SaveAllStateValues(YatzyContext *ctx, const char *filename) {
    double start_time = timer_now();
    printf("Saving state values to %s...\n", filename);

    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error creating file: %s\n", strerror(errno));
        return;
    }

    StateFileHeader header = {
        .magic = STATE_FILE_MAGIC,
        .version = STATE_FILE_VERSION,
        .total_states = NUM_STATES,
        .reserved = 0
    };
    fwrite(&header, sizeof(header), 1, f);
    fwrite(ctx->state_values, sizeof(float), NUM_STATES, f);
    fclose(f);

    double elapsed = timer_now() - start_time;
    printf("Saved %d states in %.2f ms\n", NUM_STATES, elapsed * 1000.0);
}
