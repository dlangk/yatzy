/*
 * storage.h — Binary file I/O for precomputed state values
 *
 * Provides load/save functions for both per-level state files
 * and consolidated all-states files (with optional mmap).
 */
#ifndef STORAGE_H
#define STORAGE_H

#include "context.h"
#include <stdint.h>

/* Binary file format header for consolidated state files. */
typedef struct {
    uint32_t magic;              /* File identifier: 0x59545A53 ("STZY") */
    uint32_t version;            /* Format version for future compatibility */
    uint32_t total_states;       /* Total number of states in file */
    uint32_t level_offsets[16];  /* Starting position of each level's data */
} StateFileHeader;

#define STATE_FILE_MAGIC 0x59545A53    /* "STZY" in hex */
#define STATE_FILE_VERSION 1

/* Check if a file exists on disk. */
int FileExists(const char *filename);

/* Per-level I/O: save/load state values for states with |C| = scored_count. */
void SaveStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename);
int LoadStateValuesForCount(YatzyContext *ctx, int scored_count, const char *filename);

/* Memory-mapped I/O for consolidated state files. */
int LoadAllStateValuesMmap(YatzyContext *ctx, const char *filename);
void SaveAllStateValuesMmap(YatzyContext *ctx, const char *filename);

#endif /* STORAGE_H */
