/*
 * storage.h — Binary file I/O for precomputed state values
 *
 * Provides load/save for consolidated state files using a flat v2 format.
 * The v2 format stores state_values in STATE_INDEX order for zero-copy mmap.
 */
#ifndef STORAGE_H
#define STORAGE_H

#include "context.h"
#include <stdint.h>

/* Binary file format header (v3): flat STATE_INDEX-ordered floats. */
typedef struct {
    uint32_t magic;          /* File identifier: 0x59545A53 ("STZY") */
    uint32_t version;        /* Format version: 3 */
    uint32_t total_states;   /* Total number of states (2097152) */
    uint32_t reserved;       /* Pad to 16-byte alignment */
} StateFileHeader;

#define STATE_FILE_MAGIC 0x59545A53    /* "STZY" in hex */
#define STATE_FILE_VERSION 3

/* Check if a file exists on disk. */
int FileExists(const char *filename);

/* Load state values via zero-copy mmap. Returns 1 on success, 0 on failure. */
int LoadAllStateValues(YatzyContext *ctx, const char *filename);

/* Save state values as a flat dump in STATE_INDEX order. */
void SaveAllStateValues(YatzyContext *ctx, const char *filename);

#endif /* STORAGE_H */
