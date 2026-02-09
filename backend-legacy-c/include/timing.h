/*
 * timing.h — Portable high-resolution timing utilities
 *
 * Header-only (inline functions), usable from any .c file.
 * Replaces ad-hoc timing code scattered across context.c, storage.c, etc.
 */
#ifndef TIMING_H
#define TIMING_H

#include <stdio.h>
#include <time.h>

/* Return current monotonic time in seconds (fractional). */
static inline double timer_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* Return elapsed milliseconds since 'start' (a value from timer_now()). */
static inline double timer_elapsed_ms(double start) {
    return (timer_now() - start) * 1000.0;
}

/* Block timer macro — executes 'code', then prints "label ... X.XXX ms". */
#define TIMER_BLOCK(label, code) do { \
    double _t0 = timer_now(); \
    code; \
    double _dt = timer_elapsed_ms(_t0); \
    printf("  %-42s %8.3f ms\n", label, _dt); \
} while (0)

#endif /* TIMING_H */
