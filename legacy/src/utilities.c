/*
 * utilities.c — Logging and environment helpers
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <unistd.h>

#include "utilities.h"

void LogMessage(const char *format, ...) {
    const time_t now = time(NULL);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));

    fprintf(stdout, "[%s] ", timestamp);

    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);

    fflush(stdout);
}

void SetWorkingDirectory(void) {
    char *base_path = getenv("YATZY_BASE_PATH");
    printf("YATZY_BASE_PATH=%s\n", base_path);
    if (!base_path) {
        base_path = ".";
    }

    if (chdir(base_path) != 0) {
        perror("chdir failed");
        exit(EXIT_FAILURE);
    }

    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Working directory changed to: %s\n", cwd);
    } else {
        perror("getcwd failed");
    }
}
