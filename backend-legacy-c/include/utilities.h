/*
 * utilities.h — Logging and environment helpers
 */
#pragma once

/* Log a formatted message with a timestamp. */
void LogMessage(const char *format, ...);

/* Set working directory from YATZY_BASE_PATH environment variable. */
void SetWorkingDirectory(void);
