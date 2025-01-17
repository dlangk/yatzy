#include <stdarg.h>
#include <stdio.h>
#include <time.h>

/**
 * @brief Logs a formatted message with a timestamp.
 *
 * This function adds a timestamp to every log entry and ensures consistent formatting.
 * It supports variable arguments like printf.
 *
 * @param format The format string (like printf).
 * @param ... Additional arguments for the format string.
 */
void LogMessage(const char *format, ...) {
    // Get the current timestamp
    const time_t now = time(NULL);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));

    // Print the timestamp
    fprintf(stdout, "[%s] ", timestamp);

    // Handle the variable arguments and print the log message
    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);

    // Ensure logs are flushed immediately
    fflush(stdout);
}