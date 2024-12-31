#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <limits.h>
#include "webserver.h"           // Declares answer_to_connection
#include "yatzy.h"               // Declares CreateYatzyContext and FreeYatzyContext
#include "precompute_scores.h"   // Declares ComputeAllStateValues

volatile bool running = true;

void handle_signal(int signal) {
    running = false;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Get the base path from the environment variable
    const char *base_path = getenv("YATZY_BASE_PATH");
    if (!base_path) {
        base_path = "."; // Default to current directory if not set
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

    printf("Starting yatzy API server...\n");

    printf("Initializing context...\n");
    YatzyContext *ctx = CreateYatzyContext();
    printf("Context created!\n");

    printf("Precomputing all state values...\n");
    ComputeAllStateValues(ctx);
    printf("State values computed!\n");

    struct MHD_Daemon *daemon = MHD_start_daemon(MHD_USE_INTERNAL_POLLING_THREAD,
                                                 9000,
                                                 NULL, NULL,
                                                 &answer_to_connection, ctx,
                                                 MHD_OPTION_END);

    if (NULL == daemon) {
        FreeYatzyContext(ctx);
        return 1;
    }

    signal(SIGINT, handle_signal); // Handle Ctrl+C (SIGINT) to stop the server

    printf("Server is running. Press Ctrl+C to stop.\n");
    while (running) {
        sleep(1); // Sleep to avoid busy-waiting
    }

    printf("\nStopping server...\n");
    MHD_stop_daemon(daemon);
    FreeYatzyContext(ctx);

    return 0;
}