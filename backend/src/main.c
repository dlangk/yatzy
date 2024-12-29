#include <stdio.h>
#include "webserver.h"           // Declares answer_to_connection
#include "yatzy.h"               // Declares CreateYatzyContext and FreeYatzyContext

volatile bool running = true;

void handle_signal(int signal) {
    running = false;
}

int main() {
    printf("Starting yatzy API server...\n");
    YatzyContext *ctx = CreateYatzyContext();
    struct MHD_Daemon *daemon = MHD_start_daemon(MHD_USE_INTERNAL_POLLING_THREAD,
                                                 8080,
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