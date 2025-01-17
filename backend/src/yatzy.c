#include <math.h>
#include <stdio.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#include <json-c/json.h>

#include <computations.h>
#include <context.h>
#include <webserver.h>
#include "storage.h"

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    SetWorkingDirectory();
    printf("Starting yatzy API server...\n");
    printf("Initializing context...\n");
    YatzyContext *ctx = CreateYatzyContext();
    printf("Context created!\n");
    printf("Precomputing all state values...\n");
    ComputeAllStateValues(ctx);
    printf("State values computed!\n");
    start_webserver(ctx, 9000);
    FreeYatzyContext(ctx);
    return 0;
}