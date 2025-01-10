#pragma once
#include <context.h>
#include <microhttpd.h>

#define ALLOWED_METHODS "GET, POST, OPTIONS"
#define ALLOWED_HEADERS "Content-Type"
#define MAX_LINE_LENGTH 1024
#define MAX_FIELDS 100

struct RequestContext {
    char *post_data;
    size_t post_size;
};

void AddCORSHeaders(struct MHD_Response *response);

int start_webserver(YatzyContext *ctx, int port);