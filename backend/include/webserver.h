#ifndef WEBSERVER_H
#define WEBSERVER_H

#include <microhttpd.h>
#include <json-c/json.h>
#include "yatzy.h"  // For YatzyContext

// Structure for storing request context
typedef struct RequestContext {
    char *post_data;
    size_t post_size;
} RequestContext;

// Function prototypes
void AddCORSHeaders(struct MHD_Response *response);

void handle_get_score_histogram(YatzyContext *ctx, struct MHD_Connection *connection);
void handle_get_state_value(YatzyContext *ctx, struct MHD_Connection *connection);
void handle_evaluate_category_score(YatzyContext *ctx, struct MHD_Connection *connection, struct json_object *parsed);
void handle_available_categories(YatzyContext *ctx, struct MHD_Connection *connection, struct json_object *parsed);
void handle_evaluate_all_categories(YatzyContext *ctx, struct MHD_Connection *connection, struct json_object *parsed);
void handle_evaluate_actions(YatzyContext *ctx, struct MHD_Connection *connection, struct json_object *parsed);
void handle_suggest_optimal_action(YatzyContext *ctx, struct MHD_Connection *connection, struct json_object *parsed);
void handle_evaluate_user_action(YatzyContext *ctx, struct MHD_Connection *connection, struct json_object *parsed);

// Utility functions
enum MHD_Result respond_with_error(struct MHD_Connection *connection, int status_code,
                                   const char *error_message, struct RequestContext *req_ctx);

// Main request handler
enum MHD_Result answer_to_connection(void *cls, struct MHD_Connection *connection,
                                     const char *url, const char *method,
                                     const char *version, const char *upload_data,
                                     size_t *upload_data_size, void **con_cls);

#endif // WEBSERVER_H