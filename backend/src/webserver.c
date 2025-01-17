#include <microhttpd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <signal.h>
#include <json-c/json_object.h>
#include <json-c/json_tokener.h>

#include <utilities.h>
#include <game_mechanics.h>
#include <computations.h>
#include <webserver.h>
#include <context.h>
#include <dice_mechanics.h>

void AddCORSHeaders(struct MHD_Response *response) {
    MHD_add_response_header(response, "Access-Control-Allow-Origin", "*");
    MHD_add_response_header(response, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    MHD_add_response_header(response, "Access-Control-Allow-Headers", "Content-Type");
}

void LogRequest(const char *method, const char *url, const char *payload, int status) {
    LogMessage("%s %s | Status: %d | Payload: %s\n\n",
               method, url, status, payload ? payload : "NULL");
}

void RespondWithJSON(struct MHD_Connection *connection, const int status_code, json_object *json_resp) {
    const char *response_str = json_object_to_json_string(json_resp);
    LogRequest("RESPONSE", "", response_str ? response_str : "NULL", status_code);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(
        strlen(response_str),
        (void *) response_str,
        MHD_RESPMEM_MUST_COPY);
    AddCORSHeaders(mhd_response);
    MHD_queue_response(connection, status_code, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(json_resp);
}

void RespondWithError(struct MHD_Connection *connection, const int status_code, const char *error_msg) {
    json_object *error_resp = json_object_new_object();
    json_object_object_add(error_resp, "error", json_object_new_string(error_msg));
    RespondWithJSON(connection, status_code, error_resp);
}

int ParseJSONField(const json_object *json_obj, const char *field, int *out_val) {
    json_object *field_obj = json_object_object_get(json_obj, field);
    if (!field_obj) return 0;
    *out_val = json_object_get_int(field_obj);
    return 1;
}

int ParseDiceArray(const json_object *json_obj, int dice[5]) {
    json_object *dice_arr = json_object_object_get(json_obj, "dice");
    if (!dice_arr) return 0;
    for (int i = 0; i < 5; i++) {
        json_object *d = json_object_array_get_idx(dice_arr, i);
        if (!d) return 0;
        dice[i] = json_object_get_int(d);
    }
    return 1;
}

int ValidateAndParseJSON(const json_object *parsed, const char *fields[], const int num_fields, int *out_values[]) {
    for (int i = 0; i < num_fields; i++) {
        json_object *obj = json_object_object_get(parsed, fields[i]);
        if (!obj) {
            return 0; // Missing field
        }
        *out_values[i] = json_object_get_int(obj);
    }
    return 1;
}

bool ParseDiceFromJSON(const json_object *parsed, int dice[5]) {
    json_object *dice_array;

    if (!json_object_object_get_ex(parsed, "dice", &dice_array) ||
        json_object_get_type(dice_array) != json_type_array ||
        json_object_array_length(dice_array) != 5) {
        return false;
    }

    for (int i = 0; i < 5; i++) {
        json_object *dice_value = json_object_array_get_idx(dice_array, i);
        if (!dice_value || !json_object_is_type(dice_value, json_type_int)) return false;
        dice[i] = json_object_get_int(dice_value);
    }
    return true;
}

void AddJSONField(json_object *json_resp, const char *key, const int value) {
    json_object_object_add(json_resp, key, json_object_new_int(value));
}

void AddJSONFieldDouble(json_object *json_resp, const char *key, const double value) {
    json_object_object_add(json_resp, key, json_object_new_double(value));
}

void AddJSONFieldString(json_object *json_resp, const char *key, const char *value) {
    json_object_object_add(json_resp, key, json_object_new_string(value));
}

void handle_evaluate_category_score(YatzyContext *ctx, struct MHD_Connection *connection, const json_object *parsed) {
    int dice[5];
    if (!ParseDiceFromJSON(parsed, dice)) {
        RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "Invalid or missing dice array");
        return;
    }

    int cat_id;
    if (!ParseJSONField(parsed, "category_id", &cat_id)) {
        RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "Missing category_id");
        return;
    }

    int score = ctx->precomputed_scores[FindDiceSetIndex(ctx, dice)][cat_id];

    json_object *resp = json_object_new_object();
    AddJSONField(resp, "category_id", cat_id);
    AddJSONFieldString(resp, "category_name", ctx->category_names[cat_id]);
    AddJSONField(resp, "score", score);

    RespondWithJSON(connection, MHD_HTTP_OK, resp);
}

void handle_available_categories(YatzyContext *ctx, struct MHD_Connection *connection, const json_object *parsed) {
    int scored_categories, dice[5];
    const char *fields[] = {"scored_categories"};
    int *values[] = {&scored_categories};

    // Validate only the required fields: scored_categories and dice
    if (!ValidateAndParseJSON(parsed, fields, 1, values) || !ParseDiceFromJSON(parsed, dice)) {
        RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "Missing required fields");
        return;
    }

    const int ds_index = FindDiceSetIndex(ctx, dice);
    json_object *resp = json_object_new_object();
    json_object *categories = json_object_new_array();

    for (int c = 0; c < CATEGORY_COUNT; c++) {
        const int scr = ctx->precomputed_scores[ds_index][c];
        const bool valid = !IS_CATEGORY_SCORED(scored_categories, c) && scr > 0;

        json_object *cat_obj = json_object_new_object();
        AddJSONField(cat_obj, "id", c);
        AddJSONFieldString(cat_obj, "name", ctx->category_names[c]);
        AddJSONField(cat_obj, "score", scr);
        json_object_object_add(cat_obj, "valid", json_object_new_boolean(valid));
        json_object_array_add(categories, cat_obj);
    }

    json_object_object_add(resp, "categories", categories);
    RespondWithJSON(connection, MHD_HTTP_OK, resp);
}

void handle_evaluate_all_categories(YatzyContext *ctx, struct MHD_Connection *connection, const json_object *parsed) {
    int upper_score, scored_categories, rerolls, dice[5];
    const char *fields[] = {"upper_score", "scored_categories", "rerolls_remaining"};
    int *values[] = {&upper_score, &scored_categories, &rerolls};

    if (!ValidateAndParseJSON(parsed, fields, 3, values) || !ParseDiceFromJSON(parsed, dice)) {
        RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "Missing required fields");
        return;
    }

    if (rerolls != 0) {
        RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "rerolls_remaining must be 0 for this endpoint");
        return;
    }

    int ds_index = FindDiceSetIndex(ctx, dice);
    json_object *resp = json_object_new_object();
    json_object *categories = json_object_new_array();

    for (int c = 0; c < CATEGORY_COUNT; c++) {
        if (!IS_CATEGORY_SCORED(scored_categories, c)) {
            int scr = ctx->precomputed_scores[ds_index][c];
            int new_up = UpdateUpperScore(upper_score, c, scr);
            int new_scored = scored_categories | (1 << c);
            double ev = scr + GetStateValue(ctx, new_up, new_scored);

            json_object *cat_obj = json_object_new_object();
            AddJSONField(cat_obj, "id", c);
            AddJSONFieldString(cat_obj, "name", ctx->category_names[c]);
            AddJSONField(cat_obj, "score", scr);
            AddJSONFieldDouble(cat_obj, "expected_value_if_chosen", ev);
            json_object_array_add(categories, cat_obj);
        }
    }

    json_object_object_add(resp, "categories", categories);
    RespondWithJSON(connection, MHD_HTTP_OK, resp);
}

void handle_evaluate_actions(YatzyContext *ctx, struct MHD_Connection *connection, const json_object *parsed) {
    int upper_score, scored_categories, rerolls, dice[5];
    const char *fields[] = {"upper_score", "scored_categories", "rerolls_remaining"};
    int *values[] = {&upper_score, &scored_categories, &rerolls};

    if (!ValidateAndParseJSON(parsed, fields, 3, values) || !ParseDiceFromJSON(parsed, dice)) {
        RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "Missing required fields");
        return;
    }

    if (rerolls <= 0) {
        RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "rerolls_remaining must be > 0");
        return;
    }

    const double *E_ds_for_masks = ComputeExpectedValues(ctx, upper_score, scored_categories, rerolls);
    int ds_index = FindDiceSetIndex(ctx, dice);

    json_object *resp = json_object_new_object();
    json_object *actions = json_object_new_array();

    for (int mask = 0; mask < 32; mask++) {
        EVProbabilityPair distribution[252];
        ComputeDistributionForRerollMask(ctx, ds_index, E_ds_for_masks, mask, distribution);

        double ev = ComputeEVFromDistribution(distribution, 252);
        json_object *action_obj = json_object_new_object();
        AddJSONField(action_obj, "mask", mask);
        AddJSONFieldDouble(action_obj, "expected_value", ev);
        json_object_array_add(actions, action_obj);
    }

    json_object_object_add(resp, "actions", actions);
    RespondWithJSON(connection, MHD_HTTP_OK, resp);
}

void ConvertMaskToBinaryString(int best_mask, char *mask_binary, int num_dice) {
    for (int i = 0; i < num_dice; i++) {
        mask_binary[i] = (best_mask & (1 << i)) ? '1' : '0'; // Convert each bit to '1' or '0'
    }
    mask_binary[num_dice] = '\0'; // Null-terminate the string
}

void handle_suggest_optimal_action(YatzyContext *ctx, struct MHD_Connection *connection, const json_object *parsed) {
    int upper_score, scored_categories, rerolls, dice[5];
    const char *fields[] = {"upper_score", "scored_categories", "rerolls_remaining"};
    int *values[] = {&upper_score, &scored_categories, &rerolls};

    if (!ValidateAndParseJSON(parsed, fields, 3, values) || !ParseDiceFromJSON(parsed, dice)) {
        RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "Missing required fields");
        return;
    }

    json_object *resp = json_object_new_object();
    if (rerolls > 0) {
        int best_mask;
        double ev;
        char mask_binary[6]; // 5 dice + 1 null terminator
        ComputeBestRerollStrategy(ctx, upper_score, scored_categories, dice, rerolls, &best_mask, &ev);
        ConvertMaskToBinaryString(best_mask, mask_binary, 5);

        // Build json response
        json_object *reroll_obj = json_object_new_object();
        AddJSONField(reroll_obj, "id", best_mask);
        AddJSONFieldDouble(reroll_obj, "expected_value", ev);
        json_object_object_add(resp, "best_reroll", reroll_obj);
        AddJSONFieldString(reroll_obj, "mask_binary", mask_binary);
    } else {
        double best_ev;
        const int best_category = ChooseBestCategoryNoRerolls(ctx, upper_score, scored_categories, dice, &best_ev);

        if (best_category >= 0) {
            json_object *category_obj = json_object_new_object();
            AddJSONField(category_obj, "id", best_category);
            AddJSONFieldDouble(category_obj, "expected_value", best_ev);
            json_object_object_add(resp, "best_category", category_obj);
        }
    }
    RespondWithJSON(connection, MHD_HTTP_OK, resp);
}

void handle_evaluate_user_action(const YatzyContext *ctx, struct MHD_Connection *connection, const json_object *parsed) {
    int upper_score, scored_categories, rerolls, dice[5];
    const char *fields[] = {"upper_score", "scored_categories", "rerolls_remaining"};
    int *values[] = {&upper_score, &scored_categories, &rerolls};

    if (!ValidateAndParseJSON(parsed, fields, 3, values) || !ParseDiceFromJSON(parsed, dice)) {
        RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "Missing required fields");
        return;
    }

    json_object *user_action = json_object_object_get(parsed, "user_action");
    if (!user_action) {
        RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "Missing user_action");
        return;
    }

    json_object *resp = json_object_new_object();
    if (rerolls > 0) {
        int mask;
        if (!ParseJSONField(user_action, "best_reroll", &mask)) {
            RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "Invalid best_reroll");
            return;
        }

        double ev = EvaluateChosenRerollMask(ctx, upper_score, scored_categories, dice, mask, rerolls);
        if (isnan(ev) || isinf(ev)) {
            ev = 0.0; // Replace invalid values with a default
        }
        AddJSONFieldDouble(resp, "expected_value", ev);
    } else {
        int category_id;
        if (!ParseJSONField(user_action, "best_category", &category_id)) {
            RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "Invalid best_category");
            return;
        }

        double ev = EvaluateChosenCategory(ctx, upper_score, scored_categories, dice, category_id);
        if (isnan(ev) || isinf(ev)) {
            ev = 0.0; // Replace invalid values with a default
        }
        AddJSONFieldDouble(resp, "expected_value", ev);
    }

    RespondWithJSON(connection, MHD_HTTP_OK, resp);
}

void handle_get_state_value(YatzyContext *ctx, struct MHD_Connection *connection) {
    const char *up_str = MHD_lookup_connection_value(connection, MHD_GET_ARGUMENT_KIND, "upper_score");
    const char *sc_str = MHD_lookup_connection_value(connection, MHD_GET_ARGUMENT_KIND, "scored_categories");

    if (!up_str || !sc_str) {
        RespondWithError(connection, MHD_HTTP_BAD_REQUEST,
                         "Missing upper_score or scored_categories query params");
        return;
    }

    int upper_score = atoi(up_str);
    int scored_categories = atoi(sc_str);
    double val = ctx->state_values[STATE_INDEX(upper_score, scored_categories)];

    json_object *resp = json_object_new_object();
    AddJSONField(resp, "upper_score", upper_score);
    AddJSONField(resp, "scored_categories", scored_categories);
    AddJSONFieldDouble(resp, "expected_final_score", val);

    RespondWithJSON(connection, MHD_HTTP_OK, resp);
}

void handle_get_score_histogram(YatzyContext *ctx, struct MHD_Connection *connection) {
    const char *file_path = "data/score_histogram.csv";
    FILE *file = fopen(file_path, "r");

    if (!file) {
        fprintf(stderr, "[ERROR] Failed to open file: %s\n", file_path);
        RespondWithError(connection, MHD_HTTP_NOT_FOUND, "Could not open score histogram file");
        return;
    }

    const int min_ev = 100, max_ev = 380, bin_count = 56;
    const double bin_width = (max_ev - min_ev) / (double) bin_count;
    int bins[56] = {0};

    char line[MAX_LINE_LENGTH];
    fgets(line, MAX_LINE_LENGTH, file);

    // Process file and populate bins
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        line[strcspn(line, "\n")] = 0;
        int score = atoi(line);
        if (score >= min_ev && score <= max_ev) {
            int bin_index = (int) ((score - min_ev) / bin_width);
            if (bin_index >= 0 && bin_index < bin_count) {
                bins[bin_index]++;
            }
        }
    }

    // Create JSON response
    json_object *resp = json_object_new_object();
    json_object *bins_arr = json_object_new_array();

    for (int i = 0; i < bin_count; i++) {
        json_object_array_add(bins_arr, json_object_new_int(bins[i]));
    }
    AddJSONField(resp, "min_ev", min_ev);
    AddJSONField(resp, "max_ev", max_ev);
    AddJSONField(resp, "bin_count", bin_count);
    json_object_object_add(resp, "bins", bins_arr);
    RespondWithJSON(connection, MHD_HTTP_OK, resp);
    fclose(file);
}

typedef void (*POSTHandler)(YatzyContext *, struct MHD_Connection *, const json_object *);

typedef void (*GETHandler)(YatzyContext *, struct MHD_Connection *);

typedef struct {
    const char *url;
    POSTHandler handler;
} POSTEndpoint;

typedef struct {
    const char *url;
    GETHandler handler;
} GETEndpoint;

void HandleGETRequest(YatzyContext *ctx,
                      const char *url,
                      struct MHD_Connection *connection) {
    GETEndpoint endpoints[] = {
        {"/state_value", handle_get_state_value},
        {"/score_histogram", handle_get_score_histogram},
    };

    // Search for a matching endpoint in the routing table
    for (size_t i = 0; i < sizeof(endpoints) / sizeof(endpoints[0]); i++) {
        if (strcmp(url, endpoints[i].url) == 0) {
            endpoints[i].handler(ctx, connection);
            LogRequest("GET", url, NULL, MHD_HTTP_OK);
            return;
        }
    }

    // Respond with an error if the endpoint is not found
    RespondWithError(connection, MHD_HTTP_NOT_FOUND, "Unknown endpoint");
    LogRequest("GET", url, NULL, MHD_HTTP_NOT_FOUND);
}

void HandlePOSTRequest(YatzyContext *ctx, const char *url,
                       struct MHD_Connection *connection,
                       json_object *parsed) {
    POSTEndpoint endpoints[] = {
        {"/evaluate_category_score", handle_evaluate_category_score},
        {"/available_categories", handle_available_categories},
        {"/evaluate_all_categories", handle_evaluate_all_categories},
        {"/evaluate_actions", handle_evaluate_actions},
        {"/suggest_optimal_action", handle_suggest_optimal_action},
        {"/evaluate_user_action", handle_evaluate_user_action},
    };

    // Convert JSON object to string for logging
    const char *payload = json_object_to_json_string(parsed);

    for (size_t i = 0; i < sizeof(endpoints) / sizeof(endpoints[0]); i++) {
        if (strcmp(url, endpoints[i].url) == 0) {
            LogRequest("POST", url, payload, MHD_HTTP_OK);
            endpoints[i].handler(ctx, connection, parsed);
            return;
        }
    }

    RespondWithError(connection, MHD_HTTP_NOT_FOUND, "Unknown endpoint");
    LogRequest("POST", url, payload, MHD_HTTP_NOT_FOUND);
}

enum MHD_Result answer_to_connection(void *cls,
                                     struct MHD_Connection *connection,
                                     const char *url,
                                     const char *method,
                                     const char *version,
                                     const char *upload_data,
                                     size_t *upload_data_size, void **con_cls) {
    YatzyContext *ctx = cls;

    if (*con_cls == NULL) {
        struct RequestContext *req_ctx = calloc(1, sizeof(struct RequestContext));
        req_ctx->post_data = NULL;
        req_ctx->post_size = 0;
        *con_cls = req_ctx;
        return MHD_YES;
    }

    struct RequestContext *req_ctx = *con_cls;

    if (strcmp(method, "OPTIONS") == 0) {
        struct MHD_Response *options_response = MHD_create_response_from_buffer(0, "", MHD_RESPMEM_PERSISTENT);
        AddCORSHeaders(options_response);
        MHD_add_response_header(options_response, "Access-Control-Max-Age", "86400");
        MHD_queue_response(connection, MHD_HTTP_OK, options_response);
        MHD_destroy_response(options_response);
        free(req_ctx->post_data);
        free(req_ctx);
        *con_cls = NULL;
        return MHD_YES;
    }

    if (strcmp(method, "POST") == 0) {
        if (*upload_data_size != 0) {
            size_t new_size = req_ctx->post_size + *upload_data_size;
            req_ctx->post_data = (char *) realloc(req_ctx->post_data, new_size + 1);
            memcpy(req_ctx->post_data + req_ctx->post_size, upload_data, *upload_data_size);
            req_ctx->post_size = new_size;
            req_ctx->post_data[req_ctx->post_size] = '\0';
            *upload_data_size = 0;
            return MHD_YES;
        }

        if (!req_ctx->post_data) {
            RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "No data received");
            return MHD_YES;
        }

        json_object *parsed = json_tokener_parse(req_ctx->post_data);
        if (!parsed) {
            RespondWithError(connection, MHD_HTTP_BAD_REQUEST, "Invalid JSON");
            return MHD_YES;
        }

        HandlePOSTRequest(ctx, url, connection, parsed);
        json_object_put(parsed);
        free(req_ctx->post_data);
        free(req_ctx);
        *con_cls = NULL;
        return MHD_YES;
    }

    if (strcmp(method, "GET") == 0) {
        HandleGETRequest(ctx, url, connection);
        free(req_ctx->post_data);
        free(req_ctx);
        *con_cls = NULL;
        return MHD_YES;
    }

    RespondWithError(connection, MHD_HTTP_METHOD_NOT_ALLOWED, "Only POST and GET supported");
    return MHD_YES;
}

static volatile int running = 1;

static void handle_signal(const int signal) {
    printf("Signal %d received. Stopping server...\n", signal);
    running = 0;
}

int start_webserver(YatzyContext *ctx, const int port) {
    struct MHD_Daemon *daemon = MHD_start_daemon(MHD_USE_INTERNAL_POLLING_THREAD,
                                                 port,
                                                 NULL, NULL,
                                                 &answer_to_connection, ctx,
                                                 MHD_OPTION_END);

    if (daemon == NULL) {
        perror("MHD_start_daemon failed");
        return 1;
    }

    signal(SIGINT, handle_signal);
    printf("Server is running on port %d. Press Ctrl+C to stop.\n", port);

    while (running) {
        sleep(1);
    }

    printf("\nStopping server...\n");
    MHD_stop_daemon(daemon);
    return 0;
}
