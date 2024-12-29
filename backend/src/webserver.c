#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <json-c/json.h>

#include "yatzy.h"
#include "precompute_scores.h"
#include "webserver.h"

#include "file_utilities.h"

struct RequestContext {
    char *post_data;
    size_t post_size;
};

// Add a helper function to set CORS headers on every response
void AddCORSHeaders(struct MHD_Response *response) {
    MHD_add_response_header(response, "Access-Control-Allow-Origin", "*");
    MHD_add_response_header(response, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    MHD_add_response_header(response, "Access-Control-Allow-Headers",
                            "Content-Type, Authorization, X-Requested-With, sec-ch-ua, sec-ch-ua-mobile, sec-ch-ua-platform");
    MHD_add_response_header(response, "Access-Control-Max-Age", "86400");
}


void handle_get_score_histogram(YatzyContext *ctx, struct MHD_Connection *connection) {
    FILE *file = fopen("backend/data/score_histogram.csv", "r");
    if (file == NULL) {
        const char *err = "{\"error\":\"Could not open score histogram file\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(
            strlen(err),
            (void *) err,
            MHD_RESPMEM_PERSISTENT
        );
        AddCORSHeaders(resp);
        MHD_queue_response(connection, MHD_HTTP_NOT_FOUND, resp);
        MHD_destroy_response(resp);
        return;
    }

    // Create bins for the histogram (using same parameters as frontend)
    const int min_ev = 100;
    const int max_ev = 380;
    const int bin_count = 56;
    const double bin_width = (max_ev - min_ev) / (double) bin_count;
    int bins[56] = {0}; // Initialize all bins to 0

    char line[MAX_LINE_LENGTH];
    // Skip header if present
    if (fgets(line, MAX_LINE_LENGTH, file)) {
        // Skip header
    }

    // Read and bin the data
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
    struct json_object *resp = json_object_new_object();
    struct json_object *bins_arr = json_object_new_array();

    // Add bin data
    for (int i = 0; i < bin_count; i++) {
        json_object_array_add(bins_arr, json_object_new_int(bins[i]));
    }

    // Add metadata
    json_object_object_add(resp, "bins", bins_arr);
    json_object_object_add(resp, "min_ev", json_object_new_int(min_ev));
    json_object_object_add(resp, "max_ev", json_object_new_int(max_ev));
    json_object_object_add(resp, "bin_count", json_object_new_int(bin_count));

    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(
        strlen(response_str),
        (void *) response_str,
        MHD_RESPMEM_MUST_COPY
    );
    AddCORSHeaders(mhd_response);

    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
    fclose(file);
}

// GET /state_value?upper_score=X&scored_categories=Y
void handle_get_state_value(YatzyContext *ctx, struct MHD_Connection *connection) {
    const char *up_str = MHD_lookup_connection_value(connection, MHD_GET_ARGUMENT_KIND, "upper_score");
    const char *sc_str = MHD_lookup_connection_value(connection, MHD_GET_ARGUMENT_KIND, "scored_categories");
    if (!up_str || !sc_str) {
        const char *err = "{\"error\":\"Missing upper_score or scored_categories query params\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection,MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }

    int upper_score = atoi(up_str);
    int scored_categories = atoi(sc_str);

    double val = GetStateValue(ctx, upper_score, scored_categories);

    struct json_object *resp = json_object_new_object();
    json_object_object_add(resp, "upper_score", json_object_new_int(upper_score));
    json_object_object_add(resp, "scored_categories", json_object_new_int(scored_categories));
    json_object_object_add(resp, "expected_final_score", json_object_new_double(val));

    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str), (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    MHD_queue_response(connection,MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}

void handle_evaluate_category_score(YatzyContext *ctx, struct MHD_Connection *connection,
                                           struct json_object *parsed) {
    struct json_object *dice_arr = json_object_object_get(parsed, "dice");
    struct json_object *cat_obj = json_object_object_get(parsed, "category_id");
    if (!dice_arr || !cat_obj) {
        const char *err = "{\"error\":\"Missing dice or category_id\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }
    int dice[5];
    for (int i = 0; i < 5; i++) {
        struct json_object *d = json_object_array_get_idx(dice_arr, i);
        dice[i] = json_object_get_int(d);
    }
    int cat_id = json_object_get_int(cat_obj);
    SortDiceSet(dice);
    int score = ctx->precomputed_scores[FindDiceSetIndex(ctx, dice)][cat_id];

    struct json_object *resp = json_object_new_object();
    json_object_object_add(resp, "category_id", json_object_new_int(cat_id));
    json_object_object_add(resp, "category_name", json_object_new_string(ctx->category_names[cat_id]));
    json_object_object_add(resp, "score", json_object_new_int(score));

    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str),
                                                                        (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}

void handle_available_categories(YatzyContext *ctx, struct MHD_Connection *connection,
                                        struct json_object *parsed) {
    struct json_object *upper_obj = json_object_object_get(parsed, "upper_score");
    struct json_object *scored_obj = json_object_object_get(parsed, "scored_categories");
    struct json_object *dice_arr = json_object_object_get(parsed, "dice");
    struct json_object *rerolls_obj = json_object_object_get(parsed, "rerolls_remaining");

    if (!upper_obj || !scored_obj || !dice_arr || !rerolls_obj) {
        const char *err = "{\"error\":\"Missing required fields\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        AddCORSHeaders(resp);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }

    int upper_score = json_object_get_int(upper_obj);
    int scored_categories = json_object_get_int(scored_obj);
    // rerolls not needed for scoring, but we parse it for consistency
    int rerolls = json_object_get_int(rerolls_obj);

    int dice[5];
    for (int i = 0; i < 5; i++) {
        struct json_object *d = json_object_array_get_idx(dice_arr, i);
        if (!d) {
            const char *err = "{\"error\":\"Invalid dice array\"}";
            struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err,
                                                                        MHD_RESPMEM_PERSISTENT);
            AddCORSHeaders(resp);
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
            MHD_destroy_response(resp);
            return;
        }
        dice[i] = json_object_get_int(d);
    }

    SortDiceSet(dice);
    int ds_index = FindDiceSetIndex(ctx, dice);

    // Construct response array
    struct json_object *resp = json_object_new_object();
    struct json_object *arr = json_object_new_array();

    for (int c = 0; c < CATEGORY_COUNT; c++) {
        int scr = ctx->precomputed_scores[ds_index][c];
        // valid if not scored and scr>0
        int is_already_scored = IS_CATEGORY_SCORED(scored_categories, c);
        bool valid = (!is_already_scored && scr > 0);

        struct json_object *cat_obj = json_object_new_object();
        json_object_object_add(cat_obj, "id", json_object_new_int(c));
        json_object_object_add(cat_obj, "name", json_object_new_string(ctx->category_names[c]));
        json_object_object_add(cat_obj, "score", json_object_new_int(scr));
        json_object_object_add(cat_obj, "valid", json_object_new_boolean(valid));
        json_object_array_add(arr, cat_obj);
    }

    json_object_object_add(resp, "categories", arr);

    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str), (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    AddCORSHeaders(mhd_response);
    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}

void handle_evaluate_all_categories(YatzyContext *ctx, struct MHD_Connection *connection,
                                           struct json_object *parsed) {
    struct json_object *upper_obj = json_object_object_get(parsed, "upper_score");
    struct json_object *scored_obj = json_object_object_get(parsed, "scored_categories");
    struct json_object *dice_arr = json_object_object_get(parsed, "dice");
    struct json_object *rerolls_obj = json_object_object_get(parsed, "rerolls_remaining");

    if (!upper_obj || !scored_obj || !dice_arr || !rerolls_obj) {
        const char *err = "{\"error\":\"Missing required fields\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }

    int upper_score = json_object_get_int(upper_obj);
    int scored_categories = json_object_get_int(scored_obj);
    int rerolls = json_object_get_int(rerolls_obj);

    if (rerolls != 0) {
        const char *err = "{\"error\":\"rerolls_remaining must be 0 for this endpoint\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }

    int dice[5];
    for (int i = 0; i < 5; i++) {
        struct json_object *d = json_object_array_get_idx(dice_arr, i);
        if (!d) {
            const char *err = "{\"error\":\"Invalid dice array\"}";
            struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err,
                                                                        MHD_RESPMEM_PERSISTENT);
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
            MHD_destroy_response(resp);
            return;
        }
        dice[i] = json_object_get_int(d);
    }
    SortDiceSet(dice);

    int ds_index = FindDiceSetIndex(ctx, dice);
    struct json_object *resp = json_object_new_object();
    struct json_object *arr = json_object_new_array();

    // For each unscored category, compute the EV if chosen now
    for (int c = 0; c < CATEGORY_COUNT; c++) {
        if (!IS_CATEGORY_SCORED(scored_categories, c)) {
            int scr = ctx->precomputed_scores[ds_index][c];
            int new_up = UpdateUpperScore(upper_score, c, scr);
            int new_scored = scored_categories | (1 << c);
            double val = scr + GetStateValue(ctx, new_up, new_scored);

            struct json_object *cat_obj = json_object_new_object();
            json_object_object_add(cat_obj, "id", json_object_new_int(c));
            json_object_object_add(cat_obj, "name", json_object_new_string(ctx->category_names[c]));
            json_object_object_add(cat_obj, "score", json_object_new_int(scr));
            json_object_object_add(cat_obj, "expected_value_if_chosen", json_object_new_double(val));
            json_object_array_add(arr, cat_obj);
        }
    }

    json_object_object_add(resp, "categories", arr);
    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str), (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}

void handle_evaluate_actions(YatzyContext *ctx, struct MHD_Connection *connection, struct json_object *parsed) {
    struct json_object *upper_obj = json_object_object_get(parsed, "upper_score");
    struct json_object *scored_obj = json_object_object_get(parsed, "scored_categories");
    struct json_object *dice_arr = json_object_object_get(parsed, "dice");
    struct json_object *rerolls_obj = json_object_object_get(parsed, "rerolls_remaining");

    if (!upper_obj || !scored_obj || !dice_arr || !rerolls_obj) {
        const char *err = "{\"error\":\"Missing required fields\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }

    int upper_score = json_object_get_int(upper_obj);
    int scored_categories = json_object_get_int(scored_obj);
    int rerolls = json_object_get_int(rerolls_obj);

    if (rerolls <= 0) {
        const char *err = "{\"error\":\"rerolls_remaining must be > 0\"}";
        struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err, MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
        MHD_destroy_response(resp);
        return;
    }

    int dice[5];
    for (int i = 0; i < 5; i++) {
        struct json_object *d = json_object_array_get_idx(dice_arr, i);
        if (!d) {
            const char *err = "{\"error\":\"Invalid dice array\"}";
            struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(err), (void *) err,
                                                                        MHD_RESPMEM_PERSISTENT);
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, resp);
            MHD_destroy_response(resp);
            return;
        }
        dice[i] = json_object_get_int(d);
    }
    SortDiceSet(dice);

    // Compute E_ds_0 for no rerolls scenario
    double E_ds_0[252]; {
        YatzyState state = {upper_score, scored_categories};
#pragma omp parallel for
        for (int ds_i = 0; ds_i < 252; ds_i++) {
            double best_val = -INFINITY;
            for (int c = 0; c < CATEGORY_COUNT; c++) {
                if (!IS_CATEGORY_SCORED(scored_categories, c)) {
                    int scr = ctx->precomputed_scores[ds_i][c];
                    int new_up = UpdateUpperScore(upper_score, c, scr);
                    int new_scored = scored_categories | (1 << c);
                    double val = scr + GetStateValue(ctx, new_up, new_scored);
                    if (val > best_val) best_val = val;
                }
            }
            E_ds_0[ds_i] = best_val;
        }
    }

    // If rerolls=2, we need E_ds_1 for one reroll scenario
    double E_ds_1[252];
    if (rerolls == 2) {
#pragma omp parallel for
        for (int ds_i = 0; ds_i < 252; ds_i++) {
            double best_val = -INFINITY;
            const double (*row)[252] = ctx->transition_table[ds_i];
            for (int mask = 0; mask < 32; mask++) {
                double ev = 0.0;
                for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                    ev += row[mask][ds2_i] * E_ds_0[ds2_i];
                }
                if (ev > best_val) best_val = ev;
            }
            E_ds_1[ds_i] = best_val;
        }
    }

    const double *E_ds_for_masks = (rerolls == 1) ? E_ds_0 : E_ds_1;

    int sorted[5];
    for (int i = 0; i < 5; i++) sorted[i] = dice[i];
    SortDiceSet(sorted);
    int ds_index = FindDiceSetIndex(ctx, sorted);

    struct json_object *resp = json_object_new_object();
    struct json_object *arr = json_object_new_array();
    for (int mask = 0; mask < 32; mask++) {
        EVProbabilityPair distribution[252];
        ComputeDistributionForRerollMask(ctx, ds_index, E_ds_for_masks, mask, distribution);

        double ev = 0.0;
        struct json_object *dist_arr = json_object_new_array();
        for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
            if (distribution[ds2_i].probability > 0.0) {
                ev += distribution[ds2_i].ev * distribution[ds2_i].probability;

                struct json_object *entry = json_object_new_object();
                json_object_object_add(entry, "ds2_index", json_object_new_int(distribution[ds2_i].ds2_index));
                json_object_object_add(entry, "ev", json_object_new_double(distribution[ds2_i].ev));
                json_object_object_add(entry, "probability", json_object_new_double(distribution[ds2_i].probability));
                json_object_array_add(dist_arr, entry);
            }
        }

        struct json_object *o = json_object_new_object();
        json_object_object_add(o, "mask", json_object_new_int(mask));
        char bin_str[6] = {0};
        for (int i = 0; i < 5; i++) bin_str[i] = (mask & (1 << i)) ? '1' : '0';
        json_object_object_add(o, "binary", json_object_new_string(bin_str));
        json_object_object_add(o, "expected_value", json_object_new_double(ev));
        json_object_object_add(o, "distribution", dist_arr);

        json_object_array_add(arr, o);
    }
    json_object_object_add(resp, "actions", arr);

    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str), (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    AddCORSHeaders(mhd_response);

    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}

void handle_suggest_optimal_action(YatzyContext *ctx, struct MHD_Connection *connection,
                                          struct json_object *parsed) {
    struct json_object *upper_score_obj = json_object_object_get(parsed, "upper_score");
    struct json_object *scored_categories_obj = json_object_object_get(parsed, "scored_categories");
    struct json_object *dice_arr = json_object_object_get(parsed, "dice");
    struct json_object *rerolls_obj = json_object_object_get(parsed, "rerolls_remaining");

    if (!upper_score_obj || !scored_categories_obj || !dice_arr || !rerolls_obj) {
        const char *error_msg = "{\"error\":\"Missing required fields\"}";
        struct MHD_Response *error_response = MHD_create_response_from_buffer(strlen(error_msg),
                                                                              (void *) error_msg,
                                                                              MHD_RESPMEM_PERSISTENT);
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
        MHD_destroy_response(error_response);
        return;
    }

    int upper_score = json_object_get_int(upper_score_obj);
    int scored_categories = json_object_get_int(scored_categories_obj);
    int rerolls = json_object_get_int(rerolls_obj);

    int dice[5];
    for (int i = 0; i < 5; i++) {
        struct json_object *die_val = json_object_array_get_idx(dice_arr, i);
        if (!die_val) {
            const char *error_msg = "{\"error\":\"Invalid dice array\"}";
            struct MHD_Response *error_response = MHD_create_response_from_buffer(strlen(error_msg),
                (void *) error_msg,
                MHD_RESPMEM_PERSISTENT);
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
            MHD_destroy_response(error_response);
            return;
        }
        dice[i] = json_object_get_int(die_val);
    }
    SortDiceSet(dice);


    // Create a new response object
    struct json_object *resp = json_object_new_object();

    // Initialize response keys
    json_object_object_add(resp, "dice_set", NULL);

    // Conver the dice set into a string
    char dice_str[8] = {0};
    for (int i = 0; i < 5; i++) {
        dice_str[i] = dice[i] + '0';
    }
    json_object_object_add(resp, "dice_set", json_object_new_string(dice_str));

    json_object_object_add(resp, "best_category", NULL);
    json_object_object_add(resp, "best_reroll", NULL);
    json_object_object_add(resp, "expected_value", NULL);

    if (rerolls > 0) {
        int best_mask;
        double ev;
        ComputeBestRerollStrategy(ctx, upper_score, scored_categories, dice, rerolls, &best_mask, &ev);

        // Create a "best_reroll" object
        struct json_object *reroll_obj = json_object_new_object();
        json_object_object_add(reroll_obj, "id", json_object_new_int(best_mask));

        // Binary string for clarity
        char bin_str[6] = {0};
        for (int i = 0; i < 5; i++) {
            bin_str[i] = (best_mask & (1 << i)) ? '1' : '0';
        }
        json_object_object_add(reroll_obj, "mask_binary", json_object_new_string(bin_str));
        json_object_object_add(resp, "best_reroll", reroll_obj);
        json_object_object_add(resp, "expected_value", json_object_new_double(ev));

        // Do NOT call json_object_put(reroll_obj) after adding it to `resp`
    }

    if (rerolls == 0) {
        double best_ev;
        int best_category = ChooseBestCategoryNoRerolls(ctx, upper_score, scored_categories, dice, &best_ev);

        if (best_category >= 0) {
            struct json_object *category_obj = json_object_new_object();
            json_object_object_add(category_obj, "id", json_object_new_int(best_category));
            json_object_object_add(category_obj, "name", json_object_new_string(ctx->category_names[best_category]));
            json_object_object_add(resp, "best_category", category_obj);
            json_object_object_add(resp, "expected_value", json_object_new_double(best_ev));
        }
    }

    // Send the response
    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str), (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    // Add CORS headers here
    AddCORSHeaders(mhd_response);

    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}

void handle_evaluate_user_action(YatzyContext *ctx, struct MHD_Connection *connection,
                                        struct json_object *parsed) {
    struct json_object *upper_score_obj = json_object_object_get(parsed, "upper_score");
    struct json_object *scored_categories_obj = json_object_object_get(parsed, "scored_categories");
    struct json_object *dice_arr = json_object_object_get(parsed, "dice");
    struct json_object *rerolls_obj = json_object_object_get(parsed, "rerolls_remaining");
    struct json_object *user_action_obj = json_object_object_get(parsed, "user_action");

    if (!upper_score_obj || !scored_categories_obj || !dice_arr || !rerolls_obj || !user_action_obj) {
        const char *error_msg = "{\"error\":\"Missing required fields\"}";
        struct MHD_Response *error_response = MHD_create_response_from_buffer(
            strlen(error_msg), (void *) error_msg, MHD_RESPMEM_PERSISTENT);
        AddCORSHeaders(error_response); // Add CORS headers
        MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
        MHD_destroy_response(error_response);
        return;
    }

    int upper_score = json_object_get_int(upper_score_obj);
    int scored_categories = json_object_get_int(scored_categories_obj);
    int rerolls = json_object_get_int(rerolls_obj);

    int dice[5];
    for (int i = 0; i < 5; i++) {
        struct json_object *die_val = json_object_array_get_idx(dice_arr, i);
        if (!die_val) {
            const char *error_msg = "{\"error\":\"Invalid dice array\"}";
            struct MHD_Response *error_response = MHD_create_response_from_buffer(
                strlen(error_msg), (void *) error_msg, MHD_RESPMEM_PERSISTENT);
            AddCORSHeaders(error_response); // Add CORS headers
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
            MHD_destroy_response(error_response);
            return;
        }
        dice[i] = json_object_get_int(die_val);
    }
    SortDiceSet(dice);

    struct json_object *resp = json_object_new_object();
    json_object_object_add(resp, "expected_value", NULL);
    json_object_object_add(resp, "best_category", NULL);
    json_object_object_add(resp, "best_reroll", NULL);

    if (rerolls > 0) {
        struct json_object *best_reroll_obj = json_object_object_get(user_action_obj, "best_reroll");
        if (!best_reroll_obj) {
            const char *error_msg = "{\"error\":\"User action missing best_reroll field\"}";
            struct MHD_Response *error_response = MHD_create_response_from_buffer(
                strlen(error_msg), (void *) error_msg, MHD_RESPMEM_PERSISTENT);
            AddCORSHeaders(error_response); // Add CORS headers
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
            MHD_destroy_response(error_response);
            json_object_put(resp); // Free resp
            return;
        }

        struct json_object *mask_id_obj = json_object_object_get(best_reroll_obj, "id");
        if (!mask_id_obj) {
            const char *error_msg = "{\"error\":\"User action missing id in best_reroll\"}";
            struct MHD_Response *error_response = MHD_create_response_from_buffer(
                strlen(error_msg), (void *) error_msg, MHD_RESPMEM_PERSISTENT);
            AddCORSHeaders(error_response); // Add CORS headers
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
            MHD_destroy_response(error_response);
            json_object_put(resp);
            return;
        }

        int mask = json_object_get_int(mask_id_obj);
        double ev = EvaluateChosenRerollMask(ctx, upper_score, scored_categories, dice, mask, rerolls);

        json_object_object_add(resp, "expected_value", json_object_new_double(ev));

        // Increment ref count before adding to resp
        json_object_get(best_reroll_obj);
        json_object_object_add(resp, "best_reroll", best_reroll_obj);
    } else {
        struct json_object *best_category_obj = json_object_object_get(user_action_obj, "best_category");
        if (!best_category_obj) {
            const char *error_msg = "{\"error\":\"User action missing best_category field\"}";
            struct MHD_Response *error_response = MHD_create_response_from_buffer(
                strlen(error_msg), (void *) error_msg, MHD_RESPMEM_PERSISTENT);
            AddCORSHeaders(error_response); // Add CORS headers
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
            MHD_destroy_response(error_response);
            json_object_put(resp);
            return;
        }

        struct json_object *category_id_obj = json_object_object_get(best_category_obj, "id");
        if (!category_id_obj) {
            const char *error_msg = "{\"error\":\"User action missing id in best_category\"}";
            struct MHD_Response *error_response = MHD_create_response_from_buffer(
                strlen(error_msg), (void *) error_msg, MHD_RESPMEM_PERSISTENT);
            AddCORSHeaders(error_response); // Add CORS headers
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, error_response);
            MHD_destroy_response(error_response);
            json_object_put(resp);
            return;
        }

        int category_id = json_object_get_int(category_id_obj);
        double ev = EvaluateChosenCategory(ctx, upper_score, scored_categories, dice, category_id);

        json_object_object_add(resp, "expected_value", json_object_new_double(ev));

        // Increment ref count before adding to resp
        json_object_get(best_category_obj);
        json_object_object_add(resp, "best_category", best_category_obj);
    }

    const char *response_str = json_object_to_json_string(resp);
    struct MHD_Response *mhd_response = MHD_create_response_from_buffer(strlen(response_str), (void *) response_str,
                                                                        MHD_RESPMEM_MUST_COPY);
    AddCORSHeaders(mhd_response); // Add CORS headers to the successful response
    MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
    MHD_destroy_response(mhd_response);
    json_object_put(resp);
}


// Utility function to send an error response with CORS headers
enum MHD_Result respond_with_error(struct MHD_Connection *connection, int status_code,
                                          const char *error_message, struct RequestContext *req_ctx) {
    struct MHD_Response *response = MHD_create_response_from_buffer(strlen(error_message),
                                                                    (void *) error_message,
                                                                    MHD_RESPMEM_PERSISTENT);
    AddCORSHeaders(response);
    MHD_queue_response(connection, status_code, response);
    MHD_destroy_response(response);

    if (req_ctx) {
        free(req_ctx->post_data);
        free(req_ctx);
    }
    return MHD_YES;
}

// Main request handler:
enum MHD_Result answer_to_connection(void *cls, struct MHD_Connection *connection,
                                            const char *url, const char *method,
                                            const char *version, const char *upload_data,
                                            size_t *upload_data_size, void **con_cls) {
    YatzyContext *ctx = (YatzyContext *) cls; // Server context

    // Initialize request context if not already done
    if (*con_cls == NULL) {
        struct RequestContext *req_ctx = (struct RequestContext *) calloc(1, sizeof(struct RequestContext));
        req_ctx->post_data = NULL;
        req_ctx->post_size = 0;
        *con_cls = req_ctx;
        return MHD_YES;
    }

    struct RequestContext *req_ctx = (struct RequestContext *) (*con_cls);

    // Handle OPTIONS preflight requests
    if (strcmp(method, "OPTIONS") == 0) {
        printf("OPTIONS request received for URL: %s\n", url);
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

    // Handle POST requests
    if (strcmp(method, "POST") == 0) {
        // Collect upload data
        if (*upload_data_size != 0) {
            size_t new_size = req_ctx->post_size + *upload_data_size;
            req_ctx->post_data = (char *) realloc(req_ctx->post_data, new_size + 1);
            if (!req_ctx->post_data) {
                return respond_with_error(connection, MHD_HTTP_INTERNAL_SERVER_ERROR, "Memory allocation error",
                                          req_ctx);
            }
            memcpy(req_ctx->post_data + req_ctx->post_size, upload_data, *upload_data_size);
            req_ctx->post_size = new_size;
            req_ctx->post_data[req_ctx->post_size] = '\0';
            *upload_data_size = 0;
            return MHD_YES;
        }

        // Process full POST data
        if (!req_ctx->post_data) {
            struct MHD_Response *response = MHD_create_response_from_buffer(
                strlen("No data received"),
                (void *) "No data received",
                MHD_RESPMEM_PERSISTENT
            );
            AddCORSHeaders(response);
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, response);
            MHD_destroy_response(response);
            free(req_ctx);
            *con_cls = NULL;
            return MHD_YES;
        }

        struct json_object *parsed = json_tokener_parse(req_ctx->post_data);
        if (!parsed) {
            struct MHD_Response *response = MHD_create_response_from_buffer(
                strlen("Invalid JSON"),
                (void *) "Invalid JSON",
                MHD_RESPMEM_PERSISTENT
            );
            AddCORSHeaders(response);
            MHD_queue_response(connection, MHD_HTTP_BAD_REQUEST, response);
            MHD_destroy_response(response);
            free(req_ctx->post_data);
            free(req_ctx);
            *con_cls = NULL;
            return MHD_YES;
        }

        // Route the POST request
        if (strcmp(url, "/evaluate_category_score") == 0) {
            handle_evaluate_category_score(ctx, connection, parsed);
        } else if (strcmp(url, "/available_categories") == 0) {
            handle_available_categories(ctx, connection, parsed);
        } else if (strcmp(url, "/evaluate_all_categories") == 0) {
            handle_evaluate_all_categories(ctx, connection, parsed);
        } else if (strcmp(url, "/evaluate_actions") == 0) {
            handle_evaluate_actions(ctx, connection, parsed);
        } else if (strcmp(url, "/suggest_optimal_action") == 0) {
            handle_suggest_optimal_action(ctx, connection, parsed);
        } else if (strcmp(url, "/evaluate_user_action") == 0) {
            handle_evaluate_user_action(ctx, connection, parsed);
        } else {
            struct MHD_Response *response = MHD_create_response_from_buffer(
                strlen("Unknown endpoint"),
                (void *) "Unknown endpoint",
                MHD_RESPMEM_PERSISTENT
            );
            AddCORSHeaders(response);
            MHD_queue_response(connection, MHD_HTTP_NOT_FOUND, response);
            MHD_destroy_response(response);
            json_object_put(parsed);
            free(req_ctx->post_data);
            free(req_ctx);
            *con_cls = NULL;
            return MHD_YES;
        }

        json_object_put(parsed);
        free(req_ctx->post_data);
        free(req_ctx);
        *con_cls = NULL;
        return MHD_YES;
    }

    // Handle GET requests
    if (strcmp(method, "GET") == 0) {
        if (strncmp(url, "/state_value", 12) == 0) {
            handle_get_state_value(ctx, connection);
        } else if (strcmp(url, "/score_histogram") == 0) {
            handle_get_score_histogram(ctx, connection);
        } else {
            return respond_with_error(connection, MHD_HTTP_NOT_FOUND, "Unknown endpoint", req_ctx);
        }

        free(req_ctx->post_data);
        free(req_ctx);
        *con_cls = NULL;
        return MHD_YES;
    }

    // Unsupported methods
    return respond_with_error(connection, MHD_HTTP_METHOD_NOT_ALLOWED, "Only POST and GET supported", req_ctx);
}