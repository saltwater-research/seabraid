// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Diagnostics through the C ABI: log sink, last error, version. Built as C11.
 * SEABRAID_EXPECTED_VERSION is the VERSION file's text, passed in by CMake. */
#include "seabraid/seabraid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "line %d: %s failed\n", __LINE__, #expression); \
        abort(); \
    } \
} while (0)

struct capture {
    unsigned count;
    uint32_t level;
    int32_t status;
    char message[256];
};

static void sink(void *context, uint32_t level, int32_t status, const char *message)
{
    struct capture *capture = (struct capture *)context;
    capture->count++;
    capture->level = level;
    capture->status = status;
    snprintf(capture->message, sizeof capture->message, "%s", message);
}

static int has_path_separator(const char *text)
{
    return strchr(text, '/') != NULL || strchr(text, '\\') != NULL;
}

static seabraid_session *open_prepared(void)
{
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, 2, 3, 4, 2};
    seabraid_session *session = NULL;
    CHECK(seabraid_open_reference(&session) == SEABRAID_OK && session != NULL);
    CHECK(seabraid_prepare(session, &config) == SEABRAID_OK);
    return session;
}

static void check_invalid_call_is_logged_once(struct capture *capture)
{
    seabraid_config_v1 stale = {SEABRAID_ABI_VERSION + 98, 2, 3, 4, 2};
    seabraid_session *session = open_prepared();
    seabraid_session *refused = NULL;
    char text[256];
    char shorter[6];
    size_t needed = 0;

    /* The explicit reference open is silent; a NULL provider is an error logged once at WARN. */
    CHECK(capture->count == 0);
    CHECK(seabraid_open(NULL, &refused) == SEABRAID_INVALID && refused == NULL);
    CHECK(capture->count == 1 && capture->level == SEABRAID_LOG_WARN);
    CHECK(strstr(capture->message, "provider is null") != NULL);
    capture->count = 0;
    CHECK(seabraid_prepare(session, &stale) == SEABRAID_INVALID);
    CHECK(capture->count == 1);
    CHECK(capture->level == SEABRAID_LOG_WARN && capture->status == SEABRAID_INVALID);
    CHECK(strstr(capture->message, "seabraid_prepare") != NULL && strstr(capture->message, "99") != NULL);
    CHECK(!has_path_separator(capture->message));

    CHECK(seabraid_last_error(text, sizeof text, &needed) == SEABRAID_OK);
    CHECK(strcmp(text, capture->message) == 0 && needed == strlen(text) + 1);
    CHECK(!has_path_separator(text));

    /* Truncation stays NUL-terminated and reports the full size. */
    CHECK(seabraid_last_error(shorter, sizeof shorter, &needed) == SEABRAID_OK);
    CHECK(strlen(shorter) == sizeof shorter - 1 && strncmp(shorter, text, sizeof shorter - 1) == 0);
    CHECK(needed == strlen(text) + 1);
    CHECK(seabraid_last_error(NULL, 0, &needed) == SEABRAID_OK && needed == strlen(text) + 1);
    CHECK(capture->count == 1);
    CHECK(seabraid_close(session) == SEABRAID_OK);
}

static void check_threshold_and_notices(struct capture *capture)
{
    capture->count = 0;
    CHECK(seabraid_set_log_sink(sink, capture, SEABRAID_LOG_WARN) == SEABRAID_OK);
    CHECK(seabraid_log(SEABRAID_LOG_INFO, 0, "below the threshold") == SEABRAID_OK && capture->count == 0);
    CHECK(seabraid_log(SEABRAID_LOG_WARN, 0, "adapter notice") == SEABRAID_OK && capture->count == 1);
    CHECK(capture->level == SEABRAID_LOG_WARN && strcmp(capture->message, "adapter notice") == 0);
    CHECK(seabraid_log(SEABRAID_LOG_OFF, 0, "off is a threshold") == SEABRAID_INVALID && capture->count == 2);
    CHECK(seabraid_log(SEABRAID_LOG_ERROR, SEABRAID_PROVIDER, NULL) == SEABRAID_INVALID && capture->count == 3);
    CHECK(seabraid_set_log_sink(sink, capture, 99) == SEABRAID_INVALID);
}

static void check_null_sink_restores_the_default(struct capture *capture)
{
    capture->count = 0;
    CHECK(seabraid_set_log_sink(sink, capture, SEABRAID_LOG_DEBUG) == SEABRAID_OK);
    CHECK(seabraid_set_log_sink(NULL, NULL, SEABRAID_LOG_OFF) == SEABRAID_OK);
    CHECK(seabraid_log(SEABRAID_LOG_ERROR, 0, "silenced by SEABRAID_LOG_OFF") == SEABRAID_OK);
    CHECK(capture->count == 0);
    CHECK(seabraid_set_log_sink(NULL, NULL, SEABRAID_LOG_WARN) == SEABRAID_OK);
    CHECK(seabraid_log(SEABRAID_LOG_DEBUG, 0, "dropped by the default threshold") == SEABRAID_OK);
    CHECK(capture->count == 0);
}

int main(void)
{
    struct capture capture = {0, 0, 0, ""};
    char empty[8] = "x";
    size_t needed = 0;

    CHECK(strcmp(seabraid_version(), SEABRAID_EXPECTED_VERSION) == 0);
    CHECK(seabraid_last_error(empty, sizeof empty, &needed) == SEABRAID_OK);
    CHECK(empty[0] == '\0' && needed == 1);

    CHECK(seabraid_set_log_sink(sink, &capture, SEABRAID_LOG_DEBUG) == SEABRAID_OK);
    check_invalid_call_is_logged_once(&capture);
    check_threshold_and_notices(&capture);
    check_null_sink_restores_the_default(&capture);
    puts("Seabraid diagnostics smoke passed");
    return 0;
}
