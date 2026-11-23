// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* The adapter against src/picks/golden.txt, split out of test_routes.c to keep both under the size limit.  This is an
 * include-once fragment of test_routes.c, not a library: it uses that file's `failures`, CHECK, from_bits and row
 * helpers, and it is compiled into each of the two executables that file builds.  OWNER of the table: src/picks.rs. */
#ifndef SEABRAID_TEST_ROUTES_GOLDEN_H
#define SEABRAID_TEST_ROUTES_GOLDEN_H

/* ---- src/picks/golden.txt: the pick rule's shared table (OWNER: src/picks.rs). Each `case` row must get its
 * verdict KIND from seabraid_ggml_routes_fault, except a partial=0 row that uses a skip (this adapter always advertises
 * PARTIAL_ROUTES, so it accepts the skip that row says is refused); the pick index is not reported by this adapter, so
 * only the runtime and its Rust test hold that. The two `limit` rows must equal this adapter's constants. FAILURE: a
 * missing or malformed table fails the test loudly (a table that cannot be read must never read as "nothing to check"). */
#ifndef SEABRAID_PICKS_GOLDEN
#error "SEABRAID_PICKS_GOLDEN must name src/picks/golden.txt (CMake passes it)"
#endif

enum { LINE_MAX_BYTES = 16384 };

static const char *kind_text(const char *kind) {
    if (strcmp(kind, "range") == 0) return RANGE;
    if (strcmp(kind, "skip_weight") == 0) return SKIPPED;
    if (strcmp(kind, "finite") == 0) return FINITE;
    if (strcmp(kind, "repeat") == 0) return REPEAT;
    return NULL;
}

static float golden_weight(const char *token) {
    return strcmp(token, "-0") == 0 ? NEG_ZERO : strtof(token, NULL);
}

/* Splits `line` at " | " into `fields` (five), in place; false when the row has another number of fields. */
static bool split_fields(char *line, char *fields[5]) {
    int count = 0;
    char *at = line;
    while (count < 5) {
        fields[count++] = at;
        char *bar = strstr(at, " | ");
        if (!bar) {
            break;
        }
        *bar = '\0';
        at = bar + 3;
    }
    return count == 5 && !strstr(at, " | ");
}

static void apply_overrides(char *spec, float *weights) {
    for (char *entry = strtok(spec, " "); entry; entry = strtok(NULL, " ")) {
        if (strcmp(entry, "-") != 0) {
            char *equals = strchr(entry, '=');
            CHECK(equals != NULL);
            if (equals) {
                *equals = '\0';
                weights[strtoul(entry, NULL, 10)] = golden_weight(equals + 1);
            }
        }
    }
}

/* NULL for "ok", else the adapter's text for the verdict's kind (the pick index after '@' is not this layer's). */
static const char *want_text(char *verdict) {
    if (strcmp(verdict, "ok") == 0) {
        return NULL;
    }
    char *at = strchr(verdict, '@');
    if (at) {
        *at = '\0';
    }
    const char *text = kind_text(verdict);
    return text ? text : "(a verdict kind this test does not know)";
}

/* The ids of a row of the table: every token of `list`, in a heap array of `*count` values (NULL on failure). */
static int32_t *parse_ids(char *list, size_t *count) {
    size_t capacity = 128;
    int32_t *ids = malloc(capacity * sizeof(*ids));
    *count = 0;
    for (char *token = strtok(list, " "); ids && token; token = strtok(NULL, " ")) {
        if (*count == capacity) {
            int32_t *grown = realloc(ids, 2 * capacity * sizeof(*ids));
            if (!grown) {
                free(ids);
                return NULL;
            }
            ids = grown;
            capacity *= 2;
        }
        ids[(*count)++] = (int32_t) strtol(token, NULL, 10);
    }
    return ids;
}

static bool holds_skip(const int32_t *ids, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] == SEABRAID_SKIP_EXPERT_ID) {
            return true;
        }
    }
    return false;
}

/* One `case` row, parsed. `ids` and `weights` are heap arrays of `count` values (NULL until loaded). */
struct golden_row {
    unsigned experts, partial, top_k;
    size_t count;
    int32_t *ids;
    float *weights;
};

/* Reads a row into `row`; false, after failing the test, when it is malformed or its storage cannot be had. The
 * caller frees `row` either way. */
static bool load_row(char *fields[5], struct golden_row *row) {
    memset(row, 0, sizeof(*row));
    const bool shaped = sscanf(fields[1], "%u %u %u", &row->experts, &row->partial, &row->top_k) == 3 && row->top_k > 0;
    CHECK(shaped);
    if (!shaped) {
        return false;
    }
    row->ids = parse_ids(fields[2], &row->count);
    row->weights = malloc((row->count ? row->count : 1) * sizeof(*row->weights));
    const bool whole = row->ids && row->weights && row->count % row->top_k == 0;
    CHECK(whole);
    if (!whole) {
        return false;
    }
    for (size_t i = 0; i < row->count; ++i) {
        row->weights[i] = 0.5f;
    }
    apply_overrides(fields[3], row->weights);
    return true;
}

/* The verdict kind the row names must be the text the adapter returns for it. */
static void compare_row(const struct golden_row *row, char *name, char *verdict) {
    const uint32_t tokens = (uint32_t) (row->count / row->top_k);
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, 32, row->experts, tokens, row->top_k};
    const seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, tokens, NULL, 0, row->ids, row->count, row->weights, row->count};
    const char *got = seabraid_ggml_routes_fault(&config, &batch);
    const char *want = want_text(verdict);
    if (want ? (!got || strcmp(got, want) != 0) : got != NULL) {
        fprintf(stderr, "FAIL golden row %s: expected \"%s\", got \"%s\"\n", name, want ? want : "(accepted)",
                got ? got : "(accepted)");
        ++failures;
    }
}

/* One `case` row: 1 when it was held to the adapter, 0 when it describes a provider without PARTIAL_ROUTES AND uses
 * a skip (this adapter accepts skips, so the row's refusal is not its rule) or could not be read. */
static int golden_case(char *fields[5]) {
    struct golden_row row;
    const bool loaded = load_row(fields, &row);
    const bool held = loaded && (row.partial || !holds_skip(row.ids, row.count));
    if (held) {
        compare_row(&row, fields[0], fields[4]);
    }
    free(row.weights);
    free(row.ids);
    return held;
}

static int golden_limit(const char *line, const char *name) {
    const size_t length = strlen(name);
    return strncmp(line, "limit ", 6) == 0 && strncmp(line + 6, name, length) == 0 && line[6 + length] == ' '
               ? atoi(line + 7 + length)
               : -1;
}

static void golden_table(void) {
    FILE *table = fopen(SEABRAID_PICKS_GOLDEN, "r");
    CHECK(table != NULL);
    if (!table) {
        return;
    }
    static char line[LINE_MAX_BYTES];
    int rows = 0, held = 0, narrow = -1, picks = -1;
    while (fgets(line, sizeof(line), table)) {
        line[strcspn(line, "\r\n")] = '\0';
        CHECK(strlen(line) < sizeof(line) - 1);
        char *fields[5];
        if (golden_limit(line, "narrow_row") >= 0) narrow = golden_limit(line, "narrow_row");
        if (golden_limit(line, "max_picks") >= 0) picks = golden_limit(line, "max_picks");
        if (strncmp(line, "case ", 5) == 0) {
            const bool shaped = split_fields(line, fields);
            CHECK(shaped);
            rows += shaped;
            held += shaped ? golden_case(fields) : 0;
        }
    }
    fclose(table);
    CHECK(narrow == SEABRAID_GGML_NARROW_ROW && picks == SEABRAID_GGML_MAX_PICKS);
    CHECK(rows >= 30 && held >= 30); /* a table that lost its rows must not pass as vacuous */
    printf("golden pick table: %d rows, %d held to the adapter\n", rows, held);
}

#endif
