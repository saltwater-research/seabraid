// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Route validation and skipped-pick mapping: owns what an expert id row may hold.
 * FAILURE: fail-fast. A defect is returned as a static reason and the batch is refused; when the scratch for a wide
 * row cannot be allocated the batch is refused too, so an unchecked row never passes as valid. */
#include "_seabraid_ggml_routes.h"

#include <stdlib.h>
#include <string.h>

/* IEEE-754 binary32 with an all-ones exponent is Inf or NaN. WHY a bit test and not isfinite(): a host built
 * with -ffast-math may fold isfinite() to true, and this check exists to refuse exactly those values. */
static bool weight_is_finite(float weight) {
    uint32_t bits;
    memcpy(&bits, &weight, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

/* IEEE-754 binary32 zero of either sign: every bit but the sign is clear. WHY a bit test and not
 * `weight == 0.0f`: under -ffast-math (finite-math-only) the compiler may drop the unordered case of the
 * compare, so a NaN weight compares equal to zero and a skipped pick carrying NaN would be accepted. */
static bool weight_is_zero(float weight) {
    uint32_t bits;
    memcpy(&bits, &weight, sizeof(bits));
    return (bits << 1) == 0;
}

/* Rows wider than this are checked for repeats by sorting. WHY: scanning the earlier picks of a row is
 * O(top_k^2) (src/picks.rs measured 12.7 s for one 64,000-pick row) and the ABI bounds top_k only by the
 * expert count. The same width as NARROW_ROW there, so both layers change strategy together: the value lives in
 * _seabraid_ggml_routes.h and src/picks/golden.txt holds both to one number (test_routes.c checks it). */
#define NARROW_ROW ((size_t) SEABRAID_GGML_NARROW_ROW)

/* NULL when pick `pick` of the row is acceptable, else the reason; `earlier` is how many picks before it are
 * scanned for a repeat (0 on a wide row, whose repeats wide_row_repeat finds). WHY a skip
 * must carry exactly zero weight: the ABI lets another device own the pick, and
 * a nonzero weight would silently drop real work instead of refusing it.
 * WHY range, then finite, then repeat: that is the order src/picks.rs pick_error refuses in, so both layers
 * name the same defect when one pick has several (src/picks/golden.txt holds this file to that order).
 * WHY uniqueness and finiteness are checked here although the runtime refuses both on the session path
 * (src/picks.rs pick_error): seabraid_ggml_execute is also the engine-facing direct entry and bypasses the
 * runtime, CUDA's sorted gather can only map an expert once per token, and a NaN weight would otherwise
 * come back as a silently poisoned row. On the session path these checks never fire. */
static const char *pick_fault(const seabraid_config_v1 *config, const int32_t *row, const float *weights,
                              size_t pick, size_t earlier) {
    if (row[pick] == SEABRAID_SKIP_EXPERT_ID) {
        return weight_is_zero(weights[pick]) ? NULL : "a skipped pick has a nonzero weight";
    }
    if (row[pick] < 0 || (uint32_t) row[pick] >= config->experts) {
        return "routes hold an out-of-range expert id";
    }
    if (!weight_is_finite(weights[pick])) {
        return "routes hold a non-finite weight";
    }
    for (size_t before = 0; before < earlier; ++before) {
        if (row[pick] == row[before]) {
            return "routes hold a repeated expert id";
        }
    }
    return NULL;
}

static bool has_skips(const seabraid_batch_v1 *batch) {
    for (size_t i = 0; i < batch->expert_id_count; ++i) {
        if (batch->expert_ids[i] == SEABRAID_SKIP_EXPERT_ID) {
            return true;
        }
    }
    return false;
}

struct id_at {
    int32_t id;
    uint32_t at;
};

static int by_id_then_position(const void *left, const void *right) {
    const struct id_at *a = left;
    const struct id_at *b = right;
    if (a->id != b->id) {
        return a->id < b->id ? -1 : 1;
    }
    return a->at < b->at ? -1 : a->at > b->at;
}

/* Position of the earliest pick of the row that repeats an earlier real id, or `top_k` when none: sort the ids
 * with their positions, and every neighbour with an equal id is a repeat at its own (later) position. `scratch`
 * holds top_k entries. Skips are never repeats. */
static size_t wide_row_repeat(const int32_t *row, size_t top_k, struct id_at *scratch) {
    size_t count = 0;
    for (size_t at = 0; at < top_k; ++at) {
        if (row[at] != SEABRAID_SKIP_EXPERT_ID) {
            scratch[count].id = row[at];
            scratch[count++].at = (uint32_t) at;
        }
    }
    qsort(scratch, count, sizeof(scratch[0]), by_id_then_position);
    size_t earliest = top_k;
    for (size_t i = 1; i < count; ++i) {
        if (scratch[i].id == scratch[i - 1].id && scratch[i].at < earliest) {
            earliest = scratch[i].at;
        }
    }
    return earliest;
}

/* The reason for the first defect of one row in pick order (the lowest index wins; at one pick, pick_fault's
 * order). A narrow row scans. A wide row finds its first pick defect and its first repeat separately and the
 * lower position wins, a tie going to the pick defect, exactly as src/picks.rs check does. */
static const char *row_fault(const seabraid_config_v1 *config, const int32_t *row, const float *weights,
                             struct id_at *scratch) {
    static const char repeat[] = "routes hold a repeated expert id";
    const size_t top_k = config->top_k;
    const bool wide = top_k > NARROW_ROW;
    for (size_t pick = 0; pick < top_k; ++pick) {
        const char *fault = pick_fault(config, row, weights, pick, wide ? 0 : pick);
        if (fault) {
            return wide && wide_row_repeat(row, top_k, scratch) < pick ? repeat : fault;
        }
    }
    return wide && wide_row_repeat(row, top_k, scratch) < top_k ? repeat : NULL;
}

const char *seabraid_ggml_routes_fault(const seabraid_config_v1 *config, const seabraid_batch_v1 *batch) {
    /* WHY: a skipped pick needs a distinct filler expert, so top-k must fit. */
    if (has_skips(batch) && config->top_k > config->experts) {
        return "skipped picks need top-k no larger than the expert count";
    }
    struct id_at *scratch = NULL;
    if (config->top_k > NARROW_ROW) {
        /* WHY refuse when the scratch cannot be had: a row that was not checked must never pass as valid. */
        scratch = malloc(config->top_k * sizeof(*scratch));
        if (!scratch) {
            return "cannot allocate to check routes";
        }
    }
    const char *fault = NULL;
    for (size_t token = 0; !fault && token < batch->tokens; ++token) {
        const size_t first = token * config->top_k;
        fault = row_fault(config, batch->expert_ids + first, batch->weights + first, scratch);
    }
    free(scratch);
    return fault;
}

static bool row_holds(const int32_t *row, size_t count, int32_t expert) {
    for (size_t i = 0; i < count; ++i) {
        if (row[i] == expert) {
            return true;
        }
    }
    return false;
}

/* WHY a filler expert instead of a mask in the graph: MUL_MAT_ID gathers by
 * id, so every id must be a real expert, and CUDA needs them distinct per
 * token; the pick's weight is exactly zero so its product adds nothing. */
static void fill_skipped_row(int32_t *mapped, const int32_t *row, size_t top_k) {
    int32_t candidate = 0;
    for (size_t pick = 0; pick < top_k; ++pick) {
        if (row[pick] != SEABRAID_SKIP_EXPERT_ID) {
            mapped[pick] = row[pick];
            continue;
        }
        while (row_holds(row, top_k, candidate)) {
            ++candidate;
        }
        mapped[pick] = candidate++;
    }
}

const int32_t *seabraid_ggml_graph_ids(const seabraid_config_v1 *config, const seabraid_batch_v1 *batch,
                                       int32_t **owned) {
    *owned = NULL;
    if (!has_skips(batch)) {
        return batch->expert_ids;
    }
    *owned = malloc(batch->expert_id_count * sizeof(int32_t));
    for (size_t token = 0; *owned && token < batch->tokens; ++token) {
        fill_skipped_row(*owned + token * config->top_k, batch->expert_ids + token * config->top_k,
                         config->top_k);
    }
    return *owned;
}
