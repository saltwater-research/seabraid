// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Provider v2 (seabraid_placement_apply_v2): the by-reference write. Included once by placement_smoke.c, after its
 * CHECK macro and fixtures, so the same text builds as C11 and as C++17 (docs/PLACEMENT-ABI.md).
 * The fixture is the two-card layout of the main smoke: heat {1, 100} promotes expert 1 (rung 1) over expert 0 (rung 0).
 * Expert e's pack bytes are 10 + 20 * e. calls records one letter per provider call: s stage, r reference, w replace,
 * each followed by its rung. */
#ifndef PLACEMENT_SMOKE_V2_H
#define PLACEMENT_SMOKE_V2_H

enum { V2_SLOT = 4, V2_MAX_CALLS = 16 };

struct ref_fixture {
    uint8_t bytes[2][V2_SLOT];
    uint32_t resident[2];
    char calls[V2_MAX_CALLS];
    unsigned count;
    int32_t ref_code;
};

#ifdef __cplusplus
extern "C" {
#endif

static void note(struct ref_fixture *state, char kind, uint8_t rung)
{
    CHECK(state->count + 2 <= V2_MAX_CALLS);
    state->calls[state->count++] = kind;
    state->calls[state->count++] = (char)('0' + rung);
    state->calls[state->count] = '\0';
}

static int32_t ref_stage(void *context, uint8_t rung, uint32_t layer, uint32_t expert,
                         uint8_t *output, size_t size, size_t *written)
{
    struct ref_fixture *state = (struct ref_fixture *)context;
    CHECK(rung < 2 && layer == 0 && size == V2_SLOT && state->resident[rung] == expert);
    note(state, 's', rung);
    memcpy(output, state->bytes[rung], size);
    *written = size;
    return SEABRAID_PLACEMENT_IO_ACK;
}

static int32_t ref_replace(void *context, uint8_t rung, uint32_t layer, uint32_t old_expert,
                           uint32_t new_expert, const uint8_t *input, size_t size)
{
    struct ref_fixture *state = (struct ref_fixture *)context;
    CHECK(rung < 2 && layer == 0 && size == V2_SLOT && state->resident[rung] == old_expert);
    note(state, 'w', rung);
    memcpy(state->bytes[rung], input, size);
    state->resident[rung] = new_expert;
    return SEABRAID_PLACEMENT_IO_ACK;
}

static int32_t ref_by_reference(void *context, uint8_t rung, uint32_t layer, uint32_t old_expert,
                                uint32_t new_expert)
{
    struct ref_fixture *state = (struct ref_fixture *)context;
    CHECK(rung < 2 && layer == 0 && state->resident[rung] == old_expert);
    note(state, 'r', rung);
    if (state->ref_code == SEABRAID_PLACEMENT_IO_ACK) {
        memset(state->bytes[rung], (int)(10u + 20u * new_expert), V2_SLOT); /* the rung read the pack itself */
        state->resident[rung] = new_expert;
    } else if (state->ref_code == SEABRAID_PLACEMENT_IO_FAILED) {
        state->bytes[rung][0] ^= 0xFFu; /* began loading and died: the slot is torn */
    }
    return state->ref_code;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

struct wide_provider_v2 {
    seabraid_placement_provider_v2 v2;
    uint8_t tail[16];
};

static void ref_reset(struct ref_fixture *state, int32_t ref_code)
{
    memset(state, 0, sizeof *state);
    memset(state->bytes[0], 10, V2_SLOT);
    memset(state->bytes[1], 30, V2_SLOT);
    state->resident[1] = 1;
    state->ref_code = ref_code;
}

static seabraid_placement_provider_v2 v2_provider(struct ref_fixture *state, const size_t *slots, uint64_t mask)
{
    seabraid_placement_provider_v2 provider;
    memset(&provider, 0, sizeof provider);
    provider.abi_version = SEABRAID_PLACEMENT_PROVIDER_V2_VERSION;
    provider.struct_size = sizeof provider;
    provider.context = state;
    provider.layer_slot_bytes = slots;
    provider.layer_count = 1;
    provider.stage = ref_stage;
    provider.replace = ref_replace;
    provider.from_fallback_rungs = mask;
    provider.replace_from_fallback = ref_by_reference;
    return provider;
}

/* Opens a fresh two-card placement and plans the one promotion; returns the ticket. */
static uint64_t v2_open_and_plan(seabraid_placement **placement)
{
    static const uint8_t owners[] = {0, 1};
    const seabraid_placement_layout_v1 layout = {SEABRAID_ABI_VERSION, 1, 2, 2, owners, 2};
    CHECK(seabraid_placement_open(&layout, NULL, placement) == SEABRAID_OK && *placement != NULL);
    return plan(*placement, 1);
}

static int32_t v2_apply(seabraid_placement *placement, uint64_t ticket, const void *provider,
                        seabraid_placement_outcome_v1 *outcome, seabraid_placement_applied_v1 *applied)
{
    *outcome = outcome_header();
    return seabraid_placement_apply_v2(placement, ticket, (const seabraid_placement_provider_v2 *)provider, outcome, 1,
                                       applied);
}

/* The rows of docs/PLACEMENT-ABI.md, "By-reference moves", as seen from C. */
static void check_provider_v2_outcomes(const size_t *slots)
{
    const struct {
        int32_t code;
        const char *calls;
        uint32_t kind;
        uint64_t epoch;
        uint64_t closed;
    } rows[] = {
        {SEABRAID_PLACEMENT_IO_ACK, "s0r0w1", SEABRAID_PLACEMENT_COMMITTED, 2, 0},
        {SEABRAID_PLACEMENT_IO_REFUSED, "s0r0s1w0w1", SEABRAID_PLACEMENT_COMMITTED, 2, 0},
        {SEABRAID_PLACEMENT_IO_FAILED, "s0r0w0", SEABRAID_PLACEMENT_RESTORED, 1, 0},
        {SEABRAID_PLACEMENT_IO_TIMEOUT, "s0r0", SEABRAID_PLACEMENT_QUARANTINED, 2, 1},
    };
    for (unsigned i = 0; i < sizeof rows / sizeof rows[0]; i++) {
        struct ref_fixture state;
        seabraid_placement *placement = NULL;
        seabraid_placement_outcome_v1 outcome;
        seabraid_placement_applied_v1 applied = {SEABRAID_ABI_VERSION, sizeof(seabraid_placement_applied_v1), 0, 0, 0};
        ref_reset(&state, rows[i].code);
        const seabraid_placement_provider_v2 provider = v2_provider(&state, slots, 1u << 0);
        const uint64_t ticket = v2_open_and_plan(&placement);
        CHECK(v2_apply(placement, ticket, &provider, &outcome, &applied) == SEABRAID_OK);
        CHECK(strcmp(state.calls, rows[i].calls) == 0);
        CHECK(outcome.kind == rows[i].kind && applied.epoch == rows[i].epoch && applied.quarantined_rungs == rows[i].closed);
        if (rows[i].kind == SEABRAID_PLACEMENT_COMMITTED) {
            CHECK(outcome.committed_epoch == 2 && state.resident[0] == 1 && state.resident[1] == 0);
            CHECK(state.bytes[0][3] == 30 && state.bytes[1][3] == 10);
        }
        if (rows[i].kind == SEABRAID_PLACEMENT_RESTORED) {
            CHECK(state.resident[0] == 0 && state.bytes[0][0] == 10); /* the torn slot was rewritten */
        }
        CHECK(seabraid_placement_close(placement) == SEABRAID_OK);
    }
}

/* Every refusal is SEABRAID_INVALID before any provider call and keeps the ticket. */
static void check_provider_v2_refusals(const size_t *slots)
{
    struct ref_fixture state;
    seabraid_placement *placement = NULL;
    seabraid_placement_outcome_v1 outcome;
    seabraid_placement_applied_v1 applied = {SEABRAID_ABI_VERSION, sizeof(seabraid_placement_applied_v1), 0, 0, 0};
    struct wide_provider_v2 wide;
    ref_reset(&state, SEABRAID_PLACEMENT_IO_ACK);
    const uint64_t ticket = v2_open_and_plan(&placement);

    seabraid_placement_provider_v2 bad = v2_provider(&state, slots, 1u << 2); /* the fallback rung is masked */
    CHECK(v2_apply(placement, ticket, &bad, &outcome, &applied) == SEABRAID_INVALID);
    bad.from_fallback_rungs = 1u << 0;
    bad.replace_from_fallback = NULL; /* a mask without its callback */
    CHECK(v2_apply(placement, ticket, &bad, &outcome, &applied) == SEABRAID_INVALID);
    bad = v2_provider(&state, slots, 0);
    bad.struct_size = 56; /* below the v2 size */
    CHECK(v2_apply(placement, ticket, &bad, &outcome, &applied) == SEABRAID_INVALID);
    bad = v2_provider(&state, slots, 0);
    bad.abi_version = SEABRAID_ABI_VERSION; /* the v1 version number is not a v2 struct */
    CHECK(v2_apply(placement, ticket, &bad, &outcome, &applied) == SEABRAID_INVALID);
    CHECK(v2_apply(placement, ticket, NULL, &outcome, &applied) == SEABRAID_INVALID);

    /* A caller built against a newer library: a zero tail is fine, a non-zero byte is not. */
    memset(&wide, 0, sizeof wide);
    wide.v2 = v2_provider(&state, slots, 1u << 0);
    wide.v2.struct_size = sizeof wide;
    wide.tail[15] = 1;
    CHECK(v2_apply(placement, ticket, &wide, &outcome, &applied) == SEABRAID_INVALID);
    CHECK(state.count == 0);

    wide.tail[15] = 0;
    CHECK(v2_apply(placement, ticket, &wide, &outcome, &applied) == SEABRAID_OK); /* the ticket was kept */
    CHECK(strcmp(state.calls, "s0r0w1") == 0 && outcome.kind == SEABRAID_PLACEMENT_COMMITTED);
    CHECK(seabraid_placement_close(placement) == SEABRAID_OK);
}

static void check_provider_v2(void)
{
    static const size_t slots[] = {V2_SLOT};
    CHECK(SEABRAID_PLACEMENT_PROVIDER_V2_VERSION == 2 && sizeof(seabraid_placement_provider_v2) == 64);
    check_provider_v2_outcomes(slots);
    check_provider_v2_refusals(slots);
}

#endif /* PLACEMENT_SMOKE_V2_H */
