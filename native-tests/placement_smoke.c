// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Build and run as C11 and C++17. See docs/PLACEMENT-ABI.md. */
#include "seabraid/placement.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "line %d: %s failed\n", __LINE__, #expression); \
        abort(); \
    } \
} while (0)

enum { SLOT_BYTES = 4 };

struct fixture {
    uint8_t bytes[2][SLOT_BYTES];
    uint32_t resident[2];
    unsigned stages;
    unsigned writes;
    int short_stage;
    int32_t next_lower;
};

#ifdef __cplusplus
extern "C" {
#endif

static int32_t stage(void *context, uint8_t rung, uint32_t layer, uint32_t expert,
                     uint8_t *output, size_t size, size_t *written)
{
    struct fixture *state = (struct fixture *)context;
    CHECK(rung < 2 && layer == 0 && size == SLOT_BYTES);
    CHECK(state->resident[rung] == expert);
    state->stages++;
    *written = size - (state->short_stage ? 1u : 0u);
    memcpy(output, state->bytes[rung], *written);
    return SEABRAID_PLACEMENT_IO_ACK;
}

static int32_t replace(void *context, uint8_t rung, uint32_t layer, uint32_t old_expert,
                       uint32_t new_expert, const uint8_t *input, size_t size)
{
    struct fixture *state = (struct fixture *)context;
    CHECK(rung < 2 && layer == 0 && new_expert < 2 && size == SLOT_BYTES);
    CHECK(state->resident[rung] == old_expert);
    CHECK(state->stages >= 2);
    CHECK(input[0] == (new_expert == 0 ? 10u : 30u));
    state->writes++;
    if (rung == 1 && state->next_lower != SEABRAID_PLACEMENT_IO_ACK) {
        int32_t status = state->next_lower;
        state->next_lower = SEABRAID_PLACEMENT_IO_ACK;
        return status;
    }
    memcpy(state->bytes[rung], input, size);
    state->resident[rung] = new_expert;
    return SEABRAID_PLACEMENT_IO_ACK;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

static uint64_t plan(seabraid_placement *placement, uint64_t epoch)
{
    const double heat[] = {1.0, 100.0};
    seabraid_placement_ticket_v1 ticket = {SEABRAID_ABI_VERSION, sizeof(seabraid_placement_ticket_v1), 0, 0, 0};
    CHECK(seabraid_placement_plan(placement, epoch, heat, 2, &ticket) == SEABRAID_OK);
    CHECK(ticket.ticket != 0 && ticket.epoch == epoch && ticket.promotion_count == 1);
    return ticket.ticket;
}

static void expect_executors(seabraid_placement *placement, uint64_t epoch, uint64_t closed,
                             uint8_t first, uint8_t second)
{
    uint8_t executors[] = {255, 255};
    seabraid_placement_snapshot_v1 snapshot = {SEABRAID_ABI_VERSION, sizeof(seabraid_placement_snapshot_v1), 0, 0, 0, 0, 0};
    CHECK(seabraid_placement_snapshot(placement, executors, 2, &snapshot) == SEABRAID_OK);
    CHECK(snapshot.epoch == epoch && snapshot.quarantined_rungs == closed);
    CHECK(snapshot.layers == 1 && snapshot.experts == 2 && snapshot.fallback_rung == 2);
    CHECK(executors[0] == first && executors[1] == second);
}

/* WHY a helper and not an initialiser: a partial initialiser of this 64-byte struct trips -Wmissing-field-initializers
 * in C++ under -Wextra -Werror, and this file must build clean as both C11 and C++17 (docs/PLACEMENT-ABI.md). */
static seabraid_placement_outcome_v1 outcome_header(void)
{
    seabraid_placement_outcome_v1 outcome;
    memset(&outcome, 0, sizeof outcome);
    outcome.abi_version = SEABRAID_ABI_VERSION;
    outcome.struct_size = sizeof(seabraid_placement_outcome_v1);
    return outcome;
}

static void check_commit(seabraid_placement *placement, seabraid_placement_provider_v1 *provider,
                         struct fixture *state, uint64_t epoch)
{
    seabraid_placement_applied_v1 applied = {SEABRAID_ABI_VERSION, sizeof(seabraid_placement_applied_v1), 0, 0, 0};
    seabraid_placement_outcome_v1 outcome = outcome_header();
    uint64_t ticket = plan(placement, epoch);
    CHECK(seabraid_placement_apply(placement, ticket, provider, &outcome, 1, &applied) == SEABRAID_OK);
    CHECK(applied.epoch == epoch + 1 && applied.outcome_count == 1 && applied.quarantined_rungs == 0);
    CHECK(outcome.kind == SEABRAID_PLACEMENT_COMMITTED && outcome.committed_epoch == epoch + 1);
    CHECK(outcome.layer == 0 && outcome.upper_rung == 0 && outcome.promote == 1 && outcome.displace == 0);
    CHECK(outcome.initial.error == 0 && outcome.restore.error == 0);
    CHECK(state->resident[0] == 1 && state->resident[1] == 0);
    CHECK(state->bytes[0][3] == 33 && state->bytes[1][3] == 13);
    expect_executors(placement, epoch + 1, 0, 1, 0);
    const unsigned writes = state->writes;
    CHECK(seabraid_placement_apply(placement, ticket, provider, &outcome, 1, &applied) == SEABRAID_BUSY_OR_STATE);
    CHECK(state->writes == writes);
}

/* Bad arguments are refused before any provider call and KEEP the ticket: it is returned still valid, so the next
 * check can spend it. */
static uint64_t check_refusals_keep_the_ticket(seabraid_placement *placement,
                                               seabraid_placement_provider_v1 *provider,
                                               struct fixture *state, uint64_t epoch)
{
    seabraid_placement_applied_v1 applied = {SEABRAID_ABI_VERSION, sizeof(seabraid_placement_applied_v1), 0, 0, 0};
    seabraid_placement_outcome_v1 outcome = outcome_header();
    uint64_t ticket = plan(placement, epoch);
    CHECK(seabraid_placement_apply(placement, ticket, provider, &outcome, 0, &applied) == SEABRAID_INVALID);
    /* A struct_size below the v1 size is refused before any provider call. */
    outcome.struct_size = 8;
    CHECK(seabraid_placement_apply(placement, ticket, provider, &outcome, 1, &applied) == SEABRAID_INVALID);
    outcome.struct_size = sizeof outcome;
    CHECK(state->stages == 0 && state->writes == 0);
    const size_t empty_slot[] = {0};
    const size_t *slot_bytes = provider->layer_slot_bytes;
    provider->layer_slot_bytes = empty_slot;
    CHECK(seabraid_placement_apply(placement, ticket, provider, &outcome, 1, &applied) == SEABRAID_INVALID);
    provider->layer_slot_bytes = slot_bytes;
    CHECK(state->stages == 0 && state->writes == 0);
    return ticket;
}

/* A stage callback that writes fewer bytes than asked is a malformed blob: rejected, nothing replaced. */
static void check_short_stage_is_rejected(seabraid_placement *placement,
                                          seabraid_placement_provider_v1 *provider,
                                          struct fixture *state, uint64_t epoch, uint64_t ticket)
{
    seabraid_placement_applied_v1 applied = {SEABRAID_ABI_VERSION, sizeof(seabraid_placement_applied_v1), 0, 0, 0};
    seabraid_placement_outcome_v1 outcome = outcome_header();
    state->short_stage = 1;
    CHECK(seabraid_placement_apply(placement, ticket, provider, &outcome, 1, &applied) == SEABRAID_OK);
    CHECK(outcome.kind == SEABRAID_PLACEMENT_REJECTED && outcome.initial.phase == SEABRAID_PLACEMENT_STAGE);
    CHECK(outcome.initial.error == SEABRAID_PLACEMENT_MALFORMED_BLOB && outcome.initial.rung == 1);
    CHECK(state->stages == 1 && state->writes == 0 && applied.epoch == epoch);
    state->short_stage = 0;
}

/* A refused replace on the lower rung is undone: the old experts are put back and the epoch does not move. */
static void check_refused_replace_is_restored(seabraid_placement *placement,
                                              seabraid_placement_provider_v1 *provider,
                                              struct fixture *state, uint64_t epoch)
{
    seabraid_placement_applied_v1 applied = {SEABRAID_ABI_VERSION, sizeof(seabraid_placement_applied_v1), 0, 0, 0};
    seabraid_placement_outcome_v1 outcome = outcome_header();
    uint64_t ticket = plan(placement, epoch);
    state->next_lower = SEABRAID_PLACEMENT_IO_REFUSED;
    CHECK(seabraid_placement_apply(placement, ticket, provider, &outcome, 1, &applied) == SEABRAID_OK);
    CHECK(outcome.kind == SEABRAID_PLACEMENT_RESTORED && outcome.initial.phase == SEABRAID_PLACEMENT_REPLACE);
    CHECK(outcome.initial.error == SEABRAID_PLACEMENT_REFUSED && outcome.initial.rung == 1);
    CHECK(outcome.restore.error == 0 && state->stages == 3 && state->writes == 3);
    CHECK(state->resident[0] == 0 && state->bytes[0][3] == 13);
    expect_executors(placement, epoch, 0, 0, 1);
}

/* An unacknowledged replace (timeout) quarantines the rung and moves the epoch. */
static void check_timeout_quarantines(seabraid_placement *placement,
                                      seabraid_placement_provider_v1 *provider,
                                      struct fixture *state, uint64_t epoch)
{
    seabraid_placement_applied_v1 applied = {SEABRAID_ABI_VERSION, sizeof(seabraid_placement_applied_v1), 0, 0, 0};
    seabraid_placement_outcome_v1 outcome = outcome_header();
    uint64_t ticket = plan(placement, epoch);
    state->next_lower = SEABRAID_PLACEMENT_IO_TIMEOUT;
    CHECK(seabraid_placement_apply(placement, ticket, provider, &outcome, 1, &applied) == SEABRAID_OK);
    CHECK(outcome.kind == SEABRAID_PLACEMENT_QUARANTINED && outcome.initial.error == SEABRAID_PLACEMENT_UNACKNOWLEDGED);
    CHECK(applied.epoch == epoch + 1 && applied.outcome_count == 1 && applied.quarantined_rungs == 2);
    CHECK(state->stages == 5 && state->writes == 6);
    expect_executors(placement, epoch + 1, 2, 0, 2);
}

static void check_provider_outcomes(seabraid_placement *placement,
                                    seabraid_placement_provider_v1 *provider,
                                    struct fixture *state, uint64_t epoch)
{
    const uint64_t ticket = check_refusals_keep_the_ticket(placement, provider, state, epoch);
    check_short_stage_is_rejected(placement, provider, state, epoch, ticket);
    check_refused_replace_is_restored(placement, provider, state, epoch);
    check_timeout_quarantines(placement, provider, state, epoch);
}

/* Hottest-first spillover: lead card first, then helpers, then the fallback rung. */
static void check_fill(void)
{
    const uint64_t slots[] = {1, 2};
    const double heat[] = {1.0, 5.0, 3.0, 2.0};
    const uint8_t expected[] = {2, 0, 1, 1};
    uint8_t owners[4] = {255, 255, 255, 255};
    seabraid_placement_fill_v1 request = {SEABRAID_ABI_VERSION, 1, 4, 2, slots, 2, heat, 4};
    CHECK(seabraid_placement_fill(&request, owners, 4) == SEABRAID_OK);
    CHECK(memcmp(owners, expected, sizeof expected) == 0);

    /* The result is a valid layout. */
    const seabraid_placement_layout_v1 layout = {SEABRAID_ABI_VERSION, 1, 4, 2, owners, 4};
    seabraid_placement *placement = NULL;
    CHECK(seabraid_placement_open(&layout, NULL, &placement) == SEABRAID_OK && placement != NULL);
    CHECK(seabraid_placement_close(placement) == SEABRAID_OK);

    /* Bad input is reported, and nothing is written. */
    const double cold[] = {1.0, -1.0, 3.0, 2.0};
    uint8_t untouched[4] = {255, 255, 255, 255};
    request.heat = cold;
    CHECK(seabraid_placement_fill(&request, untouched, 4) == SEABRAID_INVALID && untouched[0] == 255);
    request.heat = heat;
    CHECK(seabraid_placement_fill(&request, untouched, 3) == SEABRAID_INVALID);
    request.abi_version = SEABRAID_ABI_VERSION + 1;
    CHECK(seabraid_placement_fill(&request, untouched, 4) == SEABRAID_INVALID);
    CHECK(seabraid_placement_fill(NULL, untouched, 4) == SEABRAID_INVALID);
}


/* Policy v2: a three-hop climb from the fallback (rung 3) to the lead through the fallback boundary. Expert e's bytes
 * are e * 10 + offset; rung r's resident expert is tracked, the fallback serves any expert. */
struct climb_fixture {
    uint32_t resident[3];
    unsigned fallback_stages;
    unsigned fallback_writes;
};

#ifdef __cplusplus
extern "C" {
#endif

static int32_t climb_stage(void *context, uint8_t rung, uint32_t layer, uint32_t expert,
                           uint8_t *output, size_t size, size_t *written)
{
    struct climb_fixture *state = (struct climb_fixture *)context;
    CHECK(rung <= 3 && layer == 0 && size == SLOT_BYTES);
    if (rung == 3) {
        state->fallback_stages++;
    } else {
        CHECK(state->resident[rung] == expert);
    }
    memset(output, (int)(expert * 10u + 1u), size);
    *written = size;
    return SEABRAID_PLACEMENT_IO_ACK;
}

static int32_t climb_replace(void *context, uint8_t rung, uint32_t layer, uint32_t old_expert,
                             uint32_t new_expert, const uint8_t *input, size_t size)
{
    struct climb_fixture *state = (struct climb_fixture *)context;
    CHECK(rung <= 3 && layer == 0 && size == SLOT_BYTES && input[0] == new_expert * 10u + 1u);
    if (rung == 3) {
        state->fallback_writes++;
    } else {
        CHECK(state->resident[rung] == old_expert);
        state->resident[rung] = new_expert;
    }
    return SEABRAID_PLACEMENT_IO_ACK;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

static const uint8_t V2_OWNERS[] = {0, 1, 2, 3};
static const double V2_HEAT[] = {1.0, 2.0, 4.0, 100.0};

static seabraid_placement_policy_v2 climb_policy(void)
{
    seabraid_placement_policy_v2 policy = {SEABRAID_PLACEMENT_POLICY_V2_VERSION, sizeof(seabraid_placement_policy_v2),
                                           12, 0, 1, 3, {0, 0, 0}, 0, 1.25, 1.0};
    return policy;
}

/* Out-of-range fields are refused and leave *out NULL. */
static void check_v2_refusals(void)
{
    const seabraid_placement_layout_v1 layout = {SEABRAID_ABI_VERSION, 1, 4, 3, V2_OWNERS, 4};
    seabraid_placement *placement = NULL;
    seabraid_placement_policy_v2 policy = climb_policy();
    policy.fallback_boundary = 2;
    CHECK(seabraid_placement_open_v2(&layout, &policy, &placement) == SEABRAID_INVALID && placement == NULL);
    policy = climb_policy();
    policy.max_hops_per_round = 0;
    CHECK(seabraid_placement_open_v2(&layout, &policy, &placement) == SEABRAID_INVALID && placement == NULL);
    policy = climb_policy();
    policy.reserved[1] = 1;
    CHECK(seabraid_placement_open_v2(&layout, &policy, &placement) == SEABRAID_INVALID && placement == NULL);
    /* One card cannot enable the fallback boundary: that boundary would be the lead with the fallback. */
    const uint8_t one_card[] = {0, 1};
    const seabraid_placement_layout_v1 small = {SEABRAID_ABI_VERSION, 1, 2, 1, one_card, 2};
    policy = climb_policy();
    CHECK(seabraid_placement_open_v2(&small, &policy, &placement) == SEABRAID_INVALID && placement == NULL);
}

/* v1 behaviour: a NULL policy never swaps with the fallback. */
static void check_v2_null_policy_is_v1(void)
{
    const seabraid_placement_layout_v1 layout = {SEABRAID_ABI_VERSION, 1, 4, 3, V2_OWNERS, 4};
    seabraid_placement *placement = NULL;
    seabraid_placement_ticket_v1 ticket = {SEABRAID_ABI_VERSION, sizeof(seabraid_placement_ticket_v1), 0, 0, 0};
    CHECK(seabraid_placement_open_v2(&layout, NULL, &placement) == SEABRAID_OK && placement != NULL);
    CHECK(seabraid_placement_plan(placement, 1, V2_HEAT, 4, &ticket) == SEABRAID_OK && ticket.promotion_count == 1);
    CHECK(seabraid_placement_close(placement) == SEABRAID_OK);
}

static void check_climb_outcomes(const seabraid_placement_outcome_v1 *outcomes, const seabraid_placement_applied_v1 *applied)
{
    CHECK(applied->outcome_count == 3 && applied->epoch == 4 && applied->quarantined_rungs == 0);
    for (uint32_t i = 0; i < 3; i++) {
        /* Bottom-up: the lower rung of the first hop is the fallback, and the climber is expert 3 each time. */
        CHECK(outcomes[i].kind == SEABRAID_PLACEMENT_COMMITTED && outcomes[i].promote == 3);
        CHECK(outcomes[i].upper_rung == 2 - i && outcomes[i].displace == 2 - i);
    }
}

static void check_v2_climb(void)
{
    const size_t slot_bytes[] = {SLOT_BYTES};
    const seabraid_placement_layout_v1 layout = {SEABRAID_ABI_VERSION, 1, 4, 3, V2_OWNERS, 4};
    struct climb_fixture state = {{0, 1, 2}, 0, 0};
    seabraid_placement_provider_v1 provider = {SEABRAID_ABI_VERSION, &state, slot_bytes, 1, climb_stage, climb_replace};
    const seabraid_placement_policy_v2 policy = climb_policy();
    seabraid_placement *placement = NULL;
    CHECK(seabraid_placement_open_v2(&layout, &policy, &placement) == SEABRAID_OK && placement != NULL);
    seabraid_placement_ticket_v1 ticket = {SEABRAID_ABI_VERSION, sizeof(seabraid_placement_ticket_v1), 0, 0, 0};
    CHECK(seabraid_placement_plan(placement, 1, V2_HEAT, 4, &ticket) == SEABRAID_OK && ticket.promotion_count == 3);
    seabraid_placement_outcome_v1 outcomes[3];
    for (int i = 0; i < 3; i++) {
        outcomes[i] = outcome_header();
    }
    seabraid_placement_applied_v1 applied = {SEABRAID_ABI_VERSION, sizeof(seabraid_placement_applied_v1), 0, 0, 0};
    CHECK(seabraid_placement_apply(placement, ticket.ticket, &provider, outcomes, 3, &applied) == SEABRAID_OK);
    check_climb_outcomes(outcomes, &applied);
    CHECK(state.resident[0] == 3 && state.resident[1] == 0 && state.resident[2] == 1);
    CHECK(state.fallback_stages == 1 && state.fallback_writes == 1);
    CHECK(seabraid_placement_close(placement) == SEABRAID_OK);
}

static void check_open_v2(void)
{
    CHECK(SEABRAID_PLACEMENT_SKIPPED == 5 && SEABRAID_PLACEMENT_POLICY_V2_VERSION == 2);
    check_v2_refusals();
    check_v2_null_policy_is_v1();
    check_v2_climb();
}

#include "placement_smoke_v2.h"

int main(void)
{
    check_fill();
    check_open_v2();
    check_provider_v2();
    struct fixture state = {{{10, 11, 12, 13}, {30, 31, 32, 33}}, {0, 1}, 0, 0, 0, SEABRAID_PLACEMENT_IO_ACK};
    const uint8_t owners[] = {0, 1};
    const size_t slot_bytes[] = {SLOT_BYTES};
    const seabraid_placement_layout_v1 layout = {SEABRAID_ABI_VERSION, 1, 2, 2, owners, 2};
    seabraid_placement_provider_v1 provider = {SEABRAID_ABI_VERSION, &state, slot_bytes, 1, stage, replace};
    seabraid_placement *placement = NULL;
    CHECK(seabraid_placement_open(&layout, NULL, &placement) == SEABRAID_OK && placement != NULL);
    expect_executors(placement, 1, 0, 0, 1);

    const uint64_t stale = plan(placement, 1);
    CHECK(seabraid_placement_quarantine(placement, 1, 1) == SEABRAID_OK);
    CHECK(seabraid_placement_apply(placement, stale, NULL, NULL, 0, NULL) == SEABRAID_BUSY_OR_STATE);
    CHECK(seabraid_placement_update(placement, 1, SEABRAID_PLACEMENT_REBUILD, owners, 2) == SEABRAID_BUSY_OR_STATE);
    CHECK(seabraid_placement_update(placement, 2, SEABRAID_PLACEMENT_ADOPT, owners, 2) == SEABRAID_OK);
    expect_executors(placement, 2, 2, 0, 2);
    CHECK(seabraid_placement_update(placement, 2, SEABRAID_PLACEMENT_REBUILD, owners, 2) == SEABRAID_OK);

    check_provider_outcomes(placement, &provider, &state, 3);
    CHECK(seabraid_placement_update(placement, 4, SEABRAID_PLACEMENT_REBUILD, owners, 2) == SEABRAID_OK);
    check_commit(placement, &provider, &state, 5);
    CHECK(seabraid_placement_close(placement) == SEABRAID_OK);
    puts("Seabraid placement provider smoke passed");
    return 0;
}
