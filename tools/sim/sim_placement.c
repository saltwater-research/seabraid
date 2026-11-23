// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "sim_placement.h"

#include "seabraid/placement.h"
#include "sim_report.h"
#include "sim_util.h"

#include <stdlib.h>
#include <string.h>

enum { SLOT_BYTES = 64 };

/* In-process stand-in for the engine's expert storage: every stage and replace is acknowledged, after checking
 * the geometry the ABI promises (a full slot). Any other behaviour would be an ABI bug, not a simulated fault. */
typedef struct sim_store {
    uint64_t stages, replaces;
    int geometry_errors;
} sim_store;

static int32_t store_stage(void *context, uint8_t rung, uint32_t layer, uint32_t expert, uint8_t *output,
                           size_t output_bytes, size_t *bytes_written)
{
    sim_store *store = (sim_store *)context;
    (void)rung;
    (void)layer;
    store->geometry_errors += output_bytes != SLOT_BYTES;
    memset(output, (int)(expert & 0xff), output_bytes);
    *bytes_written = output_bytes;
    store->stages++;
    return SEABRAID_PLACEMENT_IO_ACK;
}

static int32_t store_replace(void *context, uint8_t rung, uint32_t layer, uint32_t old_expert,
                             uint32_t new_expert, const uint8_t *input, size_t input_bytes)
{
    sim_store *store = (sim_store *)context;
    (void)rung;
    (void)layer;
    (void)old_expert;
    (void)new_expert;
    (void)input;
    store->geometry_errors += input_bytes != SLOT_BYTES;
    store->replaces++;
    return SEABRAID_PLACEMENT_IO_ACK;
}

/* A Zipf-like ramp whose hot spot rotates each round, so the hottest experts of round r are cold in round
 * r + k and the planner has real promotions to make. Seeded noise breaks ties between cells. */
static void drift_heat(double *heat, uint32_t layers, uint32_t experts, uint32_t round, sim_rng *rng)
{
    uint32_t shift = experts / 7 + 1;
    for (uint32_t l = 0; l < layers; l++) {
        for (uint32_t e = 0; e < experts; e++) {
            uint32_t rank = (e + round * shift + l * 3) % experts;
            heat[(size_t)l * experts + e] = 1000.0 / (1.0 + rank) * (1.0 + 0.05 * sim_rng_unit(rng));
        }
    }
}

/* One plan/apply round. Returns 0 on success (zero promotions is a valid plan) and 1 on any unexpected result. */
static int run_round(seabraid_placement *placement, const double *heat, size_t cells, uint64_t *epoch,
                     const seabraid_placement_provider_v1 *provider, sim_placement_result *out)
{
    seabraid_placement_ticket_v1 ticket = {.abi_version = SEABRAID_ABI_VERSION, .struct_size = sizeof ticket};
    seabraid_placement_applied_v1 applied = {.abi_version = SEABRAID_ABI_VERSION, .struct_size = sizeof applied};
    seabraid_placement_outcome_v1 *outcomes;
    int failed;
    if (sim_check(seabraid_placement_plan(placement, *epoch, heat, cells, &ticket), "seabraid_placement_plan")) {
        return 1;
    }
    if (ticket.promotion_count == 0) {
        return 0;
    }
    outcomes = (seabraid_placement_outcome_v1 *)calloc(ticket.promotion_count, sizeof *outcomes);
    if (outcomes == NULL) {
        return 1;
    }
    /* The library steps through the array by element 0's struct_size, so element 0 carries the header. */
    outcomes[0].abi_version = SEABRAID_ABI_VERSION;
    outcomes[0].struct_size = (uint32_t)sizeof *outcomes;
    failed = sim_check(seabraid_placement_apply(placement, ticket.ticket, provider, outcomes,
                                                ticket.promotion_count, &applied),
                       "seabraid_placement_apply");
    for (size_t i = 0; !failed && i < applied.outcome_count; i++) {
        failed = outcomes[i].kind != SEABRAID_PLACEMENT_COMMITTED;
        out->swaps += !failed;
    }
    out->plans++;
    *epoch = failed ? *epoch : applied.epoch;
    free(outcomes);
    return failed;
}

static int snapshot_matches(seabraid_placement *placement, size_t cells, uint64_t epoch)
{
    seabraid_placement_snapshot_v1 snapshot = {.abi_version = SEABRAID_ABI_VERSION, .struct_size = sizeof snapshot};
    uint8_t *executors = (uint8_t *)malloc(cells);
    int bad = executors == NULL ||
              sim_check(seabraid_placement_snapshot(placement, executors, cells, &snapshot),
                        "seabraid_placement_snapshot") ||
              snapshot.epoch != epoch;
    free(executors);
    return bad;
}

static int open_filled(const sim_config *cfg, const double *heat0, uint8_t *owners, seabraid_placement **out)
{
    size_t cells = (size_t)cfg->layers * cfg->experts;
    uint64_t slots[SEABRAID_PLACEMENT_MAX_CARDS];
    seabraid_placement_fill_v1 fill = {SEABRAID_ABI_VERSION, cfg->layers, cfg->experts, (uint8_t)cfg->devices,
                                       slots, cfg->devices, heat0, cells};
    seabraid_placement_layout_v1 layout = {SEABRAID_ABI_VERSION, cfg->layers, cfg->experts,
                                           (uint8_t)cfg->devices, owners, cells};
    /* Half the cells fit on cards, the rest start on the fallback rung, so drift always has promotions. */
    uint64_t per_card = cells / (2 * (size_t)cfg->devices);
    for (uint32_t d = 0; d < cfg->devices; d++) {
        slots[d] = per_card == 0 ? 1 : per_card;
    }
    return sim_check(seabraid_placement_fill(&fill, owners, cells), "seabraid_placement_fill") ||
           sim_check(seabraid_placement_open(&layout, NULL, out), "seabraid_placement_open");
}

const char *sim_placement_skip_reason(const sim_config *cfg)
{
    if (cfg->placement_rounds == 0) {
        return "off";
    }
    if (cfg->devices > SEABRAID_PLACEMENT_MAX_CARDS) {
        return "skipped: devices exceed SEABRAID_PLACEMENT_MAX_CARDS";
    }
    if ((uint64_t)cfg->layers * cfg->experts > SEABRAID_PLACEMENT_MAX_CELLS) {
        return "skipped: layers * experts exceed SEABRAID_PLACEMENT_MAX_CELLS";
    }
    return NULL;
}

static int run_rounds(const sim_config *cfg, double *heat, uint8_t *owners, sim_placement_result *out)
{
    size_t cells = (size_t)cfg->layers * cfg->experts;
    sim_store store = {0, 0, 0};
    seabraid_placement_provider_v1 provider = {SEABRAID_ABI_VERSION, &store, NULL, cfg->layers, store_stage,
                                               store_replace};
    seabraid_placement *placement = NULL;
    size_t *table = (size_t *)malloc(cfg->layers * sizeof *table);
    sim_rng rng;
    uint64_t epoch = 1;
    int failed = table == NULL;
    sim_rng_seed(&rng, cfg->seed, 0x706c61ull);
    for (uint32_t l = 0; !failed && l < cfg->layers; l++) {
        table[l] = SLOT_BYTES;
    }
    provider.layer_slot_bytes = table;
    drift_heat(heat, cfg->layers, cfg->experts, 0, &rng);
    failed = failed || open_filled(cfg, heat, owners, &placement);
    for (uint32_t r = 1; !failed && r <= cfg->placement_rounds; r++) {
        drift_heat(heat, cfg->layers, cfg->experts, r, &rng);
        failed = run_round(placement, heat, cells, &epoch, &provider, out) || snapshot_matches(placement, cells, epoch);
    }
    failed = failed || store.geometry_errors != 0;
    if (placement != NULL) {
        failed |= sim_check(seabraid_placement_close(placement), "seabraid_placement_close");
    }
    free(table);
    return failed;
}

int sim_placement_run(const sim_config *cfg, sim_placement_result *out)
{
    size_t cells = (size_t)cfg->layers * cfg->experts;
    double *heat;
    uint8_t *owners;
    int failed;
    memset(out, 0, sizeof *out);
    out->note = sim_placement_skip_reason(cfg);
    if (out->note != NULL) {
        return 0;
    }
    heat = (double *)malloc(cells * sizeof *heat);
    owners = (uint8_t *)malloc(cells);
    failed = heat == NULL || owners == NULL || run_rounds(cfg, heat, owners, out);
    free(heat);
    free(owners);
    out->note = "ok";
    return failed;
}
