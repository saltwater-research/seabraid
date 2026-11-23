// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "sim_report.h"

#include "sim_budget.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

static atomic_uint_fast64_t g_records;
static atomic_uint_fast64_t g_provider_records;

int sim_check(int32_t status, const char *call)
{
    char reason[256];
    if (status == SEABRAID_OK) {
        return 0;
    }
    seabraid_last_error(reason, sizeof reason, NULL);
    fprintf(stderr, "%s failed (status %d): %s\n", call, (int)status, reason);
    return 1;
}

/* May run on any worker thread, so it only bumps atomics. */
static void count_record(void *context, uint32_t level, int32_t status, const char *message)
{
    (void)context;
    (void)level;
    (void)message;
    atomic_fetch_add(&g_records, 1);
    if (status == SEABRAID_PROVIDER) {
        atomic_fetch_add(&g_provider_records, 1);
    }
}

int sim_log_install(void)
{
    /* DEBUG so no record is filtered: the simulator's check is that every injected failure produced one. */
    return sim_check(seabraid_set_log_sink(count_record, NULL, SEABRAID_LOG_DEBUG), "seabraid_set_log_sink");
}

void sim_log_totals(uint64_t *records, uint64_t *provider_records)
{
    *records = atomic_load(&g_records);
    *provider_records = atomic_load(&g_provider_records);
}

static int compare_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* Nearest-rank percentile of a sorted sample, in microseconds. */
static double percentile_us(const uint64_t *sorted, size_t count, double p)
{
    size_t rank = (size_t)(p * (double)count + 0.999999999);
    rank = rank == 0 ? 1 : rank;
    return (double)sorted[rank - 1] / 1000.0;
}

void sim_summarize_latency(uint64_t *ns, size_t count, sim_result *out)
{
    double total = 0.0;
    size_t i;
    if (count == 0) {
        return;
    }
    qsort(ns, count, sizeof *ns, compare_u64);
    for (i = 0; i < count; i++) {
        total += (double)ns[i];
    }
    out->p50_us = percentile_us(ns, count, 0.50);
    out->p95_us = percentile_us(ns, count, 0.95);
    out->p99_us = percentile_us(ns, count, 0.99);
    out->max_us = (double)ns[count - 1] / 1000.0;
    out->layers_per_s = total > 0.0 ? (double)count / (total / 1e9) : 0.0;
}

void sim_print_json(const sim_config *c, const sim_result *r)
{
    printf("{\"ok\":%s,\"hidden\":%u,\"experts\":%u,\"top_k\":%u,\"tokens\":%u,\"devices\":%u,\"layers\":%u,"
           "\"steps\":%u,\"warmup\":%u,\"kernel\":\"%s\",\"fail_rate\":%g,\"placement_rounds\":%u,\"seed\":%llu,"
           "\"compact_rows\":%u,\"placement_epoch\":%llu,\"estimated_bytes\":%.0f,",
           r->ok ? "true" : "false", c->hidden, c->experts, c->top_k, c->tokens, c->devices, c->layers, c->steps,
           c->warmup, c->kernel, c->fail_rate, c->placement_rounds, (unsigned long long)c->seed,
           c->compact_rows, (unsigned long long)c->placement_epoch, sim_estimate_bytes(c));
    printf("\"p50_us\":%.2f,\"p95_us\":%.2f,\"p99_us\":%.2f,\"max_us\":%.2f,\"layers_per_s\":%.1f,"
           "\"rows_per_s\":%.1f,\"max_abs_err\":%.3g,\"max_scaled_err\":%.3g,\"verified_layers\":%llu,",
           r->p50_us, r->p95_us, r->p99_us, r->max_us, r->layers_per_s, r->rows_per_s, r->max_abs_err, r->max_scaled_err,
           (unsigned long long)r->verified);
    printf("\"injected_failures\":%llu,\"provider_failures_observed\":%llu,\"provider_failures_status\":%llu,"
           "\"log_records\":%llu,\"log_provider_records\":%llu,\"rows_submitted\":%llu,\"rows_shipped\":%llu,"
           "\"swaps_applied\":%llu,\"placement\":\"%s\"}\n",
           (unsigned long long)r->injected, (unsigned long long)r->failures_observed,
           (unsigned long long)r->failures_status, (unsigned long long)r->log_records,
           (unsigned long long)r->log_provider_records, (unsigned long long)r->rows_submitted,
           (unsigned long long)r->rows_shipped, (unsigned long long)r->swaps, r->placement);
}
