// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* seabraid_sim: drives the public C ABI the way an engine would, with synthetic layers, and reports one JSON line.
 *
 * Shape of a run: --devices N simulated devices, each a seabraid_session_open session whose v3 provider owns a
 * disjoint slice of the experts and accepts partial routes (--compact-rows also lets the library drop rows a device
 * has no pick in; --placement-epoch stamps every batch). For every layer the "engine" masks the picks a device
 * does not own, submits the same rows to all N sessions, collects them, sums the partial outputs and compares
 * the sum to a single-device double-precision reference. Latency is submit-of-first to collect-of-last per layer.
 *
 * Exit codes: 0 ok, 1 unexpected ABI status or setup failure (including out of memory), 2 numeric mismatch (scaled
 * error past its float32 rounding bound or not finite, see sim_scaled_error) or nothing verified, 3 bookkeeping mismatch:
 * failure counts between injected / collected / session status / log records, or row counts (rows_submitted must
 * equal tokens per submit summed; rows_shipped must equal it unless --compact-rows, and never exceed it).
 *
 * The runtime never imports this directory; it links only the installed headers and libseabraid. */
#include "seabraid/seabraid.h"
#include "sim_cli.h"
#include "sim_placement.h"
#include "sim_providers.h"
#include "sim_report.h"
#include "sim_workload.h"

#include <stdio.h>
#include <stdlib.h>

#define SIM_COLLECT_TIMEOUT_MS 60000u

typedef struct sim_run {
    sim_config cfg;
    sim_workload *workload;
    sim_device *devices;
    seabraid_provider_v3 *providers;
    seabraid_session **sessions;
    uint64_t *tickets;
    uint64_t *latency_ns;
    size_t latency_count;
    int counts_disagree;
    sim_result result;
} sim_run;

static void run_free(sim_run *run)
{
    free(run->devices);
    free(run->providers);
    free(run->sessions);
    free(run->tickets);
    free(run->latency_ns);
    sim_workload_free(run->workload);
}

/* WHY a message here: a failed allocation used to exit 1 with nothing on stderr. The latency table is layers * steps
 * entries, checked for overflow like every other size product (sim_calloc2). */
static int run_alloc(sim_run *run)
{
    uint32_t n = run->cfg.devices;
    run->workload = sim_workload_new(&run->cfg);
    run->devices = (sim_device *)calloc(n, sizeof *run->devices);
    run->providers = (seabraid_provider_v3 *)calloc(n, sizeof *run->providers);
    run->sessions = (seabraid_session **)calloc(n, sizeof *run->sessions);
    run->tickets = (uint64_t *)calloc(n, sizeof *run->tickets);
    run->latency_ns = (uint64_t *)sim_calloc2(run->cfg.layers, run->cfg.steps, sizeof *run->latency_ns);
    if (run->workload && run->devices && run->providers && run->sessions && run->tickets && run->latency_ns) {
        return 1;
    }
    fprintf(stderr, "seabraid_sim: out of memory for --hidden %u --experts %u --top-k %u --tokens %u --devices %u "
                    "--layers %u --steps %u (or a size past what this host can address); lower them\n",
            run->cfg.hidden, run->cfg.experts, run->cfg.top_k, run->cfg.tokens, run->cfg.devices, run->cfg.layers,
            run->cfg.steps);
    return 0;
}

static int open_sessions(sim_run *run)
{
    const sim_config *c = &run->cfg;
    seabraid_config_v1 config = {SEABRAID_ABI_VERSION, c->hidden, c->experts, c->tokens, c->top_k};
    uint32_t flags = SEABRAID_PROVIDER_PARTIAL_ROUTES | (c->compact_rows ? SEABRAID_PROVIDER_COMPACT_ROWS : 0u);
    for (uint32_t d = 0; d < c->devices; d++) {
        if (sim_device_init(&run->devices[d], c, d)) {
            fprintf(stderr, "unknown kernel %s\n", c->kernel);
            return 1;
        }
        /* One operation per layer: docs/ABI.md maps an operation id to a model layer. */
        run->providers[d] = (seabraid_provider_v3){SEABRAID_PROVIDER_V3_VERSION, sizeof(seabraid_provider_v3),
                                                   c->layers, flags, &config, &run->devices[d],
                                                   sim_device_execute};
        if (sim_check(seabraid_session_open(&run->providers[d], &run->sessions[d]), "seabraid_session_open")) {
            return 1;
        }
    }
    return 0;
}

static int submit_all(sim_run *run, uint32_t layer)
{
    for (uint32_t d = 0; d < run->cfg.devices; d++) {
        seabraid_batch_v2 batch;
        sim_workload_batch(run->workload, d, &batch);
        batch.operation_id = layer;
        if (sim_check(seabraid_session_submit(run->sessions[d], &batch, &run->tickets[d]),
                      "seabraid_session_submit")) {
            return 1;
        }
    }
    return 0;
}

/* Collects every device (each holds one outstanding ticket). A SEABRAID_PROVIDER status is EXPECTED only when
 * failures are being injected; it sets *failed so the caller skips the numeric check for this layer. Any
 * other status stops the run. */
static int collect_all(sim_run *run, int *failed)
{
    size_t count = sim_workload_output_count(run->workload);
    for (uint32_t d = 0; d < run->cfg.devices; d++) {
        int32_t status = seabraid_collect(run->sessions[d], run->tickets[d], SIM_COLLECT_TIMEOUT_MS,
                                          sim_workload_output(run->workload, d), count);
        if (status == SEABRAID_PROVIDER && run->cfg.fail_rate > 0.0) {
            run->result.failures_observed++;
            *failed = 1;
        } else if (sim_check(status, "seabraid_collect")) {
            return 1;
        }
    }
    return 0;
}

static int run_layer(sim_run *run, uint32_t layer, int measured)
{
    int failed = 0;
    uint64_t start, stop;
    double err, abs_err;
    sim_workload_generate(run->workload);
    start = sim_now_ns();
    if (submit_all(run, layer) || collect_all(run, &failed)) {
        return 1;
    }
    stop = sim_now_ns();
    if (measured) {
        run->latency_ns[run->latency_count++] = stop - start;
    }
    if (!failed) {
        err = sim_workload_max_error(run->workload, &abs_err);
        run->result.max_scaled_err = err > run->result.max_scaled_err ? err : run->result.max_scaled_err;
        run->result.max_abs_err = abs_err > run->result.max_abs_err ? abs_err : run->result.max_abs_err;
        run->result.verified++;
    }
    return 0;
}

static int run_steps(sim_run *run)
{
    /* WHY uint64: warmup + steps added in uint32 wrapped (--warmup 4294967295 --steps 1 ran zero steps and passed);
     * validate() also caps both, so the sum cannot reach 2^64 either. */
    uint64_t total = (uint64_t)run->cfg.warmup + run->cfg.steps;
    for (uint64_t step = 0; step < total; step++) {
        for (uint32_t layer = 0; layer < run->cfg.layers; layer++) {
            if (run_layer(run, layer, step >= run->cfg.warmup)) {
                return 1;
            }
        }
    }
    return 0;
}

/* The row invariants of one session: every submit counted its tokens, and only COMPACT_ROWS may ship fewer. */
static int rows_agree(const sim_run *run, const seabraid_session_status_v1 *status)
{
    uint64_t expected = (uint64_t)run->cfg.tokens * status->submitted;
    if (status->rows_submitted != expected || status->rows_shipped > status->rows_submitted) {
        return 0;
    }
    return run->cfg.compact_rows || status->rows_shipped == status->rows_submitted;
}

/* Reads each session's own counters, then closes it. The invariant checked here is the one the status struct
 * documents: every submitted ticket was either collected or counted as a provider failure. */
static int close_sessions(sim_run *run)
{
    int failed = 0;
    for (uint32_t d = 0; d < run->cfg.devices; d++) {
        seabraid_session_status_v1 status = {.abi_version = SEABRAID_ABI_VERSION, .struct_size = sizeof status};
        failed |= sim_check(seabraid_session_status(run->sessions[d], &status),
                            "seabraid_session_status");
        run->result.failures_status += status.provider_failures;
        run->result.rows_submitted += status.rows_submitted;
        run->result.rows_shipped += status.rows_shipped;
        if (!rows_agree(run, &status)) {
            fprintf(stderr, "device %u: row counters disagree (submitted %llu rows_submitted %llu rows_shipped %llu)\n", d,
                    (unsigned long long)status.submitted, (unsigned long long)status.rows_submitted,
                    (unsigned long long)status.rows_shipped);
            run->counts_disagree = 1;
        }
        if (status.outstanding != 0 || status.submitted != status.collected + status.provider_failures) {
            fprintf(stderr, "device %u: status counters disagree (submitted %llu collected %llu failures %llu)\n",
                    d, (unsigned long long)status.submitted, (unsigned long long)status.collected,
                    (unsigned long long)status.provider_failures);
            run->counts_disagree = 1;
        }
        failed |= sim_check(seabraid_close(run->sessions[d]), "seabraid_close");
        run->result.injected += (uint64_t)atomic_load(&run->devices[d].injected);
    }
    return failed;
}

/* Each injected failure must be (a) returned by collect, (b) counted by the session and (c) logged once. */
static int counts_agree(const sim_run *run)
{
    const sim_result *r = &run->result;
    if (r->failures_observed == r->injected && r->failures_status == r->injected &&
        r->log_provider_records == r->injected && !run->counts_disagree) {
        return 1;
    }
    fprintf(stderr, "failure counts disagree: injected %llu, collect saw %llu, session status %llu, log records %llu\n",
            (unsigned long long)r->injected, (unsigned long long)r->failures_observed,
            (unsigned long long)r->failures_status, (unsigned long long)r->log_provider_records);
    return 0;
}

static void finish_result(sim_run *run, const sim_placement_result *placement)
{
    sim_result *r = &run->result;
    sim_summarize_latency(run->latency_ns, run->latency_count, r);
    r->rows_per_s = r->layers_per_s * run->cfg.tokens;
    r->swaps = placement->swaps;
    r->placement = placement->note;
    sim_log_totals(&r->log_records, &r->log_provider_records);
}

/* 0 ok, 2 numeric mismatch or nothing verified, 3 count mismatch. Numeric first: a wrong answer outranks a
 * bookkeeping error. WHY nothing-verified fails: a run whose every layer was skipped or failed checked no number, so
 * "ok" would be a claim without evidence; only --fail-rate 1 legitimately verifies nothing (every call is made to fail). */
static int judge(sim_run *run)
{
    int code = 0;
    if (!sim_error_ok(run->result.max_scaled_err)) {
        fprintf(stderr, "scaled error %g is not within %g (max_abs_err %g; 1 = the rounding bound, inf = a non-finite output)\n", run->result.max_scaled_err, SIM_MAX_ERR,
                run->result.max_abs_err);
        code = 2;
    } else if (run->result.verified == 0 && run->cfg.fail_rate < 1.0) {
        fprintf(stderr, "no layer was verified (every layer failed or none ran), so no number was checked\n");
        code = 2;
    } else if (!counts_agree(run)) {
        code = 3;
    }
    run->result.ok = code == 0;
    return code;
}

/* What the flags alone decide: help, the caps, a refusal, or --validate-only. Returns the exit code, or -1 to run. */
static int flags_only(int parsed, const sim_run *run, const char *err)
{
    if (parsed == SIM_CLI_HELP) {
        sim_print_usage(stdout);
        return 0;
    }
    if (parsed == SIM_CLI_CAPS) {
        sim_print_caps(stdout);
        return 0;
    }
    if (parsed) {
        fprintf(stderr, "seabraid_sim: %s\n", err);
        return 1;
    }
    return run->cfg.validate_only ? 0 : -1; /* validate-only: the flags passed every rule; nothing is allocated or run */
}

static int execute(sim_run *run)
{
    sim_placement_result placement;
    int code = 1;
    if (run_alloc(run) && !sim_log_install() && !open_sessions(run) && !run_steps(run) && !close_sessions(run) &&
        !sim_placement_run(&run->cfg, &placement)) {
        finish_result(run, &placement);
        code = judge(run);
        sim_print_json(&run->cfg, &run->result);
    }
    run_free(run);
    return code;
}

int main(int argc, char **argv)
{
    sim_run run = {0};
    char err[256];
    int early = flags_only(sim_parse_args(argc, argv, &run.cfg, err, sizeof err), &run, err);
    return early >= 0 ? early : execute(&run);
}
