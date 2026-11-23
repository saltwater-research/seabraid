// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* A reference adapter walkthrough: one provider_v3 device driven through every call an engine adapter makes, one
 * printed line per step. It exits nonzero on any unexpected status and prints "ok" last. Read it top to
 * bottom next to docs/ADAPTER-GUIDE.md. Built and run by CTest as seabraid_adapter_walkthrough. */
#include "seabraid/seabraid.h"
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
enum { HIDDEN = 4, EXPERTS = 6, MAX_TOKENS = 4, TOP_K = 3, ROWS = 3 };
#define SKIP SEABRAID_SKIP_EXPERT_ID

/* What the device callback records so main() can check what the runtime actually handed it. WHY atomics: execute()
 * runs on the session's worker thread while main() reads and writes these fields, and the ABI says "the caller
 * synchronizes shared access", so state shared between a callback and the engine thread is atomic (or locked). Plain
 * fields would work here only because collect() happens to order the two; do not rely on that in an adapter. */
typedef struct {
    atomic_uint calls, last_tokens;
    atomic_int fail_next; /* simulates a device fault on the next call */
} device;

/* The "kernel": expert e computes y = x * (e + 2) + e. A real adapter launches its GPU work here. */
static float expert(float x, int32_t e) { return x * (float)(e + 2) + (float)e; }

/* Runs on the session worker thread. Every pointer expires when this returns. With COMPACT_ROWS the runtime has
 * already dropped fully skipped rows, so batch->tokens counts only the rows that still need work, and
 * output_count is tokens * hidden for that compacted count. */
static int32_t execute(void *context, const seabraid_config_v1 *cfg, const seabraid_batch_v2 *batch, float *out,
                       size_t output_count)
{
    device *dev = context;
    const float *x = batch->activations; /* F32 only: ACCEPTS_F16/BF16 are not declared */
    atomic_fetch_add(&dev->calls, 1);
    atomic_store(&dev->last_tokens, batch->tokens);
    if (atomic_exchange(&dev->fail_next, 0)) { /* read and clear in one step */
        return 7; /* any nonzero value surfaces at collect as SEABRAID_PROVIDER */
    }
    if (output_count != (size_t)batch->tokens * cfg->hidden) return 1;
    for (size_t t = 0; t < batch->tokens; t++) {
        for (size_t c = 0; c < cfg->hidden; c++) {
            float sum = 0.0f;
            for (size_t k = 0; k < cfg->top_k; k++) {
                int32_t e = batch->expert_ids[t * cfg->top_k + k];
                if (e != SKIP) sum += batch->weights[t * cfg->top_k + k] * expert(x[t * cfg->hidden + c], e);
            }
            out[t * cfg->hidden + c] = sum;
        }
    }
    return SEABRAID_OK;
}

/* The log sink: the runtime reports every refusal and failure here. stderr keeps stdout to one line per step. */
static void on_log(void *context, uint32_t level, int32_t status, const char *message)
{
    (void)context;
    fprintf(stderr, "  [log level=%u status=%d] %s\n", level, status, message);
}

static int failed;
/* One step = one printed line. A status other than the expected one fails the program. */
static void step(const char *name, int32_t got, int32_t want)
{
    printf("%-34s status=%d%s\n", name, got, got == want ? "" : "  UNEXPECTED");
    failed |= got != want;
}

static seabraid_batch_v2 make_batch(uint64_t epoch, uint32_t tokens, const float *x, const int32_t *ids, const float *w)
{
    seabraid_batch_v2 b;
    memset(&b, 0, sizeof b); /* zero first: every field you do not set means "not used" */
    b.abi_version = SEABRAID_BATCH_V2_VERSION;
    b.struct_size = sizeof b;
    b.source_modalities = SEABRAID_SOURCE_TEXT;
    b.tokens = tokens;
    b.activation_format = SEABRAID_FORMAT_ROUTED_F32;
    b.placement_epoch = epoch; /* 0 = unversioned; once nonzero, an older epoch is stale */
    b.activations = x;
    b.activation_count = (size_t)tokens * HIDDEN;
    b.expert_ids = ids;
    b.expert_id_count = (size_t)tokens * TOP_K;
    b.weights = w;
    b.weight_count = (size_t)tokens * TOP_K;
    return b;
}

int main(void)
{
    device dev = {0};
    seabraid_config_v1 cfg = {SEABRAID_ABI_VERSION, HIDDEN, EXPERTS, MAX_TOKENS, TOP_K};
    seabraid_provider_v3 prov;
    memset(&prov, 0, sizeof prov);
    prov.abi_version = SEABRAID_PROVIDER_V3_VERSION;
    prov.struct_size = sizeof prov;
    prov.operation_count = 1;
    /* PARTIAL_ROUTES: this device may hold only some experts, so the engine marks the rest with SKIP.
     * COMPACT_ROWS: rows whose picks are all skipped are not shipped to execute() at all. */
    prov.flags = SEABRAID_PROVIDER_PARTIAL_ROUTES | SEABRAID_PROVIDER_COMPACT_ROWS;
    prov.config = &cfg;
    prov.context = &dev;
    prov.execute = execute;

    /* The ABI epoch is a link-time/run-time handshake: a library from another epoch must not be driven. */
    if (seabraid_abi_epoch() != SEABRAID_ABI_EPOCH) {
        fprintf(stderr, "ABI epoch mismatch: library %u, header %u\n", (unsigned)seabraid_abi_epoch(), SEABRAID_ABI_EPOCH);
        return 2;
    }
    seabraid_set_log_sink(on_log, NULL, SEABRAID_LOG_INFO);
    printf("seabraid %s\n", seabraid_version());

    seabraid_session *session = NULL;
    step("session_open", seabraid_session_open(&prov, &session), SEABRAID_OK);
    if (!session) return 1;

    /* Three rows; row 1 is entirely skipped (another device owns those experts), row 2 is partly skipped. */
    float x[ROWS * HIDDEN], out[ROWS * HIDDEN], want[ROWS * HIDDEN];
    int32_t ids[ROWS * TOP_K] = {0, 3, 5, SKIP, SKIP, SKIP, SKIP, 2, SKIP};
    float w[ROWS * TOP_K] = {.5f, .25f, .25f, 0, 0, 0, 0, 1.f, 0}; /* a skipped pick carries weight 0 */
    for (int i = 0; i < ROWS * HIDDEN; i++) {
        x[i] = 0.5f * (float)i;
        want[i] = 0.0f;
    }
    for (int i = 0; i < ROWS * HIDDEN; i++) { /* the expected output, computed apart from execute() */
        for (int k = 0; k < TOP_K; k++) {
            int32_t e = ids[i / HIDDEN * TOP_K + k];
            if (e != SKIP) want[i] += w[i / HIDDEN * TOP_K + k] * expert(x[i], e);
        }
    }

    /* 1. A versioned batch at epoch 5. The callback must see 2 rows, not 3; the skipped row's output is zero. */
    uint64_t ticket = 0;
    seabraid_batch_v2 b = make_batch(5, ROWS, x, ids, w);
    step("submit epoch 5, 3 rows", seabraid_session_submit(session, &b, &ticket), SEABRAID_OK);
    memset(out, 0xFF, sizeof out); /* poison: a row the runtime failed to zero would show */
    step("collect", seabraid_collect(session, ticket, 1000, out, ROWS * HIDDEN), SEABRAID_OK);
    for (int i = 0; i < ROWS * HIDDEN; i++) {
        if (out[i] - want[i] > 1e-5f || want[i] - out[i] > 1e-5f) {
            failed = 1;
            printf("  MISMATCH [%d]\n", i);
        }
    }
    printf("device saw %u rows of 3 submitted\n", atomic_load(&dev.last_tokens));
    failed |= atomic_load(&dev.last_tokens) != 2;

    /* 2. Stale epoch: the engine changed placement after this batch was routed. The refusal changes nothing, so
     * the documented answer is: re-snapshot (take the new epoch, re-route) and resubmit under it. */
    b = make_batch(4, ROWS, x, ids, w);
    step("submit stale epoch 4", seabraid_session_submit(session, &b, &ticket), SEABRAID_BUSY_OR_STATE);
    b = make_batch(6, ROWS, x, ids, w); /* here: the epoch a fresh placement snapshot would report */
    step("resubmit at epoch 6", seabraid_session_submit(session, &b, &ticket), SEABRAID_OK);
    step("collect", seabraid_collect(session, ticket, 1000, out, ROWS * HIDDEN), SEABRAID_OK);

    /* 3. A malformed batch (expert 2 twice in one row) is refused whole; the reason is in the thread's last error. */
    int32_t dup[TOP_K] = {2, 2, 1};
    float wd[TOP_K] = {1, 1, 1};
    char why[200] = "";
    b = make_batch(6, 1, x, dup, wd);
    step("submit repeated expert id", seabraid_session_submit(session, &b, &ticket), SEABRAID_INVALID);
    seabraid_last_error(why, sizeof why, NULL);
    printf("  reason: %s\n", why);

    /* 4. A device failure arrives at collect as SEABRAID_PROVIDER; the session stays usable. */
    atomic_store(&dev.fail_next, 1);
    b = make_batch(6, ROWS, x, ids, w);
    step("submit (device will fail)", seabraid_session_submit(session, &b, &ticket), SEABRAID_OK);
    step("collect after device failure", seabraid_collect(session, ticket, 1000, out, ROWS * HIDDEN), SEABRAID_PROVIDER);
    step("resubmit after failure", seabraid_session_submit(session, &b, &ticket), SEABRAID_OK);
    step("collect", seabraid_collect(session, ticket, 1000, out, ROWS * HIDDEN), SEABRAID_OK);

    /* 5. Counters: the session keeps score so an engine can export them without its own bookkeeping. */
    seabraid_session_status_v1 st;
    memset(&st, 0, sizeof st);
    st.abi_version = SEABRAID_ABI_VERSION;
    st.struct_size = sizeof st;
    step("session_status", seabraid_session_status(session, &st), SEABRAID_OK);
    printf("submitted=%llu collected=%llu provider_failures=%llu outstanding=%u\n", (unsigned long long)st.submitted,
           (unsigned long long)st.collected, (unsigned long long)st.provider_failures, st.outstanding);
    failed |= st.provider_failures != 1 || st.outstanding != 0;

    step("close", seabraid_close(session), SEABRAID_OK);
    printf(failed ? "FAIL\n" : "ok\n");
    return failed;
}
