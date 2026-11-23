// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* The smallest useful Seabraid provider: it computes the reference operation,
 * output = sum over picks of weight * activation * (expert_id + 1), then drives
 * one batch through session_open -> session_submit -> collect -> close. Replace
 * the body of execute() with real expert kernels. Built and run by CTest. */
#include "seabraid/seabraid.h"

#include <stdio.h>
#include <string.h>

/* Runs on the session's worker thread. The batch and output pointers expire
 * when this returns, so every device access must be finished by then. */
static int32_t execute(void *context, const seabraid_config_v1 *config, const seabraid_batch_v2 *batch,
                       float *output, size_t output_count)
{
    const float *activations = (const float *)batch->activations; /* F32: the default activation_format */
    size_t token, column, pick;
    (void)context;
    if (output_count != (size_t)batch->tokens * config->hidden) {
        return 1; /* any nonzero value reports a provider failure */
    }
    for (token = 0; token < batch->tokens; token++) {
        for (column = 0; column < config->hidden; column++) {
            float sum = 0.0f;
            for (pick = token * config->top_k; pick < (token + 1) * config->top_k; pick++) {
                sum += batch->weights[pick] * activations[token * config->hidden + column] *
                       (float)(batch->expert_ids[pick] + 1);
            }
            output[token * config->hidden + column] = sum;
        }
    }
    return SEABRAID_OK;
}

/* Every call returns a status; seabraid_last_error says why a nonzero one failed. */
static int check(int32_t status, const char *call)
{
    char reason[256];
    if (status == SEABRAID_OK) {
        return 0;
    }
    seabraid_last_error(reason, sizeof reason, NULL);
    fprintf(stderr, "%s failed (%d): %s\n", call, (int)status, reason);
    return 1;
}

int main(void)
{
    static const float activations[] = {1.0f, 2.0f};
    static const int32_t expert_ids[] = {0, 2};
    static const float weights[] = {0.25f, 0.75f};
    static const float expected[] = {2.5f, 5.0f};
    seabraid_config_v1 config = {SEABRAID_ABI_VERSION, 2, 3, 2, 2}; /* hidden, experts, max_tokens, top_k */
    seabraid_provider_v3 provider; /* zero it, then set abi_version and struct_size: unset bytes must be 0 */
    seabraid_batch_v2 batch;
    float output[2] = {0.0f, 0.0f};
    seabraid_session *session = NULL;
    uint64_t ticket = 0;
    int failed = 0;

    memset(&provider, 0, sizeof provider);
    provider.abi_version = SEABRAID_PROVIDER_V3_VERSION;
    provider.struct_size = sizeof provider;
    provider.operation_count = 1; /* one operation id per model layer; this model has one */
    provider.config = &config;    /* copied at open: no separate prepare call */
    provider.execute = execute;
    memset(&batch, 0, sizeof batch);
    batch.abi_version = SEABRAID_BATCH_V2_VERSION;
    batch.struct_size = sizeof batch;
    batch.source_modalities = SEABRAID_SOURCE_TEXT;
    batch.placement_epoch = 0; /* 0 = unversioned; an engine with live placement passes the snapshot epoch */
    batch.tokens = 1;
    batch.activation_format = SEABRAID_FORMAT_ROUTED_F32;
    batch.activations = activations;
    batch.activation_count = 2;
    batch.expert_ids = expert_ids;
    batch.expert_id_count = 2;
    batch.weights = weights;
    batch.weight_count = 2;

    printf("seabraid %s\n", seabraid_version());
    if (seabraid_abi_epoch() != SEABRAID_ABI_EPOCH) { /* a library from another ABI epoch must not be driven */
        fprintf(stderr, "ABI epoch mismatch: library %u, header %u\n", (unsigned)seabraid_abi_epoch(), SEABRAID_ABI_EPOCH);
        return 1;
    }
    if (check(seabraid_session_open(&provider, &session), "seabraid_session_open")) {
        return 1;
    }
    /* Stop at the first failed step: a failed submit leaves no ticket to collect. close still runs to free the session. */
    failed = check(seabraid_session_submit(session, &batch, &ticket), "seabraid_session_submit") ||
             check(seabraid_collect(session, ticket, 1000, output, 2), "seabraid_collect");
    failed |= check(seabraid_close(session), "seabraid_close");
    if (!failed) {
        printf("output = [%g, %g]\n", output[0], output[1]);
        failed = (output[0] - expected[0] > 1e-6f || expected[0] - output[0] > 1e-6f ||
                  output[1] - expected[1] > 1e-6f || expected[1] - output[1] > 1e-6f);
    }
    return failed;
}
