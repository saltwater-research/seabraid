// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Run from the repository root after `cargo build --offline`.
// Linux: ln -sf libseabraid.so target/debug/libseabraid.so.0.5   (the SONAME; cargo writes only libseabraid.so)
//        c++ -std=c++17 -pthread -Iinclude native-tests/provider_smoke.cpp -Ltarget/debug -lseabraid -o target/provider_smoke
//        LD_LIBRARY_PATH=target/debug ./target/provider_smoke
// Windows (MSVC developer shell): cl /std:c++17 /EHsc /Iinclude /Fo:target\ /Fe:target\debug\provider_smoke.exe native-tests\provider_smoke.cpp target\debug\seabraid.dll.lib
//                                 target\debug\provider_smoke.exe

#include "seabraid/seabraid.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace {

struct ProviderGate {
    std::mutex mutex;
    std::condition_variable changed;
    bool released = false;

    void release()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            released = true;
        }
        changed.notify_one();
    }

    bool await_release()
    {
        // Bound the wait so a regression to synchronous submit reports failure.
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(5),
                                [this] { return released; });
    }
};

void expect(bool passed, const char *check)
{
    if (!passed) {
        std::fprintf(stderr, "%s failed\n", check);
        std::abort();
    }
}

void expect_status(int32_t actual, int32_t expected, const char *call)
{
    if (actual != expected) {
        std::fprintf(stderr, "%s returned %d, expected %d\n", call,
                     static_cast<int>(actual), static_cast<int>(expected));
        std::abort();
    }
}

}  // namespace

extern "C" int32_t smoke_provider_execute(void *context,
                                           const seabraid_config_v1 *config,
                                           const seabraid_batch_v1 *batch,
                                           float *output, size_t output_count) noexcept
{
    try {
        auto &gate = *static_cast<ProviderGate *>(context);
        if (!gate.await_release()) {
            return SEABRAID_PROVIDER;
        }
        if (config->hidden != 2 || batch->tokens != 1 || output_count != 2) {
            return SEABRAID_PROVIDER;
        }
        // Every caller input affects the result, so borrowing any caller buffer fails.
        for (size_t index = 0; index < output_count; ++index) {
            output[index] = batch->activations[index] * batch->weights[0]
                          + static_cast<float>(batch->expert_ids[0]);
        }
        return SEABRAID_OK;
    } catch (...) {
        return SEABRAID_PROVIDER;
    }
}

extern "C" int32_t smoke_multi_execute(void *, uint32_t operation_id,
                                        const seabraid_config_v1 *, const seabraid_batch_v1 *batch,
                                        float *output, size_t output_count) noexcept
{
    for (size_t index = 0; index < output_count; ++index) {
        output[index] = batch->activations[index] + static_cast<float>(operation_id);
    }
    return SEABRAID_OK;
}

static void exercise_multi_provider()
{
    const seabraid_provider_v2 provider{SEABRAID_MULTI_PROVIDER_VERSION, 2, 0, nullptr, smoke_multi_execute};
    seabraid_session *session = nullptr;
    expect_status(seabraid_open_multi(&provider, &session), SEABRAID_OK, "open multiple operations");
    const seabraid_config_v1 config{SEABRAID_ABI_VERSION, 2, 1, 1, 1};
    expect_status(seabraid_prepare(session, &config), SEABRAID_OK, "prepare shared geometry");

    const float activations[] = {2.0f, 6.0f};
    const int32_t expert_ids[] = {0};
    const float weights[] = {1.0f};
    const seabraid_batch_v1 batch{SEABRAID_ABI_VERSION, 1, activations, 2, expert_ids, 1, weights, 1};
    uint64_t ticket = 0;
    expect_status(seabraid_submit_operation(session, 2, &batch, 0, &ticket),
                  SEABRAID_INVALID, "operation outside provider table");
    expect(ticket == 0, "unknown operation leaves ticket untouched");

    float output[] = {-1.0f, -1.0f};
    expect_status(seabraid_submit_operation(session, 1, &batch, 0, &ticket),
                  SEABRAID_OK, "submit selected operation");
    expect_status(seabraid_collect(session, ticket, 5000, output, 2), SEABRAID_OK, "collect operation one");
    expect(output[0] == 3.0f && output[1] == 7.0f, "operation one result");

    expect_status(seabraid_submit(session, &batch, &ticket), SEABRAID_OK, "legacy submit to multi provider");
    expect_status(seabraid_collect(session, ticket, 5000, output, 2), SEABRAID_OK, "collect operation zero");
    expect(output[0] == 2.0f && output[1] == 6.0f, "operation zero result after reuse");
    expect_status(seabraid_close(session), SEABRAID_OK, "close multiple operations");
}

// A provider that never finishes must not hang the caller: close_timeout reports it and leaks the worker.
static void exercise_bounded_close()
{
    ProviderGate gate;
    const seabraid_provider_v1 provider{SEABRAID_ABI_VERSION, &gate, smoke_provider_execute};
    seabraid_session *session = nullptr;
    expect_status(seabraid_open(&provider, &session), SEABRAID_OK, "open for bounded close");
    const seabraid_config_v1 config{SEABRAID_ABI_VERSION, 2, 3, 1, 1};
    expect_status(seabraid_prepare(session, &config), SEABRAID_OK, "prepare for bounded close");
    const float activations[] = {1.0f, 2.0f};
    const int32_t expert_ids[] = {0};
    const float weights[] = {1.0f};
    const seabraid_batch_v1 batch{SEABRAID_ABI_VERSION, 1, activations, 2, expert_ids, 1, weights, 1};
    uint64_t ticket = 0;
    expect_status(seabraid_submit(session, &batch, &ticket), SEABRAID_OK, "submit for bounded close");
    expect_status(seabraid_close_timeout(session, 50), SEABRAID_TIMEOUT, "close_timeout on a stuck provider");
    // The leaked worker still uses `gate`, so release it before the gate goes out of scope.
    gate.release();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

static void exercise_reference_open()
{
    seabraid_session *session = nullptr;
    expect_status(seabraid_open_reference(&session), SEABRAID_OK, "open reference");
    expect_status(seabraid_close_timeout(session, 5000), SEABRAID_OK, "bounded close of an idle session");
}

extern "C" int32_t smoke_v3_execute(void *, const seabraid_config_v1 *config, const seabraid_batch_v2 *batch,
                                     float *output, size_t output_count) noexcept
{
    if (batch->abi_version != SEABRAID_BATCH_V2_VERSION || batch->struct_size != sizeof(seabraid_batch_v2)
        || batch->activation_format != SEABRAID_FORMAT_ROUTED_F32 || batch->tokens != 1
        || batch->operation_id != 1 || config->hidden != 2 || output_count != 2) {
        return SEABRAID_PROVIDER;
    }
    const float *activations = static_cast<const float *>(batch->activations);
    for (size_t index = 0; index < output_count; ++index) {
        output[index] = activations[index] * batch->weights[0] + static_cast<float>(batch->expert_ids[0]);
    }
    return SEABRAID_OK;
}

// The v2 entry points from C: open a v3 provider, submit a v2 batch, collect, and the refusals the header promises.
static void exercise_v2_entry_points()
{
    const seabraid_config_v1 config{SEABRAID_ABI_VERSION, 2, 3, 1, 1};
    seabraid_provider_v3 provider{SEABRAID_PROVIDER_V3_VERSION, sizeof(seabraid_provider_v3), 2, 0,
                                  &config, nullptr, smoke_v3_execute};
    seabraid_session *session = nullptr;
    expect_status(seabraid_session_open(&provider, &session), SEABRAID_OK, "v3 open");
    expect(session != nullptr, "v3 session handle");

    const float activations[] = {4.0f, 8.0f};
    const int32_t expert_ids[] = {1};
    const float weights[] = {0.5f};
    seabraid_batch_v2 batch{SEABRAID_BATCH_V2_VERSION, sizeof(seabraid_batch_v2), 1, 0, 0, 1,
                            SEABRAID_FORMAT_ROUTED_F32, activations, 2, expert_ids, 1, weights, 1};
    uint64_t ticket = 0;
    batch.activation_format = SEABRAID_FORMAT_ROUTED_F16;
    expect_status(seabraid_session_submit(session, &batch, &ticket), SEABRAID_INVALID,
                  "F16 without ACCEPTS_F16");
    expect(ticket == 0, "refused v2 submit creates no ticket");
    batch.activation_format = SEABRAID_FORMAT_ROUTED_F32;
    batch.operation_id = 2;
    expect_status(seabraid_session_submit(session, &batch, &ticket), SEABRAID_INVALID, "operation outside table");
    batch.operation_id = 1;
    expect_status(seabraid_session_submit(session, &batch, &ticket), SEABRAID_OK, "v2 submit");
    float output[] = {-1.0f, -1.0f};
    expect_status(seabraid_collect(session, ticket, 5000, output, 2), SEABRAID_OK, "collect v2");
    expect(output[0] == 3.0f && output[1] == 5.0f, "v2 provider output");
    expect_status(seabraid_close(session), SEABRAID_OK, "close v3 session");

    provider.flags = SEABRAID_PROVIDER_COMPACT_ROWS;
    // A non-null stale value (never dereferenced) proves a failed open overwrites *out with NULL.
    session = reinterpret_cast<seabraid_session *>(&provider);
    expect_status(seabraid_session_open(&provider, &session), SEABRAID_INVALID, "COMPACT_ROWS without PARTIAL_ROUTES");
    expect(session == nullptr, "refused v3 open leaves no handle");

    const seabraid_provider_v3 reference{SEABRAID_PROVIDER_V3_VERSION, sizeof(seabraid_provider_v3), 1,
                                         SEABRAID_PROVIDER_REFERENCE, &config, nullptr, nullptr};
    expect_status(seabraid_session_open(&reference, &session), SEABRAID_OK, "v3 reference open");
    batch.operation_id = 0;
    expect_status(seabraid_session_submit(session, &batch, &ticket), SEABRAID_OK, "v2 submit to reference");
    expect_status(seabraid_collect(session, ticket, 5000, output, 2), SEABRAID_OK, "collect reference");
    expect(output[0] == 4.0f && output[1] == 8.0f, "reference output");
    expect_status(seabraid_close(session), SEABRAID_OK, "close v3 reference session");
}

// Records what the v3 view carried so the C side can assert it (context points at a V3Seen).
struct V3Seen {
    uint32_t tokens;
    uint32_t format;
    uint64_t epoch;
    uint16_t halves[2];
};

// Doubles each activation, or, for F16, returns the raw 16-bit element as a float so the bytes are checkable.
extern "C" int32_t smoke_v3_record_execute(void *context, const seabraid_config_v1 *config,
                                            const seabraid_batch_v2 *batch, float *output,
                                            size_t output_count) noexcept
{
    V3Seen *seen = static_cast<V3Seen *>(context);
    seen->tokens = batch->tokens;
    seen->format = batch->activation_format;
    seen->epoch = batch->placement_epoch;
    if (output_count != static_cast<size_t>(batch->tokens) * config->hidden) {
        return SEABRAID_PROVIDER;
    }
    if (batch->activation_format == SEABRAID_FORMAT_ROUTED_F16) {
        const uint16_t *halves = static_cast<const uint16_t *>(batch->activations);
        for (size_t index = 0; index < output_count; ++index) {
            output[index] = static_cast<float>(halves[index]);
            if (index < 2) {
                seen->halves[index] = halves[index];
            }
        }
        return SEABRAID_OK;
    }
    const float *activations = static_cast<const float *>(batch->activations);
    for (size_t index = 0; index < output_count; ++index) {
        output[index] = activations[index] * 2.0f;
    }
    return SEABRAID_OK;
}

// ACCEPTS_F16 hands over the 16-bit bytes untouched; COMPACT_ROWS hides fully skipped rows; epochs are enforced.
static void exercise_v2_formats_rows_and_epochs()
{
    const seabraid_config_v1 config{SEABRAID_ABI_VERSION, 2, 3, 4, 1};
    V3Seen seen{};
    seabraid_provider_v3 provider{SEABRAID_PROVIDER_V3_VERSION, sizeof(seabraid_provider_v3), 1,
                                  SEABRAID_PROVIDER_ACCEPTS_F16, &config, &seen, smoke_v3_record_execute};
    seabraid_session *session = nullptr;
    expect_status(seabraid_session_open(&provider, &session), SEABRAID_OK, "v3 ACCEPTS_F16 open");

    const uint16_t halves[] = {0x3C00, 0xC000};
    const int32_t one_id[] = {1};
    const float one_weight[] = {0.5f};
    seabraid_batch_v2 batch{SEABRAID_BATCH_V2_VERSION, sizeof(seabraid_batch_v2), 0, 0, 9, 1,
                            SEABRAID_FORMAT_ROUTED_F16, halves, 2, one_id, 1, one_weight, 1};
    uint64_t ticket = 0;
    expect_status(seabraid_session_submit(session, &batch, &ticket), SEABRAID_OK, "F16 submit");
    float output[6] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    expect_status(seabraid_collect(session, ticket, 5000, output, 2), SEABRAID_OK, "collect F16");
    expect(seen.format == SEABRAID_FORMAT_ROUTED_F16 && seen.halves[0] == 0x3C00 && seen.halves[1] == 0xC000,
           "F16 bytes arrive unconverted");
    expect(output[0] == 15360.0f && output[1] == 49152.0f && seen.epoch == 9, "F16 output and epoch");

    batch.placement_epoch = 3;
    expect_status(seabraid_session_submit(session, &batch, &ticket), SEABRAID_BUSY_OR_STATE, "stale epoch");
    batch.placement_epoch = 0;
    expect_status(seabraid_session_submit(session, &batch, &ticket), SEABRAID_INVALID, "epoch 0 after nonzero");
    batch.activation_format = SEABRAID_FORMAT_ROUTED_BF16;
    batch.placement_epoch = 9;
    expect_status(seabraid_session_submit(session, &batch, &ticket), SEABRAID_INVALID, "BF16 without ACCEPTS_BF16");
    expect_status(seabraid_close(session), SEABRAID_OK, "close F16 session");

    provider.flags = SEABRAID_PROVIDER_PARTIAL_ROUTES | SEABRAID_PROVIDER_COMPACT_ROWS;
    expect_status(seabraid_session_open(&provider, &session), SEABRAID_OK, "v3 COMPACT_ROWS open");
    const float rows[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const int32_t ids[] = {1, SEABRAID_SKIP_EXPERT_ID, 0};
    const float weights[] = {0.5f, 0.0f, 1.0f};
    seabraid_batch_v2 wide{SEABRAID_BATCH_V2_VERSION, sizeof(seabraid_batch_v2), 0, 0, 0, 3,
                           SEABRAID_FORMAT_ROUTED_F32, rows, 6, ids, 3, weights, 3};
    expect_status(seabraid_session_submit(session, &wide, &ticket), SEABRAID_OK, "COMPACT_ROWS submit");
    expect_status(seabraid_collect(session, ticket, 5000, output, 6), SEABRAID_OK, "collect compacted");
    expect(seen.tokens == 2, "provider sees only the kept rows");
    expect(output[0] == 2.0f && output[1] == 4.0f && output[2] == 0.0f && output[3] == 0.0f
           && output[4] == 10.0f && output[5] == 12.0f,
           "compacted output lands in the original rows");
    expect_status(seabraid_close(session), SEABRAID_OK, "close compacted session");
}

int main()
{
    seabraid_capabilities_v1 capabilities{SEABRAID_ABI_VERSION, sizeof(seabraid_capabilities_v1), 0, 0, 0, 0};
    expect_status(seabraid_query_capabilities(&capabilities), SEABRAID_OK, "query capabilities");
    expect(seabraid_abi_epoch() == SEABRAID_ABI_EPOCH, "layout epoch");
    seabraid_session *refused = nullptr;
    expect_status(seabraid_open(nullptr, &refused), SEABRAID_INVALID, "null provider is refused");
    expect(refused == nullptr, "refused open leaves no handle");
    expect(capabilities.operation == SEABRAID_OPERATION_ROUTED_MOE
           && capabilities.input_format == SEABRAID_FORMAT_ROUTED_F32
           && capabilities.output_format == SEABRAID_FORMAT_PARTIAL_F32,
           "routed activation formats");
    constexpr uint32_t origins = SEABRAID_SOURCE_AUDIO | SEABRAID_SOURCE_VIDEO;
    expect((capabilities.source_modalities & origins) == origins, "mixed sources");

    ProviderGate gate;
    const seabraid_provider_v1 provider{SEABRAID_ABI_VERSION, &gate, smoke_provider_execute};
    seabraid_session *session = nullptr;
    expect_status(seabraid_open(&provider, &session), SEABRAID_OK, "open");
    expect(session != nullptr, "session handle");

    const seabraid_config_v1 config{SEABRAID_ABI_VERSION, 2, 3, 1, 1};
    expect_status(seabraid_prepare(session, &config), SEABRAID_OK, "prepare");

    float activations[] = {4.0f, 8.0f};
    int32_t expert_ids[] = {1};
    float weights[] = {0.5f};
    const seabraid_batch_v1 batch{SEABRAID_ABI_VERSION, 1, activations, 2, expert_ids, 1, weights, 1};
    uint64_t ticket = 0;
    expect_status(seabraid_submit_routed(session, &batch, UINT32_C(0x80000000), &ticket),
                  SEABRAID_INVALID, "unknown modality");
    expect(ticket == 0, "invalid metadata creates no ticket");
    expect_status(seabraid_submit_routed(session, &batch, origins, &ticket), SEABRAID_OK, "submit");

    // The provider can read only after release, so this tests the submit copy.
    activations[0] = 10.0f;
    activations[1] = 20.0f;
    expert_ids[0] = 2;
    weights[0] = 1.0f;

    float output[] = {-1.0f, -1.0f};
    expect_status(seabraid_collect(session, ticket + 1, 0, output, 2),
                  SEABRAID_INVALID, "wrong ticket");
    expect(output[0] == -1.0f && output[1] == -1.0f, "wrong-ticket output");
    expect_status(seabraid_collect(session, ticket, 0, output, 2), SEABRAID_TIMEOUT, "poll");
    expect(output[0] == -1.0f && output[1] == -1.0f, "timeout output");

    gate.release();
    expect_status(seabraid_collect(session, ticket, 5000, output, 2), SEABRAID_OK, "retry collect");
    expect(output[0] == 3.0f && output[1] == 5.0f, "provider output from copied inputs");
    expect_status(seabraid_collect(session, ticket, 0, output, 2),
                  SEABRAID_BUSY_OR_STATE, "consumed ticket");
    expect_status(seabraid_close(session), SEABRAID_OK, "close");
    exercise_multi_provider();
    exercise_bounded_close();
    exercise_reference_open();
    exercise_v2_entry_points();
    exercise_v2_formats_rows_and_epochs();
    std::puts("Seabraid C++ provider smoke passed");
}
