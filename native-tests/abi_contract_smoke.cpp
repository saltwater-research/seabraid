// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The header's versioning promises, checked from C++ against the built library: the 16-bit BF16 accept path, the
// struct_size tail-byte rule, a 48-byte session-status caller, and the placement snapshot epoch -> batch placement_epoch
// handoff.  OWNER: this file owns those four contracts' native tests (provider_smoke.cpp owns the F16/COMPACT_ROWS ones).
// CHECKER: ctest seabraid_abi_contract.  FAILURE: any broken promise prints the check name and aborts.
#include "seabraid/placement.h"
#include "seabraid/seabraid.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

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
        std::fprintf(stderr, "%s returned %d, expected %d\n", call, static_cast<int>(actual), static_cast<int>(expected));
        std::abort();
    }
}

// What the provider saw, so the C++ side can assert it (context points at one of these).
struct Seen {
    uint32_t format;
    uint64_t epoch;
    uint16_t raw[2];
};

// Returns each 16-bit element as a float so the bytes are checkable, or doubles an F32 element.
extern "C" int32_t record_execute(void *context, const seabraid_config_v1 *config, const seabraid_batch_v2 *batch,
                                   float *output, size_t output_count) noexcept
{
    Seen *seen = static_cast<Seen *>(context);
    seen->format = batch->activation_format;
    seen->epoch = batch->placement_epoch;
    if (output_count != static_cast<size_t>(batch->tokens) * config->hidden) {
        return SEABRAID_PROVIDER;
    }
    if (batch->activation_format == SEABRAID_FORMAT_ROUTED_F32) {
        const float *values = static_cast<const float *>(batch->activations);
        for (size_t i = 0; i < output_count; ++i) {
            output[i] = values[i] * 2.0f;
        }
        return SEABRAID_OK;
    }
    const uint16_t *halves = static_cast<const uint16_t *>(batch->activations);
    for (size_t i = 0; i < output_count; ++i) {
        output[i] = static_cast<float>(halves[i]);
        seen->raw[i] = halves[i];
    }
    return SEABRAID_OK;
}

const seabraid_config_v1 kConfig{SEABRAID_ABI_VERSION, 2, 3, 1, 1};
const int32_t kIds[] = {1};
const float kWeights[] = {0.5f};

seabraid_session *open_session(Seen *seen, uint32_t flags)
{
    seabraid_provider_v3 provider{SEABRAID_PROVIDER_V3_VERSION, sizeof(seabraid_provider_v3), 1, flags, &kConfig, seen,
                                  record_execute};
    seabraid_session *session = nullptr;
    expect_status(seabraid_session_open(&provider, &session), SEABRAID_OK, "open");
    return session;
}

seabraid_batch_v2 batch_of(const void *activations, uint32_t format, uint64_t epoch)
{
    return seabraid_batch_v2{SEABRAID_BATCH_V2_VERSION, sizeof(seabraid_batch_v2), 0, 0, epoch, 1, format, activations, 2,
                             kIds, 1, kWeights, 1};
}

void submit_and_collect(seabraid_session *session, const seabraid_batch_v2 *batch, float *output, const char *what)
{
    uint64_t ticket = 0;
    expect_status(seabraid_session_submit(session, batch, &ticket), SEABRAID_OK, what);
    expect_status(seabraid_collect(session, ticket, 5000, output, 2), SEABRAID_OK, what);
}

// ACCEPTS_BF16 admits ROUTED_BF16 and hands the 16-bit bytes over unconverted; it does not admit F16.
void bf16_accept_path()
{
    Seen seen{};
    seabraid_session *session = open_session(&seen, SEABRAID_PROVIDER_ACCEPTS_BF16);
    const uint16_t bf16[] = {0x3F80, 0xC000};  // 1.0 and -2.0 as bfloat16
    float output[2] = {-1.0f, -1.0f};
    seabraid_batch_v2 batch = batch_of(bf16, SEABRAID_FORMAT_ROUTED_BF16, 0);
    submit_and_collect(session, &batch, output, "BF16 submit");
    expect(seen.format == SEABRAID_FORMAT_ROUTED_BF16 && seen.raw[0] == 0x3F80 && seen.raw[1] == 0xC000, "BF16 bytes arrive unconverted");
    expect(output[0] == 16256.0f && output[1] == 49152.0f, "BF16 provider output");
    uint64_t ticket = 0;
    batch.activation_format = SEABRAID_FORMAT_ROUTED_F16;
    expect_status(seabraid_session_submit(session, &batch, &ticket), SEABRAID_INVALID, "F16 with only ACCEPTS_BF16");
    expect(ticket == 0, "refused F16 creates no ticket");
    expect_status(seabraid_close(session), SEABRAID_OK, "close BF16 session");
}

// struct_size larger than the library's struct: zero tail bytes are accepted, a nonzero tail byte is INVALID and changes nothing.
struct BigBatch {
    seabraid_batch_v2 head;
    uint64_t tail;
};
struct BigProvider {
    seabraid_provider_v3 head;
    uint64_t tail;
};

void struct_size_tail_rule()
{
    Seen seen{};
    seabraid_session *session = open_session(&seen, 0);
    const float values[] = {4.0f, 8.0f};
    BigBatch big{batch_of(values, SEABRAID_FORMAT_ROUTED_F32, 0), 0};
    big.head.struct_size = sizeof(BigBatch);
    big.tail = 1;
    uint64_t ticket = 0;
    expect_status(seabraid_session_submit(session, &big.head, &ticket), SEABRAID_INVALID, "nonzero batch tail");
    expect(ticket == 0, "refused tail creates no ticket");
    big.tail = 0;
    float output[2] = {-1.0f, -1.0f};
    submit_and_collect(session, &big.head, output, "zero batch tail");
    expect(output[0] == 8.0f && output[1] == 16.0f, "zero-tail batch output");
    expect_status(seabraid_close(session), SEABRAID_OK, "close tail session");

    BigProvider provider{{SEABRAID_PROVIDER_V3_VERSION, sizeof(BigProvider), 1, 0, &kConfig, &seen, record_execute}, 1};
    seabraid_session *opened = reinterpret_cast<seabraid_session *>(&provider);  // stale non-null value, never dereferenced
    expect_status(seabraid_session_open(&provider.head, &opened), SEABRAID_INVALID, "nonzero provider tail");
    expect(opened == nullptr, "refused open leaves no handle");
    provider.tail = 0;
    expect_status(seabraid_session_open(&provider.head, &opened), SEABRAID_OK, "zero provider tail");
    expect_status(seabraid_close(opened), SEABRAID_OK, "close zero-tail provider session");
}

// A caller built when the status struct was 48 bytes gets its 48 bytes and a struct_size of 48; the appended counters stay untouched.
void status_48_byte_caller()
{
    static_assert(offsetof(seabraid_session_status_v1, rows_submitted) == 48, "the appended counters start at byte 48");
    Seen seen{};
    seabraid_session *session = open_session(&seen, 0);
    const float values[] = {4.0f, 8.0f};
    float output[2];
    const seabraid_batch_v2 batch = batch_of(values, SEABRAID_FORMAT_ROUTED_F32, 0);
    submit_and_collect(session, &batch, output, "status batch");

    seabraid_session_status_v1 old_caller;
    std::memset(&old_caller, 0xAB, sizeof old_caller);
    old_caller.abi_version = SEABRAID_ABI_VERSION;
    old_caller.struct_size = 48;
    expect_status(seabraid_session_status(session, &old_caller), SEABRAID_OK, "status for a 48-byte caller");
    expect(old_caller.struct_size == 48, "struct_size written back as 48");
    expect(old_caller.submitted == 1 && old_caller.collected == 1 && old_caller.outstanding == 0, "the 48 known bytes are filled");
    const unsigned char *tail = reinterpret_cast<const unsigned char *>(&old_caller) + 48;
    bool untouched = true;
    for (size_t i = 0; i < sizeof old_caller - 48; ++i) {
        untouched = untouched && tail[i] == 0xAB;
    }
    expect(untouched, "counters past byte 48 are not written for a 48-byte caller");

    seabraid_session_status_v1 current{SEABRAID_ABI_VERSION, sizeof current, 0, 0, 0, 0, 0, 0, 0, 0};
    expect_status(seabraid_session_status(session, &current), SEABRAID_OK, "status for a current caller");
    expect(current.struct_size == sizeof current && current.rows_submitted == 1 && current.rows_shipped == 1, "a current caller sees the appended counters");
    expect_status(seabraid_close(session), SEABRAID_OK, "close status session");
}

// The snapshot's epoch is what a batch carries: the current epoch is accepted and reaches the provider, a stale one is refused.
void placement_epoch_handoff()
{
    const uint8_t owners[] = {0, 1};
    const seabraid_placement_layout_v1 layout{SEABRAID_ABI_VERSION, 1, 2, 2, owners, 2};
    seabraid_placement *placement = nullptr;
    expect_status(seabraid_placement_open(&layout, nullptr, &placement), SEABRAID_OK, "placement open");
    uint8_t executors[2] = {255, 255};
    seabraid_placement_snapshot_v1 snapshot{SEABRAID_ABI_VERSION, sizeof snapshot, 0, 0, 0, 0, 0};
    expect_status(seabraid_placement_snapshot(placement, executors, 2, &snapshot), SEABRAID_OK, "snapshot at the start");
    expect(snapshot.epoch == 1, "a new placement starts at epoch 1");

    Seen seen{};
    seabraid_session *session = open_session(&seen, 0);
    const float values[] = {4.0f, 8.0f};
    float output[2];
    seabraid_batch_v2 batch = batch_of(values, SEABRAID_FORMAT_ROUTED_F32, snapshot.epoch);
    submit_and_collect(session, &batch, output, "batch at the snapshot epoch");
    expect(seen.epoch == snapshot.epoch, "provider sees the snapshot epoch");

    expect_status(seabraid_placement_quarantine(placement, snapshot.epoch, 1), SEABRAID_OK, "quarantine moves the epoch");
    seabraid_placement_snapshot_v1 later{SEABRAID_ABI_VERSION, sizeof later, 0, 0, 0, 0, 0};
    expect_status(seabraid_placement_snapshot(placement, executors, 2, &later), SEABRAID_OK, "snapshot after the change");
    expect(later.epoch == snapshot.epoch + 1, "the placement change advanced the epoch");

    // The session learns the new epoch from the first batch that carries it; only then is the old one stale.
    batch.placement_epoch = later.epoch;
    submit_and_collect(session, &batch, output, "batch at the new snapshot epoch");
    expect(seen.epoch == later.epoch, "provider sees the new snapshot epoch");
    uint64_t ticket = 0;
    batch.placement_epoch = snapshot.epoch;
    expect_status(seabraid_session_submit(session, &batch, &ticket), SEABRAID_BUSY_OR_STATE, "batch at the stale epoch");
    expect(ticket == 0, "stale epoch creates no ticket");
    expect_status(seabraid_close(session), SEABRAID_OK, "close epoch session");
    expect_status(seabraid_placement_close(placement), SEABRAID_OK, "placement close");
}

}  // namespace

int main()
{
    bf16_accept_path();
    struct_size_tail_rule();
    status_48_byte_caller();
    placement_epoch_handoff();
    std::puts("Seabraid ABI contract smoke passed");
    return 0;
}
