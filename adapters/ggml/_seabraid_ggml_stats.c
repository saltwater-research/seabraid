// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Owner of the adapter's transfer counter: the bytes the direct execute path really moves across the
 * host/device boundary. Only run_graph moves data (three uploads and one download per batch). The expert
 * weights are resident and never staged, and the append_swiglu path leaves its copies to the host's
 * scheduler, which the adapter cannot see, so neither appears here.
 * FAILURE: counting cannot fail and never refuses a batch; only a read into a null, unversioned or too-small
 * struct fails (fail-fast, as SEABRAID_INVALID). */
#include "_seabraid_ggml_stats.h"

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include "seabraid_ggml.h"

/* WHY atomics: the Seabraid worker thread runs execute while the host may run another cache's execute (or
 * read the counter) on its own thread, and the counter is process-wide by contract. WHY relaxed: the
 * counters order nothing else; a reader needs only an untorn value of each independent total. */
static struct {
    atomic_uint_fast64_t sent_bytes;
    atomic_uint_fast64_t sent_transfers;
    atomic_uint_fast64_t returned_bytes;
    atomic_uint_fast64_t returned_transfers;
    atomic_uint_fast64_t batches;
} totals;

static void add(atomic_uint_fast64_t *counter, uint64_t amount) {
    atomic_fetch_add_explicit(counter, amount, memory_order_relaxed);
}

static uint64_t load(atomic_uint_fast64_t *counter) {
    return atomic_load_explicit(counter, memory_order_relaxed);
}

static void zero(atomic_uint_fast64_t *counter) {
    atomic_store_explicit(counter, 0, memory_order_relaxed);
}

void seabraid_ggml_stats_sent(uint64_t bytes, uint64_t transfers) {
    add(&totals.sent_bytes, bytes);
    add(&totals.sent_transfers, transfers);
}

void seabraid_ggml_stats_returned(uint64_t bytes) {
    add(&totals.returned_bytes, bytes);
    add(&totals.returned_transfers, 1);
}

void seabraid_ggml_stats_batch_done(void) {
    add(&totals.batches, 1);
}

int32_t seabraid_ggml_get_transfer_stats(seabraid_ggml_transfer_stats *out) {
    /* Same contract as the runtime's library-filled structs (seabraid.h): refuse an unknown version or a
     * struct_size below the v1 size (the whole struct today), write at most min(struct_size, current size)
     * bytes, and set the struct_size written back to that count. WHY refuse below v1 instead of clamping:
     * every field of this v1 struct is documented, so a shorter caller is a bug in the caller, and writing
     * our own size into its struct would claim bytes it never received. */
    if (!out || out->abi_version != SEABRAID_ABI_VERSION || out->struct_size < sizeof(seabraid_ggml_transfer_stats)) {
        return SEABRAID_INVALID;
    }
    seabraid_ggml_transfer_stats now = {SEABRAID_ABI_VERSION, sizeof(now)};
    now.sent_bytes = load(&totals.sent_bytes);
    now.sent_transfers = load(&totals.sent_transfers);
    now.returned_bytes = load(&totals.returned_bytes);
    now.returned_transfers = load(&totals.returned_transfers);
    now.batches = load(&totals.batches);
    /* weight_bytes_staged stays 0: the adapter keeps weights resident. The field is reserved so a future
     * stager adds a value instead of breaking the layout. */
    const uint32_t caller_size = out->struct_size;
    const uint32_t written = caller_size < sizeof(now) ? caller_size : (uint32_t) sizeof(now);
    now.struct_size = written; /* the clamp: a newer, larger caller reads this back as the filled extent */
    memcpy(out, &now, written);
    return SEABRAID_OK;
}

void seabraid_ggml_reset_transfer_stats(void) {
    zero(&totals.sent_bytes);
    zero(&totals.sent_transfers);
    zero(&totals.returned_bytes);
    zero(&totals.returned_transfers);
    zero(&totals.batches);
}

size_t seabraid_ggml_stats_summary(char *buffer, size_t size) {
    seabraid_ggml_transfer_stats now = {SEABRAID_ABI_VERSION, sizeof(now)};
    if (!buffer || !size || seabraid_ggml_get_transfer_stats(&now) != SEABRAID_OK || now.batches == 0) {
        return 0;
    }
    const int written = snprintf(buffer, size,
                                 "transfers: %llu batches, %llu bytes sent in %llu uploads, %llu bytes returned in %llu downloads",
                                 (unsigned long long) now.batches, (unsigned long long) now.sent_bytes,
                                 (unsigned long long) now.sent_transfers, (unsigned long long) now.returned_bytes,
                                 (unsigned long long) now.returned_transfers);
    return written > 0 && (size_t) written < size ? (size_t) written : 0;
}
