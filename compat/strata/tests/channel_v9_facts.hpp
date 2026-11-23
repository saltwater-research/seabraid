// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// The wire v9 facts channel_v9_golden.txt stores, from the compiler that builds the tests.  ONE list: seabraid_layout_print --v9 writes
// it, channel_v9_test.cpp reads the golden and compares each row with the same list, so the printer and the check cannot disagree.
#include "strata/seabraid/channel.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace seabraid::test {

inline std::vector<std::pair<const char*, long long>> v9_facts() {
    return {
        {"version", kVersion},
        {"header_bytes", (long long) sizeof(Header)},
        {"header_file_io_offset", (long long) offsetof(Header, file_io)},
        {"header_file_count_offset", (long long) offsetof(Header, file_count)},
        {"header_file_table_id_offset", (long long) offsetof(Header, file_table_id)},
        {"header_file_direct_bytes_offset", (long long) offsetof(Header, file_direct_bytes)},
        {"header_file_buffered_bytes_offset", (long long) offsetof(Header, file_buffered_bytes)},
        {"header_file_reads_offset", (long long) offsetof(Header, file_reads)},
        {"tier_wire_op_bytes", (long long) sizeof(TierWireOp)},
        {"file_extent_bytes", (long long) sizeof(FileExtent)},
        {"file_ref_bytes", (long long) sizeof(FileRef)},
        {"file_ref_sum_offset", (long long) offsetof(FileRef, sum)},
        {"file_ref_blob_bytes_offset", (long long) offsetof(FileRef, blob_bytes)},
        {"file_ref_n_extents_offset", (long long) offsetof(FileRef, n_extents)},
        {"file_ref_extent_offset", (long long) offsetof(FileRef, extent)},
        {"tier_swap_file", kTierSwapFile},
        {"tier_bad_sum", kTierBadSum},
        {"file_ref_max_extents", kFileRefMaxExtents},
        {"file_table_max", kFileTableMax},
        {"sum_xxh3_64", kSumXxh3_64},
        {"layout_total_qwen", (long long) Layout::of(2560, 160, 48, 512, 1510400).total},   // unchanged since v8: the shm size
    };
}

}  // namespace seabraid::test
