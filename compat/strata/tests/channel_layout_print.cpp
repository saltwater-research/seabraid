// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/channel_layout_print.cpp - prints the v6 channel's real sizes (built by the same compiler as the helper and main),
// in the format tests/channel_v6_golden.txt stores.  This is the honest SOURCE of the numbers install/_plan_shm.py mirrors:
//   seabraid_layout_print > tests/channel_v6_golden.txt
// Rows: H window k L E blob_max total   (cap = window * k, exactly how the helpers size it)
#include "channel_v10_facts.hpp"
#include "channel_v9_facts.hpp"
#include "strata/seabraid/channel.hpp"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--v8") == 0) {   // the file tests/channel_v8_golden.txt stores: seabraid_layout_print --v8 > it
        std::printf("version %u\nheader_bytes %zu\ntier_wire_op_bytes %zu\ntier_wire_op_blob_slot_offset %zu\ntier_swap_blob %d\nblob_batch %d\n",
                    (unsigned) seabraid::kVersion, sizeof(seabraid::Header), sizeof(seabraid::TierWireOp),
                    offsetof(seabraid::TierWireOp, blob_slot), (int) seabraid::kTierSwapBlob, seabraid::kBlobBatch);
        return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--v9") == 0) {   // the file tests/channel_v9_golden.txt stores: seabraid_layout_print --v9 > it
        for (const auto& f : seabraid::test::v9_facts()) std::printf("%s %lld\n", f.first, f.second);   // the one list channel_v9_test.cpp checks
        return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--v10") == 0) {   // the file tests/channel_v10_golden.txt stores: seabraid_layout_print --v10 > it
        for (const auto& f : seabraid::test::v10_facts()) std::printf("%s %lld\n", f.first, f.second);   // the one list channel_v10_test.cpp checks
        return 0;
    }
    std::printf("header_bytes %zu\n", sizeof(seabraid::Header));
    struct G { int64_t H, window, k, L, E, blob; };
    const G rows[] = {
        {2560, 16, 10, 48, 512, 1510400},  // Qwen IQ2_XS pack (the v3/v4 measurements' geometry)
        {2560, 8, 10, 48, 512, 1510400},
        {64, 4, 8, 4, 8, 256},              // the unit-test geometry
    };
    for (const G& g : rows) {
        const auto l = seabraid::Layout::of(g.H, g.window * g.k, g.L, g.E, g.blob);
        std::printf("%lld %lld %lld %lld %lld %lld %zu\n", (long long) g.H, (long long) g.window, (long long) g.k, (long long) g.L,
                    (long long) g.E, (long long) g.blob, l.total);
    }
    return 0;
}
