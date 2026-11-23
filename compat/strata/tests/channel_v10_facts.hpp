// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// The wire v10 facts channel_v10_golden.txt stores, from the compiler that builds the tests.  ONE list: seabraid_layout_print --v10 writes
// it, channel_v10_test.cpp reads the golden and compares each row with the same list, so the printer and the check cannot disagree.
#include "strata/seabraid/channel.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace seabraid::test {

inline std::vector<std::pair<const char*, long long>> v10_facts() {
    return {
        {"version", kVersion},
        {"magic", (long long) kMagic},
        {"header_bytes", (long long) sizeof(Header)},
        {"header_helper_pid_offset", (long long) offsetof(Header, helper_pid)},
        {"header_file_reads_offset", (long long) offsetof(Header, file_reads)},
        {"header_helper_start_offset", (long long) offsetof(Header, helper_start)},
        {"layout_total_qwen", (long long) Layout::of(2560, 160, 48, 512, 1510400).total},   // unchanged since v8: the shm size
    };
}

}  // namespace seabraid::test
