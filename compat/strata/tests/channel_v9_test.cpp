// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/channel_v9_test.cpp - the wire v9 bytes (kTierSwapFile, FileRef, the Header's file_* fields), pinned to channel_v9_golden.txt
// (`seabraid_layout_print --v9 > channel_v9_golden.txt`, written by the real compiler), and the refusal of a v8 peer naming both
// versions.  CPU only.   usage: channel_v9_test tests/channel_v9_golden.txt
// Guards: a field moved in FileRef or the Header shifts every file move; a version that stays 8 lets a v8 helper ACK a kind it ignores
// (the bytes never read); a refusal that omits a version leaves an operator guessing which side is old.
#include "channel_v9_facts.hpp"

#include <cstdio>
#include <fstream>
#include <map>
#include <string>

#include <unistd.h>

using namespace seabraid;

static int g_fail = 0;
#define CHECK(cond, what)                                                                       \
    do {                                                                                        \
        if (!(cond)) { std::printf("FAIL: %s (%s:%d)\n", (what), __FILE__, __LINE__); ++g_fail; } \
    } while (0)

static void t_golden(const char* path) {
    std::map<std::string, long long> g;
    std::ifstream in(path);
    std::string k;
    long long v;
    while (in >> k >> v) g[k] = v;
    const auto facts = test::v9_facts();
    CHECK(g.size() == facts.size(), "the golden has one row per fact");
    for (const auto& f : facts) {
        const auto it = g.find(f.first);
        char what[160];
        std::snprintf(what, sizeof what, "golden row %s equals the compiler's value", f.first);
        CHECK(it != g.end() && it->second == f.second, what);
    }
    CHECK(kVersion == 11 && kTierSwapFile == 4 && kTierBadSum == 4 && sizeof(FileRef) == 72 && sizeof(FileExtent) == 16, "the v9 constants");
    CHECK(sizeof(TierWireOp) == 20 && sizeof(Header) == 2752, "TierWireOp is v8's 20 bytes; the Header is v9's plus the v10 helper_start (the shm layout totals are unchanged)");
}

// a v8 peer is refused at connect, and the refusal names both versions
static void t_v8_refused() {
    const std::string name = "/seabraid_v9_" + std::to_string(getpid());
    Channel helper, main_side;
    std::string err;
    CHECK(helper.create(name, 64, 32, 4, 8, 4, 256, err), "create");
    helper.header()->version = 8;
    CHECK(!main_side.connect(name, 64, 32, 4, 8, 4, err), "a v8 helper is refused by a v9 main");
    CHECK(err.find("v8") != std::string::npos && err.find("v11") != std::string::npos, "naming both versions (helper v8, main v11)");
    helper.header()->version = kVersion;
    CHECK(main_side.connect(name, 64, 32, 4, 8, 4, err), "and the right version connects");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: channel_v9_test channel_v9_golden.txt\n"); return 2; }
    t_golden(argv[1]);
    t_v8_refused();
    std::printf(g_fail ? "channel_v9_test: %d FAILED checks\n" : "channel_v9_test: all checks passed\n", g_fail);
    return g_fail != 0;
}
