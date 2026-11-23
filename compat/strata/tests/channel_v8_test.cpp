// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/channel_v8_test.cpp - the wire v8 additions, pinned to channel_v8_golden.txt (written by `seabraid_layout_print --v8`
// with the real compiler): kTierSwapBlob and TierWireOp::blob_slot, the Header (unchanged, so the shm size is), and the
// refusal of a v7 peer naming both versions.  CPU only.
//   usage: channel_v8_test tests/channel_v8_golden.txt
// Guards: a field moved in TierWireOp shifts every op on the wire; a version that stays 7 lets a v7 helper ignore a blob op and
// ACK it (the bytes never written); a refusal that omits a version leaves an operator guessing which side is old.
#include "strata/seabraid/channel.hpp"

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

static std::map<std::string, long long> read_golden(const char* path) {
    std::map<std::string, long long> m;
    std::ifstream in(path);
    std::string k;
    long long v;
    while (in >> k >> v) m[k] = v;
    return m;
}

// Row `key` of the golden file must equal `actual`.
static void golden_is(const std::map<std::string, long long>& g, const char* key, long long actual, const char* what) {
    const auto it = g.find(key);
    CHECK(it != g.end() && it->second == actual, what);
}

static void t_golden(const char* path) {
    const auto g = read_golden(path);
    CHECK(g.size() == 6, "the golden file has its six rows");
    golden_is(g, "version", (long long) kVersion, "the wire version is the golden's");
    CHECK(kVersion == 11, "wire version is 11 (this golden still pins v8's rows; its version and header_bytes rows follow kVersion)");
    golden_is(g, "header_bytes", (long long) sizeof(Header), "Header is unchanged at v8 (the shm size is)");
    golden_is(g, "tier_wire_op_bytes", (long long) sizeof(TierWireOp), "TierWireOp size");
    CHECK(sizeof(TierWireOp) == 20, "TierWireOp stays 20 bytes");
    golden_is(g, "tier_wire_op_blob_slot_offset", (long long) offsetof(TierWireOp, blob_slot), "blob_slot offset");
    CHECK(offsetof(TierWireOp, blob_slot) == 2, "blob_slot is where pad_ was");
    golden_is(g, "tier_swap_blob", (long long) kTierSwapBlob, "kTierSwapBlob value");
    CHECK(kTierSwapBlob == 3 && kTierSwapBlob != kTierLoad && kTierSwapBlob != kTierSwap, "kTierSwapBlob = 3, distinct");
    golden_is(g, "blob_batch", kBlobBatch, "blob slots per channel");
}

// a v7 peer is refused at connect, and the refusal names both versions
static void t_v7_refused() {
    const std::string name = "/seabraid_v8_" + std::to_string(getpid());
    Channel helper, main_side;
    std::string err;
    CHECK(helper.create(name, 64, 32, 4, 8, 4, 256, err), "create");
    helper.header()->version = 7;
    CHECK(!main_side.connect(name, 64, 32, 4, 8, 4, err), "a v7 helper is refused by a v9 main");
    CHECK(err.find("v7") != std::string::npos && err.find("v11") != std::string::npos, "naming both versions (helper v7, main v11)");
    helper.header()->version = kVersion;
    CHECK(main_side.connect(name, 64, 32, 4, 8, 4, err), "and the right version connects");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: channel_v8_test channel_v8_golden.txt\n"); return 2; }
    t_golden(argv[1]);
    t_v7_refused();
    std::printf(g_fail ? "channel_v8_test: %d FAILED checks\n" : "channel_v8_test: all checks passed\n", g_fail);
    return g_fail != 0;
}
