// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/channel_v10_test.cpp - the wire v10 bytes (Header::helper_start, the helper's identity), pinned to channel_v10_golden.txt
// (`seabraid_layout_print --v10 > channel_v10_golden.txt`, written by the real compiler), the identity a helper publishes, and the
// refusal of a v9 peer naming both versions.  CPU only.   usage: channel_v10_test tests/channel_v10_golden.txt
// Guards: a field moved in the Header shifts every later read; a version that stays 9 lets a v9 helper (which never writes
// helper_start) be trusted with an identity of zero; an identity written after kHelperUp is read as zero.
#include "channel_v10_facts.hpp"
#include "strata/seabraid/peer_watch.hpp"

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
    const auto facts = test::v10_facts();
    CHECK(g.size() == facts.size(), "the golden has one row per fact");
    for (const auto& f : facts) {
        const auto it = g.find(f.first);
        char what[160];
        std::snprintf(what, sizeof what, "golden row %s equals the compiler's value", f.first);
        CHECK(it != g.end() && it->second == f.second, what);
    }
    CHECK(kVersion == 11 && sizeof(Header) == 2752 && sizeof(Header) <= 4096, "the v11 constants (the Header stays under the page Layout::of reserves)");
    CHECK(kMagic == 0x31445242414553ull && std::memcmp(&kMagic, "SEABRD1", 7) == 0, "the magic is the bytes SEABRD1");
}

// the identity a helper publishes, and a v9 peer refused at connect naming both versions
static void t_identity_and_v9_refused() {
    const std::string name = "/seabraid_v10_" + std::to_string(getpid());
    Channel helper, main_side;
    std::string err;
    CHECK(helper.create(name, 64, 32, 4, 8, 4, 256, err), "create");
    Header* h = helper.header();
    CHECK(h->helper_pid == 0 && h->helper_start == 0, "a new channel carries no identity");
    announce_identity(*h);
    uint64_t start = 0;
    CHECK(h->helper_pid == (int32_t) getpid() && read_starttime(getpid(), start) && h->helper_start == start && start != 0,
          "announce_identity writes this process's (pid, starttime)");
    h->version = 9;
    CHECK(!main_side.connect(name, 64, 32, 4, 8, 4, err), "a v9 helper is refused by a v11 main");
    CHECK(err.find("magic/version") != std::string::npos && err.find("v9") != std::string::npos && err.find("v11") != std::string::npos,
          "naming both versions (helper v9, main v11)");
    h->version = kVersion;
    CHECK(main_side.connect(name, 64, 32, 4, 8, 4, err), "and the right version connects");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: channel_v10_test channel_v10_golden.txt\n"); return 2; }
    t_golden(argv[1]);
    t_identity_and_v9_refused();
    std::printf(g_fail ? "channel_v10_test: %d FAILED checks\n" : "channel_v10_test: all checks passed\n", g_fail);
    return g_fail != 0;
}
