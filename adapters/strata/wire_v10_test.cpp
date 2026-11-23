// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Wire v10 host fixture (W5; no GPU, no Strata engine): the layout golden, and the refusal of a v9 peer naming both versions.
// v10 appends Header::helper_start (the helper's identity, see peer_watch.hpp); every earlier offset and every Layout total stays.
// usage: strata_wire_v10_test wire_v10_golden.txt      |      strata_wire_v10_test --print > wire_v10_golden.txt
// Guards: a field moved in Header shifts every later read; a version still 9 lets a v9 helper (which never writes helper_start)
// be trusted with an identity of zero; a refusal that omits a version leaves an operator guessing which side is old.
#include "test_support.hpp"

#include "strata/seabraid/channel.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unistd.h>

namespace {
using seabraid::strata_adapter::testing::check;
using namespace seabraid;

std::map<std::string, long long> facts() {
    return {{"version", kVersion},
            {"magic", (long long) kMagic},
            {"header_bytes", (long long) sizeof(Header)},
            {"header_helper_pid_offset", (long long) offsetof(Header, helper_pid)},
            {"header_file_reads_offset", (long long) offsetof(Header, file_reads)},
            {"header_helper_start_offset", (long long) offsetof(Header, helper_start)},
            {"layout_total_qwen", (long long) Layout::of(2560, 160, 48, 512, 1510400).total},   // unchanged since v8: the shm size
            {"layout_total_unit", (long long) Layout::of(64, 32, 4, 8, 256).total}};
}

void golden(const char* path) {
    std::ifstream in(path);
    check(in.good(), "cannot read the golden file (argument 1)");
    std::map<std::string, long long> want;
    std::string key;
    long long value;
    while (in >> key >> value) want[key] = value;
    const auto have = facts();
    check(want.size() == have.size(), "golden has exactly the rows the test prints");
    for (const auto& [k, v] : have) {
        const auto it = want.find(k);
        if (it == want.end() || it->second != v) {
            std::fprintf(stderr, "golden row %s: file %lld, build %lld\n", k.c_str(), it == want.end() ? -1 : it->second, v);
            check(false, "the wire bytes differ from the golden file");
        }
    }
    check(kVersion == 11, "wire version is 11 (v10's layout under the SEABRD1 tag)");
    check(kMagic == 0x31445242414553ull && std::memcmp(&kMagic, "SEABRD1", 7) == 0, "the magic is the bytes SEABRD1");
    check(sizeof(Header) <= 4096, "the header fits the page Layout::of reserves, so the shm size is unchanged");
}

void v9_peer_refused() {
    const std::string name = "/seabraid_wv10_" + std::to_string((long long) ::getpid());
    std::string err;
    Channel helper;
    check(helper.create(name, 64, 32, 4, 8, 4, 256, err), ("create: " + err).c_str());
    Header* hh = helper.header();
    check(hh->version == 11, "a new channel announces v11");
    check(hh->helper_pid == 0 && hh->helper_start == 0, "a new channel has no identity until the helper announces one");
    hh->version = 9;
    Channel main_side;
    check(!main_side.connect(name, 64, 32, 4, 8, 4, err), "connect refuses a v9 helper");
    check(err.find("wrong magic/version") != std::string::npos && err.find("helper v9") != std::string::npos && err.find("main v11") != std::string::npos,
          ("the refusal names both versions: " + err).c_str());
    hh->version = 11;
    check(main_side.connect(name, 64, 32, 4, 8, 4, err), ("connect works once the versions agree: " + err).c_str());
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--print") == 0) {
        for (const auto& [k, v] : facts()) std::printf("%s %lld\n", k.c_str(), v);
        return 0;
    }
    check(argc > 1, "usage: strata_wire_v10_test wire_v10_golden.txt");
    golden(argv[1]);
    v9_peer_refused();
    std::puts("wire v10: all checks passed");
    return 0;
}
