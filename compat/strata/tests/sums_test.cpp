// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/sums_test.cpp - the sums tool's logic (strata-ext/src/sums_build.hpp) on a synthetic two-shard pack: the manifest it builds
// holds the XXH3-64 of every expert's reference blob; it survives serialize / write_atomic / parse; a flipped byte shows as exactly
// that cell in --verify; a changed file makes the manifest stale (matches_files); an unaddressable or unreadable cell stops the build
// and names (layer, expert); the write is a rename of a fresh exclusive temp (none left behind, and a link planted at the old fixed
// `.tmp` name is never written through).  The stamps are the build's own descriptors' (FileTable::stamps).  CPU only, no Strata (the Strata half, FileExpertSource, is
// reached through the ExtentsOf callback, which here is a fixed table).
#include "file_pack_fixture.hpp"
#include "sums_build.hpp"

#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <iterator>
#include <set>
#include <sys/stat.h>

using namespace seabraid;
using namespace seabraid::test;

static int g_fail = 0;
#define CHECK(cond, what)                                                                       \
    do {                                                                                        \
        if (!(cond)) { std::printf("FAIL: %s (%s:%d)\n", (what), __FILE__, __LINE__); ++g_fail; } \
    } while (0)

static constexpr uint32_t L = 2, E = 3, kBlob = 64 + 96 + 32;
static constexpr uint64_t S0 = 3000, S1 = 2000;

// Where cell (l, e) lives: three role slices, the first and last in shard 0, the middle in shard 1 (the GGUF-in-place shape).
static bool where(int64_t l, int64_t e, FileExtent* out, int& n) {
    const uint32_t c = (uint32_t) (l * E + e);
    out[0] = {0, 64, c * 64};
    out[1] = {1, 96, c * 96};
    out[2] = {0, 32, 1000 + c * 32};
    n = 3;
    return true;
}

static std::vector<uint64_t> reference(const Pack& pack) {
    std::vector<uint64_t> sums;
    for (uint32_t l = 0; l < L; ++l)
        for (uint32_t e = 0; e < E; ++e) {
            FileExtent ex[3];
            int n = 0;
            where(l, e, ex, n);
            const std::vector<uint8_t> b = pack.blob_of(ex, n);
            sums.push_back(blob_sum(b.data(), b.size()));
        }
    return sums;
}

struct Ctx {
    Pack pack;
    FileTable files;
    std::vector<uint8_t> staging = std::vector<uint8_t>(256);
    const std::vector<uint32_t> blob = std::vector<uint32_t>(L, kBlob);
    std::vector<uint64_t> sums;
    sums::Manifest manifest, parsed;
    std::string err, why;
    std::vector<std::string> bad;
};

// True when `dir` holds exactly `names` (plus . and ..).
static bool only_these(const std::string& dir, std::set<std::string> names) {
    DIR* d = ::opendir(dir.c_str());
    if (d == nullptr) return false;
    bool ok = true;
    for (dirent* e = ::readdir(d); e != nullptr; e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n != "." && n != "..") ok = names.erase(n) == 1 && ok;
    }
    ::closedir(d);
    return ok && names.empty();
}

// SOLFIX 2: a link planted at the old fixed temp name (`expert-sums.v1.tmp`) must not redirect the write into another file, and the
// manifest must land as a regular file of its own.
static void t_planted_temp_link(Ctx& c) {
    const std::string path = c.pack.dir + "/" + sums::kFileName, victim = c.pack.dir + "/victim.txt", link = path + ".tmp";
    std::FILE* f = std::fopen(victim.c_str(), "wb");
    CHECK(f != nullptr && std::fputs("VICTIM", f) >= 0 && std::fclose(f) == 0, "make the victim");
    CHECK(::symlink(victim.c_str(), link.c_str()) == 0, "plant the link");
    CHECK(sums_build::write_atomic(path, "MANIFEST", c.err), "write_atomic");
    std::ifstream v(victim, std::ios::binary);
    const std::string kept((std::istreambuf_iterator<char>(v)), std::istreambuf_iterator<char>());
    CHECK(kept == "VICTIM", "the planted link's target is untouched");
    struct stat st{};
    CHECK(::lstat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_size == 8, "the manifest is a regular file holding the bytes");
    CHECK((st.st_mode & 0777) == 0644, "readable by the engine and the helper (0644)");
    CHECK(only_these(c.pack.dir, {"shard0.bin", "shard1.bin", sums::kFileName, "victim.txt", sums::kFileName + std::string(".tmp")}),
          "and no temp file of its own is left");
    ::unlink(victim.c_str());
    ::unlink(link.c_str());
}

static bool setup(Ctx& c) {
    std::string warn;
    if (!c.pack.make({S0, S1}, "sums")) { std::printf("FAIL: cannot make the pack dir\n"); return false; }
    if (!c.files.open_all(c.pack.paths, kFileIoBuffered, c.staging.data(), (int64_t) c.staging.size(), c.err, warn)) { std::printf("FAIL: open: %s\n", c.err.c_str()); return false; }
    return true;
}

static void t_build(Ctx& c) {
    CHECK(sums_build::build_sums(c.files, where, L, E, c.blob, c.sums, c.err), "the build reads every cell");
    CHECK(c.sums == reference(c.pack), "every sum is the XXH3-64 of the reference blob, from the in-memory bytes");
}

static void t_roundtrip(Ctx& c) {
    c.manifest.layers = L; c.manifest.experts = E; c.manifest.blob_bytes = c.blob; c.manifest.sums = c.sums;
    c.manifest.files = c.files.stamps();   // the stamps of the descriptors the build read (what seabraid-sums writes)
    std::string bytes;
    CHECK(sums::serialize(c.manifest, bytes, c.err), "serialize");
    const std::string path = c.pack.dir + "/" + sums::kFileName;
    CHECK(sums_build::write_atomic(path, bytes, c.err), "write_atomic");
    std::ifstream in(path, std::ios::binary);
    const std::string back((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(back == bytes, "the file holds the manifest");
    CHECK(only_these(c.pack.dir, {"shard0.bin", "shard1.bin", sums::kFileName}), "no temp file is left behind");
    CHECK(sums::parse((const uint8_t*) back.data(), back.size(), c.parsed, c.err), "it parses");
    CHECK(c.parsed.sums == c.sums && c.parsed.blob_bytes == c.blob, "back to the same sums");
}

static void t_verify(Ctx& c) {
    CHECK(sums_build::verify_sums(c.files, where, c.parsed, c.bad, c.err), "verify: the pack matches its manifest");
    CHECK(c.bad.empty(), "and nothing is reported");
    CHECK(sums::matches_files(c.parsed, c.manifest.files.data(), c.manifest.files.size(), c.why), "the files are fresh");
    const uint32_t wrong[L] = {kBlob, kBlob + 1};
    CHECK(!sums::matches_geometry(c.parsed, L, E, wrong, c.why), "a manifest whose layer blob size differs from the engine's is stale");
    // one byte of cell (1, 2)'s middle slice flipped after the manifest was built
    c.pack.corrupt(1, (1 * E + 2) * 96 + 5);
    CHECK(!sums_build::verify_sums(c.files, where, c.parsed, c.bad, c.err), "a flipped byte makes verify fail");
    CHECK(c.bad.size() == 1 && c.bad[0] == "layer 1 expert 2", "naming exactly the cell whose byte changed");
}

static void t_stale(Ctx& c) {
    c.pack.bytes[0].push_back(0);   // the shard grew
    c.pack.write_file(0);
    std::vector<sums::FileStat> now;
    CHECK(sums::stamp_files(c.pack.paths, now, c.err), "stat again (the engine's check at start)");
    CHECK(!sums::matches_files(c.parsed, now.data(), now.size(), c.why), "a file that grew makes the manifest stale");
    CHECK(c.why.find("file 0") != std::string::npos, "naming the file");
}

static void t_unreadable(Ctx& c) {
    auto missing = [](int64_t l, int64_t e, FileExtent* out, int& n) {
        if (l == 0 && e == 1) { n = 0; return false; }   // the engine cannot place this one
        return where(l, e, out, n);
    };
    CHECK(!sums_build::build_sums(c.files, missing, L, E, c.blob, c.sums, c.err), "a cell the engine cannot place stops the build");
    CHECK(c.err.find("layer 0 expert 1") != std::string::npos, "named");
    auto past_end = [](int64_t, int64_t, FileExtent* out, int& n) { out[0] = {1, 192, 1900}; n = 1; return true; };
    CHECK(!sums_build::build_sums(c.files, past_end, L, E, c.blob, c.sums, c.err), "an extent past the end of its file stops the build");
    CHECK(c.err.find("layer 0 expert 0") != std::string::npos, "named");
    CHECK(c.err.find("unreadable") != std::string::npos, "as unreadable");
}

int main() {
    Ctx c;
    if (!setup(c)) return 1;
    t_build(c);
    t_roundtrip(c);
    t_verify(c);
    t_stale(c);
    t_unreadable(c);
    t_planted_temp_link(c);
    c.pack.manifest = c.pack.dir + "/" + sums::kFileName;   // the fixture removes it with the pack
    std::printf(g_fail ? "sums_test: %d FAILED checks\n" : "sums_test: all checks passed\n", g_fail);
    return g_fail != 0;
}
