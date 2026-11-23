// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/file_fetch_test.cpp - the helper's reader of a file op (strata-ext/src/file_fetch.cpp), on a synthetic two-shard pack in a
// disk-backed directory with an expert-sums.v1 written for a small model over it.  CPU only, no Strata.
// What it guards: a 3-extent blob assembled from two shards is byte-equal to the reference in BOTH read modes (buffered pread and
// O_DIRECT through a bounce buffer, including an extent that ends inside the file's last block); a file that no longer holds what the
// manifest summed is kTierBadSum; a ref that is not the op's OWN cell's (another expert's extents and that expert's valid sum, or the
// right extents with another sum) is kTierRejected before one byte is read; an unbound table refuses; a stale, foreign or oversized
// manifest is not bound; every bound is refused before one byte is read (the read counter does not move); a file that shrank after
// launch is a short read; a symlink or directory is not opened; a path swapped between the buffered and the O_DIRECT open is refused;
// the stamps are the descriptors' and unchanged() sees a swap; --file-io off / auto / explicit behave as documented.  Each guard is
// sabotaged once (tests/SABOTAGE.md).
#include "file_pack_fixture.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace seabraid;
using namespace seabraid::test;

static int g_fail = 0;
#define CHECK(cond, what)                                                                       \
    do {                                                                                        \
        if (!(cond)) { std::printf("FAIL: %s (%s:%d)\n", (what), __FILE__, __LINE__); ++g_fail; } \
    } while (0)

static constexpr uint64_t S0 = 100003, S1 = 70001;   // neither is a multiple of a block: the last block of each is partial
// A 3-extent blob across two shards: 32-byte aligned offsets (a GGUF tensor's), then one that ends at the very end of file 0.
static const FileExtent kExt[2] = {{0, 96, 4096 + 32}, {1, 160, 32}};
static constexpr int64_t kBlob = 96 + 160 + 224;

static FileExtent tail_extent() { return {0, 224, S0 - 224}; }
static void three(FileExtent* e) { e[0] = kExt[0]; e[1] = kExt[1]; e[2] = tail_extent(); }

// The model the manifest describes: layer 0 = two 480-byte experts (B = the three extents above, C = three others of the same size),
// layer 1 = two 1-byte experts (one in the middle of a block, one the very last byte of file 1).
static Pack::Model model() {
    FileExtent e[3];
    three(e);
    return {{{e[0], e[1], e[2]}, {{0, 96, 8192}, {1, 160, 4000}, {0, 224, 60000}}},
            {{{1, 1, 5}}, {{1, 1, S1 - 1}}}};
}

static int g_skips = 0;   // arms that could not run here (no O_DIRECT on this filesystem); SEABRAID_STRATA_FR_REQUIRE_DIRECT=1 makes one a failure

struct Rig {
    Pack pack;
    FileTable ft;
    Pack::Model m = model();
    std::vector<uint8_t> staging = std::vector<uint8_t>(4096);
    std::string err, warn;
    /// The rig for one arm: false (and counted, with the reason printed) when this mode cannot be honoured on this filesystem.
    bool arm(FileIo mode, const char* name) {
        if (up(mode) && ft.io() == mode) return bound();
        std::printf("SKIP %s arm: %s%s\n", name, err.c_str(), warn.c_str());
        ++g_skips;
        return false;
    }
    bool up(FileIo mode) {
        if (pack.paths.empty() && !pack.make({S0, S1}, "fetch")) { std::printf("FAIL: cannot make the pack dir\n"); ++g_fail; return false; }
        return ft.open_all(pack.paths, mode, staging.data(), (int64_t) staging.size(), err, warn);
    }
    bool bound() {   // the launch step the helper runs after open_all
        if (pack.bind_table(ft, m, err)) return true;
        std::printf("FAIL: bind the manifest: %s\n", err.c_str());
        ++g_fail;
        return false;
    }
};

// ---- the reference round trip, in each mode
static void rt_blob(Rig& r, FileIo mode) {
    std::string why;
    CHECK(r.ft.fetch(0, 0, r.pack.ref_of(r.m, 0, 0), kBlob, why) == kTierApplied, "a three-extent blob across two shards is fetched");
    FileExtent e[3];
    three(e);
    const std::vector<uint8_t> want = r.pack.blob_of(e, 3);
    CHECK(std::memcmp(r.ft.staging(), want.data(), want.size()) == 0, "staging holds the reference blob byte for byte");
    CHECK(r.ft.reads() == 3, "one read per extent");
    const uint64_t mine = mode == kFileIoDirect ? r.ft.direct_bytes() : r.ft.buffered_bytes();
    const uint64_t other = mode == kFileIoDirect ? r.ft.buffered_bytes() : r.ft.direct_bytes();
    CHECK(mine == (uint64_t) kBlob, "the bytes are counted in the mode that read them, extent bytes only");
    CHECK(other == 0, "and none in the other");
}

// an odd single-byte extent inside a block, and one that is exactly the last byte of the file
static void rt_edges(Rig& r) {
    std::string why;
    CHECK(r.ft.fetch(1, 0, r.pack.ref_of(r.m, 1, 0), 1, why) == kTierApplied, "a one-byte extent in the middle of a block is fetched");
    CHECK(r.ft.staging()[0] == r.pack.bytes[1][5], "and is that byte");
    CHECK(r.ft.fetch(1, 1, r.pack.ref_of(r.m, 1, 1), 1, why) == kTierApplied, "the last byte of the file (end of file inside the block) is fetched");
    CHECK(r.ft.staging()[0] == r.pack.bytes[1][S1 - 1], "and is that byte");
}

static void rt_header(Rig& r, FileIo mode) {
    Header h{};
    r.ft.announce(h);
    CHECK(h.file_io == (int32_t) mode && h.file_count == 2, "announce: mode and file count");
    CHECK(h.file_table_id == file_table_id(r.pack.sizes().data(), 2), "announce: the table id of the launch sizes");
    CHECK(h.file_reads == r.ft.reads() && h.file_direct_bytes == r.ft.direct_bytes(), "announce mirrors the counters");
    CHECK(h.file_buffered_bytes == r.ft.buffered_bytes(), "all three");
}

static void t_round_trip(FileIo mode, const char* name) {
    Rig r;
    if (!r.arm(mode, name)) return;
    rt_blob(r, mode);
    rt_edges(r);
    rt_header(r, mode);
}

// ---- SOLFIX 4: the sum and the extents a file op is checked against are the OP's cell's, from the helper's own manifest and layout
static void t_cell_binding(FileIo mode, const char* name) {
    Rig r;
    if (!r.arm(mode, name)) return;
    const FileRef c_ref = r.pack.ref_of(r.m, 0, 1);   // expert C: same layer, same size, its own valid manifest sum
    const uint64_t reads = r.ft.reads();
    std::string why;
    CHECK(r.ft.fetch(0, 0, c_ref, kBlob, why) == kTierRejected, "B's op carrying C's extents and C's valid sum is refused (never C's bytes into B)");
    CHECK(why.find("extents") != std::string::npos, "naming the extents");
    FileRef b_with_c_sum = r.pack.ref_of(r.m, 0, 0);
    b_with_c_sum.sum = c_ref.sum;
    CHECK(r.ft.fetch(0, 0, b_with_c_sum, kBlob, why) == kTierRejected, "B's extents with another expert's sum are refused");
    CHECK(why.find("manifest sum") != std::string::npos, "naming the sum");
    CHECK(r.ft.reads() == reads, "neither read one byte");
    CHECK(r.ft.fetch(0, 7, c_ref, kBlob, why) == kTierRejected && r.ft.reads() == reads, "a cell the manifest does not have is refused");
    CHECK(r.ft.fetch(0, 1, c_ref, kBlob, why) == kTierApplied, "C's ref on C's own op is served");
}

static void t_unbound(FileIo mode, const char* name) {
    Rig r;
    if (!r.up(mode) || r.ft.io() != mode) { std::printf("SKIP unbound %s arm\n", name); ++g_skips; return; }
    std::string why;
    CHECK(r.ft.fetch(0, 0, r.pack.ref_of(r.m, 0, 0), kBlob, why) == kTierRejected && r.ft.reads() == 0, "a table with no manifest bound refuses every file op");
    CHECK(why.find("expert-sums") != std::string::npos, "and says it has no manifest");
}

// A manifest that is not about these files or this model is never bound; neither is one longer than any manifest can be.
static void t_manifest_refusals() {
    Rig r;
    if (!r.up(kFileIoBuffered)) { std::printf("FAIL: open\n"); ++g_fail; return; }
    std::string why;
    CHECK(r.pack.write_manifest(r.m), "write the manifest");
    const auto blob_bytes = r.pack.layer_bytes(r.m);
    CHECK(!r.ft.bind_manifest(r.pack.manifest, 2, 3, blob_bytes, Pack::where(r.m), why) && why.find("stale") != std::string::npos, "another geometry is not bound");
    std::vector<uint32_t> other = blob_bytes;
    other[1] += 1;
    CHECK(!r.ft.bind_manifest(r.pack.manifest, 2, 2, other, Pack::where(r.m), why), "another blob size is not bound");
    CHECK(!r.ft.bind_manifest(r.pack.manifest, 2, 2, blob_bytes, FileTable::ExtentsOf{}, why), "a table with no pack layout is not bound");
    CHECK(r.ft.fetch(0, 0, r.pack.ref_of(r.m, 0, 0), kBlob, why) == kTierRejected, "and a refused bind leaves the table refusing");
    CHECK(::truncate(r.pack.manifest.c_str(), (off_t) (64ull << 20)) == 0, "grow the manifest to a sparse 64 MB file");
    CHECK(!r.ft.bind_manifest(r.pack.manifest, 2, 2, blob_bytes, Pack::where(r.m), why), "an oversized manifest is not bound");
    std::vector<uint8_t> bytes;
    CHECK(sums::read_manifest(r.pack.manifest, bytes, why) && bytes.size() == sums::kMaxManifestBytes + 1, "and the read stopped one byte past the longest manifest (SOLFIX 5)");
    // stale: the manifest was built before a file changed, and this table opened the changed file
    Rig s;
    CHECK(s.up(kFileIoOff), "pack");
    CHECK(s.pack.write_manifest(s.m), "manifest first");
    s.pack.bytes[1].push_back(7);
    CHECK(s.pack.write_file(1), "then shard 1 grows");
    CHECK(s.ft.open_all(s.pack.paths, kFileIoBuffered, s.staging.data(), 4096, s.err, s.warn), "open");
    CHECK(!s.ft.bind_manifest(s.pack.manifest, 2, 2, s.pack.layer_bytes(s.m), Pack::where(s.m), why) && why.find("file 1 changed") != std::string::npos,
          "a manifest older than the files opened is stale and not bound");
}

// ---- a file that no longer holds what the manifest summed
static void t_corrupt(FileIo mode, const char* name) {
    Rig r;
    if (!r.arm(mode, name)) return;
    const FileRef ref = r.pack.ref_of(r.m, 0, 0);   // main's ref: taken from the good bytes, like the manifest
    r.pack.corrupt(1, 100);   // one byte inside extent 1 (file 1, 32..192) flipped after the manifest was built
    std::string why;
    CHECK(r.ft.fetch(0, 0, ref, kBlob, why) == kTierBadSum, "one flipped byte in extent 1 is kTierBadSum, not an applied blob");
    CHECK(why.find("sum") != std::string::npos, "and the reason says it is the sum");
}

// ---- every bound is refused before a byte is read: by fetch (it is not the cell's ref) and by read (file_ref_valid)
static void refused(Rig& r, FileRef ref, int64_t want, const char* what) {
    const uint64_t reads = r.ft.reads();
    std::string why;
    CHECK(!r.ft.read(ref, want, why) && !why.empty(), what);
    CHECK(r.ft.fetch(0, 0, ref, want, why) == kTierRejected, "and a file op carrying it is refused");
    CHECK(r.ft.reads() == reads, "nothing was read for it");
}

static void t_bounds(FileIo mode, const char* name) {
    Rig r;
    if (!r.arm(mode, name)) return;
    const FileRef good = r.pack.ref_of(r.m, 0, 0);
    FileRef b = good;
    b.extent[1].offset = S1 - 100;                                       refused(r, b, kBlob, "offset + length past the end of the file");
    b = good; b.extent[2].offset = UINT64_MAX - 10;                      refused(r, b, kBlob, "an offset whose sum with the length would wrap");
    b = good; b.extent[0].file = 2;                                      refused(r, b, kBlob, "a file index outside the table");
    b = good; b.extent[2].file = 0xFFFFFFFFu;                            refused(r, b, kBlob, "a huge file index");
    b = good; b.n_extents = 4;                                           refused(r, b, kBlob, "more than three extents");
    b = good; b.n_extents = 0;                                           refused(r, b, kBlob, "no extents");
    b = good; b.blob_bytes += 1;                                         refused(r, b, kBlob, "extent lengths that do not sum to blob_bytes");
    refused(r, good, kBlob + 1, "a blob size that is not the layer's own");
    b = good; b.extent[1].length = 0; b.blob_bytes -= 160;               refused(r, b, kBlob - 160, "a zero-length extent");
    b = good; b.ref_bytes = 64;                                          refused(r, b, kBlob, "a FileRef of another size");
    b = good; b.sum_kind = 2;                                            refused(r, b, kBlob, "another sum kind");
    FileExtent e[3];
    three(e);
    e[0].length = 5000;
    const FileRef big = r.pack.ref_for(e, 3);                            refused(r, big, 5000 + 160 + 224, "a blob bigger than the staging buffer");
    std::string why;
    CHECK(r.ft.fetch(0, 0, good, kBlob, why) == kTierApplied, "and the good ref is still served after all the refusals");
}

// ---- the file shrank after launch
static void t_shrunk(FileIo mode, const char* name) {
    Rig r;
    if (!r.arm(mode, name)) return;
    CHECK(::truncate(r.pack.paths[0].c_str(), (off_t) (S0 / 2)) == 0, "truncate the shard");
    std::string why;
    CHECK(r.ft.fetch(0, 0, r.pack.ref_of(r.m, 0, 0), kBlob, why) == kTierRejected, "a shard that shrank after launch is a rejected short read");
    CHECK(why.find("short") != std::string::npos && why.find("file 0") != std::string::npos, "naming the file and that it was short");
}

// ---- SOLFIX 3: one inode per table entry, and the stamps are the descriptors'
static std::string g_swap_from;   // the hook renames this file over the path between the two opens
static void swap_in(const std::string& path) {
    if (!g_swap_from.empty()) ::rename(g_swap_from.c_str(), path.c_str());
    g_swap_from.clear();
}

static void t_two_opens() {
    Rig probe;
    if (!probe.up(kFileIoDirect) || probe.ft.io() != kFileIoDirect) { std::printf("SKIP two-opens arm: no O_DIRECT here\n"); ++g_skips; return; }
    Rig r;
    CHECK(r.up(kFileIoOff), "pack");
    g_swap_from = r.pack.dir + "/other.bin";
    std::FILE* f = std::fopen(g_swap_from.c_str(), "wb");
    const std::vector<uint8_t> junk(S0, 0xA5);
    CHECK(f != nullptr && std::fwrite(junk.data(), 1, junk.size(), f) == junk.size() && std::fclose(f) == 0, "make another file");
    g_between_opens_for_test = swap_in;
    std::string err, warn;
    FileTable t;
    const bool opened = t.open_all(r.pack.paths, kFileIoDirect, r.staging.data(), 4096, err, warn);
    g_between_opens_for_test = nullptr;
    CHECK(!opened && err.find("changed between its two opens") != std::string::npos, "a path swapped between the buffered and the direct open is refused");
}

static void t_stamps_are_the_descriptors() {
    Rig r;
    CHECK(r.up(kFileIoBuffered), "open");
    std::vector<sums::FileStat> at_open;
    std::string why;
    CHECK(sums::stamp_files(r.pack.paths, at_open, why) && r.ft.stamps().size() == 2, "stamp");
    CHECK(r.ft.stamps()[1].size == at_open[1].size && r.ft.stamps()[1].mtime_ns == at_open[1].mtime_ns, "the table's stamp is the file it opened");
    CHECK(r.ft.unchanged(r.pack.paths, why), "an untouched pack is unchanged");
    const std::string other = r.pack.dir + "/other.bin";
    std::FILE* f = std::fopen(other.c_str(), "wb");
    CHECK(f != nullptr && std::fputs("a different, shorter file", f) >= 0 && std::fclose(f) == 0, "make another file");
    CHECK(::rename(other.c_str(), r.pack.paths[1].c_str()) == 0, "swap it in under shard 1's path");
    CHECK(r.ft.stamps()[1].size == S1, "the stamp still describes the bytes the table reads, not the new path");
    CHECK(!r.ft.unchanged(r.pack.paths, why) && why.find("file 1 was replaced") != std::string::npos, "and unchanged() sees the swap");
    Rig g;
    CHECK(g.up(kFileIoBuffered), "open");
    g.pack.bytes[0][9] ^= 1;
    CHECK(g.pack.write_file(0), "rewrite shard 0 in place");
    const struct timespec old_times[2] = {{1, 0}, {1, 0}};   // a coarse clock could give the rewrite the launch mtime: make it differ
    CHECK(::utimensat(AT_FDCWD, g.pack.paths[0].c_str(), old_times, 0) == 0, "and its mtime moves");
    CHECK(!g.ft.unchanged(g.pack.paths, why) && why.find("file 0 changed") != std::string::npos, "a file rewritten while read is changed");
}

// ---- launch: modes, and the files that are never opened
static void launch_off() {
    Rig r;
    CHECK(r.up(kFileIoOff), "--file-io off opens");
    CHECK(r.ft.io() == kFileIoOff && r.ft.count() == 0, "nothing");
    std::string why;
    CHECK(r.ft.fetch(0, 0, r.pack.ref_of(r.m, 0, 0), kBlob, why) == kTierRejected, "and refuses every ref");
    CHECK(why.find("file_io 0") != std::string::npos, "naming file_io 0");
    CHECK(!r.ft.bind_manifest(r.pack.dir + "/none", 2, 2, {480, 1}, Pack::where(r.m), why), "an off table binds nothing");
    Header h{};
    r.ft.announce(h);
    CHECK(h.file_io == 0 && h.file_count == 0 && h.file_table_id == 0, "an off table announces file_io 0, no files");
}

static void launch_refusals() {
    Rig m;
    CHECK(m.up(kFileIoOff), "pack");
    std::vector<std::string> paths = m.pack.paths;
    paths.push_back(m.pack.dir + "/missing.bin");
    std::string err, warn;
    FileTable a;
    CHECK(a.open_all(paths, kFileIoAuto, m.staging.data(), 4096, err, warn), "auto with a missing file is not a launch failure");
    CHECK(a.io() == kFileIoOff && !warn.empty(), "it serves file_io 0 and says why");
    FileTable b;
    CHECK(!b.open_all(paths, kFileIoBuffered, m.staging.data(), 4096, err, warn), "explicit buffered with a missing file fails");
    CHECK(err.find("missing.bin") != std::string::npos, "and names it");
    const std::string link = m.pack.dir + "/link.bin";
    CHECK(::symlink(m.pack.paths[0].c_str(), link.c_str()) == 0, "make a symlink");
    FileTable c;
    CHECK(!c.open_all({link}, kFileIoBuffered, m.staging.data(), 4096, err, warn), "a symlink is not followed (O_NOFOLLOW)");
    CHECK(err.find("cannot open") != std::string::npos, "it is not opened");
    FileTable d;
    CHECK(!d.open_all({m.pack.dir}, kFileIoBuffered, m.staging.data(), 4096, err, warn), "a directory is not a pack file");
    CHECK(err.find("regular") != std::string::npos, "it says so");
    ::unlink(link.c_str());
    FileTable g;
    CHECK(!g.open_all(m.pack.paths, kFileIoGds, m.staging.data(), 4096, err, warn), "gds is refused at launch in this build");
    FileTable s;
    CHECK(!s.open_all(m.pack.paths, kFileIoBuffered, nullptr, 0, err, warn), "a table with no staging buffer is refused");
}

static void t_launch() {
    launch_off();
    launch_refusals();
}

static void t_sum_and_table_id() {
    CHECK(blob_sum((const uint8_t*) "", 0) == 0x2D06800538D394C2ull, "blob_sum is XXH3-64 seed 0 (the empty-input vector)");
    const uint64_t one[2] = {1, 2}, two[2] = {2, 1};
    CHECK(file_table_id(one, 2) != file_table_id(two, 2) && file_table_id(one, 2) == file_table_id(one, 2), "the table id depends on the sizes and their order");
    const uint8_t le[16] = {1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0};
    CHECK(file_table_id(one, 2) == blob_sum(le, 16), "and is the sum of the sizes as little-endian u64s (what main computes)");
}

int main() {
    t_sum_and_table_id();
    t_launch();
    t_manifest_refusals();
    t_stamps_are_the_descriptors();
    t_two_opens();
    for (const auto& arm : {std::make_pair(kFileIoBuffered, "buffered"), std::make_pair(kFileIoDirect, "direct")}) {
        t_round_trip(arm.first, arm.second);
        t_cell_binding(arm.first, arm.second);
        t_unbound(arm.first, arm.second);
        t_corrupt(arm.first, arm.second);
        t_bounds(arm.first, arm.second);
        t_shrunk(arm.first, arm.second);
    }
    if (g_skips != 0 && std::getenv("SEABRAID_STRATA_FR_REQUIRE_DIRECT") != nullptr) { std::printf("FAIL: %d arms skipped and SEABRAID_STRATA_FR_REQUIRE_DIRECT is set\n", g_skips); ++g_fail; }
    std::printf(g_fail ? "file_fetch_test: %d FAILED checks\n" : "file_fetch_test: all checks passed\n", g_fail);
    return g_fail != 0;
}
