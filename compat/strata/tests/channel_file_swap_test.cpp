// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/channel_file_swap_test.cpp - a kTierSwapFile served over a REAL shared-memory channel: main writes a FileRef into blob slot 0
// and submits one op; an in-process helper (its own thread, its own mapping) runs serve_tier_channel against a fake cache and a real
// FileTable over a synthetic two-shard pack.  CPU only; no Strata.
// What it guards (finish line 1a, 1b, 1d): the bytes the helper reads from the FILES land in the displaced expert's slot byte for
// byte (buffered and O_DIRECT), held[] moves; a file that no longer holds what main summed is kTierBadSum with the slot, held[] and
// the cache untouched; a ref naming ANOTHER cell (B's op, C's extents and C's valid sum: SOLFIX 4) is kTierRejected with no read and
// no slot write, because the helper checks it against its own manifest for B; every bad ref (bounds, index, >3 extents, length mismatch) is kTierRejected before the cache is called; a round
// holding a file op and anything else is refused whole; a helper with no file table refuses; a slot that does not read back is
// kTierFailed.  Each guard sabotaged once (tests/SABOTAGE.md).
#include "file_pack_fixture.hpp"
#include "tier_round_fake.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unistd.h>

using namespace seabraid;
using namespace seabraid::test;

static int g_fail = 0, g_skips = 0;   // arms that could not run here; SEABRAID_STRATA_FR_REQUIRE_DIRECT=1 makes a skip a failure
#define CHECK(cond, what)                                                                       \
    do {                                                                                        \
        if (!(cond)) { std::printf("FAIL: %s (%s:%d)\n", (what), __FILE__, __LINE__); ++g_fail; } \
    } while (0)

static constexpr int64_t H = 64, CAP = 32, NL = 4, NE = 8, BLOB_MAX = 1024, kBlob = 480;
static constexpr uint32_t WMT = 4;
static constexpr uint64_t S0 = 100003, S1 = 70001;

static const Pack::Model kModel = three_slice_model((int) NL, (int) NE);   // every cell 480 bytes, three slices over two shards

// Did the helper report a failure whose text contains this?  (The real helper copies the first into tier_err; here Reports keeps them all.)
static bool said(const Reports& rep, const char* needle) {
    for (const std::string& w : rep.why)
        if (w.find(needle) != std::string::npos) return true;
    return false;
}

struct Rig {
    Channel helper, main_side;
    FakeCache cache;
    Reports rep;
    Pack pack;
    FileTable files;
    std::vector<uint8_t> staging = std::vector<uint8_t>(BLOB_MAX);
    std::atomic<bool> stop{false};
    std::thread th;
    bool serve_files = true;   // false: the helper has no file table (the 4-argument serve)
    std::string name;
    static void loop(Rig* r) {
        while (!r->stop.load()) {
            const uint64_t seq = r->helper.tier_pending();
            if (seq == 0) { usleep(200); continue; }
            if (r->serve_files) serve_tier_channel(r->helper, seq, r->cache, r->rep, r->files);
            else serve_tier_channel(r->helper, seq, r->cache, r->rep);
        }
    }
    bool up(FileIo mode, const char* tag) {
        name = std::string("/seabraid_fileswap_") + tag + "_" + std::to_string(getpid());
        std::string err, warn;
        if (!pack.make({S0, S1}, tag) || !files.open_all(pack.paths, mode, staging.data(), BLOB_MAX, err, warn) || files.io() != mode) {
            std::printf("SKIP %s arm: %s%s\n", tag, err.c_str(), warn.c_str());
            ++g_skips;
            return false;
        }
        if (!pack.bind_table(files, kModel, err)) { std::printf("FAIL: bind the manifest: %s\n", err.c_str()); ++g_fail; return false; }
        if (!helper.create(name, H, CAP, NL, NE, WMT, BLOB_MAX, err) || !main_side.connect(name, H, CAP, NL, NE, WMT, err)) {
            std::printf("FAIL: channel setup: %s\n", err.c_str()); ++g_fail; return false;
        }
        files.announce(*helper.header());   // as the helper does before kHelperUp: file_io, file_count, file_table_id and the live counters
        cache.bytes = kBlob;
        for (int32_t l = 0; l < NL; ++l)
            for (int32_t e = 0; e < 4; ++e) { cache.put(l, e); helper.held()[l * NE + e] = 1; }
        th = std::thread(loop, this);
        return true;
    }
    ~Rig() { stop = true; if (th.joinable()) th.join(); }
    // Main: write the FileRef into slot 0, submit `n` ops, wait for the ack.  res[0] is the first op's result.
    bool round(const TierWireOp* ops, int n, const FileRef& ref, uint8_t* res) {
        FileRef copy = ref;
        file_ref_to(main_side.blob(0), BLOB_MAX, copy);
        const uint64_t seq = main_side.tier_submit(ops, n);
        if (!Channel::wait([&] { return main_side.tier_done(seq); }, 10)) { std::printf("FAIL: the helper never acked\n"); ++g_fail; return false; }
        for (int i = 0; i < n; ++i) res[i] = main_side.header()->tier_res[i];
        return main_side.header()->tier_ok != 0;
    }
    bool held(int32_t l, int32_t e) { return main_side.held()[l * NE + e] != 0; }
    std::vector<uint8_t> slot_of(int32_t l, int32_t e) const { return cache.slots[(size_t) cache.slot_of.at({l, e})]; }
};

static void t_round_trip(Rig& r) {
    const FileRef ref = r.pack.ref_of(kModel, 1, 6);
    const std::vector<uint8_t> want = r.pack.blob_of(kModel[1][6].data(), 3);
    uint8_t res = 0;
    const TierWireOp op = swap_op(kTierSwapFile, 1, 0, 6, 0);
    CHECK(r.round(&op, 1, ref, &res) && res == kTierApplied, "a file swap over the channel is applied");
    CHECK(r.cache.slot_of.count({1, 6}) != 0 && r.slot_of(1, 6) == want, "the slot holds what the helper read from the FILES, byte for byte (finish line 1a)");
    CHECK(r.held(1, 6) && !r.held(1, 0), "held[] as main reads it: new in, displaced out");
    CHECK(r.cache.applies_from_blob == 1, "the cache was given the helper's staging blob");
    const TierWireOp plain = as_swap(op);
    std::string why;
    CHECK(tier_residency_consistent(r.main_side.held(), NL, NE, &plain, &res, 1, why), "main's residency check accepts the round");
    const Header* h = r.main_side.header();
    CHECK(__atomic_load_n(&h->file_reads, __ATOMIC_RELAXED) == 3 && __atomic_load_n(&h->file_direct_bytes, __ATOMIC_RELAXED) +
          __atomic_load_n(&h->file_buffered_bytes, __ATOMIC_RELAXED) == (uint64_t) kBlob, "the Header counters grew by the blob");
}

// SOLFIX 4 (the trust boundary): main's op names cell B, its FileRef names cell C (same layer, same size) with C's VALID manifest sum.
// Before the fix the helper read C's bytes, "verified" them against C's sum and wrote them into B's slot.  Now the ref must be the
// helper's own for B, so it is refused before one byte is read: no read, no slot write, held[] unchanged.
static void t_other_cells_ref(Rig& r) {
    const Header* h = r.main_side.header();
    const uint64_t reads = __atomic_load_n(&h->file_reads, __ATOMIC_RELAXED);
    const int applies = r.cache.applies;
    const std::vector<uint8_t> before = r.slot_of(1, 3);
    const TierWireOp b_op = swap_op(kTierSwapFile, 1, 3, 5, 0);   // B = expert 5 of layer 1, displacing expert 3
    const FileRef c_ref = r.pack.ref_of(kModel, 1, 7);             // C = expert 7 of layer 1: its extents, its valid sum
    uint8_t res = 0;
    CHECK(!r.round(&b_op, 1, c_ref, &res) && res == kTierRejected, "B's op with C's extents and C's valid sum is kTierRejected");
    CHECK(said(r.rep, "extents"), "naming the extents as not B's");
    FileRef b_ext_c_sum = r.pack.ref_of(kModel, 1, 5);
    b_ext_c_sum.sum = c_ref.sum;
    CHECK(!r.round(&b_op, 1, b_ext_c_sum, &res) && res == kTierRejected, "B's extents with C's sum are kTierRejected");
    CHECK(__atomic_load_n(&h->file_reads, __ATOMIC_RELAXED) == reads, "neither read one byte from the files");
    CHECK(r.cache.applies == applies && r.slot_of(1, 3) == before, "no slot write: B's destination slot still holds the displaced expert");
    CHECK(r.held(1, 3) && !r.held(1, 5) && !r.held(1, 7), "held[] unchanged");
    CHECK(r.round(&b_op, 1, r.pack.ref_of(kModel, 1, 5), &res) && res == kTierApplied, "B's own ref on B's op is applied");
    CHECK(r.slot_of(1, 5) == r.pack.blob_of(kModel[1][5].data(), 3), "with B's bytes");
}

static void t_bad_sum(Rig& r) {
    const FileRef ref = r.pack.ref_of(kModel, 2, 7);   // main's sum, from the good bytes
    r.pack.corrupt(1, 32 + (2 * NE + 7) * 160 + 5);      // a byte of cell (2, 7)'s middle slice changes after the manifest was built
    const std::vector<uint8_t> before = r.slot_of(2, 1);
    const int applies = r.cache.applies;
    uint8_t res = 0;
    const TierWireOp op = swap_op(kTierSwapFile, 2, 1, 7, 0);
    CHECK(!r.round(&op, 1, ref, &res) && res == kTierBadSum, "a file that does not match the manifest is kTierBadSum, tier_ok 0 (finish line 1b)");
    CHECK(r.cache.applies == applies && r.slot_of(2, 1) == before, "the cache was never asked to write: the slot is byte-equal to the old expert");
    CHECK(r.held(2, 1) && !r.held(2, 7) && !r.cache.suspect(2, 1), "held[] unchanged and nothing quarantined");
    CHECK(said(r.rep, "sum"), "and the helper reported that it is the sum");
}

static void t_bounds(Rig& r) {
    const FileRef good = r.pack.ref_of(kModel, 3, 5);
    const int applies = r.cache.applies;
    const std::vector<uint8_t> before = r.slot_of(3, 0);
    const TierWireOp op = swap_op(kTierSwapFile, 3, 0, 5, 0);
    FileRef bad[6];
    for (FileRef& b : bad) b = good;
    bad[0].extent[1].offset = S1 - 100;       // offset + length past the end of the file
    bad[1].extent[0].file = 7;                // a file index outside the table
    bad[2].n_extents = 4;                     // more than three extents
    bad[3].blob_bytes += 1;                   // lengths do not sum to blob_bytes
    bad[4].extent[2].offset = UINT64_MAX - 3; // a wrapping offset
    bad[5].ref_bytes = 8;                     // another FileRef size
    for (int i = 0; i < 6; ++i) {
        uint8_t res = 0;
        char what[96];
        std::snprintf(what, sizeof what, "bad ref %d is kTierRejected, never kTierApplied (finish line 1d)", i);
        CHECK(!r.round(&op, 1, bad[i], &res) && res == kTierRejected, what);
    }
    CHECK(r.cache.applies == applies && r.slot_of(3, 0) == before && r.held(3, 0) && !r.held(3, 5), "none of them reached the cache or the slot");
    // an op that does not fit the model never reads the file either
    uint8_t res = 0;
    const TierWireOp across = [&] { TierWireOp o = op; o.layer_new = 2; return o; }();
    CHECK(!r.round(&across, 1, good, &res) && res == kTierRejected && r.cache.applies == applies, "a cross-layer file swap is refused like a blob swap");
}

static void t_round_shape(Rig& r) {
    const FileRef ref = r.pack.ref_of(kModel, 0, 6);
    const TierWireOp ops[2] = {swap_op(kTierSwapFile, 0, 1, 6, 0), swap_op(kTierSwap, 1, 1, 6, 0)};
    const int applies = r.cache.applies;
    uint8_t res[2] = {0, 0};
    CHECK(!r.round(ops, 2, ref, res) && res[0] == kTierRejected && res[1] == kTierRejected, "a round holding a file op and another op is refused whole");
    CHECK(r.cache.applies == applies && r.held(0, 1) && r.held(1, 1), "before anything was read or written");
    CHECK(said(r.rep, "exactly one op"), "naming the one-op rule");
}

static void t_readback(Rig& r) {
    const FileRef ref = r.pack.ref_of(kModel, 0, 4);
    r.cache.corrupt_expert_new = 4;
    uint8_t res = 0;
    const TierWireOp op = swap_op(kTierSwapFile, 0, 2, 4, 0);
    CHECK(!r.round(&op, 1, ref, &res) && res == kTierFailed, "a slot that does not read back against the staging blob is kTierFailed");
    CHECK(!r.held(0, 4) && !r.held(0, 2) && r.cache.suspect(0, 4), "published as holding nothing (main closes the card)");
    r.cache.corrupt_expert_new = -1;
}

static void t_arm(FileIo mode, const char* tag) {
    {
        Rig r;
        if (!r.up(mode, tag)) return;
        t_round_trip(r);
        t_other_cells_ref(r);
        t_bounds(r);
        t_round_shape(r);
        t_readback(r);
    }
    {
        Rig r;   // a fresh pack for the corruption (the others need good files)
        if (!r.up(mode, tag)) return;
        t_bad_sum(r);
    }
}

static void t_no_file_table() {
    Rig r;
    r.serve_files = false;
    if (!r.up(kFileIoBuffered, "nofiles")) return;
    uint8_t res = 0;
    const int applies = r.cache.applies;
    const TierWireOp op = swap_op(kTierSwapFile, 1, 0, 6, 0);
    CHECK(!r.round(&op, 1, r.pack.ref_of(kModel, 1, 6), &res) && res == kTierRejected && r.cache.applies == applies, "a helper with no file table refuses a file swap");
    CHECK(said(r.rep, "no file moves"), "and says so");
}


int main() {
    t_arm(kFileIoBuffered, "buf");
    t_arm(kFileIoDirect, "dio");
    t_no_file_table();
    if (g_skips != 0 && std::getenv("SEABRAID_STRATA_FR_REQUIRE_DIRECT") != nullptr) { std::printf("FAIL: %d arms skipped and SEABRAID_STRATA_FR_REQUIRE_DIRECT is set\n", g_skips); ++g_fail; }
    std::printf(g_fail ? "channel_file_swap_test: %d FAILED checks\n" : "channel_file_swap_test: all checks passed\n", g_fail);
    return g_fail != 0;
}
