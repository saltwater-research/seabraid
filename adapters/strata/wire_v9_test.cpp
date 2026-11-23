// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Wire v9 host fixture (no GPU, no Strata engine): the layout golden, the refusal of a v8 peer, every FileRef validation
// rule, the one-op round rule, the blob_slot rule for the new kind, the --file-io option, and the residency replay of a file
// swap.  usage: strata_wire_v9_test wire_v9_golden.txt      |      strata_wire_v9_test --print > wire_v9_golden.txt
// Guards: a field moved in Header/FileRef shifts every v9 read; a version still 8 lets a v8 helper ACK a kind it ignores;
// a FileRef rule that is missing lets main (or a corrupt shm) make the helper read outside its files.
#include "test_support.hpp"

#include "helper_options.hpp"
#include "strata/seabraid/channel.hpp"
#include "strata/seabraid/file_ref.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
using seabraid::strata_adapter::testing::check;
using namespace seabraid;

// ---- 1. golden: the bytes of v9, printed by this very test so the file has one honest source
std::map<std::string, long long> facts() {
    return {{"version", kVersion},
            {"header_bytes", (long long) sizeof(Header)},
            {"header_file_io_offset", (long long) offsetof(Header, file_io)},
            {"header_file_count_offset", (long long) offsetof(Header, file_count)},
            {"header_file_table_id_offset", (long long) offsetof(Header, file_table_id)},
            {"header_file_direct_bytes_offset", (long long) offsetof(Header, file_direct_bytes)},
            {"header_file_buffered_bytes_offset", (long long) offsetof(Header, file_buffered_bytes)},
            {"header_file_reads_offset", (long long) offsetof(Header, file_reads)},
            {"header_pf_svc_n_offset", (long long) offsetof(Header, pf_svc_n)},
            {"tier_wire_op_bytes", (long long) sizeof(TierWireOp)},
            {"tier_wire_op_blob_slot_offset", (long long) offsetof(TierWireOp, blob_slot)},
            {"file_extent_bytes", (long long) sizeof(FileExtent)},
            {"file_ref_bytes", (long long) sizeof(FileRef)},
            {"file_ref_sum_offset", (long long) offsetof(FileRef, sum)},
            {"file_ref_blob_bytes_offset", (long long) offsetof(FileRef, blob_bytes)},
            {"file_ref_n_extents_offset", (long long) offsetof(FileRef, n_extents)},
            {"file_ref_extent_offset", (long long) offsetof(FileRef, extent)},
            {"tier_swap_blob", kTierSwapBlob},
            {"tier_swap_file", kTierSwapFile},
            {"tier_bad_sum", kTierBadSum},
            {"file_ref_max_extents", kFileRefMaxExtents},
            {"sum_xxh3_64", kSumXxh3_64},
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
    check(kVersion == 11, "wire version is 11 (this golden still pins v9's rows; its version and header_bytes rows follow kVersion)");
    check(sizeof(Header) <= 4096, "the header fits the page Layout::of reserves, so shm size is unchanged");
}

// ---- 2. a v8 peer is refused at connect, naming both versions
void v8_peer_refused() {
    const std::string name = "/seabraid_wv9_" + std::to_string((long long) ::getpid());
    std::string err;
    Channel helper;
    check(helper.create(name, 64, 32, 4, 8, 4, 256, err), ("create: " + err).c_str());
    Header* hh = helper.header();
    check(hh->version == kVersion, "a new channel announces the current version");
    hh->version = 8;
    Channel main_side;
    check(!main_side.connect(name, 64, 32, 4, 8, 4, err), "connect refuses a v8 helper");
    check(err.find("helper v8") != std::string::npos && err.find("main v11") != std::string::npos,
          ("the refusal names both versions: " + err).c_str());
    hh->version = kVersion;
    check(main_side.connect(name, 64, 32, 4, 8, 4, err), ("connect works once the versions agree: " + err).c_str());
    check(hh->file_io == 0 && hh->file_count == 0 && hh->file_table_id == 0 && hh->file_direct_bytes == 0 &&
          hh->file_buffered_bytes == 0 && hh->file_reads == 0, "the v9 fields start at zero (file moves off until the helper says otherwise)");
}

// ---- 3. FileRef validation: every rule refuses, each naming itself; the legal ref passes
constexpr int64_t kBlob = 1000, kStaging = 4096;
const uint64_t kSizes[2] = {10000, 5000};

FileRef good() {   // 3 extents over two files: 400 + 300 + 300
    const FileExtent e[3] = {{0, 400, 32}, {1, 300, 4700}, {0, 300, 9700}};
    return make_file_ref(e, 3, 0x1234);
}

bool refused(const FileRef& r, const char* needle, int32_t files = 2, int64_t want = kBlob, int64_t staging = kStaging) {
    std::string why;
    return !file_ref_valid(r, kSizes, files, want, staging, why) && why.find(needle) != std::string::npos;
}

void file_ref_rules() {
    std::string why;
    FileRef r = good();
    check(file_ref_valid(r, kSizes, 2, kBlob, kStaging, why), ("a legal 3-extent ref passes: " + why).c_str());
    check(r.blob_bytes == 1000 && r.ref_bytes == 72 && r.sum_kind == kSumXxh3_64, "make_file_ref fills the derived fields");
    const FileExtent one[1] = {{1, 1000, 4000}};   // last byte of file 1 exactly
    check(file_ref_valid(make_file_ref(one, 1, 1), kSizes, 2, kBlob, kStaging, why), "an extent ending exactly at end of file passes");

    r = good(); r.ref_bytes = 64;                 check(refused(r, "FileRef size"), "ref_bytes other than 72 is refused");
    r = good(); r.sum_kind = 2;                   check(refused(r, "sum kind"), "an unknown sum kind is refused");
    r = good(); r.n_extents = 0;                  check(refused(r, "extents"), "zero extents is refused");
    r = good(); r.n_extents = 4;                  check(refused(r, "extents"), "four extents is refused (the array holds three)");
    r = good(); r.n_extents = 0xFFFFFFFFu;        check(refused(r, "extents"), "a huge extent count is refused before any index");
    r = good(); r.extent[1].file = 2;             check(refused(r, "file index"), "a file index at the table size is refused");
    r = good(); r.extent[0].file = 0xFFFFFFFFu;   check(refused(r, "file index"), "a huge file index is refused");
    check(refused(good(), "file index", 1), "an index past a one-file table is refused");
    check(refused(good(), "bad file table", 65), "a table larger than kFileTableMax is refused");
    r = good(); r.extent[2].length = 0; r.blob_bytes = 700;
    check(refused(r, "zero length", 2, 700), "a zero-length extent is refused");
    r = good(); r.extent[0].offset = 9700;        check(refused(r, "passes the end"), "an extent running past end of file is refused");
    r = good(); r.extent[0].offset = UINT64_MAX;  check(refused(r, "passes the end"), "offset = 2^64-1 is refused, not wrapped");
    r = good(); r.extent[0].offset = UINT64_MAX - 100; r.extent[0].length = 400;
    check(refused(r, "passes the end"), "offset + length overflowing 2^64 is refused, not wrapped");
    r = good(); r.extent[1].length = 0xFFFFFFFFu; check(refused(r, "passes the end"), "a length larger than the file is refused");
    r = good(); r.blob_bytes = 999;               check(refused(r, "sum to"), "lengths that do not add up to blob_bytes are refused");
    r = good(); r.extent[0].length = 0xFFFFFFFFu - 100; r.extent[1].length = 0xFFFFFFFFu - 100; r.blob_bytes = 1000;
    check(refused(r, "passes the end"), "lengths whose u32 sum would wrap are refused (bounded by the file first)");
    check(refused(good(), "layer needs", 2, 1001), "a blob size other than the layer's is refused");
    check(refused(good(), "staging", 2, kBlob, 999), "a blob larger than the staging buffer is refused");
    const FileExtent big[2] = {{0, 0xFFFFFFFFu, 0}, {0, 0xFFFFFFFFu, 0}};
    const uint64_t huge[1] = {1ull << 40};
    r = make_file_ref(big, 2, 1);
    check(r.blob_bytes == 0 && !file_ref_valid(r, huge, 1, 0, 1ll << 40, why) && why.find("sum to") != std::string::npos,
          "two u32 lengths that overflow blob_bytes yield a ref that sums to a mismatch, never a short read");
    check(!file_ref_valid(good(), nullptr, 2, kBlob, kStaging, why), "a missing size table is refused, not dereferenced");
}

void slot_access() {
    std::vector<uint8_t> slot(200, 0xAB);
    FileRef out{};
    check(!file_ref_from(slot.data(), 71, out), "a slot smaller than a FileRef cannot be read as one");
    check(!file_ref_to(slot.data(), 71, good()) && slot[0] == 0xAB, "a slot smaller than a FileRef is not written");
    check(file_ref_to(slot.data(), 200, good()) && file_ref_from(slot.data(), 200, out), "a FileRef round-trips through a slot");
    check(out.sum == 0x1234 && out.extent[2].offset == 9700 && out.n_extents == 3 && out.blob_bytes == 1000, "the copy equals the original");
    check(slot[72] == 0xAB, "writing a FileRef touches exactly 72 bytes of the slot");
    check(!file_ref_from(nullptr, 200, out), "a null slot is refused");
}

// ---- 4. the op kind: slot rule, one-op rounds, residency
TierWireOp op(uint8_t kind, uint16_t slot, int l_old, int e_old, int l_new, int e_new) {
    TierWireOp o{};
    o.kind = kind; o.blob_slot = slot; o.layer_old = l_old; o.expert_old = e_old; o.layer_new = l_new; o.expert_new = e_new;
    return o;
}

void kind_rules() {
    check(blob_slot_valid(op(kTierSwapFile, 0, 1, 2, 1, 3)) && blob_slot_valid(op(kTierSwapFile, kBlobBatch - 1, 1, 2, 1, 3)),
          "a file swap may name any slot below kBlobBatch");
    check(!blob_slot_valid(op(kTierSwapFile, kBlobBatch, 1, 2, 1, 3)), "a file swap naming slot kBlobBatch is refused");
    check(!blob_slot_valid(op(kTierSwap, 1, 1, 2, 1, 3)) && !blob_slot_valid(op(kTierLoad, 1, -1, -1, 1, 3)), "other kinds keep blob_slot 0");
    check(blob_slot_valid(op(kTierSwapBlob, 15, 1, 2, 1, 3)) && !blob_slot_valid(op(kTierSwapBlob, 16, 1, 2, 1, 3)), "v8 blob swap rule is unchanged");

    std::string why;
    const TierWireOp one[1] = {op(kTierSwapFile, 0, 1, 2, 1, 3)};
    const TierWireOp two[2] = {op(kTierSwapFile, 0, 1, 2, 1, 3), op(kTierSwap, 0, 1, 4, 1, 5)};
    const TierWireOp swaps[2] = {op(kTierSwapBlob, 0, 1, 2, 1, 3), op(kTierSwap, 0, 1, 4, 1, 5)};
    check(tier_round_shape_ok(one, 1, why), "a round of one file swap is legal");
    check(!tier_round_shape_ok(two, 2, why) && why.find("exactly one op") != std::string::npos, "a file swap beside another op is refused");
    check(tier_round_shape_ok(swaps, 2, why), "a multi-op round of v8 kinds is untouched");
}

void residency() {
    constexpr int64_t L = 4, E = 8;
    std::vector<uint8_t> held(L * E, 0);
    held[1 * E + 2] = 1;
    const TierWireOp ops[1] = {op(kTierSwapFile, 0, 1, 2, 1, 3)};
    check(tier_op_displaces(ops[0]), "a file swap displaces its old cell");
    publish_tier_residency(held.data(), L, E, ops, 1, [&](int32_t l, int32_t e) { return l == 1 && e == 3; });
    check(held[1 * E + 2] == 0 && held[1 * E + 3] == 1, "publishing a file swap clears the old cell and sets the new one");
    std::string why;
    const uint8_t applied[1] = {kTierApplied}, bad[1] = {kTierBadSum}, rejected[1] = {kTierRejected};
    check(tier_residency_consistent(held.data(), L, E, ops, applied, 1, why), ("an applied file swap replays as a swap: " + why).c_str());
    held[1 * E + 2] = 1;   // a helper that forgot to displace
    check(!tier_residency_consistent(held.data(), L, E, ops, applied, 1, why) && why.find("layer 1 expert 2") != std::string::npos,
          "an applied file swap whose old cell is still marked resident is a contradiction (it was not replayed as a load)");
    // a bad sum / rejected op changed nothing: the old cell stays resident, the new one absent, and that is consistent
    std::vector<uint8_t> untouched(L * E, 0);
    untouched[1 * E + 2] = 1;
    check(tier_residency_consistent(untouched.data(), L, E, ops, bad, 1, why), "a kTierBadSum op is not replayed: the slot was never written");
    check(tier_residency_consistent(untouched.data(), L, E, ops, rejected, 1, why), "a rejected file op is not replayed either");
}

// ---- 5. --file-io on the helper's command line
bool parse(std::vector<std::string> args, QwenHelperOptions& o, std::string& err) {
    std::vector<std::string> store{"helper", "--pack", "/p"};
    store.insert(store.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& s : store) argv.push_back(s.data());
    err.clear();
    return o.parse((int) argv.size(), argv.data(), err);
}

void file_io_option() {
    QwenHelperOptions o;
    std::string err;
    check(parse({}, o, err) && o.file_io == kFileIoAuto, "the default is auto");
    const struct { const char* word; FileIo mode; } modes[] = {{"auto", kFileIoAuto}, {"direct", kFileIoDirect}, {"buffered", kFileIoBuffered}, {"off", kFileIoOff}};
    for (const auto& m : modes) {
        QwenHelperOptions p;
        check(parse({"--file-io", m.word}, p, err) && p.file_io == m.mode, (std::string("--file-io ") + m.word + " parses").c_str());
    }
    QwenHelperOptions g, bad;
    check(!parse({"--file-io", "gds"}, g, err) && err.find("not served") != std::string::npos, "gds is named and refused");
    check(!parse({"--file-io", "turbo"}, bad, err) && err.find("auto, direct, buffered or off") != std::string::npos, "an unknown mode is refused with the choices");
    QwenHelperOptions e;
    check(!parse({"--file-io"}, e, err) && err.find("requires a value") != std::string::npos, "a trailing --file-io is refused");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--print") == 0) {
        for (const auto& [k, v] : facts()) std::printf("%s %lld\n", k.c_str(), v);
        return 0;
    }
    check(argc > 1, "usage: strata_wire_v9_test wire_v9_golden.txt");
    golden(argv[1]);
    v8_peer_refused();
    file_ref_rules();
    slot_access();
    kind_rules();
    residency();
    file_io_option();
    std::puts("wire v9: all checks passed");
    return 0;
}
