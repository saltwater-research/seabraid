// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// CHECKER for file_moves.hpp: the manifest load (each refusal is ONE warning and file moves stay off), every refusal of a file move,
// the helper's answers (applied, rejected, bad sum, failed, closed), the switch-off after a bad sum, main's own read check, the
// counter line, and the freshness stamp.  A fake FileBackend stands in for the engine; no GPU, no helper, no pack file but a temp one.
#include "file_moves.hpp"

#include "diag_capture.hpp"
#include "strata/seabraid/file_ref.hpp"
#include "strata/seabraid/file_stamp.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::fprintf(stderr, "line %d: %s failed\n%s", __LINE__, #expression, capture::dump().c_str()); \
        std::abort(); \
    } \
} while (false)

namespace capture = seabraid::strata_adapter::capture;
namespace sums = seabraid::sums;
using namespace seabraid::strata_adapter;
using strata::core::IoStatus;

namespace {

constexpr int32_t kLayers = 2, kExperts = 3;
constexpr size_t kBlob = 64;
const std::vector<uint64_t> kFileSizes{4096, 8192};

// The pack: expert (l, e)'s bytes are a function of the cell, so a wrong cell can never match by luck.
std::vector<uint8_t> blob(int32_t l, int32_t e) {
    std::vector<uint8_t> b(kBlob);
    for (size_t i = 0; i < b.size(); ++i) b[i] = (uint8_t) (l * 97 + e * 31 + i * 7 + 1);
    return b;
}

PackSums::Facts facts() {
    PackSums::Facts f;
    f.layers = kLayers; f.experts = kExperts;
    f.files = {{kFileSizes[0], 111}, {kFileSizes[1], 222}};
    f.blob_bytes.assign(kLayers, (uint32_t) kBlob);
    return f;
}

std::vector<uint8_t> manifest_bytes(const PackSums::Facts& f) {
    sums::Manifest m;
    m.layers = f.layers; m.experts = f.experts; m.files = f.files; m.blob_bytes = f.blob_bytes;
    for (int32_t l = 0; l < kLayers; ++l)
        for (int32_t e = 0; e < kExperts; ++e) { const auto b = blob(l, e); m.sums.push_back(blob_sum(b.data(), b.size())); }
    std::string out, err;
    CHECK(sums::serialize(m, out, err));
    return {out.begin(), out.end()};
}

PackSums::Reader bytes_reader(std::vector<uint8_t> bytes) {
    return [b = std::move(bytes)](std::vector<uint8_t>& out) { out = b; return true; };
}

std::shared_ptr<PackSums> good_sums() {
    auto s = PackSums::load(bytes_reader(manifest_bytes(facts())), facts());
    CHECK(s != nullptr);
    return s;
}

// What the engine would answer; every answer and every call is observable.
struct FakeBackend final : FileBackend {
    HelperFacts helper{2, (int32_t) kFileSizes.size(), file_table_id(kFileSizes), 100, 10, 3};
    bool in_ram_answer = false, extents_ok = true, send_ok = true, throw_in_extents = false, throw_in_send = false;
    uint8_t result = seabraid::kTierApplied;
    int closed_state = 0;
    int sends = 0;
    seabraid::FileRef last_ref{};
    std::vector<uint64_t> last_sizes;
    int32_t last_count = -1, last_layer = -1, last_old = -1, last_new = -1;

    HelperFacts helper_facts() noexcept override { return helper; }
    int32_t extents_layer = -1, extents_expert = -1;   // the cell main asked the engine to address
    bool extents(int32_t l, int32_t e, seabraid::FileExtent out[3], int& n) override {
        extents_layer = l; extents_expert = e;
        if (throw_in_extents) throw std::runtime_error("bad_alloc stand-in");
        if (!extents_ok) { n = 0; return false; }
        out[0] = {0, (uint32_t) (kBlob / 2), 128};
        out[1] = {1, (uint32_t) (kBlob / 2), 256};
        n = 2;
        return true;
    }
    bool in_ram(int32_t, int32_t) override { return in_ram_answer; }
    bool send(int32_t l, int32_t old_e, int32_t new_e, const seabraid::FileRef& ref, const uint64_t* sizes, int32_t count,
              uint8_t& res, std::string& err) override {
        ++sends;
        if (throw_in_send) throw std::runtime_error("bad_alloc stand-in");
        last_ref = ref; last_layer = l; last_old = old_e; last_new = new_e; last_count = count;
        last_sizes.assign(sizes, sizes + count);
        if (!send_ok) { err = "refused before the channel"; return false; }
        res = result;
        return true;
    }
    int closed() noexcept override { return closed_state; }
};

struct Rig {
    std::shared_ptr<PackSums> sums = good_sums();
    FakeBackend* backend = new FakeBackend();
    FileMoves moves{sums, std::unique_ptr<FileBackend>(backend), 1};
};

// ---- the manifest load: every refusal is one WARN, no manifest object, nothing else changes ----
void load_refusals() {
    capture::clear();
    CHECK(PackSums::load(PackSums::Reader{}, facts()) == nullptr);                                  // no reader at all
    CHECK(PackSums::load([](std::vector<uint8_t>&) { return false; }, facts()) == nullptr);         // the seam says "no manifest"
    capture::require("file moves OFF", 2);
    capture::require("seabraid-sums", 2);
    capture::clear();
    auto bad = manifest_bytes(facts());
    bad[0] ^= 0xFF;                                                                                  // magic
    CHECK(PackSums::load(bytes_reader(bad), facts()) == nullptr);
    capture::require("file moves OFF: expert-sums: bad magic", 1);
    capture::clear();
    PackSums::Facts stale = facts();
    stale.files[1].mtime_ns += 1;                                                                    // a pack file changed after the manifest
    CHECK(PackSums::load(bytes_reader(manifest_bytes(facts())), stale) == nullptr);
    capture::require("stale or about another model: file 1 changed", 1);
    capture::clear();
    PackSums::Facts other = facts();
    other.blob_bytes[1] += 4;                                                                        // another model's geometry
    CHECK(PackSums::load(bytes_reader(manifest_bytes(facts())), other) == nullptr);
    capture::require("layer 1 blob size differs", 1);
    capture::require("manifest ok", 0);
    capture::clear();
    CHECK(good_sums() != nullptr);
    capture::require("manifest ok, 2 layers x 3 experts, 2 files", 1);
    capture::require("file moves OFF", 0);
}

// The default reader reads <pack>/expert-sums.v1 from disk; a pack without one is the same refusal.
void pack_file_reader() {
    char dir[] = "/tmp/seabraid_sums_XXXXXX";
    CHECK(mkdtemp(dir) != nullptr);
    capture::clear();
    CHECK(PackSums::load(PackSums::pack_file(dir), facts()) == nullptr);                             // no file there
    capture::require("file moves OFF", 1);
    const auto bytes = manifest_bytes(facts());
    const std::string path = std::string(dir) + "/" + sums::kFileName;
    { std::ofstream out(path, std::ios::binary); out.write((const char*) bytes.data(), (std::streamsize) bytes.size()); }
    CHECK(PackSums::load(PackSums::pack_file(dir), facts()) != nullptr);
    // SOLFIX 5: a planted file far longer than any manifest is read only one byte past the longest manifest, then refused by length
    CHECK(truncate(path.c_str(), (off_t) (64ull << 20)) == 0);                                      // a sparse 64 MB file
    std::vector<uint8_t> got;
    CHECK(PackSums::pack_file(dir)(got) && got.size() == sums::kMaxManifestBytes + 1);
    capture::clear();
    CHECK(PackSums::load(PackSums::pack_file(dir), facts()) == nullptr);
    capture::require("file moves OFF", 1);
    CHECK(unlink(path.c_str()) == 0 && rmdir(dir) == 0);
}

// ---- main's own read of a blob against the manifest ----
void own_read_check() {
    auto s = good_sums();
    std::string err;
    capture::clear();
    auto b = blob(1, 2);
    CHECK(s->read_ok(1, 2, b.data(), b.size(), err) && !s->cell_bad(1, 2) && s->read_mismatches() == 0);
    b[5] ^= 1;                                                                                       // one flipped byte in the pack
    CHECK(!s->read_ok(1, 2, b.data(), b.size(), err) && s->cell_bad(1, 2) && s->read_mismatches() == 1);
    capture::require("layer 1 expert 2 as read from the pack disagrees", 1, 0);
    CHECK(!s->cell_bad(1, 1) && s->cell_bad(2, 0) && s->cell_bad(0, 3));                             // other cells fine; outside the model is "bad"
    auto short_blob = blob(0, 0);
    short_blob.pop_back();
    CHECK(!s->read_ok(0, 0, short_blob.data(), short_blob.size(), err) && s->cell_bad(0, 0));        // a short read never matches
    CHECK(!s->read_ok(5, 0, b.data(), b.size(), err));                                               // no sum for that cell
}

// ---- one file move ----
void applied_move() {
    Rig r;
    std::string err;
    CHECK(r.moves.active());
    CHECK(r.moves.move(1, 0, 2, err) == IoStatus::kOk);
    CHECK(r.backend->sends == 1 && r.moves.attempts() == 1 && r.moves.applied() == 1 && r.moves.refused() == 0);
    const auto want = blob(1, 2);
    CHECK(r.backend->last_ref.sum == blob_sum(want.data(), want.size()));                            // the manifest's sum of the NEW expert
    CHECK(r.backend->last_ref.n_extents == 2 && r.backend->last_ref.blob_bytes == kBlob);
    CHECK(r.backend->last_layer == 1 && r.backend->last_old == 0 && r.backend->last_new == 2);
    CHECK(r.backend->extents_layer == 1 && r.backend->extents_expert == 2);                         // the ref's extents are the op's own new cell
    CHECK(r.backend->last_count == 2 && r.backend->last_sizes == kFileSizes);                        // main's sizes from the manifest, in table order
}

// The counters a helper announced reach the totals unchanged.
void announced_counters() {
    Rig r;
    FileIoTotals total;
    r.moves.add_totals(total);
    CHECK(total.direct_bytes == 100 && total.buffered_bytes == 10 && total.reads == 3 && total.bad_sum == 0);
}

void refusals_send_nothing() {
    capture::clear();
    {   Rig r; std::string err;                                                                       // the RAM copy is cheaper than a read
        r.backend->in_ram_answer = true;
        CHECK(r.moves.move(0, 1, 2, err) == IoStatus::kRefused && r.backend->sends == 0 && err.find("RAM") != std::string::npos); }
    {   Rig r; std::string err;                                                                       // the engine names no file bytes
        r.backend->extents_ok = false;
        CHECK(r.moves.move(0, 1, 2, err) == IoStatus::kRefused && r.backend->sends == 0); }
    {   Rig r; std::string err;                                                                       // an in-place rewrite has no file form
        CHECK(r.moves.move(0, 2, 2, err) == IoStatus::kRefused && r.backend->sends == 0); }
    {   Rig r; std::string err;                                                                       // a cell outside the manifest
        CHECK(r.moves.move(0, 1, 9, err) == IoStatus::kRefused && r.backend->sends == 0); }
    {   Rig r; std::string err;                                                                       // a helper that announced no file io
        r.backend->helper.io = seabraid::kFileIoOff;
        CHECK(!r.moves.active() && r.moves.move(0, 1, 2, err) == IoStatus::kRefused && r.backend->sends == 0); }
    capture::require("file moves OFF", 0);                                                            // none of those is worth a warning
    capture::require("file moves: helper", 0);
    {   Rig r; std::string err;                                                                       // main's own read of the cell disagreed once: it never moves
        auto b = blob(0, 2);
        b[0] ^= 1;
        CHECK(!r.sums->read_ok(0, 2, b.data(), b.size(), err));
        CHECK(r.moves.move(0, 1, 2, err) == IoStatus::kRefused && r.backend->sends == 0);
        CHECK(r.moves.move(0, 2, 1, err) == IoStatus::kOk); }                                         // the neighbours still move
}

// A helper whose file table is not the manifest's opened other files: bytes only, said ONCE.
void wrong_file_table() {
    Rig r;
    std::string err;
    capture::clear();
    r.backend->helper.table_id ^= 1;
    CHECK(!r.moves.active());
    for (int i = 0; i < 3; ++i) CHECK(r.moves.move(0, 1, 2, err) == IoStatus::kRefused);
    CHECK(r.backend->sends == 0);
    capture::require("opened other files than the manifest names", 1);
    r.backend->helper.table_id = file_table_id(kFileSizes);
    r.backend->helper.count = 1;                                                                      // a different number of files
    CHECK(!r.moves.active());
}

// The helper's answers.
void helper_answers() {
    {   Rig r; std::string err; r.backend->result = seabraid::kTierRejected;                          // "use bytes", and file moves stay on
        CHECK(r.moves.move(0, 1, 2, err) == IoStatus::kRefused && r.moves.refused() == 1 && r.moves.active() && r.moves.bad_sums() == 0); }
    {   Rig r; std::string err; r.backend->send_ok = false;                                           // refused before the channel
        CHECK(r.moves.move(0, 1, 2, err) == IoStatus::kRefused && err == "refused before the channel" && r.moves.active()); }
    {   Rig r; std::string err; r.backend->result = seabraid::kTierFailed;
        CHECK(r.moves.move(0, 1, 2, err) == IoStatus::kFailed); }
    {   Rig r; std::string err; r.backend->closed_state = 1;                                          // the card closed itself on a timeout
        CHECK(r.moves.move(0, 1, 2, err) == IoStatus::kTimeout); }
    {   Rig r; std::string err; r.backend->closed_state = 2; r.backend->result = seabraid::kTierApplied;   // a torn slot outranks any result
        CHECK(r.moves.move(0, 1, 2, err) == IoStatus::kFailed); }
    {   Rig r; std::string err; r.backend->throw_in_extents = true;                                   // nothing was sent: REFUSED
        CHECK(r.moves.move(0, 1, 2, err) == IoStatus::kRefused && r.backend->sends == 0); }
    {   Rig r; std::string err; r.backend->throw_in_send = true;                                      // the send began: state unknown, TIMEOUT
        CHECK(r.moves.move(0, 1, 2, err) == IoStatus::kTimeout); }
}

// kTierBadSum: that helper reads the wrong file.  One WARN, the move is REFUSED (the byte path follows), and it never asks again.
void bad_sum_switches_off() {
    Rig r;
    std::string err;
    capture::clear();
    r.backend->result = seabraid::kTierBadSum;
    CHECK(r.moves.active());
    CHECK(r.moves.move(1, 0, 2, err) == IoStatus::kRefused);
    CHECK(r.moves.bad_sums() == 1 && r.moves.switched_off() && !r.moves.active());
    capture::require("file moves OFF for helper rung 1", 1, 0);
    capture::require("layer 1 expert 2", 1);
    r.backend->result = seabraid::kTierApplied;
    for (int i = 0; i < 3; ++i) CHECK(r.moves.move(1, 0, 2, err) == IoStatus::kRefused);              // switched off: not even asked
    CHECK(r.backend->sends == 1);
    capture::require("file moves OFF for helper rung", 1);                                            // ONE warning for the session
    FileIoTotals total;
    r.moves.add_totals(total);
    CHECK(total.bad_sum == 1);
}

// The counter line placement-qual reads: cumulative, summed over helpers, nothing for an empty list.
void counter_line() {
    Rig a, b;
    b.backend->helper = {2, (int32_t) kFileSizes.size(), file_table_id(kFileSizes), 1000, 20, 4};
    b.backend->result = seabraid::kTierBadSum;
    std::string err;
    CHECK(b.moves.move(0, 1, 2, err) == IoStatus::kRefused);
    capture::clear();
    log_file_io({});
    capture::require("helper file io", 0);
    log_file_io({&a.moves, &b.moves});
    capture::require("helper file io: direct_bytes=1100 buffered_bytes=30 reads=7 bad_sum=1", 1, 0);
}

// The freshness stamp is the manifest writer's formula (the wire's own function): size and mtime of each file, in order.
void stamp() {
    char path[] = "/tmp/seabraid_stamp_XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0 && write(fd, "abcd", 4) == 4);
    close(fd);
    std::vector<sums::FileStat> out;
    std::string err;
    CHECK(sums::stamp_files({path}, out, err) && out.size() == 1 && out[0].size == 4 && out[0].mtime_ns > 0);
    CHECK(!sums::stamp_files({path, "/nonexistent/seabraid/none"}, out, err) && err == "cannot stat /nonexistent/seabraid/none");
    unlink(path);
}

}  // namespace

int main() {
    capture::start();
    load_refusals();
    pack_file_reader();
    own_read_check();
    applied_move();
    announced_counters();
    refusals_send_nothing();
    wrong_file_table();
    helper_answers();
    bad_sum_switches_off();
    counter_line();
    stamp();
    std::puts("Seabraid Strata file moves fixture passed");
}
