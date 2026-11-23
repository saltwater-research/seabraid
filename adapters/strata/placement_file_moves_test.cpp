// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// File moves through the REAL Seabraid runtime and the driver: a scripted 1 layer x 3 expert world (owners {0,1,2}: the lead, one
// helper, the fallback) where expert 2 on the fallback is hot and the helper's expert 1 is cold, so the library plans the swap
// (helper <- 2, fallback <- 1).  CHECKER for placement.cpp's provider_v2 path (rung mask, replace_from_fallback trampoline), the
// by-reference trace record, the round's fr counters, and the engine's contract that a helper's bad sum clears its mask bit.
#include "placement_driver.hpp"

#include "diag_capture.hpp"
#include "file_moves.hpp"

#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::fprintf(stderr, "line %d: %s failed\n%s", __LINE__, #expression, capture::dump().c_str()); \
        std::abort(); \
    } \
} while (false)

namespace capture = seabraid::strata_adapter::capture;
namespace trace = seabraid::strata_adapter::trace;
using namespace seabraid::strata_adapter;
using strata::core::IoStatus;

namespace {

constexpr size_t kExperts = 3, kBytes = 4;
const std::vector<float> kHeat{100, 1, 50};      // lead keeps 0; fallback's 2 beats the helper's 1
const std::vector<float> kHeatBack{100, 50, 1};  // later: 1 (on the fallback by then) beats the helper's 2

std::vector<uint8_t> blob(int32_t expert) { return {uint8_t(expert), uint8_t(expert + 1), uint8_t(expert + 2), 9}; }

struct Log {
    std::vector<std::string> calls;
    void add(const char* who, const char* what, int a = -1, int b = -1) {
        calls.push_back(std::string(who) + "." + what + (a >= 0 ? "(" + std::to_string(a) + (b >= 0 ? ">" + std::to_string(b) : "") + ")" : ""));
    }
    bool is(std::initializer_list<const char*> want) const {
        if (want.size() != calls.size()) return false;
        size_t i = 0;
        for (const char* w : want) if (calls[i++] != w) return false;
        return true;
    }
    std::string text() const { std::string t; for (const auto& c : calls) t += c + " "; return t; }
};

// Any rung: stage and replace work on a residency table; the fallback holds everything (it serves from the pack).
class Card : public strata::core::RungIO {
public:
    Card(const char* name, Log& log, int expert) : name_(name), log_(log) { if (expert >= 0) resident[expert] = 1; }
    IoStatus stage(int32_t, int32_t expert, std::vector<uint8_t>& out, std::string&) override {
        log_.add(name_, "stage", expert);
        out = blob(expert);
        return IoStatus::kOk;
    }
    IoStatus replace(int32_t, int32_t old_e, int32_t new_e, const std::vector<uint8_t>& bytes, std::string&) override {
        log_.add(name_, "replace", old_e, new_e);
        CHECK(old_e == new_e || bytes == blob(new_e));                 // a card only ever writes the right bytes
        if (old_e != new_e) { resident[old_e] = 0; resident[new_e] = 1; }
        return IoStatus::kOk;
    }
    std::array<uint8_t, kExperts> resident{};
protected:
    const char* name_;
    Log& log_;
};

// The helper: loads from the file when asked, as the answer script says.
class Helper : public Card {
public:
    enum class Ask { kAck, kRefuse, kFail };
    Helper(Log& log, int expert) : Card("h", log, expert) {}
    IoStatus replace_from_fallback(int32_t, int32_t old_e, int32_t new_e, std::string& err) override {
        log_.add(name_, "rff", old_e, new_e);
        if (ask == Ask::kRefuse) { err = "use bytes"; return IoStatus::kRefused; }
        if (ask == Ask::kFail) { err = "torn"; return IoStatus::kFailed; }
        resident[old_e] = 0;
        resident[new_e] = 1;
        return IoStatus::kOk;
    }
    bool reads_files() const override { return files_on; }
    Ask ask = Ask::kAck;
    bool files_on = true;
};

struct Host : PlacementHost {
    Host(Card& lead, Card& helper) : lead_(lead), helper_(helper) {}
    bool drain(std::string&) noexcept override { return true; }
    void lead_cells(std::vector<uint8_t>& out) noexcept override { out.assign(lead_.resident.begin(), lead_.resident.end()); }
    bool helper_cells(uint8_t, std::vector<uint8_t>& out) noexcept override { out.assign(helper_.resident.begin(), helper_.resident.end()); return true; }
    bool helper_alive(uint8_t) noexcept override { return true; }
    void after_commit(const seabraid_placement_outcome_v1&) noexcept override {}
    void after_round() noexcept override { ++rounds; }
    Card &lead_, &helper_;
    int rounds = 0;
};

seabraid_placement_policy_v2 policy() {
    seabraid_placement_policy_v2 p{};
    p.abi_version = SEABRAID_PLACEMENT_POLICY_V2_VERSION;
    p.struct_size = sizeof p;
    p.max_moves_per_round = 4;
    p.ladder_boundaries = 0;       // every boundary
    p.fallback_boundary = 1;       // helper <-> fallback swaps
    p.max_hops_per_round = 1;
    p.promote_margin = 1.0;
    p.max_helper_layer_share = 1.0;
    return p;
}

struct Rig {
    using Make = std::function<std::unique_ptr<Helper>(Log&)>;
    explicit Rig(Helper::Ask answer = Helper::Ask::kAck, const Make& make = nullptr) {
        capture::clear();
        helper = make ? make(log) : std::make_unique<Helper>(log, 1);
        helper->ask = answer;
        PlacementDriverConfig config;
        config.every_windows = 1;
        config.policy = policy();
        driver = PlacementDriver::open(1, kExperts, {0, 1, 2}, {&lead, helper.get()}, &fallback, {kBytes}, config, view, err);
        CHECK(driver && err.empty());
        host = std::make_unique<Host>(lead, *helper);
    }
    bool tick(const std::vector<float>& heat = kHeat) { return driver->tick(heat.data(), heat.size(), true, *host); }
    Log log;
    ExecutorView view;
    Card lead{"l", log, 0}, fallback{"F", log, -1};
    std::unique_ptr<Helper> helper;
    std::unique_ptr<Host> host;
    std::unique_ptr<PlacementDriver> driver;
    std::string err;
};

std::string temp_path() {
    char path[] = "/tmp/seabraid_fmoves_XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    close(fd);
    return path;
}

std::string slurp(const std::string& path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

bool has(const std::string& text, const char* needle) { return text.find(needle) != std::string::npos; }

// ACK: the helper loads expert 2 itself, so the fallback's expert 2 is NEVER staged; only the displaced expert 1 is, for recovery.
void by_reference_commit() {
    const std::string path = temp_path();
    trace::configure(path.c_str(), 0);
    Rig rig;
    CHECK(rig.tick());
    CHECK(rig.log.is({"h.stage(1)", "h.rff(1>2)", "F.replace(2>1)"}));
    CHECK(rig.helper->resident[2] == 1 && rig.helper->resident[1] == 0);
    capture::require("placement move 0/2 2->1 src=fallback bytes=4 verified=0", 1);   // the diag line is unchanged: tools parse it
    capture::require("placement round=1 fr=1 fr_ok=1 fr_bytes=0", 1);
    capture::require("failed", 0);                                                      // a by-reference commit logs no failure
    CHECK(rig.host->rounds == 1);                                                       // the host hook ran after the round
    const std::string text = slurp(path);
    CHECK(has(text, "\"kind\":\"committed\"") && has(text, "\"src\":\"file\",\"dst\":\"fallback\""));
    trace::close_trace();
    unlink(path.c_str());
}

// REFUSED is not a failure: the byte path runs in the same swap, nothing is logged as an error, the record says fallback.
void refusal_takes_the_byte_path() {
    const std::string path = temp_path();
    trace::configure(path.c_str(), 0);
    Rig rig(Helper::Ask::kRefuse);
    CHECK(rig.tick());
    CHECK(rig.log.is({"h.stage(1)", "h.rff(1>2)", "F.stage(2)", "h.replace(1>2)", "F.replace(2>1)"}));
    CHECK(rig.helper->resident[2] == 1);
    capture::require("placement round=1 fr=1 fr_ok=0 fr_bytes=1", 1);
    capture::require("replace_from_fallback failed", 0);                                // not logged as a failure
    capture::require("placement move 0/2 2->1 src=fallback", 1);
    const std::string text = slurp(path);
    CHECK(has(text, "\"kind\":\"committed\"") && has(text, "\"src\":\"fallback\"") && !has(text, "\"src\":\"file\""));
    trace::close_trace();
    unlink(path.c_str());
}

// FAILED: a write that began did not complete; the swap is rolled back and the failure is logged like a failed replace.
void failure_is_a_failure() {
    Rig rig(Helper::Ask::kFail);
    CHECK(rig.tick());
    capture::require("placement replace_from_fallback failed: rung 1 layer 0 expert 2, io status 2", 1);
    capture::require("placement move", 0);                                              // nothing committed
    CHECK(rig.helper->resident[1] == 1 && rig.helper->resident[2] == 0);                // the helper still holds what it held
    capture::require("placement round=1 fr=1 fr_ok=0 fr_bytes=1", 1);
}

// The rung mask: exactly the cards that report reads_files(); never the fallback, recomputed every time.
void mask_bits() {
    Log log;
    Card lead("l", log, 0), plain("p", log, 1), fallback("F", log, -1);
    Helper reader(log, 2);
    const std::vector<strata::core::RungIO*> cards{&lead, &plain, &reader};
    const std::vector<size_t> sizes{kBytes};
    const PlacementRungs rungs{cards, sizes, &fallback};
    CHECK(from_fallback_mask(rungs) == 4u);          // card 2 only
    reader.files_on = false;
    CHECK(from_fallback_mask(rungs) == 0u);          // it left the mask the moment it stopped reading files
    reader.files_on = true;
    CHECK(from_fallback_mask(rungs) == 4u);
    const std::vector<strata::core::RungIO*> none;
    CHECK(from_fallback_mask(PlacementRungs{none, sizes}) == 0u);
}

// The engine's contract end to end: a REAL FileMoves over a backend whose helper answers kTierBadSum.  The move is refused, the BYTE
// path commits in the same swap, the mask loses the helper's bit, and the next round never asks (v1 call, zero file ops).
struct BadSumBackend final : FileBackend {
    HelperFacts helper_facts() noexcept override { return {2, 1, file_table_id({4096}), 0, 0, 0}; }
    bool extents(int32_t, int32_t, seabraid::FileExtent out[3], int& n) override { out[0] = {0, (uint32_t) kBytes, 64}; n = 1; return true; }
    bool in_ram(int32_t, int32_t) override { return false; }
    bool send(int32_t, int32_t, int32_t, const seabraid::FileRef&, const uint64_t*, int32_t, uint8_t& res, std::string&) override {
        ++sends;
        res = seabraid::kTierBadSum;
        return true;
    }
    int closed() noexcept override { return 0; }
    int sends = 0;
};

class FileHelper final : public Helper {
public:
    FileHelper(Log& log, int expert, FileMoves& mover) : Helper(log, expert), mover_(mover) {}
    IoStatus replace_from_fallback(int32_t l, int32_t old_e, int32_t new_e, std::string& err) override {
        log_.add(name_, "rff", old_e, new_e);
        const IoStatus status = mover_.move(l, old_e, new_e, err);
        if (status == IoStatus::kOk) { resident[old_e] = 0; resident[new_e] = 1; }
        return status;
    }
    bool reads_files() const override { return mover_.active(); }
private:
    FileMoves& mover_;
};

std::shared_ptr<PackSums> manifest_for_three() {
    seabraid::sums::Manifest m;
    m.layers = 1; m.experts = (uint32_t) kExperts; m.files = {{4096, 7}}; m.blob_bytes = {(uint32_t) kBytes};
    for (int e = 0; e < (int) kExperts; ++e) { const auto b = blob(e); m.sums.push_back(blob_sum(b.data(), b.size())); }
    std::string bytes, err;
    CHECK(seabraid::sums::serialize(m, bytes, err));
    PackSums::Facts facts;
    facts.layers = 1; facts.experts = (int64_t) kExperts; facts.files = m.files; facts.blob_bytes = m.blob_bytes;
    auto sums = PackSums::load([&](std::vector<uint8_t>& out) { out.assign(bytes.begin(), bytes.end()); return true; }, facts);
    CHECK(sums != nullptr);
    return sums;
}

void bad_sum_clears_the_mask() {
    auto* backend = new BadSumBackend();
    FileMoves mover(manifest_for_three(), std::unique_ptr<FileBackend>(backend), 1);
    Rig rig(Helper::Ask::kAck, [&](Log& log) { return std::make_unique<FileHelper>(log, 1, mover); });
    const std::vector<strata::core::RungIO*> cards{&rig.lead, rig.helper.get()};
    const std::vector<size_t> sizes{kBytes};
    const PlacementRungs rungs{cards, sizes, &rig.fallback};
    CHECK(from_fallback_mask(rungs) == 2u);                                              // the helper reads files
    CHECK(rig.tick());
    CHECK(rig.log.is({"h.stage(1)", "h.rff(1>2)", "F.stage(2)", "h.replace(1>2)", "F.replace(2>1)"}));   // refused, then bytes
    CHECK(rig.helper->resident[2] == 1 && backend->sends == 1 && mover.bad_sums() == 1);
    capture::require("file moves OFF for helper rung 1", 1);
    capture::require("placement move 0/2 2->1", 1);                                      // the swap committed through the byte path
    CHECK(from_fallback_mask(rungs) == 0u);                                              // the bit left the mask
    rig.log.calls.clear();
    for (int i = 0; i < 24 && rig.log.calls.empty(); ++i) CHECK(rig.tick(kHeatBack));   // the planner smooths heat: a few rounds until it moves 1
    CHECK(rig.log.is({"F.stage(1)", "h.stage(2)", "h.replace(2>1)", "F.replace(1>2)"}));   // v1 call: no file op at all (and no rff)
    CHECK(backend->sends == 1);
    capture::require("file moves OFF for helper rung", 1);                               // ONE warning for the session
}

}  // namespace

int main() {
    capture::start();
    by_reference_commit();
    refusal_takes_the_byte_path();
    failure_is_a_failure();
    mask_bits();
    bad_sum_clears_the_mask();
    std::puts("Seabraid Strata file-move placement fixture passed");
}
