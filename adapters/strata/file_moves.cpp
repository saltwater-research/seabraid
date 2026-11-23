// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// See file_moves.hpp for the owner, contract and failure rules.
// WHY XXH_INLINE_ALL: Strata links its own copy of xxhash.c; inlining every symbol into this TU keeps the two from colliding.
#define XXH_INLINE_ALL
#include "xxhash.h"

#include "file_moves.hpp"

#include "diag.hpp"

#include "strata/seabraid/file_ref.hpp"
#include "strata/seabraid/file_stamp.hpp"

#include <utility>

namespace seabraid::strata_adapter {
namespace {

using diag::kError;
using diag::kWarn;
using strata::core::IoStatus;

// The first rule the loaded manifest breaks against this engine, or "" (expert_sums.hpp's fresh_for, which the helper uses too).
std::string manifest_fault(const seabraid::sums::Manifest& m, const PackSums::Facts& engine) {
    std::string why;
    return seabraid::sums::fresh_for(m, engine.layers, engine.experts, engine.blob_bytes.data(), engine.files.data(), engine.files.size(), why)
               ? std::string() : why;
}

const char* kRebuild = "rebuild it with seabraid-sums";

}  // namespace

uint64_t blob_sum(const uint8_t* bytes, size_t n) noexcept { return (uint64_t) XXH3_64bits(bytes, n); }

// XXH3-64 over the file sizes as little-endian u64 in table order (the helper's file_fetch.cpp computes the same).
uint64_t file_table_id(const std::vector<uint64_t>& sizes) {
    std::vector<uint8_t> le;
    le.reserve(sizes.size() * 8);
    for (const uint64_t s : sizes)
        for (int b = 0; b < 8; ++b) le.push_back((uint8_t) (s >> (8 * b)));
    return blob_sum(le.data(), le.size());
}

PackSums::Reader PackSums::pack_file(const std::string& pack_dir) {
    // Bounded (file_stamp.hpp read_manifest): a file longer than any manifest comes back one byte too long and parse() refuses it.
    return [path = pack_dir + "/" + seabraid::sums::kFileName](std::vector<uint8_t>& bytes) {
        std::string err;
        return seabraid::sums::read_manifest(path, bytes, err);
    };
}

std::shared_ptr<PackSums> PackSums::load(const Reader& read, const Facts& engine) {
    std::vector<uint8_t> bytes;
    std::string err;
    std::shared_ptr<PackSums> sums;
    try {
        sums.reset(new PackSums());
        if (!read || !read(bytes)) {
            diag::logf(kWarn, 0, "file moves OFF: the pack has no readable expert-sums.v1 (%s)", kRebuild);
            return nullptr;
        }
        if (!seabraid::sums::parse(bytes.data(), bytes.size(), sums->manifest_, err)) {
            diag::logf(kWarn, 0, "file moves OFF: %s (%s)", err.c_str(), kRebuild);
            return nullptr;
        }
        if (const std::string fault = manifest_fault(sums->manifest_, engine); !fault.empty()) {
            diag::logf(kWarn, 0, "file moves OFF: expert-sums.v1 is stale or about another model: %s (%s)", fault.c_str(), kRebuild);
            return nullptr;
        }
        for (const auto& f : sums->manifest_.files) sums->sizes_.push_back(f.size);
        sums->table_id_ = file_table_id(sums->sizes_);
        sums->bad_.assign((size_t) sums->manifest_.layers * sums->manifest_.experts, 0);
    } catch (...) {  // allow-error-path: out of memory while loading: logged below, file moves stay off
        diag::logf(kWarn, 0, "file moves OFF: out of memory reading expert-sums.v1");
        return nullptr;
    }
    diag::logf(diag::kInfo, 0, "file moves: manifest ok, %u layers x %u experts, %zu files", sums->manifest_.layers,
               sums->manifest_.experts, sums->sizes_.size());
    return sums;
}

bool PackSums::cell_bad(int32_t layer, int32_t expert) const noexcept {
    if (layer < 0 || expert < 0 || (uint32_t) layer >= manifest_.layers || (uint32_t) expert >= manifest_.experts) return true;
    return bad_[(size_t) layer * manifest_.experts + (size_t) expert] != 0;
}

bool PackSums::read_ok(int32_t layer, int32_t expert, const uint8_t* bytes, size_t n, std::string& err) {
    uint64_t want = 0;
    if (!manifest_.sum_of(layer, expert, want) || bytes == nullptr) { err = "the manifest names no expert at that cell"; return false; }
    if (n == manifest_.blob_bytes[(size_t) layer] && blob_sum(bytes, n) == want) return true;
    // WHY mark and keep serving: the engine cannot tell whether the pack or the manifest is wrong, and refusing to serve would take
    // the model down.  The cell is left where it is (never moved) and the operator can run seabraid-sums --verify.
    ++mismatches_;
    bad_[(size_t) layer * manifest_.experts + (size_t) expert] = 1;
    err = "the pack's bytes for this expert disagree with expert-sums.v1";
    diag::logf(kError, 0, "file moves: layer %d expert %d as read from the pack disagrees with expert-sums.v1; it keeps serving "
               "and never moves (check the pack with seabraid-sums --verify)", (int) layer, (int) expert);
    return false;
}

const char* FileMoves::unfit() const {
    if (off_) return "file moves are off for this helper";
    const FileBackend::HelperFacts h = backend_->helper_facts();
    if (h.io == seabraid::kFileIoOff) return "the helper serves no file moves";
    if ((size_t) h.count != sums_->file_sizes().size() || h.table_id != sums_->table_id()) return "the helper's file table is not the manifest's";
    return nullptr;
}

bool FileMoves::active() const { return unfit() == nullptr; }

// Everything that can say "bytes, please" before anything is sent.  false: `err` says why; nothing was written or sent.
bool FileMoves::prepare(int32_t layer, int32_t old_e, int32_t new_e, seabraid::FileRef& ref, std::string& err) {
    if (const char* why = unfit()) {
        note_wrong_table();
        err = why;
        return false;
    }
    uint64_t sum = 0;
    seabraid::FileExtent ext[seabraid::kFileRefMaxExtents];
    int n = 0;
    if (old_e == new_e || sums_->cell_bad(layer, new_e) || !sums_->sum_of(layer, new_e, sum)) err = "no checked sum for that expert";
    else if (backend_->in_ram(layer, new_e)) err = "the expert is in the RAM copy, which is cheaper than a read";
    else if (!backend_->extents(layer, new_e, ext, n)) err = "the engine names no file bytes for that expert";
    else {
        ref = seabraid::make_file_ref(ext, n, sum);
        return true;
    }
    return false;
}

// A helper whose table is not the manifest's opened other files than the pack: said once, and it takes bytes only.
void FileMoves::note_wrong_table() {
    if (said_table_ || off_ || backend_->helper_facts().io == seabraid::kFileIoOff) return;
    said_table_ = true;
    diag::logf(kWarn, 0, "file moves: helper rung %u opened other files than the manifest names; it takes bytes only", rung_);
}

IoStatus FileMoves::move(int32_t layer, int32_t old_e, int32_t new_e, std::string& err) {
    bool sent = false;
    try {
        seabraid::FileRef ref{};
        if (!prepare(layer, old_e, new_e, ref, err)) return IoStatus::kRefused;
        ++attempts_;
        sent = true;   // from here a throw must not read as "nothing happened"
        return send(layer, old_e, new_e, ref, err);
    } catch (...) {  // allow-error-path: only allocation can throw here; after the send began the state is unknown, so TIMEOUT
        err = "file move: allocation failed";
        return sent ? IoStatus::kTimeout : IoStatus::kRefused;
    }
}

IoStatus FileMoves::send(int32_t layer, int32_t old_e, int32_t new_e, const seabraid::FileRef& ref, std::string& err) {
    uint8_t res = (uint8_t) seabraid::kTierPending;
    std::string why;
    const bool ok = backend_->send(layer, old_e, new_e, ref, sums_->file_sizes().data(), (int32_t) sums_->file_sizes().size(), res, why);
    // The engine closes the card itself on a timeout or a torn slot, exactly as for a blob swap: that outranks any result.
    if (const int closed = backend_->closed(); closed != 0) return closed == 1 ? IoStatus::kTimeout : IoStatus::kFailed;
    if (!ok) { ++refused_; err = why; return IoStatus::kRefused; }   // refused before anything was sent
    return answer(res, layer, new_e, err);
}

IoStatus FileMoves::answer(uint8_t res, int32_t layer, int32_t expert, std::string& err) {
    switch (res) {
    case seabraid::kTierApplied: ++applied_; return IoStatus::kOk;
    case seabraid::kTierFailed: return IoStatus::kFailed;
    case seabraid::kTierBadSum: return bad_sum(layer, expert);
    default: break;   // kTierRejected: the helper refused before touching a slot
    }
    ++refused_;
    err = "the helper refused the file reference";
    return IoStatus::kRefused;
}

IoStatus FileMoves::bad_sum(int32_t layer, int32_t expert) {
    ++bad_sums_;
    ++refused_;
    if (!off_) {
        off_ = true;   // WHY sticky: a helper that read the wrong bytes once is reading the wrong file; it takes bytes only for the session
        diag::logf(kWarn, 0, "file moves OFF for helper rung %u: it read layer %d expert %d from its file table and got other bytes than "
                   "the manifest; this helper now takes bytes only", rung_, (int) layer, (int) expert);
    }
    return IoStatus::kRefused;   // the byte path follows in the same swap and re-verifies main's own read
}

void FileMoves::add_totals(FileIoTotals& total) const {
    const FileBackend::HelperFacts h = backend_->helper_facts();
    total.direct_bytes += h.direct_bytes;
    total.buffered_bytes += h.buffered_bytes;
    total.reads += h.reads;
    total.bad_sum += bad_sums_;
}

void log_file_io(const std::vector<const FileMoves*>& helpers) noexcept {
    if (helpers.empty()) return;
    FileIoTotals total;
    for (const FileMoves* h : helpers) h->add_totals(total);
    diag::logf(diag::kInfo, 0, "helper file io: direct_bytes=%llu buffered_bytes=%llu reads=%llu bad_sum=%llu",
               (unsigned long long) total.direct_bytes, (unsigned long long) total.buffered_bytes,
               (unsigned long long) total.reads, (unsigned long long) total.bad_sum);
}

}  // namespace seabraid::strata_adapter
