// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// File-reference expert moves, engine side (track FR-E): the pack's checksum manifest and the rule for when a helper may
// load a promoted expert from the model file instead of receiving its bytes.
// OWNER: "may THIS expert move by file reference, and what does the helper's answer mean".  The format of the manifest is
//   wire/strata/seabraid/expert_sums.hpp, the FileRef rules are wire/.../file_ref.hpp, the addressing of an expert in the files
//   is Strata's FileExpertSource::file_extents; this file owns none of those, it only decides.
// CONTRACT: PackSums (one per pack: manifest, freshness, main's own read check, the cells that must never move) and
//   FileMoves (one per helper rung: the decision, the session switch-off, the counters).  The engine facts they need
//   (extents, the RAM copy, the helper's announcement, the send) come through FileBackend, implemented over RemoteExperts and
//   FileExpertSource in src/program/generate.cpp and faked in file_moves_test.  Engine thread only.
// SIZE BUDGET: header 150, file_moves.cpp 280.
// FAILURE CONTRACT: fail open to bytes.  Every refusal is IoStatus::kRefused (nothing written; Seabraid's swap then runs the
//   byte path in the same swap); a missing, malformed, stale or mismatched manifest turns file moves OFF with one WARN and never
//   stops serving; a helper's bad sum turns file moves off for that helper for the session; a cell whose main-side read
//   disagrees with the manifest keeps serving from the pack, logs ERROR and never moves.  Only a timeout or a torn slot (the
//   helper's own result) is anything but a refusal, exactly as for a blob swap.
// CHECKER: file_moves_test (each refusal, the bad-sum switch-off, the WARN-off seam, the counter line).
#include "strata/seabraid/channel_layout.hpp"
#include "strata/seabraid/expert_sums.hpp"
#include "strata/core/ownership_map.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace seabraid::strata_adapter {

// The two hashes of the manifest scheme, so the engine, the tests and any other reader share one implementation (XXH3-64, seed 0).
uint64_t blob_sum(const uint8_t* bytes, size_t n) noexcept;                 // a blob's sum, as the manifest stores it
uint64_t file_table_id(const std::vector<uint64_t>& file_sizes);            // the helper's Header::file_table_id for these file sizes

// The manifest of one pack, loaded and checked once at engine start.
class PackSums final : public strata::core::ReadCheck {
public:
    // The seam: the manifest file's bytes.  False = there is no manifest to read.
    using Reader = std::function<bool(std::vector<uint8_t>& bytes)>;
    // What the engine knows about the pack right now: the stamp of each file, the geometry, every layer's blob size.
    struct Facts {
        std::vector<seabraid::sums::FileStat> files;
        int64_t layers = 0, experts = 0;
        std::vector<uint32_t> blob_bytes;
    };
    // nullptr, with ONE WARN naming the rule that failed, when the manifest is missing, malformed, stale or about another model.
    // The caller then runs without file moves; nothing else changes.
    static std::shared_ptr<PackSums> load(const Reader& read, const Facts& engine);
    // The reader for `<pack_dir>/expert-sums.v1`.
    static Reader pack_file(const std::string& pack_dir);

    bool sum_of(int64_t layer, int64_t expert, uint64_t& out) const { return manifest_.sum_of(layer, expert, out); }
    const std::vector<uint64_t>& file_sizes() const noexcept { return sizes_; }
    uint64_t table_id() const noexcept { return table_id_; }
    // Main's own read of a blob against the manifest.  false: the pack disagrees (ERROR logged, the cell is marked and never moves).
    bool read_ok(int32_t layer, int32_t expert, const uint8_t* bytes, size_t n, std::string& err) override;
    bool cell_bad(int32_t layer, int32_t expert) const noexcept;
    uint64_t read_mismatches() const noexcept { return mismatches_; }

private:
    PackSums() = default;
    seabraid::sums::Manifest manifest_;
    std::vector<uint64_t> sizes_;
    uint64_t table_id_ = 0;
    std::vector<uint8_t> bad_;      // layers x experts, 1 = a read of that cell disagreed with the manifest
    uint64_t mismatches_ = 0;
};

// What only the engine can answer.
struct FileBackend {
    struct HelperFacts { int32_t io = 0, count = 0; uint64_t table_id = 0, direct_bytes = 0, buffered_bytes = 0, reads = 0; };
    virtual ~FileBackend() = default;
    virtual HelperFacts helper_facts() noexcept = 0;                  // what the helper announced, plus its counters
    // These three may throw (allocation): FileMoves::move turns a throw into REFUSED before the send and TIMEOUT after it.
    virtual bool extents(int32_t layer, int32_t expert, seabraid::FileExtent out[3], int& n) = 0;
    virtual bool in_ram(int32_t layer, int32_t expert) = 0;           // the engine's RAM copy holds it (cheaper than a read)
    // One kTierSwapFile round.  false: refused before anything was sent.  `res` is a seabraid::kTier* result otherwise.
    virtual bool send(int32_t layer, int32_t old_e, int32_t new_e, const seabraid::FileRef& ref, const uint64_t* file_size,
                      int32_t file_count, uint8_t& res, std::string& err) = 0;
    virtual int closed() noexcept = 0;                                // RemoteExperts::seabraid_closed(): 0 open, 1 timeout, 2 torn
};

struct FileIoTotals { uint64_t direct_bytes = 0, buffered_bytes = 0, reads = 0, bad_sum = 0; };

// One helper rung's file moves.
class FileMoves final : public strata::core::FileMover {
public:
    FileMoves(std::shared_ptr<PackSums> sums, std::unique_ptr<FileBackend> backend, unsigned rung) noexcept
        : sums_(std::move(sums)), backend_(std::move(backend)), rung_(rung) {}

    // The helper announced file io, its file table is the manifest's, and nothing has switched file moves off for it.
    bool active() const override;
    strata::core::IoStatus move(int32_t layer, int32_t old_e, int32_t new_e, std::string& err) override;

    uint64_t attempts() const noexcept { return attempts_; }
    uint64_t applied() const noexcept { return applied_; }
    uint64_t refused() const noexcept { return refused_; }
    uint64_t bad_sums() const noexcept { return bad_sums_; }
    bool switched_off() const noexcept { return off_; }
    void add_totals(FileIoTotals& total) const;   // this helper's announced counters plus main's bad-sum count

private:
    const char* unfit() const;                    // why this helper cannot take a file move now, or nullptr
    bool prepare(int32_t layer, int32_t old_e, int32_t new_e, seabraid::FileRef& ref, std::string& err);
    void note_wrong_table();
    strata::core::IoStatus send(int32_t layer, int32_t old_e, int32_t new_e, const seabraid::FileRef& ref, std::string& err);
    strata::core::IoStatus answer(uint8_t res, int32_t layer, int32_t expert, std::string& err);
    strata::core::IoStatus bad_sum(int32_t layer, int32_t expert);

    std::shared_ptr<PackSums> sums_;
    std::unique_ptr<FileBackend> backend_;
    unsigned rung_;
    bool off_ = false, said_table_ = false;
    uint64_t attempts_ = 0, applied_ = 0, refused_ = 0, bad_sums_ = 0;
};

// One INFO line, cumulative over every helper: "helper file io: direct_bytes=N buffered_bytes=N reads=N bad_sum=N"
// (the line placement-qual's arms.json reads).  Nothing is logged for an empty list.
void log_file_io(const std::vector<const FileMoves*>& helpers) noexcept;

}  // namespace seabraid::strata_adapter
