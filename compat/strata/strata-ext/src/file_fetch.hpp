// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid helper: read an expert's bytes straight from the model files and prove they are that expert's.  DESIGN GATE
//   Owner    : "how this process turns a file op into a verified blob": the file table, the O_DIRECT window arithmetic, the staging
//              blob, the binding of every cell to the FileRef THIS helper expects for it (its own pack layout + its own
//              expert-sums.v1), the XXH3-64 check and the Header file_* counters.  What a legal FileRef is, is the wire's (file_ref.hpp);
//              the manifest's bytes are the wire's (expert_sums.hpp); the slot write and read-back are tier_round.hpp's.
//   Contract : FileTable::{open_all, bind_manifest, read, fetch, announce, stamps, unchanged, close_all}, blob_sum, file_table_id.  No
//              Strata, no GPU runtime: the staging memory is the caller's and the pack layout is a callback, so host tests drive this code.
//   Size     : hpp <= 125 lines, cpp <= 320.
//   Failure  : open_all fails (explicit mode) or serves file_io 0 with a warning (auto); bind_manifest fails with the reason and the
//              caller decides (helper.cpp: launch failure when explicit, file_io 0 when auto); fetch refuses before it reads
//              (kTierRejected, with the reason) and a sum mismatch is kTierBadSum.  Neither ever touches a slot.
//   Checker  : tests/file_fetch_test.cpp, tests/channel_file_swap_test.cpp, tests/file_cache_test.cpp, tests/sums_test.cpp,
//              tests/test_file_fetch_open_rule.py (no open() on the serve path).
// SECURITY: files are opened ONCE, at launch, read-only, O_NOFOLLOW, regular files only, from the paths the helper's own pack loader
// opened, and the O_DIRECT descriptor must be the same inode as the buffered one; a FileRef holds integers, so no path ever arrives
// from the wire.  TRUST: the sum a read is checked against and the extents read are the helper's OWN for the op's (layer, expert):
// the wire's FileRef must equal them (file_ref_mismatch) and is never used for either, because a sum riding beside its extents
// proves only that the sender knew some expert's sum.  GPUDirect Storage is a named seam: kFileIoGds is refused at launch in this build.
#include "strata/seabraid/channel.hpp"
#include "strata/seabraid/expert_sums.hpp"
#include "strata/seabraid/file_ref.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace seabraid {

/// XXH3-64, seed 0, over `n` bytes: the wire's kSumXxh3_64.  The same function builds the manifest, checks a read and (for main)
/// checks its own reads, so there is one definition of "the sum of a blob".
uint64_t blob_sum(const uint8_t* p, size_t n);
/// Header::file_table_id: blob_sum over the file sizes as little-endian u64s in table order.
uint64_t file_table_id(const uint64_t* sizes, size_t n);

/// Test seam, never set in production: called with the path between the buffered and the O_DIRECT open of one file.  WHY a seam:
/// the race it exposes (the path swapped between the two opens) cannot be produced deterministically from outside the process.
extern void (*g_between_opens_for_test)(const std::string& path);

class FileTable {
public:
    /// Where the helper's own pack keeps cell (layer, expert): 1..3 extents (FileExpertSource::file_extents in the helper).
    using ExtentsOf = std::function<bool(int64_t layer, int64_t expert, FileExtent out[3], int& n)>;

    FileTable() = default;
    ~FileTable() { close_all(); }
    FileTable(const FileTable&) = delete;
    FileTable& operator=(const FileTable&) = delete;

    /// LAUNCH ONLY: opens `paths` (read-only, O_NOFOLLOW, regular files) and picks the mode.  `staging` (`staging_bytes`, >= the
    /// largest blob) is where a blob is assembled.  kFileIoOff opens nothing.  kFileIoAuto = direct where every file has a DIO
    /// alignment, else buffered; if the table cannot be opened it serves kFileIoOff and says why in `warn` (returns true).  An
    /// explicit direct or buffered mode that cannot be honoured returns false with the reason, naming the file, in `err`.
    bool open_all(const std::vector<std::string>& paths, FileIo want, uint8_t* staging, int64_t staging_bytes, std::string& err,
                  std::string& warn);
    /// LAUNCH ONLY, after open_all: read the manifest at `manifest_path` (bounded), require it fresh for the files THIS table opened
    /// (stamps from its descriptors) and for this model's geometry, and from then on answer every cell from it and `where`.  Until it
    /// succeeds, fetch refuses everything.  False + `why`: nothing is bound.
    bool bind_manifest(const std::string& manifest_path, int64_t layers, int64_t experts, const std::vector<uint32_t>& blob_bytes,
                       ExtentsOf where, std::string& why);
    /// Serve file_io 0 from now on (and release every descriptor).
    void close_all();

    FileIo io() const { return io_; }
    int32_t count() const { return (int32_t) slots_.size(); }
    uint64_t table_id() const { return table_id_; }
    const uint8_t* staging() const { return staging_; }
    /// The size fstat gave at launch of each file, in table order (what file_ref_valid bounds against).
    const std::vector<uint64_t>& sizes() const { return sizes_; }
    /// The freshness stamp of each file as fstat gave it at launch: the stamp of the bytes this table reads, not of whatever the
    /// path names now.
    const std::vector<sums::FileStat>& stamps() const { return stamps_; }
    /// True when every descriptor still has its launch stamp and every path still names the inode this table reads.  The sums tool
    /// asks after its scan, so a pack that changed while it was read never gets a manifest.
    bool unchanged(const std::vector<std::string>& paths, std::string& why) const;

    /// Publish file_io, file_count and file_table_id (before kHelperUp), and from now on mirror the counters into `h`.
    void announce(Header& h);

    /// Validate `ref` against this table (file_ref_valid) and assemble its blob into staging().  False + `why`: nothing was read
    /// when the ref is invalid; an I/O error or short read (the file shrank since launch) names the file index, offset and errno.
    /// No sum is checked: this is the sums tool's raw read; the serve path uses fetch.
    bool read(const FileRef& ref, int64_t want_blob_bytes, std::string& why);
    /// The serve path.  The FileRef this helper expects for (layer, expert) is built from the bound manifest and pack layout; `wire`
    /// must equal it; it is read, then summed against the manifest.  kTierApplied = staging() holds that expert's blob; kTierRejected =
    /// refused or unreadable (nothing read when refused); kTierBadSum = the bytes read are not the manifest's.  It writes no slot.
    TierRes fetch(int32_t layer, int32_t expert, const FileRef& wire, int64_t want_blob_bytes, std::string& why);

    uint64_t direct_bytes() const { return direct_bytes_; }
    uint64_t buffered_bytes() const { return buffered_bytes_; }
    uint64_t reads() const { return reads_; }

private:
    struct Slot { int fd = -1, dfd = -1; uint32_t mem_align = 0, off_align = 0; };   // dfd >= 0 only when a DIO alignment is known
    bool open_files(const std::vector<std::string>& paths, bool want_direct, std::string& err);
    bool open_slot(const std::string& path, uint32_t idx, bool want_direct, Slot& s, struct stat& st, std::string& err);
    bool pick_mode(FileIo want, const std::vector<std::string>& paths, std::string& err);
    bool alloc_bounce(int64_t staging_bytes, std::string& err);
    bool expected_ref(int32_t layer, int32_t expert, FileRef& own, std::string& why) const;
    bool read_buffered(const Slot& s, const FileExtent& e, uint8_t* dst, std::string& why);
    bool read_direct(const Slot& s, const FileExtent& e, uint8_t* dst, std::string& why);
    void publish();

    std::vector<Slot> slots_;
    std::vector<uint64_t> sizes_;
    std::vector<sums::FileStat> stamps_;
    sums::Manifest manifest_;
    ExtentsOf where_;   // empty = no manifest bound: fetch refuses
    FileIo io_ = kFileIoOff;
    uint64_t table_id_ = 0, direct_bytes_ = 0, buffered_bytes_ = 0, reads_ = 0;
    uint8_t* staging_ = nullptr;
    int64_t staging_bytes_ = 0;
    uint8_t* bounce_ = nullptr;
    uint64_t bounce_bytes_ = 0, bounce_align_ = 0;
    Header* hdr_ = nullptr;
};

}  // namespace seabraid
