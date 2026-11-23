// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// See file_fetch.hpp for the design gate.  Layout of this file: the sum, the LAUNCH-ONLY section (the only code that opens a file;
// tests/test_file_fetch_open_rule.py fails on an open( anywhere else), then the serve path (reads only).
// WHY XXH_INLINE_ALL: Strata links its own copy of xxhash.c; inlining every symbol into this TU keeps the two from ever colliding.
#define XXH_INLINE_ALL
#include "xxhash.h"

#include "file_fetch.hpp"
#include "strata/seabraid/file_stamp.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace seabraid {

uint64_t blob_sum(const uint8_t* p, size_t n) { return (uint64_t) XXH3_64bits(p, n); }

uint64_t file_table_id(const uint64_t* sizes, size_t n) {
    std::string le;   // explicit little-endian, so the id never depends on the host's byte order
    for (size_t i = 0; i < n; ++i)
        for (int b = 0; b < 8; ++b) le.push_back((char) (sizes[i] >> (8 * b)));
    return blob_sum((const uint8_t*) le.data(), le.size());
}

namespace {
bool pow2(uint32_t v) { return v != 0 && (v & (v - 1)) == 0; }

std::string io_text(uint32_t file, uint64_t offset, const char* what, int err) {
    return "file " + std::to_string(file) + " offset " + std::to_string(offset) + ": " + what +
           (err != 0 ? std::string(" failed: ") + std::strerror(err) : std::string());
}

enum class Dio { kNo, kYes, kUnknown };

// What the kernel says about direct I/O on this file (statx STATX_DIOALIGN, Linux 6.1+).  kUnknown = it does not say.
Dio dio_alignment(int fd, uint32_t& mem, uint32_t& off) {
#ifdef STATX_DIOALIGN
    struct statx sx{};
    if (::statx(fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0 && (sx.stx_mask & STATX_DIOALIGN) != 0) {
        mem = sx.stx_dio_mem_align; off = sx.stx_dio_offset_align;
        return pow2(off) && (mem == 0 || pow2(mem)) ? Dio::kYes : Dio::kNo;   // offset align 0 = no direct I/O on this file
    }
#else
    (void) fd; (void) mem; (void) off;
#endif
    return Dio::kUnknown;
}

// An older kernel or a filesystem that does not report the alignment: assume 4096 and let one aligned read decide.
bool probe_direct(int dfd) {
    void* buf = nullptr;
    if (posix_memalign(&buf, 4096, 4096) != 0) return false;
    const ssize_t r = ::pread(dfd, buf, 4096, 0);
    std::free(buf);
    return r >= 0;
}
}  // namespace

// ---- LAUNCH ONLY: every open() of this file is between here and END LAUNCH ONLY ------------------------------------------

void (*g_between_opens_for_test)(const std::string& path) = nullptr;

namespace {
bool same_inode(const struct stat& a, const struct stat& b) { return a.st_dev == b.st_dev && a.st_ino == b.st_ino; }

// The O_DIRECT descriptor of a file, when the kernel offers direct I/O on it.  A failed open or no alignment just leaves the file
// buffered-only; a descriptor that is not the buffered one's inode is a launch refusal (false).
// WHY refuse and not fall back to buffered: the path was swapped between the two opens, so the pack is changing under the helper,
// and the sizes and stamps taken from the first descriptor would describe a file the reads never touch.
bool open_direct(const std::string& path, int fd, const struct stat& st, int& dfd, uint32_t& mem_align, uint32_t& off_align) {
    if (g_between_opens_for_test != nullptr) g_between_opens_for_test(path);
    const int d = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECT);
    if (d < 0) return true;
    struct stat dst{};
    if (::fstat(d, &dst) != 0 || !same_inode(st, dst)) { ::close(d); return false; }
    uint32_t mem = 0, off = 0;
    Dio dio = dio_alignment(fd, mem, off);
    if (dio == Dio::kUnknown) { mem = off = 4096; dio = probe_direct(d) ? Dio::kYes : Dio::kNo; }
    if (dio != Dio::kYes) { ::close(d); return true; }
    dfd = d; mem_align = mem; off_align = off;
    return true;
}
}  // namespace

bool FileTable::open_slot(const std::string& path, uint32_t idx, bool want_direct, Slot& s, struct stat& st, std::string& err) {
    const std::string at = "file " + std::to_string(idx) + " (" + path + "): ";
    s.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (s.fd < 0) { err = at + "cannot open: " + std::strerror(errno); return false; }
    if (::fstat(s.fd, &st) != 0 || !S_ISREG(st.st_mode)) { err = at + "not a regular file"; return false; }
    if (want_direct && !open_direct(path, s.fd, st, s.dfd, s.mem_align, s.off_align)) {
        err = at + "the path changed between its two opens (another file is there now)";
        return false;
    }
    return true;
}

// The manifest is read here, at launch, through the wire's bounded reader; nothing is read from disk for it after this.
bool FileTable::bind_manifest(const std::string& manifest_path, int64_t layers, int64_t experts, const std::vector<uint32_t>& blob_bytes,
                              ExtentsOf where, std::string& why) {
    where_ = nullptr;
    manifest_ = sums::Manifest{};
    std::vector<uint8_t> bytes;
    sums::Manifest m;
    if (io_ == kFileIoOff) { why = "this helper serves no file moves (file_io 0)"; return false; }
    if (!where || (int64_t) blob_bytes.size() != layers) { why = "no pack layout to bind the manifest to"; return false; }
    if (!sums::read_manifest(manifest_path, bytes, why) || !sums::parse(bytes.data(), bytes.size(), m, why)) return false;
    if (!sums::fresh_for(m, layers, experts, blob_bytes.data(), stamps_.data(), stamps_.size(), why)) {
        why = "expert-sums.v1 is stale or about another model: " + why + "; rebuild it with seabraid-sums";
        return false;
    }
    manifest_ = std::move(m);
    where_ = std::move(where);
    return true;
}
// ---- END LAUNCH ONLY -------------------------------------------------------------------------------------------------------

bool FileTable::open_files(const std::vector<std::string>& paths, bool want_direct, std::string& err) {
    if (paths.empty() || paths.size() > (size_t) kFileTableMax) { err = "the pack names " + std::to_string(paths.size()) + " files, not 1.." + std::to_string(kFileTableMax); return false; }
    slots_.reserve(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
        slots_.emplace_back();
        struct stat st{};
        if (!open_slot(paths[i], (uint32_t) i, want_direct, slots_.back(), st, err)) return false;
        sizes_.push_back((uint64_t) st.st_size);
        stamps_.push_back(sums::stamp_of(st));   // from the descriptor that is read, never from the path again
    }
    table_id_ = file_table_id(sizes_.data(), sizes_.size());
    return true;
}

bool FileTable::unchanged(const std::vector<std::string>& paths, std::string& why) const {
    if (paths.size() != slots_.size()) { why = "the table holds " + std::to_string(slots_.size()) + " files, not " + std::to_string(paths.size()); return false; }
    for (size_t i = 0; i < paths.size(); ++i) {
        struct stat now{}, at{};
        const std::string f = "file " + std::to_string(i);
        if (::fstat(slots_[i].fd, &now) != 0 || ::lstat(paths[i].c_str(), &at) != 0) { why = f + " cannot be stat'ed"; return false; }
        const sums::FileStat s = sums::stamp_of(now);
        if (s.size != stamps_[i].size || s.mtime_ns != stamps_[i].mtime_ns) { why = f + " changed while it was read"; return false; }
        if (!same_inode(now, at)) { why = f + " was replaced while it was read"; return false; }
    }
    return true;
}

bool FileTable::pick_mode(FileIo want, const std::vector<std::string>& paths, std::string& err) {
    size_t first_without = slots_.size();
    for (size_t i = slots_.size(); i-- > 0;)
        if (slots_[i].dfd < 0) first_without = i;
    const bool all_direct = first_without == slots_.size();
    if (want == kFileIoDirect && !all_direct) {
        err = "--file-io direct: file " + std::to_string(first_without) + " (" + paths[first_without] + ") has no direct I/O alignment on its filesystem";
        return false;
    }
    io_ = want == kFileIoBuffered || !all_direct ? kFileIoBuffered : kFileIoDirect;
    if (io_ == kFileIoBuffered)
        for (Slot& s : slots_) { if (s.dfd >= 0) ::close(s.dfd); s.dfd = -1; }
    return true;
}

bool FileTable::alloc_bounce(int64_t staging_bytes, std::string& err) {
    if (io_ != kFileIoDirect) return true;
    for (const Slot& s : slots_) bounce_align_ = std::max<uint64_t>(bounce_align_, std::max(s.off_align, s.mem_align));
    bounce_bytes_ = (uint64_t) staging_bytes + 2 * bounce_align_;   // the largest extent's window: its length plus a block each side
    void* p = nullptr;
    if (posix_memalign(&p, (size_t) bounce_align_, (size_t) bounce_bytes_) != 0) { err = "cannot allocate the direct-read bounce buffer"; return false; }
    bounce_ = (uint8_t*) p;
    return true;
}

bool FileTable::open_all(const std::vector<std::string>& paths, FileIo want, uint8_t* staging, int64_t staging_bytes, std::string& err,
                         std::string& warn) {
    close_all();
    err.clear(); warn.clear();
    if (want == kFileIoOff) return true;
    if (want == kFileIoGds) { err = "--file-io gds is not served by this build"; return false; }
    if (staging == nullptr || staging_bytes <= 0) { err = "file moves need a staging buffer"; return false; }
    staging_ = staging; staging_bytes_ = staging_bytes;
    std::string why;
    if (open_files(paths, want != kFileIoBuffered, why) && pick_mode(want, paths, why) && alloc_bounce(staging_bytes, why)) return true;
    close_all();
    if (want == kFileIoAuto) { warn = "file moves are off: " + why; return true; }   // auto never fails a launch
    err = why;
    return false;
}

void FileTable::close_all() {
    for (Slot& s : slots_) {
        if (s.fd >= 0) ::close(s.fd);
        if (s.dfd >= 0) ::close(s.dfd);
    }
    slots_.clear(); sizes_.clear(); stamps_.clear();
    manifest_ = sums::Manifest{};
    where_ = nullptr;
    std::free(bounce_);
    bounce_ = nullptr; bounce_bytes_ = bounce_align_ = 0;
    staging_ = nullptr; staging_bytes_ = 0; table_id_ = 0; io_ = kFileIoOff;
}

// ---- serve path: reads only ------------------------------------------------------------------------------------------------

void FileTable::publish() {
    if (hdr_ == nullptr) return;
    __atomic_store_n(&hdr_->file_direct_bytes, direct_bytes_, __ATOMIC_RELAXED);
    __atomic_store_n(&hdr_->file_buffered_bytes, buffered_bytes_, __ATOMIC_RELAXED);
    __atomic_store_n(&hdr_->file_reads, reads_, __ATOMIC_RELAXED);
}

void FileTable::announce(Header& h) {
    h.file_io = (int32_t) io_;
    h.file_count = count();
    h.file_table_id = table_id_;
    hdr_ = &h;
    publish();
}

bool FileTable::read_buffered(const Slot& s, const FileExtent& e, uint8_t* dst, std::string& why) {
    uint64_t done = 0;
    while (done < e.length) {
        const ssize_t r = ::pread(s.fd, dst + done, (size_t) (e.length - done), (off_t) (e.offset + done));
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) { why = io_text(e.file, e.offset + done, "read", errno); return false; }
        if (r == 0) { why = io_text(e.file, e.offset + done, "short read (the file is shorter than at launch)", 0); return false; }
        done += (uint64_t) r;
    }
    buffered_bytes_ += e.length;
    return true;
}

// The window [floor(offset), ceil(offset + length)) in blocks of `u`, read O_DIRECT into the aligned bounce buffer, then the
// extent's own bytes copied out.  A short read is accepted only when it still covers the extent (end of file inside the last block).
bool FileTable::read_direct(const Slot& s, const FileExtent& e, uint8_t* dst, std::string& why) {
    const uint64_t u = std::max(s.off_align, s.mem_align);   // powers of two (dio_alignment), so u is a multiple of both
    const uint64_t start = e.offset / u * u, need = (uint64_t) e.offset + e.length - start;
    const uint64_t window = (need + u - 1) / u * u;
    if (window > bounce_bytes_) { why = io_text(e.file, e.offset, "extent window exceeds the bounce buffer", 0); return false; }
    uint64_t got = 0;
    while (got < need) {
        const ssize_t r = ::pread(s.dfd, bounce_ + got, (size_t) (window - got), (off_t) (start + got));
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) { why = io_text(e.file, e.offset, "direct read", errno); return false; }
        if (r == 0) break;
        got += (uint64_t) r;
        if (got % u != 0) break;   // the end of the file fell inside a block
    }
    if (got < need) { why = io_text(e.file, e.offset, "short direct read (the file is shorter than at launch)", 0); return false; }
    std::memcpy(dst, bounce_ + (e.offset - start), e.length);
    direct_bytes_ += e.length;
    return true;
}

bool FileTable::read(const FileRef& ref, int64_t want_blob_bytes, std::string& why) {
    if (io_ == kFileIoOff) { why = "this helper serves no file moves (file_io 0)"; return false; }
    // Every refusal of the ref is HERE, before the first read, and so before any slot is written.
    if (!file_ref_valid(ref, sizes_.data(), count(), want_blob_bytes, staging_bytes_, why)) return false;
    uint64_t pos = 0;
    bool ok = true;
    for (uint32_t i = 0; ok && i < ref.n_extents; ++i) {
        const FileExtent& e = ref.extent[i];
        ok = io_ == kFileIoDirect ? read_direct(slots_[e.file], e, staging_ + pos, why) : read_buffered(slots_[e.file], e, staging_ + pos, why);
        pos += e.length;
        ++reads_;
    }
    publish();
    return ok;
}

// The FileRef this helper expects for the cell: its own pack's extents and its own manifest's sum.  False = it has none.
bool FileTable::expected_ref(int32_t layer, int32_t expert, FileRef& own, std::string& why) const {
    if (!where_) { why = "this helper has no expert-sums.v1 bound, so it cannot check a file move"; return false; }
    uint64_t sum = 0;
    FileExtent ext[kFileRefMaxExtents];
    int n = 0;
    if (!manifest_.sum_of(layer, expert, sum)) { why = "the manifest has no expert " + std::to_string(expert) + " in layer " + std::to_string(layer); return false; }
    if (!where_(layer, expert, ext, n) || n < 1 || n > kFileRefMaxExtents) { why = "this helper's pack names no file bytes for that expert"; return false; }
    own = make_file_ref(ext, n, sum);
    return true;
}

// WHY a ref that is not the cell's own is kTierRejected and not kTierBadSum: nothing was read, so it is a refused ref like any other
// bound failure (main fails open to bytes for this swap); kTierBadSum keeps its one meaning, "the file held other bytes than the
// manifest", which is what makes main switch file moves off for the helper.
TierRes FileTable::fetch(int32_t layer, int32_t expert, const FileRef& wire, int64_t want_blob_bytes, std::string& why) {
    if (io_ == kFileIoOff) { why = "this helper serves no file moves (file_io 0)"; return kTierRejected; }
    FileRef own{};
    if (!expected_ref(layer, expert, own, why)) return kTierRejected;
    why = file_ref_mismatch(wire, own);
    if (!why.empty()) return kTierRejected;
    if (!read(own, want_blob_bytes, why)) return kTierRejected;   // the helper's own ref: the wire's is never read from
    if (blob_sum(staging_, own.blob_bytes) == own.sum) return kTierApplied;
    why = "the bytes read from the file table are not the manifest's for this expert (sum mismatch)";
    return kTierBadSum;
}

}  // namespace seabraid
