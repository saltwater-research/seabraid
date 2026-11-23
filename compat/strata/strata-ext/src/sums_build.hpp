// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid-sums, the part that is not Strata: read every expert of a pack through the helper's own file reader and sum it
// (build), or compare a manifest with the files (verify).  DESIGN GATE
//   Owner    : "the sums of a pack": the loop over (layer, expert), the freshness stamps, the atomic write.  The manifest's BYTES are
//              the wire's (expert_sums.hpp); the read and the hash are file_fetch.hpp's; WHERE an expert lives is the engine's
//              (the tool passes a callback over FileExpertSource::file_extents), so this header needs no Strata and is host-tested.
//   Contract : build_sums, verify_sums, write_atomic.  `ExtentsOf` is `bool(int64_t layer, int64_t expert, FileExtent out[3], int& n)`.
//              The stamps a manifest carries are FileTable::stamps() (from the descriptors read), and the caller asks
//              FileTable::unchanged() after the scan: a path is never stat'ed again to describe bytes read through a descriptor.
//   Size     : <= 120 lines.
//   Failure  : the first unreadable or unaddressable blob stops the build and names (layer, expert); nothing is written then.
//              The manifest is written to a fresh exclusive temp file, synced and renamed, so a reader sees all of it or the old one.
//   Checker  : tests/sums_test.cpp (round trip, corruption, stale, atomic write, a planted temp link) and a run on the Linux test
//              machine with a synthetic pack.
#include "file_fetch.hpp"
#include "strata/seabraid/expert_sums.hpp"

#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace seabraid::sums_build {

namespace detail {
template <class ExtentsOf>
bool sum_of_cell(FileTable& files, ExtentsOf& extents_of, int64_t l, int64_t e, uint32_t blob_bytes, uint64_t& out, std::string& err) {
    FileExtent ex[kFileRefMaxExtents];
    int n = 0;
    const std::string cell = "layer " + std::to_string(l) + " expert " + std::to_string(e) + ": ";
    if (!extents_of(l, e, ex, n) || n < 1) { err = cell + "the pack names no bytes for it"; return false; }
    std::string why;
    if (!files.read(make_file_ref(ex, n, 0), blob_bytes, why)) { err = cell + "unreadable: " + why; return false; }
    out = blob_sum(files.staging(), blob_bytes);
    return true;
}
}  // namespace detail

/// Every expert's sum, layer-major.  False names the first cell that cannot be read; `sums` is then incomplete.
template <class ExtentsOf>
bool build_sums(FileTable& files, ExtentsOf&& extents_of, uint32_t layers, uint32_t experts, const std::vector<uint32_t>& blob_bytes,
                std::vector<uint64_t>& sums, std::string& err) {
    sums.assign((size_t) layers * experts, 0);
    for (uint32_t l = 0; l < layers; ++l)
        for (uint32_t e = 0; e < experts; ++e)
            if (!detail::sum_of_cell(files, extents_of, l, e, blob_bytes[l], sums[(size_t) l * experts + e], err)) return false;
    return true;
}

/// Compares `m` with the files: every cell that reads back differently, as "layer L expert E".  True = all equal.  An unreadable
/// cell is an error (false, `err`), not a mismatch.
template <class ExtentsOf>
bool verify_sums(FileTable& files, ExtentsOf&& extents_of, const sums::Manifest& m, std::vector<std::string>& mismatched, std::string& err) {
    mismatched.clear();
    for (uint32_t l = 0; l < m.layers; ++l)
        for (uint32_t e = 0; e < m.experts; ++e) {
            uint64_t now = 0;
            if (!detail::sum_of_cell(files, extents_of, l, e, m.blob_bytes[l], now, err)) return false;
            if (now != m.sums[(size_t) l * m.experts + e]) mismatched.push_back("layer " + std::to_string(l) + " expert " + std::to_string(e));
        }
    return mismatched.empty();
}

namespace detail {
inline bool write_all(int fd, const std::string& bytes) {
    size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t w = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        done += (size_t) w;
    }
    return true;
}
}  // namespace detail

/// Writes `bytes` to `path` through a temp file in the same directory and a rename, so a reader never sees half a manifest.
/// WHY mkstemp and not a fixed `path.tmp`: mkstemp creates a NEW name with O_CREAT|O_EXCL, so a link planted at a predictable name is
/// never followed (the write cannot land in another file) and two concurrent builds never truncate each other's temp.  WHY 0644: the
/// manifest is not secret and the engine and the helper may read it as other users.  WHY fsync before the rename: after a crash the
/// name points at a complete manifest or at the old one, never at an empty file.
inline bool write_atomic(const std::string& path, const std::string& bytes, std::string& err) {
    std::string tmp = path + ".XXXXXX";
    const int fd = ::mkstemp(&tmp[0]);
    if (fd < 0) { err = "cannot create a temp file beside " + path; return false; }
    bool ok = ::fchmod(fd, 0644) == 0 && detail::write_all(fd, bytes) && ::fsync(fd) == 0;
    ok = ::close(fd) == 0 && ok;
    if (!ok || ::rename(tmp.c_str(), path.c_str()) != 0) { ::unlink(tmp.c_str()); err = "cannot write " + path; return false; }
    return true;
}

}  // namespace seabraid::sums_build
