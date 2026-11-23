// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// The pack-file facts both sides read from disk: the freshness stamp of a pack's files (size and mtime of each file, in table order:
// what expert_sums.hpp matches_files compares) and the bounded read of the manifest itself.
// OWNER: the formula of the stamp, for its writer (the manifest tool, from the descriptors it read), the helper (from the descriptors it
//   serves) and the engine at start (from the paths): one formula, so no two of them can disagree about what "unchanged" means.  And
//   the one manifest read, bounded by expert_sums.hpp's kMaxManifestBytes, for all three readers.
// SIZE BUDGET: 60 lines.  FAILURE: false with the first unreadable file named in `err` (callers decide whether a path may be shown).
// CHECKER: the manifest tool's sums_test, the helper's file_fetch_test (bind_manifest) and adapters/strata/file_moves_test.cpp.
#include "strata/seabraid/expert_sums.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace seabraid::sums {

/// The stamp of one file from its stat (or fstat) record.
inline FileStat stamp_of(const struct stat& st) {
    return {(uint64_t) st.st_size, (uint64_t) st.st_mtim.tv_sec * 1000000000ull + (uint64_t) st.st_mtim.tv_nsec};
}

/// The stamps of `paths` as they are now (the engine's check at start: it holds no descriptors of the pack files).
inline bool stamp_files(const std::vector<std::string>& paths, std::vector<FileStat>& out, std::string& err) {
    out.clear();
    for (const std::string& p : paths) {
        struct stat st{};
        if (::stat(p.c_str(), &st) != 0) { err = "cannot stat " + p; return false; }
        out.push_back(stamp_of(st));
    }
    return true;
}

/// The bytes of the manifest at `path`, never more than kMaxManifestBytes + 1 of them.  A longer file comes back exactly one byte
/// too long, which parse() refuses by length: the reader's memory is bounded by the format, not by whatever sits at the path.
inline bool read_manifest(const std::string& path, std::vector<uint8_t>& out, std::string& err) {
    out.clear();
    std::ifstream in(path, std::ios::binary);
    if (!in) { err = "cannot open " + path; return false; }
    out.resize((size_t) kMaxManifestBytes + 1);
    in.read((char*) out.data(), (std::streamsize) out.size());
    out.resize((size_t) in.gcount());
    if (in.bad()) { err = "cannot read " + path; return false; }
    return true;
}

}  // namespace seabraid::sums
