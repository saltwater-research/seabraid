// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid wire v9: the `expert-sums.v1` manifest, <pack_dir>/expert-sums.v1.  OWNER of the FORMAT only: parse, serialize and
// the freshness checks.  WHY no hashing here: the hash (XXH3-64, kSumXxh3_64) lives in the consumers that link Strata, so this
// header stays pure and a second reader needs no hash library to validate a manifest's shape.
// WHY a manifest at install time and not sums computed when needed: file moves exist for NVMe-only experts main never read,
// so main has no bytes to hash at move time; hashing them then would be the very read the move removes.
//
// Bytes (all little-endian, no struct is ever mapped, so no padding or alignment assumption):
//   0   magic "SBSUMS1\0" (8)      8  version u32 (=1)   12  sum_kind u32 (=kSumXxh3_64)   16  layers u32   20  experts u32
//   24  n_files u32                28 reserved u32 (=0)   32  n_files x { size u64, mtime_ns u64 }
//       layers x blob_bytes u32, then layers x experts x sum u64 (layer-major).   ~49 KB for 48 x 128.
#include "strata/seabraid/channel_layout.hpp"

#include <cstdint>
#include <cstring>
#include <utility>
#include <string>
#include <vector>

namespace seabraid::sums {

inline constexpr char kMagic[8] = {'S', 'B', 'S', 'U', 'M', 'S', '1', '\0'};
inline constexpr uint32_t kFormatVersion = 1;
inline constexpr size_t kHeaderBytes = 32, kFileEntryBytes = 16;
inline constexpr uint32_t kMaxLayers = 1024, kMaxExperts = 512;   // = helper_options.hpp's bounds: a bigger manifest is not ours
inline constexpr const char* kFileName = "expert-sums.v1";

struct FileStat { uint64_t size; uint64_t mtime_ns; };

struct Manifest {
    uint32_t sum_kind = kSumXxh3_64, layers = 0, experts = 0;
    std::vector<FileStat> files;
    std::vector<uint32_t> blob_bytes;   // per layer
    std::vector<uint64_t> sums;         // layers * experts, layer-major

    bool sum_of(int64_t layer, int64_t expert, uint64_t& out) const {
        if (layer < 0 || expert < 0 || layer >= layers || expert >= experts) return false;
        out = sums[(size_t) layer * experts + (size_t) expert];
        return true;
    }
};

namespace detail {
inline uint32_t rd32(const uint8_t* p) { return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24; }
inline uint64_t rd64(const uint8_t* p) { return (uint64_t) rd32(p) | (uint64_t) rd32(p + 4) << 32; }
inline void wr32(std::string& o, uint32_t v) { for (int i = 0; i < 4; ++i) o.push_back((char) (v >> (8 * i))); }
inline void wr64(std::string& o, uint64_t v) { wr32(o, (uint32_t) v); wr32(o, (uint32_t) (v >> 32)); }
}  // namespace detail

/// Exact byte length a manifest of this shape must have (0 when the shape is out of range).  64-bit math, bounded inputs.
inline constexpr uint64_t expected_bytes(uint32_t layers, uint32_t experts, uint32_t n_files) {
    if (layers == 0 || layers > kMaxLayers || experts == 0 || experts > kMaxExperts || n_files == 0 || n_files > (uint32_t) kFileTableMax) return 0;
    return kHeaderBytes + (uint64_t) n_files * kFileEntryBytes + (uint64_t) layers * 4 + (uint64_t) layers * experts * 8;
}

/// The longest manifest any reader accepts (~4.2 MB).  WHY every reader bounds its read by it: no valid manifest is longer, so
/// reading further only lets a planted multi-GB file exhaust the reader's memory before parse() refuses it by length.
inline constexpr uint64_t kMaxManifestBytes = expected_bytes(kMaxLayers, kMaxExperts, (uint32_t) kFileTableMax);

namespace detail {
/// The header's rules; on success `m` carries the geometry and `n_files` the file count.  Counts are bounded here, before
/// any length they imply is computed or trusted.
inline bool parse_header(const uint8_t* bytes, size_t n, Manifest& m, uint32_t& n_files, std::string& err) {
    if (bytes == nullptr || n < kHeaderBytes) { err = "expert-sums: " + std::to_string(n) + " bytes is shorter than the " + std::to_string(kHeaderBytes) + "-byte header"; return false; }
    if (std::memcmp(bytes, kMagic, sizeof(kMagic)) != 0) { err = "expert-sums: bad magic"; return false; }
    if (rd32(bytes + 8) != kFormatVersion) { err = "expert-sums: format version " + std::to_string(rd32(bytes + 8)) + ", this build reads " + std::to_string(kFormatVersion); return false; }
    m.sum_kind = rd32(bytes + 12); m.layers = rd32(bytes + 16); m.experts = rd32(bytes + 20);
    n_files = rd32(bytes + 24);
    if (m.sum_kind != kSumXxh3_64) { err = "expert-sums: sum kind " + std::to_string(m.sum_kind) + " is not XXH3-64"; return false; }
    if (rd32(bytes + 28) != 0) { err = "expert-sums: reserved field is not zero"; return false; }
    const uint64_t want = expected_bytes(m.layers, m.experts, n_files);
    if (want == 0) { err = "expert-sums: geometry " + std::to_string(m.layers) + " layers x " + std::to_string(m.experts) + " experts x " + std::to_string(n_files) + " files is out of range"; return false; }
    if ((uint64_t) n != want) { err = "expert-sums: " + std::to_string(n) + " bytes, the header implies " + std::to_string(want); return false; }
    return true;
}
}  // namespace detail

/// Parses `n` bytes.  On any refusal `out` is left empty and `err` names the first broken rule.  Never trusts a count before
/// the length it implies has been checked against `n`.
inline bool parse(const uint8_t* bytes, size_t n, Manifest& out, std::string& err) {
    out = Manifest{};
    Manifest m;
    uint32_t n_files = 0;
    if (!detail::parse_header(bytes, n, m, n_files, err)) return false;
    const uint8_t* p = bytes + kHeaderBytes;
    for (uint32_t i = 0; i < n_files; ++i, p += kFileEntryBytes) m.files.push_back({detail::rd64(p), detail::rd64(p + 8)});
    for (uint32_t l = 0; l < m.layers; ++l, p += 4) {
        const uint32_t b = detail::rd32(p);
        if (b == 0) { err = "expert-sums: layer " + std::to_string(l) + " has a zero blob size"; return false; }
        m.blob_bytes.push_back(b);
    }
    for (size_t i = 0; i < (size_t) m.layers * m.experts; ++i, p += 8) m.sums.push_back(detail::rd64(p));
    out = std::move(m);
    return true;
}

/// The writer's half (the seabraid-sums tool and tests): the manifest's bytes.  The caller fills `sums` with its own hashes.
inline bool serialize(const Manifest& m, std::string& out, std::string& err) {
    const uint64_t want = expected_bytes(m.layers, m.experts, (uint32_t) m.files.size());
    if (want == 0 || m.sum_kind != kSumXxh3_64 || m.blob_bytes.size() != m.layers || m.sums.size() != (size_t) m.layers * m.experts) {
        err = "expert-sums: manifest shape is inconsistent"; return false;
    }
    out.assign(kMagic, sizeof(kMagic));
    detail::wr32(out, kFormatVersion); detail::wr32(out, m.sum_kind); detail::wr32(out, m.layers); detail::wr32(out, m.experts);
    detail::wr32(out, (uint32_t) m.files.size()); detail::wr32(out, 0);
    for (const FileStat& f : m.files) { detail::wr64(out, f.size); detail::wr64(out, f.mtime_ns); }
    for (uint32_t b : m.blob_bytes) detail::wr32(out, b);
    for (uint64_t s : m.sums) detail::wr64(out, s);
    return out.size() == want;
}

/// Fresh = every file the pack names has the size and mtime the manifest saw, in the same order.  A stale manifest turns file
/// moves off with one WARN naming the rebuild command; it is never repaired in place.
inline bool matches_files(const Manifest& m, const FileStat* now, size_t n, std::string& why) {
    if (n != m.files.size() || (n != 0 && now == nullptr)) { why = "pack has " + std::to_string(n) + " files, manifest has " + std::to_string(m.files.size()); return false; }
    for (size_t i = 0; i < n; ++i)
        if (now[i].size != m.files[i].size || now[i].mtime_ns != m.files[i].mtime_ns) { why = "file " + std::to_string(i) + " changed since the manifest was built"; return false; }
    return true;
}

/// Same, for the engine's own geometry and per-layer blob sizes (`blob_bytes` has `layers` entries).
inline bool matches_geometry(const Manifest& m, int64_t layers, int64_t experts, const uint32_t* blob_bytes, std::string& why) {
    if (layers != m.layers || experts != m.experts) { why = "manifest is " + std::to_string(m.layers) + "x" + std::to_string(m.experts) + ", engine is " + std::to_string(layers) + "x" + std::to_string(experts); return false; }
    for (uint32_t l = 0; l < m.layers; ++l)
        if (blob_bytes == nullptr || blob_bytes[l] != m.blob_bytes[l]) { why = "layer " + std::to_string(l) + " blob size differs from the manifest"; return false; }
    return true;
}

/// Everything a loaded manifest must match before any of its sums is used: this model's geometry, then these files' stamps.  WHY one
/// function: the engine (at start) and the helper (at launch) both gate on it, so they cannot disagree about "this manifest is ours".
inline bool fresh_for(const Manifest& m, int64_t layers, int64_t experts, const uint32_t* blob_bytes, const FileStat* files, size_t n_files,
                      std::string& why) {
    return matches_geometry(m, layers, experts, blob_bytes, why) && matches_files(m, files, n_files, why);
}

}  // namespace seabraid::sums
