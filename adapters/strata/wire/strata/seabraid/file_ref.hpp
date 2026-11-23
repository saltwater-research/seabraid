// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid wire v9: the FileRef rules.  OWNER of how a FileRef is built, moved through a blob slot and validated, and of the
// one-op round rule of kTierSwapFile; the bytes (FileRef, FileExtent, kTierSwapFile) are channel_layout.hpp's.
// WHY a file of its own: helper and engine validate with these same functions (so they cannot disagree about a legal ref),
// and channel_layout.hpp is the bytes, kept under its split point.  Pure: no I/O, no hashing, no GPU runtime.
#include "strata/seabraid/channel_layout.hpp"

namespace seabraid {

/// Main (and the helper, for the ref it expects): the FileRef for `extents` (n in 1..3); `sum` comes from the manifest, nothing is
/// hashed here.  The helper never trusts a sum or extents it received: see file_ref_mismatch.
inline FileRef make_file_ref(const FileExtent* extents, int n, uint64_t sum) {
    FileRef r{};
    r.ref_bytes = (uint32_t) sizeof(FileRef); r.sum_kind = kSumXxh3_64; r.sum = sum; r.n_extents = (uint32_t) n;
    uint64_t total = 0;
    for (int i = 0; i < n && i < kFileRefMaxExtents; ++i) { r.extent[i] = extents[i]; total += extents[i].length; }
    r.blob_bytes = total > UINT32_MAX ? 0 : (uint32_t) total;
    return r;
}

/// Both sides: move a FileRef through a blob slot.  The reader copies it out ONCE and uses only the copy (the slot is
/// shared, so untrusted, memory).  False when the slot cannot hold one.
inline bool file_ref_from(const uint8_t* slot, int64_t slot_bytes, FileRef& out) {
    if (slot == nullptr || slot_bytes < (int64_t) sizeof(FileRef)) return false;
    std::memcpy(&out, slot, sizeof(FileRef));
    return true;
}
inline bool file_ref_to(uint8_t* slot, int64_t slot_bytes, const FileRef& ref) {
    if (slot == nullptr || slot_bytes < (int64_t) sizeof(FileRef)) return false;
    std::memcpy(slot, &ref, sizeof(FileRef));
    return true;
}

namespace detail {
// Each returns "" when its rule holds, else the reason.  Split so no function carries more than a handful of rules.
inline std::string ref_shape_error(const FileRef& r) {
    if (r.ref_bytes != (uint32_t) sizeof(FileRef)) return "FileRef size " + std::to_string(r.ref_bytes) + " is not " + std::to_string(sizeof(FileRef));
    if (r.sum_kind != kSumXxh3_64) return "FileRef sum kind " + std::to_string(r.sum_kind) + " is not XXH3-64";
    if (r.n_extents < 1 || r.n_extents > (uint32_t) kFileRefMaxExtents) return "FileRef has " + std::to_string(r.n_extents) + " extents, not 1.." + std::to_string(kFileRefMaxExtents);
    return "";
}
inline std::string file_table_error(const uint64_t* file_size, int32_t file_count) {
    if (file_count < 0 || file_count > kFileTableMax) return "bad file table: " + std::to_string(file_count) + " entries";
    return file_count > 0 && file_size == nullptr ? "bad file table: no sizes" : "";
}
inline std::string extent_error(const FileExtent& e, uint32_t i, const uint64_t* file_size, int32_t file_count) {
    const std::string at = "extent " + std::to_string(i) + ": ";
    if (e.file >= (uint32_t) file_count) return at + "file index " + std::to_string(e.file) + " >= table size " + std::to_string(file_count);
    if (e.length == 0) return at + "zero length";
    const uint64_t size = file_size[e.file];
    // offset + length is never computed: `offset <= size - length` cannot wrap once `length <= size` holds.
    if (e.length > size || e.offset > size - e.length)
        return at + "offset " + std::to_string(e.offset) + " + length " + std::to_string(e.length) + " passes the end of file " + std::to_string(e.file) + " (" + std::to_string(size) + " bytes)";
    return "";
}
inline std::string blob_size_error(const FileRef& r, uint64_t total, int64_t want_blob_bytes, int64_t staging_bytes) {
    if (total != r.blob_bytes) return "extent lengths sum to " + std::to_string(total) + ", FileRef says " + std::to_string(r.blob_bytes);
    if ((int64_t) r.blob_bytes != want_blob_bytes) return "FileRef blob is " + std::to_string(r.blob_bytes) + " bytes, layer needs " + std::to_string(want_blob_bytes);
    if ((int64_t) r.blob_bytes > staging_bytes) return "FileRef blob " + std::to_string(r.blob_bytes) + " exceeds the staging buffer " + std::to_string(staging_bytes);
    return "";
}
}  // namespace detail

/// The one validation of a FileRef, shared by the helper (before any read) and main (before it submits), so the two can
/// never disagree about a legal ref.  `file_size[i]` is the size fstat gave the helper at launch for table entry i;
/// `want_blob_bytes` the helper's own blob_bytes(layer_new); `staging_bytes` its staging buffer.  Every rule is a refusal
/// before any read and so before any slot write; `why` names the first broken one.
inline bool file_ref_valid(const FileRef& r, const uint64_t* file_size, int32_t file_count, int64_t want_blob_bytes,
                           int64_t staging_bytes, std::string& why) {
    why = detail::ref_shape_error(r);
    if (why.empty()) why = detail::file_table_error(file_size, file_count);
    uint64_t total = 0;                                     // 64-bit: three u32 lengths cannot overflow it
    for (uint32_t i = 0; why.empty() && i < r.n_extents; ++i) {
        why = detail::extent_error(r.extent[i], i, file_size, file_count);
        total += r.extent[i].length;
    }
    if (why.empty()) why = detail::blob_size_error(r, total, want_blob_bytes, staging_bytes);
    return why.empty();
}

/// Helper: "" when the FileRef main sent (`wire`) is exactly the one the helper builds for the op's own cell (`own`: extents from its
/// own pack layout, sum from its own expert-sums.v1), else what differs.  WHY the wire's sum and extents are only ever COMPARED, never
/// used: a sum that rides beside the extents proves nothing (a ref for expert B can carry expert C's extents and C's valid sum), so the
/// checksum is trusted only when it comes from the helper's own manifest, looked up by the op's (layer, expert).
namespace detail {
inline bool same_extents(const FileRef& a, const FileRef& b) {
    if (a.n_extents != b.n_extents || a.blob_bytes != b.blob_bytes || b.n_extents > (uint32_t) kFileRefMaxExtents) return false;
    for (uint32_t i = 0; i < b.n_extents; ++i)
        if (a.extent[i].file != b.extent[i].file || a.extent[i].length != b.extent[i].length || a.extent[i].offset != b.extent[i].offset) return false;
    return true;
}
}  // namespace detail

inline std::string file_ref_mismatch(const FileRef& wire, const FileRef& own) {
    if (wire.ref_bytes != own.ref_bytes || wire.sum_kind != own.sum_kind) return "the FileRef's size or sum kind is not this helper's";
    if (!detail::same_extents(wire, own)) return "the FileRef's extents are not where this helper's pack keeps that expert";
    return wire.sum == own.sum ? std::string() : "the FileRef's sum is not this helper's manifest sum for that expert";
}

/// The one round-shape rule: a round holding a kTierSwapFile holds exactly one op (one staging buffer).
inline bool tier_round_shape_ok(const TierWireOp* ops, int n, std::string& why) {
    if (n <= 1) return true;
    for (int i = 0; i < n; ++i)
        if (ops[i].kind == kTierSwapFile) { why = "a round with a file swap must hold exactly one op, this one holds " + std::to_string(n); return false; }
    return true;
}

} // namespace seabraid
