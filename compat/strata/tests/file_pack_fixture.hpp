// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// A synthetic pack for the file-move tests: a few shard files of deterministic bytes in a disk-backed directory (SEABRAID_STRATA_FR_TEST_DIR,
// default the current directory, which ctest sets to the build dir), the reference blob of any extent list, and the FileRef main
// would send for it.  Host only: no Strata, no GPU.  The reference is computed here from the in-memory bytes, never by the code
// under test, so a reader that addresses the wrong bytes fails against it.
#include "file_fetch.hpp"
#include "strata/seabraid/file_stamp.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace seabraid::test {

inline uint8_t pack_pattern(int file, uint64_t i) { return (uint8_t) (131u * (uint32_t) i + 17u * (uint32_t) file + (uint32_t) (i >> 9) + 1u); }

struct Pack {
    std::string dir;
    std::vector<std::string> paths;
    std::vector<std::vector<uint8_t>> bytes;   ///< what each file holds

    /// `sizes[i]` bytes for file i.  False when the directory cannot be made.
    bool make(const std::vector<uint64_t>& sizes, const char* tag) {
        const char* base = std::getenv("SEABRAID_STRATA_FR_TEST_DIR");
        static int serial = 0;   // two packs of one tag in one process (the launch test holds two) must not share a directory
        dir = std::string(base != nullptr && *base != '\0' ? base : ".") + "/seabraid_fr_" + tag + "_" + std::to_string(::getpid()) + "_" + std::to_string(serial++);
        if (::mkdir(dir.c_str(), 0700) != 0) return false;
        for (size_t f = 0; f < sizes.size(); ++f) {
            bytes.emplace_back((size_t) sizes[f]);
            for (uint64_t i = 0; i < sizes[f]; ++i) bytes.back()[(size_t) i] = pack_pattern((int) f, i);
            paths.push_back(dir + "/shard" + std::to_string(f) + ".bin");
            if (!write_file(f)) return false;
        }
        return true;
    }
    bool write_file(size_t f) const {
        std::FILE* fp = std::fopen(paths[f].c_str(), "wb");
        if (fp == nullptr) return false;
        const bool ok = std::fwrite(bytes[f].data(), 1, bytes[f].size(), fp) == bytes[f].size();
        return std::fclose(fp) == 0 && ok;
    }
    std::vector<uint64_t> sizes() const {
        std::vector<uint64_t> s;
        for (const auto& b : bytes) s.push_back(b.size());
        return s;
    }
    /// The blob an extent list names, from the in-memory bytes.
    std::vector<uint8_t> blob_of(const FileExtent* e, int n) const {
        std::vector<uint8_t> out;
        for (int i = 0; i < n; ++i) out.insert(out.end(), bytes[e[i].file].begin() + (long) e[i].offset, bytes[e[i].file].begin() + (long) (e[i].offset + e[i].length));
        return out;
    }
    /// The FileRef main sends: the extents and the sum of the reference blob (what the manifest holds).
    FileRef ref_for(const FileExtent* e, int n) const {
        const std::vector<uint8_t> b = blob_of(e, n);
        return make_file_ref(e, n, blob_sum(b.data(), b.size()));
    }
    /// Flip one byte of file `f` on disk AFTER the manifest-side sum was taken: the file no longer holds what main summed.
    void corrupt(size_t f, uint64_t at) {
        bytes[f][(size_t) at] ^= 0xFF;
        write_file(f);
    }

    /// A model over this pack: cells[l][e] is where expert e of layer l lives (1..3 extents; one total length per layer).
    using Model = std::vector<std::vector<std::vector<FileExtent>>>;
    std::string manifest;   ///< <dir>/expert-sums.v1 once write_manifest ran
    std::vector<uint32_t> layer_bytes(const Model& m) const {
        std::vector<uint32_t> b;
        for (const auto& layer : m) {
            uint32_t n = 0;
            for (const FileExtent& e : layer[0]) n += e.length;
            b.push_back(n);
        }
        return b;
    }
    /// What the manifest tool would write for `m`: the sums of the reference blobs (in-memory bytes), the stamps of the files now.
    bool write_manifest(const Model& m) {
        sums::Manifest out;
        out.layers = (uint32_t) m.size(); out.experts = (uint32_t) m[0].size(); out.blob_bytes = layer_bytes(m);
        for (const auto& layer : m)
            for (const auto& cell : layer) out.sums.push_back(ref_for(cell.data(), (int) cell.size()).sum);
        std::string bytes, err;
        if (!sums::stamp_files(paths, out.files, err) || !sums::serialize(out, bytes, err)) return false;
        return write_raw(bytes);
    }
    bool write_raw(const std::string& bytes) {
        manifest = dir + "/" + sums::kFileName;
        std::FILE* fp = std::fopen(manifest.c_str(), "wb");
        if (fp == nullptr) return false;
        const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), fp) == bytes.size();
        return std::fclose(fp) == 0 && ok;
    }
    static FileTable::ExtentsOf where(const Model& m) {
        return [m](int64_t l, int64_t e, FileExtent* out, int& n) {
            if (l < 0 || e < 0 || l >= (int64_t) m.size() || e >= (int64_t) m[(size_t) l].size()) return false;
            const auto& cell = m[(size_t) l][(size_t) e];
            n = (int) cell.size();
            for (int i = 0; i < n; ++i) out[i] = cell[(size_t) i];
            return true;
        };
    }
    /// The launch step the helper runs after open_all: write the manifest for `m`, then bind the table to it.
    bool bind_table(FileTable& ft, const Model& m, std::string& err) {
        if (!write_manifest(m)) { err = "cannot write the manifest"; return false; }
        return ft.bind_manifest(manifest, (int64_t) m.size(), (int64_t) m[0].size(), layer_bytes(m), where(m), err);
    }
    /// The FileRef main sends for cell (l, e) of `m`.
    FileRef ref_of(const Model& m, int l, int e) const { return ref_for(m[(size_t) l][(size_t) e].data(), (int) m[(size_t) l][(size_t) e].size()); }

    ~Pack() {
        for (const std::string& p : paths) ::unlink(p.c_str());
        if (!manifest.empty()) ::unlink(manifest.c_str());
        if (!dir.empty()) ::rmdir(dir.c_str());
    }
};

/// A model of `layers` x `experts` cells of 480 bytes each over a two-shard pack (shard 0 >= 100 KB, shard 1 >= 70 KB): three slices
/// per cell, the first and last in shard 0 and the middle in shard 1 (the GGUF-in-place shape).  Every cell is distinct bytes.
inline Pack::Model three_slice_model(int layers, int experts) {
    Pack::Model m((size_t) layers, std::vector<std::vector<FileExtent>>((size_t) experts));
    for (int l = 0; l < layers; ++l)
        for (int e = 0; e < experts; ++e) {
            const uint64_t c = (uint64_t) (l * experts + e);
            m[(size_t) l][(size_t) e] = {{0, 96, 8192 + c * 96}, {1, 160, 32 + c * 160}, {0, 224, 60000 + c * 224}};
        }
    return m;
}

}  // namespace seabraid::test
