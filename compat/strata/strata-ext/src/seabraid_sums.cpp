// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid-sums: write <pack>/expert-sums.v1, the manifest file moves check every read against (wire v9, expert_sums.hpp), or with
// --verify compare an existing one with the pack.  Run once per pack by the installer: it reads every expert once, O_DIRECT when
// the filesystem allows so the read does not churn the page cache.
//   seabraid-sums PACK_DIR [--native SHARD1.gguf] [--layers N] [--experts N] [--verify]     (layers/experts default to the helper's)
// Exit: 0 done / verified, 1 cannot open the pack, 2 a blob could not be read (named) or a file changed while it was read,
// 3 --verify found the manifest stale, wrong or not a manifest, 4 bad command line.  FAILURE: nothing is written unless every blob
// was read AND every file still is, after the scan, the file the scan read (FileTable::unchanged); the stamps written are fstat's of
// the descriptors read, never a second stat of the paths.
#include "log_text.hpp"
#include "pack_extents.hpp"
#include "sums_build.hpp"
#include "strata/seabraid/file_stamp.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {
constexpr int kOk = 0, kCannotOpen = 1, kUnreadable = 2, kBadManifest = 3, kUsage = 4;

struct Args { std::string pack, native; int64_t layers = 48, experts = 512; bool verify = false; };

int fail(int code, const std::string& why) {
    std::fprintf(stderr, "seabraid-sums: %s\n", seabraid::logtext::scrub_paths(why).c_str());
    return code;
}

// The options that take a value; false for anything else.
bool take_option(Args& a, const std::string& option, const std::string& value) {
    if (option == "--native") a.native = value;
    else if (option == "--layers") a.layers = std::atoll(value.c_str());
    else if (option == "--experts") a.experts = std::atoll(value.c_str());
    else return false;
    return true;
}

bool parse(int argc, char** argv, Args& a) {
    if (argc < 2) return false;
    a.pack = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string o = argv[i];
        if (o == "--verify") a.verify = true;
        else if (i + 1 >= argc || !take_option(a, o, argv[++i])) return false;
    }
    return a.layers > 0 && a.experts > 0 && a.layers <= seabraid::sums::kMaxLayers && a.experts <= seabraid::sums::kMaxExperts;
}

using Extents = seabraid::PackExtents;   // the engine's own addressing (one owner: FileExpertSource::file_extents)

// After a scan: every file must still be the one read, unchanged.  0 = it is.
int still_same(const seabraid::FileTable& files, const std::vector<std::string>& paths) {
    std::string why;
    return files.unchanged(paths, why) ? kOk : fail(kUnreadable, "the pack changed while it was read: " + why + "; run seabraid-sums again");
}

seabraid::sums::Manifest geometry(const Args& a, const std::vector<seabraid::sums::FileStat>& stamps) {
    seabraid::sums::Manifest m;
    m.layers = (uint32_t) a.layers; m.experts = (uint32_t) a.experts; m.files = stamps;
    for (int64_t l = 0; l < a.layers; ++l) m.blob_bytes.push_back((uint32_t) strata::kernels::cpu::expert_layout().blob_bytes(l));
    return m;
}

int verify(const Args& a, seabraid::FileTable& files, const Extents& ex, const seabraid::sums::Manifest& want, const std::string& path,
           const std::vector<std::string>& paths) {
    std::vector<uint8_t> bytes;   // bounded: a file longer than any manifest comes back one byte too long and parse refuses it
    seabraid::sums::Manifest have;
    std::string err, why;
    if (!seabraid::sums::read_manifest(path, bytes, err) || !seabraid::sums::parse(bytes.data(), bytes.size(), have, err))
        return fail(kBadManifest, "no usable manifest: " + err);
    if (!seabraid::sums::fresh_for(have, a.layers, a.experts, want.blob_bytes.data(), want.files.data(), want.files.size(), why))
        return fail(kBadManifest, "manifest is stale: " + why + "; rebuild it with seabraid-sums");
    std::vector<std::string> bad;
    err.clear();
    const bool same = seabraid::sums_build::verify_sums(files, ex, have, bad, err);
    if (!same && !err.empty()) return fail(kUnreadable, err);   // a blob that cannot be read is not a mismatch
    if (const int code = still_same(files, paths)) return code;
    for (size_t i = 0; i < bad.size() && i < 20; ++i) std::fprintf(stderr, "seabraid-sums: MISMATCH %s\n", bad[i].c_str());
    if (!bad.empty()) return fail(kBadManifest, std::to_string(bad.size()) + " experts differ from the manifest");
    std::printf("seabraid-sums: verified %lld x %lld experts\n", (long long) a.layers, (long long) a.experts);
    return kOk;
}

int build(const Args& a, seabraid::FileTable& files, const Extents& ex, seabraid::sums::Manifest& m, const std::string& path,
          const std::vector<std::string>& paths) {
    std::string err, bytes;
    const auto t0 = std::chrono::steady_clock::now();
    if (!seabraid::sums_build::build_sums(files, ex, m.layers, m.experts, m.blob_bytes, m.sums, err)) return fail(kUnreadable, err);
    if (const int code = still_same(files, paths)) return code;   // the stamps must describe the bytes summed
    if (!seabraid::sums::serialize(m, bytes, err) || !seabraid::sums_build::write_atomic(path, bytes, err)) return fail(kUnreadable, err);
    std::printf("seabraid-sums: wrote %s (%lld x %lld experts, %d files, %.1f s)\n", path.c_str(), (long long) a.layers, (long long) a.experts,
                files.count(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return kOk;
}

// The pack, opened as the helper opens it (same layout, same loader).  0 = open; `blob_max` is the largest blob.
int open_pack(const Args& a, strata::core::FileExpertSource& src, int64_t& blob_max, std::string& err) {
    if (!strata::kernels::cpu::expert_layout_load(a.pack, a.layers, a.experts, err)) return fail(kCannotOpen, "expert layout: " + err);
    if (!a.native.empty()) src.set_gguf(a.native);
    if (!src.open(a.pack, a.layers, a.experts, err)) return fail(kCannotOpen, "expert source: " + err);
    for (int64_t l = 0; l < a.layers; ++l) blob_max = std::max<int64_t>(blob_max, (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(l));
    return kOk;
}
}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) { std::fprintf(stderr, "usage: seabraid-sums PACK_DIR [--native SHARD1.gguf] [--layers N] [--experts N] [--verify]\n"); return kUsage; }
    std::string err, warn;
    strata::core::FileExpertSource src;
    int64_t blob_max = 0;
    if (const int code = open_pack(a, src, blob_max, err)) return code;
    std::vector<uint8_t> staging((size_t) blob_max);
    seabraid::FileTable files;   // auto = O_DIRECT where the filesystem has an alignment: the read must not fill the page cache
    if (!files.open_all(src.file_paths(), seabraid::kFileIoAuto, staging.data(), blob_max, err, warn) || files.io() == seabraid::kFileIoOff)
        return fail(kCannotOpen, "file table: " + (err.empty() ? warn : err));
    seabraid::sums::Manifest m = geometry(a, files.stamps());   // fstat of the descriptors the scan reads
    const std::string path = a.pack + "/" + seabraid::sums::kFileName;
    const Extents ex{src};
    return a.verify ? verify(a, files, ex, m, path, src.file_paths()) : build(a, files, ex, m, path, src.file_paths());
}
