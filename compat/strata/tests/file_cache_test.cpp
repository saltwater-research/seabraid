// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/file_cache_test.cpp - DESIGN 7: an O_DIRECT file move reads the SSD and leaves the page cache alone; a buffered one fills it.
// A synthetic file in a disk-backed directory (SEABRAID_STRATA_FR_TEST_DIR, default the build dir): fdatasync, then posix_fadvise(DONTNEED) on
// THIS file only (unprivileged, touches nothing else; never /proc/sys/vm/drop_caches, which is root and machine-wide on a shared box),
// then mincore over a read-only mapping of it (the pattern of Strata's tests/core/page_cache_drop_test.cpp).
// Asserts: eviction took (0 resident pages) - else SKIP with the reason, e.g. tmpfs, where pages are always resident; the direct arm
// returns the right bytes, counts file_direct_bytes == blob bytes and file_buffered_bytes == 0, and the file is STILL 0 pages resident;
// the buffered arm returns the same bytes and leaves pages resident.  CPU only, no Strata.  SEABRAID_STRATA_FR_REQUIRE_DIRECT=1: a skip fails.
#include "file_pack_fixture.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace seabraid;
using namespace seabraid::test;

static int g_fail = 0, g_skips = 0;
#define CHECK(cond, what)                                                                       \
    do {                                                                                        \
        if (!(cond)) { std::printf("FAIL: %s (%s:%d)\n", (what), __FILE__, __LINE__); ++g_fail; } \
    } while (0)

static constexpr uint64_t kFile = 1 << 20;            // 1 MiB: 256 pages
static constexpr uint32_t kBlobBytes = 64 << 10;      // the extent: 16 pages, page aligned in the middle of the file
static constexpr uint64_t kAt = 256 << 10;

// Pages of `path` in the page cache right now: mincore over a read-only mapping (mapping a page does not read it).  -1 = cannot tell.
static long resident_pages(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    void* m = ::mmap(nullptr, kFile, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) return -1;
    std::vector<unsigned char> vec(kFile / 4096);
    long n = ::mincore(m, kFile, vec.data()) == 0 ? 0 : -1;
    for (size_t i = 0; n >= 0 && i < vec.size(); ++i) n += vec[i] & 1;
    ::munmap(m, kFile);
    return n;
}

// Push this file's pages out of the cache.  True only if mincore agrees they are gone.
static bool evict(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    ::fdatasync(fd);   // dirty pages cannot be dropped
    const int rc = ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
    return rc == 0 && resident_pages(path) == 0;
}

static void arm(FileIo mode, const char* name) {
    Pack pack;
    if (!pack.make({kFile}, name)) { std::printf("FAIL: cannot make the pack dir\n"); ++g_fail; return; }
    std::vector<uint8_t> staging(kBlobBytes);
    FileTable ft;
    std::string err, warn;
    if (!ft.open_all(pack.paths, mode, staging.data(), (int64_t) staging.size(), err, warn) || ft.io() != mode) {
        std::printf("SKIP %s arm: %s%s\n", name, err.c_str(), warn.c_str()); ++g_skips; return;
    }
    const Pack::Model model = {{{{0, kBlobBytes, kAt}}}};   // one layer, one expert: the extent below
    if (!pack.bind_table(ft, model, err)) { std::printf("FAIL: bind the manifest: %s\n", err.c_str()); ++g_fail; return; }
    if (!evict(pack.paths[0])) { std::printf("SKIP %s arm: this filesystem will not evict its pages (tmpfs?): set SEABRAID_STRATA_FR_TEST_DIR to a disk\n", name); ++g_skips; return; }
    std::string why;
    CHECK(ft.fetch(0, 0, pack.ref_of(model, 0, 0), kBlobBytes, why) == kTierApplied && std::memcmp(ft.staging(), pack.bytes[0].data() + kAt, kBlobBytes) == 0, "the move returns the right bytes");
    const long after = resident_pages(pack.paths[0]);
    std::printf("%s: %ld of %llu pages resident after the move (extent = %u pages)\n", name, after, (unsigned long long) (kFile / 4096), kBlobBytes / 4096);
    if (mode == kFileIoDirect) {
        CHECK(ft.direct_bytes() == kBlobBytes && ft.buffered_bytes() == 0, "file_direct_bytes == the blob, file_buffered_bytes == 0 (DESIGN 7)");
        CHECK(after == 0, "O_DIRECT left the file's page cache empty: the bytes came off the device");
    } else {
        CHECK(ft.buffered_bytes() == kBlobBytes && ft.direct_bytes() == 0, "the buffered arm counts buffered bytes");
        CHECK(after >= (long) (kBlobBytes / 4096), "the buffered read filled the page cache (the contrast that makes the direct result mean something)");
    }
}

int main() {
    arm(kFileIoDirect, "direct");
    arm(kFileIoBuffered, "buffered");
    if (g_skips != 0 && std::getenv("SEABRAID_STRATA_FR_REQUIRE_DIRECT") != nullptr) { std::printf("FAIL: %d arms skipped and SEABRAID_STRATA_FR_REQUIRE_DIRECT is set\n", g_skips); ++g_fail; }
    std::printf(g_fail ? "file_cache_test: %d FAILED checks\n" : "file_cache_test: all checks passed\n", g_fail);
    return g_fail != 0;
}
