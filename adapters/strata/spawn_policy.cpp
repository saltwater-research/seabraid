// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "spawn_policy.hpp"

#include "diag.hpp"

#include "seabraid/seabraid.h"

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace seabraid::strata_adapter {
namespace {
using diag::kError;

// Match complete variable names. Unknown settings (including future server credentials)
// stay in the engine in both modes. Loader paths come from the trusted launcher.
constexpr const char* kExact[] = {
    "PATH", "LD_LIBRARY_PATH", "HOME", "LANG", "TMPDIR", "XDG_RUNTIME_DIR",
    "LC_ALL", "LC_COLLATE", "LC_CTYPE", "LC_MESSAGES", "LC_MONETARY", "LC_NUMERIC", "LC_TIME",
    "LC_ADDRESS", "LC_IDENTIFICATION", "LC_MEASUREMENT", "LC_NAME", "LC_PAPER", "LC_TELEPHONE",
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL", "GPU_MAX_HEAP_SIZE", "GPU_MAX_HW_QUEUES", "GPU_PINNED_MIN_XFER_SIZE",
    "HSA_ENABLE_SDMA", "HSA_USERPTR_FOR_PAGED_MEM", "HSA_SVM_FOR_PAGED_MEM",
    "STRATA_HIP_ARENA_NONCOHERENT", "STRATA_FILL_AHEAD",
    "SEABRAID_STRATA_SHM", "SEABRAID_STRATA_HOT_SHARE", "SEABRAID_STRATA_LINK_CHECK", "SEABRAID_STRATA_LINK_CHECK_EVERY",
    "SEABRAID_STRATA_CONNECT_S", "SEABRAID_STRATA_LOAD_S", "SEABRAID_STRATA_TIMEOUT_S", "SEABRAID_STRATA_EXCLUDE_HELPER_RAM",
};
constexpr const char* kDebug[] = {
    "STRATA_DUMP_MOE_INPUT", "STRATA_ROUTE_HIST", "STRATA_DECODE_TIMING", "STRATA_D4_CHECK",
    "SEABRAID_STRATA_CARD_BENCH", "SEABRAID_STRATA_CARD_BENCH_ITERS", "SEABRAID_STRATA_CARD_BENCH_LAYER",
};

// True when the NAME part (first `length` bytes) equals one of `names`.
template <size_t N>
bool listed(const char* entry, size_t length, const char* const (&names)[N]) noexcept {
    for (const char* name : names)
        if (std::strlen(name) == length && std::strncmp(entry, name, length) == 0) return true;
    return false;
}

bool allowed(const char* entry, bool debug) noexcept {
    const char* equals = std::strchr(entry, '=');
    if (!equals) return false;
    const size_t length = static_cast<size_t>(equals - entry);
    return listed(entry, length, kExact) || (debug && listed(entry, length, kDebug));
}

const char* refuse(const char* reason, int code = 0) noexcept {
    diag::logf(kError, SEABRAID_INVALID, "helper rejected: %s (errno %d)", reason, code);
    return reason;
}

// Owned by this user or root, and nobody else may rewrite it.
const char* file_fault(const struct stat& info) noexcept {
    if (!S_ISREG(info.st_mode)) return "helper is not a regular file";
    if (info.st_uid != ::geteuid() && info.st_uid != 0) return "helper owner is neither this user nor root";
    if (info.st_mode & (S_IWGRP | S_IWOTH)) return "helper is group- or world-writable";
    // WHY: a setuid/setgid bit would make exec change identity under the checks above.
    if (info.st_mode & (S_ISUID | S_ISGID)) return "helper is setuid or setgid";
    return nullptr;
}

// WHY: in a world-writable, non-sticky directory anyone can swap the file at this path,
// so every path-based use (logs, the next launch) would name a stranger's file.
const char* directory_fault(const std::string& path) noexcept {
    const std::string dir = path.substr(0, path.rfind('/') == 0 ? 1 : path.rfind('/'));
    struct stat info {};
    if (::stat(dir.c_str(), &info) != 0) return "helper directory cannot be inspected";
    if ((info.st_mode & S_IWOTH) && !(info.st_mode & S_ISVTX)) return "helper directory is world-writable without the sticky bit";
    return nullptr;
}

}  // namespace

const char* vet_helper(const std::string& path, VettedHelper& vetted) {
    vetted.fd.reset();
    if (path.empty() || path[0] != '/') return refuse("helper path is not absolute");
    char* real = ::realpath(path.c_str(), nullptr);
    if (!real) return refuse("helper path cannot be resolved", errno);
    vetted.path.assign(real);
    std::free(real);
    // WHY O_PATH|O_NOFOLLOW: the descriptor pins the inode we vet and later exec;
    // a symlink swapped in after realpath() is opened as the link itself and is
    // refused below as "not a regular file".
    vetted.fd.reset(::open(vetted.path.c_str(), O_PATH | O_NOFOLLOW | O_CLOEXEC));
    if (!vetted.fd.valid()) return refuse("helper cannot be opened", errno);
    struct stat info {};
    if (::fstat(vetted.fd.get(), &info) != 0) return refuse("helper cannot be inspected", errno);
    if (const char* fault = file_fault(info)) return refuse(fault);
    if (const char* fault = directory_fault(vetted.path)) return refuse(fault);
    // access() through the descriptor's magic link checks the pinned inode, and
    // honours noexec mounts and ACLs that the mode bits alone would miss.
    const std::string link = "/proc/self/fd/" + std::to_string(vetted.fd.get());
    if (::access(link.c_str(), X_OK) != 0) return refuse("helper is not executable", errno);
    return nullptr;
}

std::vector<std::string> allowlisted_environment(char* const* env, bool debug) {
    std::vector<std::string> kept;
    for (; env && *env; ++env)
        if (allowed(*env, debug)) kept.emplace_back(*env);
    return kept;
}

}  // namespace seabraid::strata_adapter
