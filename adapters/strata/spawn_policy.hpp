// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Helper-spawn policy: which binary may be launched and what it may inherit.
// OWNER: the only definition of the SEABRAID_QWEN_HELPER trust rules and the
//   child environment allow-list. CONTRACT: pure functions over a path and an
//   environ array; no global state; used by PrivateQwenWorker::spawn.
// SIZE BUDGET: <=150 lines. FAILURE CONTRACT: fail closed - any doubt about the helper refuses the
//   launch; every refusal is logged once through diag.
// CHECKER: private_qwen_test (relative, group-writable, world-writable,
//   non-executable, directory paths; environment filter on a synthetic environ),
//   helper_hardening_test (setuid/setgid helper, environment gate end to end) and
//   helper_process_test (a file swapped after vetting is not what runs).
#pragma once

#include "unique_fd.hpp"

#include <string>
#include <vector>

namespace seabraid::strata_adapter {

// The helper as it was vetted: the symlink-free path (for argv[0] and logs only)
// and an O_PATH|O_CLOEXEC descriptor on the very inode that passed the checks.
// Exec THIS descriptor, never the path: a file swapped in after vetting then
// cannot be what runs (see HelperProcess::spawn).
struct VettedHelper {
    std::string path;
    UniqueFd fd;
};

// Returns nullptr when `path` may be exec'd as the helper; otherwise a static
// reason (already logged). The ownership, mode and executability checks run on
// the opened descriptor (fstat / access via /proc/self/fd), not on the path, so
// there is no window between "vetted" and "the file that is launched".
const char* vet_helper(const std::string& path, VettedHelper& vetted);

// Copies only the allow-listed "NAME=value" entries of `env`. `debug` also passes
// the explicitly listed diagnostic switches (default: off).
std::vector<std::string> allowlisted_environment(char* const* env, bool debug = false);

}  // namespace seabraid::strata_adapter
