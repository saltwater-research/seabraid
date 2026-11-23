// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// OWNER: the raw pidfd_open(2) syscall, the one place SYS_pidfd_open is named (checks/check_pid_probes.py fails on any other).
//   WHY a wrapper and not glibc's pidfd_open: glibc gained it in 2.36, and the helpers build against older ones.
// CONTRACT: sys_pidfd_open(pid) returns a close-on-exec pidfd, or -1 with errno (ENOSYS when this build has no syscall number).
//   SEABRAID_STRATA_HAVE_PIDFD is defined exactly when the real syscall is compiled in.  No signalling, no waiting: those stay with
//   the supervisor (adapters/strata/helper_kernel.cpp), which owns the test seam for them.
// SIZE BUDGET: <=40 lines.
// FAILURE CONTRACT: fail-open-loud, the caller decides; this file never logs.
// CHECKER: adapters/strata/peer_watch_test.cpp (W2) and helper_hardening_test (the refused-call seam still reaches the syscall).
#include <cerrno>

#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__linux__) && defined(SYS_pidfd_open)
#define SEABRAID_STRATA_HAVE_PIDFD 1
#endif

namespace seabraid {

inline int sys_pidfd_open(pid_t pid) noexcept {
#if defined(SEABRAID_STRATA_HAVE_PIDFD)
    return static_cast<int>(::syscall(SYS_pidfd_open, pid, 0u));
#else
    (void) pid;
    errno = ENOSYS;
    return -1;
#endif
}

}  // namespace seabraid
