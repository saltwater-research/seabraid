// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Non-consuming exit probe for the helper process.
// OWNER: the only place that turns a helper pidfd into "is it dead, and how"; the
//   kernel calls underneath it live in helper_kernel.hpp.
// CONTRACT: the probe never reaps (WNOWAIT), so the process owner can still collect
//   the status; safe from any thread.
// SIZE BUDGET: header <=80, source <=120 lines.
// FAILURE CONTRACT: a pidfd that is gone (ECHILD) counts as exited with unknown
//   status; an unavailable waitid(P_PIDFD) falls back to poll (logged once by
//   helper_kernel); an unreadable stat_fd counts as "not seen" (the pidfd stays
//   the authority).
// CHECKER: helper_hardening_test (waitid EINVAL, probe), private_qwen_test (a SIGKILLed
//   helper retires in <1 s, "signal 9" logged once) and helper_process_test.
#pragma once

#include <csignal>
#include <cstddef>

namespace seabraid::strata_adapter {

// Two views of one helper; a negative descriptor means "not watched".
// WHY two: the pidfd fires only when the LAST thread is gone. A SIGKILLed GPU
// helper's main thread is a zombie within microseconds, but the kernel then spends
// ~0.6 s (measured 2026-11-16, 22 GB on an RTX 3090) releasing device memory before
// the pidfd fires. /proc/<pid>/stat shows the main thread's exit at once, with its
// exit code. It is opened while the pid is an unreaped child, so the descriptor
// names this process even after the pid could be reused.
struct HelperWatch {
    int pidfd = -1;
    int stat_fd = -1;   // open /proc/<pid>/stat
};

struct HelperExit {
    bool known = false;          // false: the status could not be read
    bool signaled = false;       // killed by `code` rather than exiting with it
    int code = 0;
    bool still_exiting = false;  // the main thread is gone, other threads are not yet
};

// True when the helper's process has terminated (pidfd) or its main thread has
// (stat_fd). The pidfd is asked first because only it reports a finished exit.
// `out` is filled only when true.
bool helper_exited(const HelperWatch& watch, HelperExit& out) noexcept;

// Parses one /proc/<pid>/stat line: true when the task is a zombie or dead ('Z'
// or 'X'), filling `out` from field 52 (exit_code, a wait status). Exposed for tests.
bool parse_stat_exit(const char* stat, HelperExit& out) noexcept;

// " (signal 9)", " (status 3)" or " (status unknown)", plus ", process still
// exiting" when only the main thread has gone; never contains a path.
// Writes at most `size` bytes including the terminator and returns `buffer`.
const char* describe_exit(const HelperExit& exit, char* buffer, std::size_t size) noexcept;

}  // namespace seabraid::strata_adapter
