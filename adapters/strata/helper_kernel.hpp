// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The kernel pidfd calls the helper supervisor makes, and their test seam.
// OWNER: the only caller of waitid(P_PIDFD)/pidfd_open/pidfd_send_signal (one decision
//   about what each call means), the launch-time capability gate, and the EINTR retry.
// CONTRACT: thin wrappers returning the syscall's (-1, errno) convention; tests swap
//   the three fallible ones via KernelCalls. Knows nothing about helper exit parsing.
//   There is NO wrapper for kill(2): the supervisor never signals or reaps by number (F1, audit 2026-11-19).
// SIZE BUDGET: header <=80, source <=200 lines.
// FAILURE CONTRACT: fail-fast at launch: pidfd_supervision_fault() names the refused call and the launch is
//   refused before any process exists.  After launch only ESRCH/ECHILD (gone) are expected; anything else is
//   logged once. Exit probes poll for liveness and retirement keeps the pidfd until collection.
// CHECKER: helper_hardening_test (launch refusal per refused call with kill(2) interposed and counted, teardown
//   with a refused pidfd signal, slow kernel) and helper_process_test.
#pragma once

#include <cerrno>
#include <csignal>
#include <string>

#include <sys/types.h>

namespace seabraid::strata_adapter {

// waitid() idtype for a pidfd (Linux 5.4); glibc before 2.36 has no P_PIDFD name.
inline constexpr int kWaitPidfd = 3;
// The syscalls an old kernel or seccomp can remove, like the syscalls (-1, errno); the supervisor's whole seam.
// `pidfd_open` null = the real syscall (tests model ENOSYS/EPERM with it).
struct KernelCalls {
    int (*waitid_pidfd)(int pidfd, siginfo_t* info, int options);
    int (*send_signal)(int pidfd, int number);
    int (*pidfd_open)(pid_t pid) = nullptr;
};
void set_kernel_calls_for_test(const KernelCalls* calls) noexcept;   // nullptr restores
int kernel_waitid_pidfd(int pidfd, siginfo_t* info, int options) noexcept;
int kernel_send_signal(int pidfd, int number) noexcept;
int kernel_pidfd_open(pid_t pid) noexcept;

// The launch gate: nullptr when pidfd_open, waitid(P_PIDFD) and pidfd_send_signal(fd, 0) all work here (probed on
// this process, so no child exists yet); else the name of the first refused call, with its errno in `error`.
// Uncached: launches are rare, and a cached "yes" would outlive a seccomp profile installed later.
const char* pidfd_supervision_fault(int& error) noexcept;
// HelperProcess::spawn's gate: true when supported; else one error record and `error` = "... refuses pidfd signalling;
// the private helper needs it" naming the refused call.
bool pidfd_launch_gate(std::string& error);
// Logs once that waitid(P_PIDFD) failed after launch (`error`); observation may poll but collection retries waitid on the same pidfd.
void note_waitid_fallback(int error) noexcept;
pid_t pid_of_pidfd(int pidfd) noexcept;   // -1 when unreadable or already reaped
// True when `pidfd` (opened after the spawn) names our still-unreaped child (waitid(P_PIDFD) succeeds on it).
bool names_unreaped_child(int pidfd) noexcept;

// WHY retry here: every caller wants the answer, never the interruption.
template <typename Call>
int retry_eintr(Call call) noexcept {
    int result;
    do { result = call(); } while (result < 0 && errno == EINTR);
    return result;
}

}  // namespace seabraid::strata_adapter
