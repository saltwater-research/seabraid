// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "helper_kernel.hpp"

#include "diag.hpp"

#include "seabraid/seabraid.h"
#include "strata/seabraid/pidfd.hpp"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

namespace seabraid::strata_adapter {
namespace {

using diag::kError;
using diag::kWarn;

int real_waitid(int pidfd, siginfo_t* info, int options) {
    return ::waitid(static_cast<idtype_t>(kWaitPidfd), static_cast<id_t>(pidfd), info, options);
}

int real_send_signal(int pidfd, int number) {
#ifdef SYS_pidfd_send_signal
    return static_cast<int>(::syscall(SYS_pidfd_send_signal, pidfd, number, nullptr, 0));
#else
    (void)pidfd; (void)number;
    errno = ENOSYS;
    return -1;
#endif
}

// WHY the wire's wrapper and not a second syscall here: SYS_pidfd_open is named in one place only (wire pidfd.hpp), which the
// engine's PeerWatch shares; this file keeps the KernelCalls seam around it.
int real_pidfd_open(pid_t pid) { return seabraid::sys_pidfd_open(pid); }

constexpr KernelCalls kRealCalls{real_waitid, real_send_signal, real_pidfd_open};
std::atomic<const KernelCalls*> g_calls{&kRealCalls};
std::atomic_flag g_fallback_noted = ATOMIC_FLAG_INIT;

}  // namespace

void set_kernel_calls_for_test(const KernelCalls* calls) noexcept {
    g_calls.store(calls ? calls : &kRealCalls);
    g_fallback_noted.clear();
}

int kernel_waitid_pidfd(int pidfd, siginfo_t* info, int options) noexcept {
    return g_calls.load()->waitid_pidfd(pidfd, info, options);
}

int kernel_send_signal(int pidfd, int number) noexcept { return g_calls.load()->send_signal(pidfd, number); }

int kernel_pidfd_open(pid_t pid) noexcept {
    const KernelCalls* calls = g_calls.load();
    return calls->pidfd_open ? calls->pidfd_open(pid) : real_pidfd_open(pid);
}

// ECHILD is the supported waitid answer: our own pidfd names a process that is not our child.  EINVAL (kernel < 5.4:
// no P_PIDFD), ENOSYS or EPERM (seccomp) are the unsupported ones.  Signal 0 is a permission check, nothing is sent.
const char* pidfd_supervision_fault(int& error) noexcept {
    const int fd = kernel_pidfd_open(::getpid());
    if (fd < 0) { error = errno; return "pidfd_open"; }
    siginfo_t info{};
    const char* fault = nullptr;
    if (kernel_waitid_pidfd(fd, &info, WEXITED | WNOHANG | WNOWAIT) < 0 && errno != ECHILD) fault = "waitid(P_PIDFD)";
    else if (kernel_send_signal(fd, 0) < 0) fault = "pidfd_send_signal";
    error = fault ? errno : 0;
    ::close(fd);
    return fault;
}

bool pidfd_launch_gate(std::string& error) {
    int code = 0;
    const char* call = pidfd_supervision_fault(code);
    if (call == nullptr) return true;
    diag::logf(kError, SEABRAID_PROVIDER, "private Qwen: refusing to launch the helper: %s failed (errno %d); this "
               "kernel or seccomp profile refuses pidfd signalling; the private helper needs it", call, code);
    error = std::string("refusing to launch private Qwen helper: ") + call +
            " failed; this kernel or seccomp profile refuses pidfd signalling; the private helper needs it";
    return false;
}

void note_waitid_fallback(int error) noexcept {
    if (!g_fallback_noted.test_and_set())
        diag::logf(kWarn, SEABRAID_PROVIDER,
                   "private Qwen: waitid(P_PIDFD) failed (errno %d); exit probes poll the pidfd and retirement retries collection", error);
}

pid_t pid_of_pidfd(int pidfd) noexcept {
    char path[48], text[512];
    std::snprintf(path, sizeof(path), "/proc/self/fdinfo/%d", pidfd);
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    const ssize_t got = ::read(fd, text, sizeof(text) - 1);
    ::close(fd);
    if (got <= 0) return -1;
    text[got] = '\0';
    const char* line = std::strstr(text, "Pid:");
    return line ? static_cast<pid_t>(std::strtol(line + 4, nullptr, 10)) : -1;   // "-1" once reaped
}

// WHY waitid(P_PIDFD) alone: a pidfd opened after the spawn names whatever had that pid then; only "our child, still
// waitable" proves it is ours, and an unreaped pid cannot be reused.  ECHILD (a stranger, or reaped by another waiter)
// and any other error answer "not ours": the caller then never signals it.
bool names_unreaped_child(int pidfd) noexcept {
    siginfo_t info{};
    return retry_eintr([&] { return kernel_waitid_pidfd(pidfd, &info, WEXITED | WNOHANG | WNOWAIT); }) == 0;
}

}  // namespace seabraid::strata_adapter
