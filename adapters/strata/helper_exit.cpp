// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "helper_exit.hpp"

#include "helper_kernel.hpp"

#include "strata/seabraid/proc_stat.hpp"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

namespace seabraid::strata_adapter {
namespace {

// WHY poll: a pidfd turns readable exactly when the process has exited, and poll
// needs neither waitid(P_PIDFD) nor the status. The owner collects it through the pidfd (never by pid, F1).
bool pidfd_readable(int pidfd) noexcept {
    pollfd watch{pidfd, POLLIN, 0};
    return ::poll(&watch, 1, 0) > 0;
}

bool process_exited(int pidfd, HelperExit& out) noexcept {
    if (pidfd < 0) return false;
    siginfo_t info{};
    // WNOWAIT: observe the status without consuming it; the owner reaps later.
    const int result = kernel_waitid_pidfd(pidfd, &info, WEXITED | WNOHANG | WNOWAIT);
    if (result < 0 && errno == EINTR) return false;   // a nonblocking probe may try again on its next tick
    if (result == 0 && info.si_pid == 0) return false;   // still running
    out = {};
    if (result < 0 && errno != ECHILD) {   // EINVAL/ENOSYS/EPERM: not "gone", the call is unavailable
        note_waitid_fallback(errno);
        return pidfd_readable(pidfd);
    }
    // ECHILD (someone else reaped it) still means "gone", status unknown.
    if (result == 0) {
        out.known = true;
        out.signaled = info.si_code != CLD_EXITED;
        out.code = info.si_status;
    }
    return true;
}

// A read error (ESRCH once the child is reaped) answers "not seen here"; the
// pidfd probe, which runs first, owns that case.
bool main_thread_exited(int stat_fd, HelperExit& out) noexcept {
    if (stat_fd < 0) return false;
    char line[1024];
    const ssize_t got = ::pread(stat_fd, line, sizeof(line) - 1, 0);
    if (got <= 0) return false;
    line[got] = '\0';
    return parse_stat_exit(line, out);
}

}  // namespace

bool parse_stat_exit(const char* stat, HelperExit& out) noexcept {
    char state = '?';
    if (!seabraid::procstat::state_of(stat, state) || !seabraid::procstat::dead_state(state)) return false;
    out = {};
    out.still_exiting = true;
    // Kernels before 3.5 have no field 52; the exit is still real, its code unknown.
    long status = 0;
    if (!seabraid::procstat::exit_status_of(stat, status)) return true;
    const seabraid::procstat::WaitStatus decoded = seabraid::procstat::decode_wait_status(status);
    out.known = true;
    out.signaled = decoded.signaled;
    out.code = decoded.code;
    return true;
}

bool helper_exited(const HelperWatch& watch, HelperExit& out) noexcept {
    return process_exited(watch.pidfd, out) || main_thread_exited(watch.stat_fd, out);
}

const char* describe_exit(const HelperExit& exit, char* buffer, std::size_t size) noexcept {
    if (size == 0) return buffer;
    const char* tail = exit.still_exiting ? ", process still exiting" : "";
    if (!exit.known) std::snprintf(buffer, size, " (status unknown)%s", tail);
    else std::snprintf(buffer, size, " (%s %d)%s", exit.signaled ? "signal" : "status", exit.code, tail);
    return buffer;
}

}  // namespace seabraid::strata_adapter
