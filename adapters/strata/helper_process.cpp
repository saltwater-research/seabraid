// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "helper_process.hpp"

#include "diag.hpp"
#include "helper_kernel.hpp"

#include "seabraid/seabraid.h"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <spawn.h>
#include <unistd.h>

extern char** environ;

namespace seabraid::strata_adapter {
namespace {

using diag::kError;
using diag::kWarn;

// Descriptors 0-2 are the helper's stdio (its stderr is the operator's log). Exact guarantee: every descriptor ABOVE
// the handoff target is closed in the child (bar one opened by another thread mid-spawn, see close_inherited);
// descriptors in [3, target) are NOT closed, so "stdio plus target only" holds only when target is 3 or 4.
constexpr int kFirstNonStdio = 3;

struct ChildActions {
    posix_spawn_file_actions_t actions;
    bool live = false;
    ~ChildActions() {
        if (live) posix_spawn_file_actions_destroy(&actions);
    }
    int init() {
        const int code = posix_spawn_file_actions_init(&actions);
        live = code == 0;
        return code;
    }
};

// Open descriptors >= `floor`, except `keep`, as the kernel lists them now.
int enumerate_open(int floor, int keep, std::vector<int>& out) {
    DIR* dir = ::opendir("/proc/self/fd");
    if (!dir) return errno;
    const int own = ::dirfd(dir);   // the listing's own descriptor is closed with it
    while (const dirent* entry = ::readdir(dir)) {
        char* end = nullptr;
        const long value = std::strtol(entry->d_name, &end, 10);
        if (*end == '\0' && end != entry->d_name && value >= floor && value != keep && value != own)
            out.push_back(static_cast<int>(value));
    }
    ::closedir(dir);
    return 0;
}

int add_closes(ChildActions& child, const std::vector<int>& fds) {
    for (const int fd : fds) {
        const int code = posix_spawn_file_actions_addclose(&child.actions, fd);
        if (code != 0) return code;
    }
    return 0;
}

// Everything at or above `floor` except `keep`. WHY an explicit list below `keep`:
// `keep` was allocated by F_DUPFD at the lowest free slot >= floor, so every slot
// in [floor, keep) was open when listed, while addclosefrom_np covers only the
// slots above it. Measured on glibc 2.43: a close action on an in-range descriptor
// that is already closed is tolerated (a racing close in another thread cannot fail
// the spawn); only an out-of-range number is refused (EBADF, at add time). A dup2
// from a closed source DOES fail the spawn (EBADF), so the handoff source is checked.
int close_inherited(ChildActions& child, int floor, int keep, CloseStrategy strategy) {
    std::vector<int> below;
#if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 34)
    if (strategy == CloseStrategy::kAuto) {
        for (int fd = floor; fd < keep; ++fd) below.push_back(fd);
        const int code = add_closes(child, below);
        return code != 0 ? code : posix_spawn_file_actions_addclosefrom_np(&child.actions, keep + 1);
    }
#endif
#endif
    // Fallback and kEnumerate: a descriptor another thread opens after this
    // listing is still inherited. Documented in README; CLOEXEC discipline of
    // the host is the remaining defence on glibc older than 2.34.
    const int code = enumerate_open(floor, keep, below);
    return code != 0 ? code : add_closes(child, below);
}

// WHY: posix_spawn with null attributes lets the child inherit this thread's blocked-signal mask, and a helper that
// starts with SIGTERM blocked ignores stop() until the SIGKILL escalation. Reset the mask and the key dispositions.
int init_spawn_attr(posix_spawnattr_t& attr) {
    sigset_t none;
    sigemptyset(&none);
    sigset_t defaults;
    sigemptyset(&defaults);
    sigaddset(&defaults, SIGTERM);
    sigaddset(&defaults, SIGINT);
    sigaddset(&defaults, SIGPIPE);
    int code = posix_spawnattr_setsigmask(&attr, &none);
    if (code == 0) code = posix_spawnattr_setsigdefault(&attr, &defaults);
    if (code == 0) code = posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
    if (code != 0) diag::logf(kError, SEABRAID_PROVIDER, "private Qwen: cannot set helper signal state (errno %d)", code);
    return code;
}

int build_actions(ChildActions& child, FdHandoff handoff, int keep, int floor, CloseStrategy strategy) {
    int code = 0;
    if (handoff.source >= 0) {
        // WHY first: the source may itself be one of the descriptors closed below.
        code = posix_spawn_file_actions_adddup2(&child.actions, handoff.source, handoff.target);
        // A source under the closing range (e.g. fd 3 mapped to 4) is closed here.
        if (code == 0 && handoff.source >= kFirstNonStdio && handoff.source < floor)
            code = posix_spawn_file_actions_addclose(&child.actions, handoff.source);
    }
    return code != 0 ? code : close_inherited(child, floor, keep, strategy);
}

// Returns an errno value (0 = ok). `pidfd` is set only by pidfd_spawn (glibc >= 2.39), which leaves no window
// in which the child could be reaped before it is watched; `pid` is then -1 (see pid_of_pidfd).
int launch(pid_t& pid, int& pidfd, const VettedHelper& vetted, const std::vector<std::string>& args, int exec_fd,
           const ChildActions& child, bool debug_env) {
    std::vector<std::string> words{vetted.path};
    words.insert(words.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& word : words) argv.push_back(word.data());
    argv.push_back(nullptr);
    // WHY an allow-list instead of `environ`: the helper only needs the GPU
    // runtime and Strata/Seabraid settings; credentials in the engine's
    // environment must not reach a process that loads vendor libraries.
    auto kept = allowlisted_environment(environ, debug_env);
    std::vector<char*> envp;
    for (auto& entry : kept) envp.push_back(entry.data());
    envp.push_back(nullptr);
    // WHY /proc/self/fd/N and not the path: posix_spawn cannot fexecve, but the
    // child's own fd table holds the vetted inode (kept CLOEXEC, closed by the
    // kernel at exec), so the exec cannot be redirected by a file swapped in
    // after vetting. Requires /proc; without it the spawn fails closed (ENOENT).
    const std::string target = "/proc/self/fd/" + std::to_string(exec_fd);
    posix_spawnattr_t attr;
    int code = posix_spawnattr_init(&attr);
    if (code != 0) {
        diag::logf(kError, SEABRAID_PROVIDER, "private Qwen: cannot init spawn attributes (errno %d)", code);
        return code;
    }
    code = init_spawn_attr(attr);
#ifdef SEABRAID_HAVE_PIDFD_SPAWN
    pid = -1;
    if (code == 0) code = pidfd_spawn(&pidfd, target.c_str(), &child.actions, &attr, argv.data(), envp.data());
#else
    pidfd = -1;
    if (code == 0) code = posix_spawn(&pid, target.c_str(), &child.actions, &attr, argv.data(), envp.data());
#endif
    posix_spawnattr_destroy(&attr);
    return code;
}

// WHY refuse these before forking: with SIGCHLD ignored (or SA_NOCLDWAIT) the kernel
// reaps the helper by itself, so waitid has no status and its pid can be reused; and a
// handoff whose source equals its target would be closed by the "close the source"
// action in build_actions, handing the helper a dead descriptor. Refusing beats handling
// it: dup2(fd, fd) only clears CLOEXEC on glibc >= 2.29.
const char* launch_refusal(const FdHandoff& handoff) noexcept {
    struct sigaction current {};
    if (::sigaction(SIGCHLD, nullptr, &current) != 0) return "cannot read the SIGCHLD disposition";
    const bool ignored = (current.sa_flags & SA_SIGINFO) == 0 && current.sa_handler == SIG_IGN;
    if (ignored || (current.sa_flags & SA_NOCLDWAIT) != 0)
        return "SIGCHLD is ignored or SA_NOCLDWAIT is set, so the kernel would reap the helper and reuse its pid";
    if (handoff.source >= 0 && handoff.source == handoff.target)
        return "the descriptor handoff has the same source and target";
    return nullptr;
}

// Logs once and fills `error`; every failed spawn ends here so each logs exactly one record.
bool fail_spawn(std::string& error, const char* what, int code) {
    diag::logf(kError, SEABRAID_PROVIDER, "private Qwen: cannot %s the helper (errno %d)", what, code);
    error = std::string("cannot ") + what + " private Qwen helper: " + std::strerror(code);
    return false;
}

// Logs once and fills `error` for a child that another waiter reaped before we held it by pidfd.
bool refuse_reaped(pid_t& pid, std::string& error) {
    pid = -1;
    diag::logf(kError, SEABRAID_PROVIDER,
               "private Qwen: the helper was reaped before it could be watched (host reaping with waitpid(-1)?)");
    error = "private Qwen helper was reaped by another waiter during startup";
    return false;
}

// pidfd_spawn gives us the child's identity atomically. With older glibc, a host waitpid(-1) can race pidfd_open.
// Check the later-opened pidfd still names a waitable child. On failure do not signal a possible stranger by
// PID. If no pidfd could be opened, the helper's --controller-pid watch ends it with this engine.
bool watch_child(pid_t& pid, UniqueFd& pidfd, int spawned, std::string& error) {
    pidfd.reset(spawned);
    if (pid < 0 && pidfd.valid()) pid = pid_of_pidfd(pidfd.get());
    if (!pidfd.valid() && pid >= 0) pidfd.reset(kernel_pidfd_open(pid));
    if (!pidfd.valid()) {
        const int saved = errno;
        if (saved == ESRCH) return refuse_reaped(pid, error);
        pid = -1;
        return fail_spawn(error, "watch (left to its controller watch, never signalled by pid)", saved);
    }
    if (pid >= 0 && names_unreaped_child(pidfd.get())) return true;
    // A pidfd delivered at spawn always belongs to this launch. An unverified later-opened fd does not.
    if (spawned < 0) pidfd.reset();
    return refuse_reaped(pid, error);
}

// Builds the child's descriptor actions and starts it; returns an errno value.
int start_child(pid_t& pid, int& pidfd, const VettedHelper& vetted, const std::vector<std::string>& args,
                FdHandoff handoff, CloseStrategy strategy, bool debug_env) {
    const int floor = std::max(handoff.target, kFirstNonStdio - 1) + 1;
    UniqueFd exec_fd(::fcntl(vetted.fd.get(), F_DUPFD_CLOEXEC, floor));
    int code = exec_fd.valid() ? 0 : errno;
    ChildActions child;
    if (code == 0) code = child.init();
    if (code == 0) code = build_actions(child, handoff, exec_fd.get(), floor, strategy);
    if (code == 0) code = launch(pid, pidfd, vetted, args, exec_fd.get(), child, debug_env);
    return code;
}

}  // namespace

bool HelperProcess::spawn(const VettedHelper& vetted, const std::vector<std::string>& args,
                          FdHandoff handoff, std::string& error, CloseStrategy strategy, bool debug_env) {
    if (pid_ >= 0) return fail_spawn(error, "replace running", EBUSY);
    if (const char* reason = launch_refusal(handoff)) {
        diag::logf(kError, SEABRAID_PROVIDER, "private Qwen: refusing to launch the helper: %s", reason);
        error = std::string("refusing to launch private Qwen helper: ") + reason;
        return false;
    }
    // WHY gate before forking (F1): a refusal that comes before any process exists needs no signal to undo.
    if (!pidfd_launch_gate(error)) return false;
    const int reserved = reserve_retirement();
    if (reserved != 0) return fail_spawn(error, "reserve a reaper for", reserved);
    int spawned_pidfd = -1;
    const int code = start_child(pid_, spawned_pidfd, vetted, args, handoff, strategy, debug_env);
    if (code != 0) {
        pid_ = -1;
        release_retirement();
        return fail_spawn(error, "launch", code);
    }
    if (!watch_child(pid_, child_fd(), spawned_pidfd, error)) {
        if (spawned_pidfd >= 0) defer_retirement(true);
        release_retirement();
        return false;
    }
    // Same window as the pidfd: the child is unreaped, so this path names it.
    char stat_path[40];
    std::snprintf(stat_path, sizeof(stat_path), "/proc/%d/stat", static_cast<int>(pid_));
    stat_fd_.reset(::open(stat_path, O_RDONLY | O_CLOEXEC));
    if (!stat_fd_.valid())
        diag::logf(kWarn, SEABRAID_OK,
                   "private Qwen: no /proc stat view of the helper (errno %d); death is seen only once its "
                   "last thread exits", errno);
    return true;
}

}  // namespace seabraid::strata_adapter
