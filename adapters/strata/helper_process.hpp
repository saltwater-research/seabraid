// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// One spawned helper child: launch, pidfd, reap, signal.
// OWNER: the only code that creates or signals the helper process. CONTRACT:
//   takes a VettedHelper (spawn_policy.hpp) plus argv and one descriptor handoff;
//   exposes a borrowed pidfd for liveness probes (helper_exit.hpp); depends on
//   diag, spawn_policy, helper_exit only - never on Strata or the wire.
// SIZE BUDGET: header <=80; each source <=250 lines except helper_process.cpp (spawn,
//   <=265: it now verifies its child); teardown: helper_process_stop.cpp.
// FAILURE CONTRACT: fail closed and report the cause where it is detected.
// CHECKER: helper_process_test (inherited descriptors, swapped file, pidfd
//   teardown, both strategies), helper_hardening_test and private_qwen_test.
#pragma once

#include "helper_exit.hpp"
#include "spawn_policy.hpp"
#include "unique_fd.hpp"

#include <memory>
#include <string>
#include <vector>

#include <sys/types.h>

namespace seabraid::strata_adapter {

// Descriptor `source` of the parent appears as `target` in the child.
struct FdHandoff {
    int source = -1;   // negative: nothing to hand over
    int target = -1;
};

// kAuto uses posix_spawn_file_actions_addclosefrom_np when glibc has it. kEnumerate
// closes the descriptors listed by /proc/self/fd and is selectable in tests.
enum class CloseStrategy { kAuto, kEnumerate };

struct HelperProcessTestAccess;

class HelperProcess {
public:
    HelperProcess();
    ~HelperProcess();
    HelperProcess(const HelperProcess&) = delete;
    HelperProcess& operator=(const HelperProcess&) = delete;

    // Runs the vetted inode with argv = {vetted.path, args...}, an allow-listed
    // environment, an empty signal mask (SIGTERM/SIGINT/SIGPIPE default), and descriptors 0-2, `handoff.target`
    // and any in [3, target) inherited; everything above target is closed.
    // Refused (logged) while SIGCHLD is ignored or handoff.source == handoff.target.
    bool spawn(const VettedHelper& vetted, const std::vector<std::string>& args, FdHandoff handoff,
               std::string& error, CloseStrategy strategy = CloseStrategy::kAuto, bool debug_env = false);

    pid_t pid() const noexcept { return pid_; }   // -1 once reaped or never spawned
    // Borrowed until destruction or another spawn. Stays valid after collection or handoff.
    int pidfd() const noexcept;
    HelperWatch watch() const noexcept { return {pidfd(), stat_fd_.get()}; }
    // True once this object has no child left to collect (including a deferred reap).
    bool reap(bool block) noexcept;
    // SIGTERM grace is 300 ms; after SIGKILL, wait up to 20 ms before the reserved
    // worker takes responsibility for signalling/collection through the same pidfd.
    void stop() noexcept;
    const HelperExit& last_exit() const noexcept { return exit_; }

private:
    friend struct HelperProcessTestAccess;
    bool signal(int number) noexcept;
    void stop_dying() noexcept;

    struct Retirement;
    int reserve_retirement() noexcept;
    void release_retirement() noexcept;
    UniqueFd& child_fd() noexcept;
    void defer_retirement(bool retry_signal) noexcept;
    pid_t pid_ = -1;
    std::unique_ptr<Retirement> retirement_;
    UniqueFd stat_fd_;   // /proc/<pid>/stat; invalid when it could not be opened
    HelperExit exit_;
};

}  // namespace seabraid::strata_adapter
