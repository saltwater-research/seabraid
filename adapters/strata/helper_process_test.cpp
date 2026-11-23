// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Spawn hardening fixture: the controller launches a scripted copy of this binary
// through HelperProcess and the child reports what it inherited and what ran.
// No channel, model or GPU is involved.
#include "helper_process.hpp"

#include "diag_capture.hpp"
#include "test_support.hpp"

#include <csignal>
#include <cstdlib>
#include <string>

#include <dirent.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace seabraid::strata_adapter {
// Lets the fixture point the recorded pid at an unrelated process, the state a
// recycled pid number produces in a host that reaps children elsewhere.
struct HelperProcessTestAccess {
    static void set_pid(HelperProcess& process, pid_t pid) { process.pid_ = pid; }
};
}  // namespace seabraid::strata_adapter

namespace {

using namespace seabraid::strata_adapter;
using testing::check;
using testing::ScratchFile;
namespace capture = seabraid::strata_adapter::capture;

constexpr int kProbeExtraDescriptors = 30;
constexpr int kProbeHandoffBroken = 31;
constexpr int kProbeStillRan = 99;   // what the swapped-in script exits with

// Descriptors above stderr other than `target` and the listing's own.
int extra_descriptors(int target) {
    DIR* dir = opendir("/proc/self/fd");
    if (!dir) return 1;
    const int own = dirfd(dir);
    int extra = 0;
    while (const dirent* entry = readdir(dir)) {
        const int fd = std::atoi(entry->d_name);
        if (fd > 2 && fd != own && fd != target) ++extra;
    }
    closedir(dir);
    return extra;
}

// Child side: `--spawn-probe <handoff target or -1> [pause]`. Exit 0 only when the
// handoff descriptor is the single descriptor above stderr and it carries the byte.
int probe(int argc, char** argv) {
    alarm(8);   // a broken parent must not leave this child behind
    const int target = std::atoi(argv[2]);
    if (argc > 3) { pause(); return 7; }
    if (extra_descriptors(target) != 0) return kProbeExtraDescriptors;
    char byte = 0;
    return target < 0 || (read(target, &byte, 1) == 1 && byte == 'x') ? 0 : kProbeHandoffBroken;
}

VettedHelper vet(const std::string& path) {
    VettedHelper vetted;
    check(vet_helper(path, vetted) == nullptr, "the fixture copy must be a trusted helper");
    return vetted;
}

void expect_exit(HelperProcess& process, int code, const char* message) {
    check(process.reap(true), "the probe child must be collectable");
    check(!process.last_exit().signaled && process.last_exit().code == code, message);
}

// Descriptors that are not CLOEXEC at the moment of the spawn, one in the gap
// below the exec slot and one far above it.
void inherited_descriptors_are_closed(const std::string& program, CloseStrategy strategy) {
    const VettedHelper vetted = vet(program);
    UniqueFd base(open("/dev/null", O_RDONLY));
    UniqueFd low(fcntl(base.get(), F_DUPFD, 5)), high(fcntl(base.get(), F_DUPFD, 60));
    check(base.valid() && low.valid() && high.valid(), "open the leaked descriptors");
    check((fcntl(low.get(), F_GETFD) & FD_CLOEXEC) == 0, "the leak must be inheritable");
    int ends[2];
    check(pipe2(ends, O_CLOEXEC) == 0, "open the handoff pipe");
    UniqueFd reader(ends[0]), writer(ends[1]);
    const int target = reader.get() == 3 ? 4 : 3;
    HelperProcess process;
    std::string error;
    check(process.spawn(vetted, {"--spawn-probe", std::to_string(target)}, FdHandoff{reader.get(), target},
                        error, strategy), error.c_str());
    check(write(writer.get(), "x", 1) == 1, "feed the handoff pipe");
    expect_exit(process, 0, "the helper must inherit only stdio and the handoff descriptor");
}

// The path is swapped after vetting; the vetted inode must still be what runs.
void swapped_file_does_not_change_what_runs(const char* program) {
    ScratchFile helper;
    helper.copy_of(program);
    const VettedHelper vetted = vet(helper.path);
    helper.replace_with("#!/bin/sh\nexit 99\n");
    HelperProcess process;
    std::string error;
    check(process.spawn(vetted, {"--spawn-probe", "-1"}, FdHandoff{}, error), error.c_str());
    check(process.reap(true), "collect the launched helper");
    check(process.last_exit().code != kProbeStillRan, "the swapped-in file ran instead of the vetted one");
    expect_exit(process, 0, "the vetted binary must run to completion");
}

// Teardown signals the process it spawned even if the recorded pid names another.
void teardown_signals_through_the_pidfd(const char* program) {
    ScratchFile helper;
    helper.copy_of(program);
    const VettedHelper vetted = vet(helper.path);
    HelperProcess process;
    std::string error;
    check(process.spawn(vetted, {"--spawn-probe", "-1", "pause"}, FdHandoff{}, error), error.c_str());
    const pid_t decoy = fork();
    check(decoy >= 0, "fork a decoy");
    if (decoy == 0) {
        alarm(8);
        pause();
        _exit(0);
    }
    HelperProcessTestAccess::set_pid(process, decoy);
    process.stop();
    check(process.pid() == -1, "teardown must leave the helper reaped");
    check(process.last_exit().signaled && process.last_exit().code == SIGTERM,
          "the helper itself must receive SIGTERM");
    int status = 0;
    const bool decoy_alive = waitpid(decoy, &status, WNOHANG) == 0;
    kill(decoy, SIGKILL);
    waitpid(decoy, &status, 0);
    check(decoy_alive, "teardown must not signal a process that merely has the recorded pid");
}

// helper_exit.hpp reads field 52 of /proc/<pid>/stat; comm (field 2) may hold ") ".
// "known/signaled/code" of a parsed exit, or "running" when the line is not one.
std::string parsed(const std::string& line) {
    HelperExit out;
    if (!parse_stat_exit(line.c_str(), out)) return "running";
    const std::string tail = out.still_exiting ? "" : " (not exiting?)";
    if (!out.known) return "unknown" + tail;
    return (out.signaled ? "signal " : "status ") + std::to_string(out.code) + tail;
}

void parses_stat_exit_codes() {
    std::string fields;
    for (int field = 4; field < 52; ++field) fields += " 0";
    check(parsed("7 (a) b) R 1 0") == "running", "a running task is not exited");
    check(parsed("7 (a) b) Z" + fields + " 9") == "signal 9", "zombie killed by signal 9");
    check(parsed("7 (x) X" + fields + " 768") == "status 3", "dead task that exited with status 3");
    check(parsed("7 (x) Z 1 0") == "unknown", "a line without field 52 is exited, status unknown");
    check(parsed("no paren") == "running", "a malformed line is not an exit");
}

// The probe sees a live child as running through both views, then its exit.
void watch_reports_a_live_then_exited_child(const char* program) {
    VettedHelper vetted;
    check(vet_helper(program, vetted) == nullptr, "the fixture copy must be a trusted helper");
    HelperProcess process;
    std::string error;
    check(process.spawn(vetted, {"--spawn-probe", "-1", "pause"}, FdHandoff{}, error), error.c_str());
    const HelperWatch watch = process.watch();
    check(watch.pidfd >= 0 && watch.stat_fd >= 0, "spawn must open both liveness views");
    HelperExit out;
    check(!helper_exited(watch, out), "a running helper is not exited");
    check(kill(process.pid(), SIGKILL) == 0, "kill the probe child");
    const auto gone = [&] { return helper_exited(watch, out); };
    for (int tries = 0; tries < 1000 && !gone(); ++tries) usleep(1000);
    check(out.known && out.signaled && out.code == SIGKILL, "the watch must report signal 9");
    check(process.reap(true), "collect the killed helper");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 && std::string(argv[1]) == "--spawn-probe") return probe(argc, argv);
    capture::start();
    ScratchFile trusted;
    trusted.copy_of(argv[0]);
    inherited_descriptors_are_closed(trusted.path, CloseStrategy::kAuto);
    inherited_descriptors_are_closed(trusted.path, CloseStrategy::kEnumerate);
    swapped_file_does_not_change_what_runs(trusted.path.c_str());
    teardown_signals_through_the_pidfd(trusted.path.c_str());
    parses_stat_exit_codes();
    watch_reports_a_live_then_exited_child(trusted.path.c_str());
    std::puts("Strata helper process spawn hardening fixture passed");
}
