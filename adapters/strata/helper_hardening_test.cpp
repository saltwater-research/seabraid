// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Spawn/teardown hardening fixture (security review 2026-11-17): SIGCHLD policy,
// the waitid(P_PIDFD) fallback, bounded teardown, setuid and directory vetting,
// the child environment gate. The controller launches scripted copies of this
// binary through HelperProcess; no channel, model or GPU is involved.
#include "helper_process.hpp"

#include "diag_capture.hpp"
#include "helper_kernel.hpp"
#include "helper_hardening_child.hpp"
#include "test_support.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

// F1 (audit 2026-11-19): kill(2) is interposed for the whole binary, so ANY call the supervisor makes - a restored
// numeric-pid fallback included - is counted while armed.  The tests' own cleanup calls run unarmed.
static std::atomic<int> g_kill_calls{0};
static std::atomic<bool> g_count_kill{false};
extern "C" int kill(pid_t pid, int number) noexcept {
    using Real = int (*)(pid_t, int);
    static const Real real = reinterpret_cast<Real>(dlsym(RTLD_NEXT, "kill"));
    if (g_count_kill.load()) g_kill_calls.fetch_add(1);
    if (real == nullptr) { errno = ENOSYS; return -1; }
    return real(pid, number);
}

// Force thread exhaustion without changing the process limit of the test runner.
static std::atomic<bool> g_refuse_thread{false};
extern "C" int pthread_create(pthread_t* thread, const pthread_attr_t* attr,
                              void* (*entry)(void*), void* arg) noexcept {
    using Real = int (*)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
    static const Real real = reinterpret_cast<Real>(dlsym(RTLD_NEXT, "pthread_create"));
    if (g_refuse_thread.load()) return EAGAIN;
    return real ? real(thread, attr, entry, arg) : ENOSYS;
}

namespace {

using namespace seabraid::strata_adapter;
using testing::check;
using testing::ScratchFile;
namespace capture = seabraid::strata_adapter::capture;

namespace probe = seabraid::strata_adapter::testing::hardening_child;

VettedHelper vet(const std::string& path) {
    VettedHelper vetted;
    check(vet_helper(path, vetted) == nullptr, "the fixture copy must be a trusted helper");
    return vetted;
}

// A trusted copy of this binary, so `mode` children can be launched from it.
struct Fixture {
    ScratchFile copy;
    VettedHelper vetted;
    explicit Fixture(const char* self) {
        copy.copy_of(self);
        vetted = vet(copy.path);
    }
};

// Reports readiness (one byte) through a pipe the child writes on descriptor 20.
struct ReadyPipe {
    UniqueFd reader, source;   // `source` is the parent's write end, numbered above 99
    ReadyPipe() {
        int ends[2];
        check(pipe2(ends, O_CLOEXEC) == 0, "open the ready pipe");
        reader.reset(ends[0]);
        UniqueFd writer(ends[1]);
        source.reset(fcntl(writer.get(), F_DUPFD_CLOEXEC, 100));
        check(source.valid(), "number the ready pipe above the handoff target");
    }
    FdHandoff handoff() const { return FdHandoff{source.get(), probe::kReadyTarget}; }
    void wait() const {
        pollfd watch{reader.get(), POLLIN, 0};
        char byte = 0;
        check(poll(&watch, 1, 5000) == 1 && read(reader.get(), &byte, 1) == 1, "the child must report ready");
    }
};

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void spawn_ok(HelperProcess& process, const Fixture& fixture, const std::vector<std::string>& args,
              FdHandoff handoff = {}, bool debug_env = false) {
    std::string error;
    check(process.spawn(fixture.vetted, args, handoff, error, CloseStrategy::kAuto, debug_env), error.c_str());
}

// Real waitid, optionally lying about "still running" and slow to block, to model a
// kernel that is slow to release a killed GPU helper.
std::atomic<bool> g_slow_kernel{false};
int slow_waitid(int pidfd, siginfo_t* info, int options) {
    if (g_slow_kernel.load() && (options & WNOHANG)) {
        std::memset(info, 0, sizeof(*info));   // si_pid == 0: nothing to collect yet
        return 0;
    }
    if (g_slow_kernel.load()) usleep(1200 * 1000);
    return ::waitid(static_cast<idtype_t>(kWaitPidfd), static_cast<id_t>(pidfd), info, options);
}
int real_signal(int pidfd, int number) {
    return static_cast<int>(syscall(SYS_pidfd_send_signal, pidfd, number, nullptr, 0));
}
int einval_waitid(int, siginfo_t*, int) { errno = EINVAL; return -1; }
int eintr_waitid(int, siginfo_t*, int) { errno = EINTR; return -1; }
int eperm_signal(int, int) { errno = EPERM; return -1; }

// 1. SIGCHLD ignored or SA_NOCLDWAIT: refuse, with the reason in the log.
void ignored_sigchld_refuses_the_spawn(const Fixture& fixture) {
    for (const bool nocldwait : {false, true}) {
        struct sigaction ours {}, previous {};
        sigemptyset(&ours.sa_mask);
        ours.sa_handler = nocldwait ? SIG_DFL : SIG_IGN;
        ours.sa_flags = nocldwait ? SA_NOCLDWAIT : 0;
        check(sigaction(SIGCHLD, &ours, &previous) == 0, "set the SIGCHLD disposition");
        capture::clear();
        HelperProcess process;
        std::string error;
        const bool launched = process.spawn(fixture.vetted, {"--child", "exit"}, FdHandoff{}, error);
        check(sigaction(SIGCHLD, &previous, nullptr) == 0, "restore SIGCHLD");
        check(!launched && process.pid() == -1, "an auto-reaped helper must not be launched");
        check(error.find("SIGCHLD") != std::string::npos, "the refusal must name SIGCHLD");
        capture::require("SIGCHLD is ignored or SA_NOCLDWAIT", 1);
    }
}

// 4. A handoff onto itself would be closed in the child.
void same_source_and_target_is_refused(const Fixture& fixture) {
    UniqueFd fd(open("/dev/null", O_RDONLY));
    capture::clear();
    HelperProcess process;
    std::string error;
    check(!process.spawn(fixture.vetted, {"--child", "exit"}, FdHandoff{fd.get(), fd.get()}, error),
          "a handoff with equal source and target must be refused");
    capture::require("same source and target", 1);
}

int enosys_signal(int, int) { errno = ENOSYS; return -1; }
int einval_signal(int, int) { errno = EINVAL; return -1; }
int enosys_open(pid_t) { errno = ENOSYS; return -1; }
int eperm_open(pid_t) { errno = EPERM; return -1; }
int einval_open(pid_t) { errno = EINVAL; return -1; }

// No child of this process exists (reaped or not): the refusal came before any spawn.
bool no_children() {
    int status = 0;
    return waitpid(-1, &status, WNOHANG) < 0 && errno == ECHILD;
}

// 2. F1: where the kernel or seccomp refuses any pidfd call the launch fails, loud and before any process exists, and
// kill(2) is never called.  Runs first, so "no children" is exact.
void launch_refused_without_pidfd_signalling(const Fixture& fixture) {
    const struct { KernelCalls calls; const char* call; } cases[] = {
        {{slow_waitid, enosys_signal}, "pidfd_send_signal failed (errno 38)"},
        {{slow_waitid, eperm_signal}, "pidfd_send_signal failed (errno 1)"},
        {{slow_waitid, einval_signal}, "pidfd_send_signal failed (errno 22)"},
        {{slow_waitid, real_signal, enosys_open}, "pidfd_open failed (errno 38)"},
        {{slow_waitid, real_signal, eperm_open}, "pidfd_open failed (errno 1)"},
        {{slow_waitid, real_signal, einval_open}, "pidfd_open failed (errno 22)"},
        {{einval_waitid, real_signal}, "waitid(P_PIDFD) failed (errno 22)"},
    };
    for (const auto& c : cases) {
        set_kernel_calls_for_test(&c.calls);
        capture::clear();
        g_kill_calls = 0;
        g_count_kill = true;
        HelperProcess process;
        std::string error;
        const bool launched = process.spawn(fixture.vetted, {"--child", "exit"}, FdHandoff{}, error);
        if (launched) process.stop();
        g_count_kill = false;
        set_kernel_calls_for_test(nullptr);
        if (launched) process.stop();
        check(!launched && process.pid() == -1, "a launch where pidfd signalling is refused must fail");
        check(error.find("refuses pidfd signalling; the private helper needs it") != std::string::npos, error.c_str());
        capture::require(c.call, 1);   // one record, naming the refused call and its errno
        check(g_kill_calls.load() == 0, "kill(2) must never be called");
        check(no_children(), "the refusal must come before any process exists");
    }
}

// 2b. F1: a pidfd signal refused after launch (a seccomp profile installed later, or a kernel fault) is reported and
// never retried by number; the helper is not signalled by pid, so it stays alive until a pidfd signal works.
void teardown_never_signals_by_number(const Fixture& fixture) {
    ReadyPipe ready;
    HelperProcess process;
    spawn_ok(process, fixture, {"--child", "ignore-term"}, ready.handoff());
    ready.wait();
    const int pidfd = process.pidfd();
    const KernelCalls refused{slow_waitid, eperm_signal};
    set_kernel_calls_for_test(&refused);
    capture::clear();
    g_kill_calls = 0;
    g_count_kill = true;
    process.stop();
    g_count_kill = false;
    check(g_kill_calls.load() == 0, "teardown must never fall back to kill(pid)");
    capture::require("SIGTERM failed (errno 1)", 1);
    capture::require("SIGKILL failed (errno 1)", 1);
    check(process.pid() == -1, "the object hands the unsignalled child to the background reap");
    pollfd watch{pidfd, POLLIN, 0};
    check(poll(&watch, 1, 0) == 0, "the child must still be alive: nothing signalled it by number");
    set_kernel_calls_for_test(nullptr);   // the reserved worker must retry SIGKILL itself
    for (int waited = 0; waited < 2000 && capture::count("ended by signal 9") == 0; waited += 10) usleep(10 * 1000);
    capture::require("ended by signal 9", 1);   // the background reap collected it, by pidfd
}

// 8. The launch gate names what is missing, uncached, and passes on this host.
void gate_names_the_refused_call() {
    int error = 0;
    const KernelCalls blocked{einval_waitid, real_signal};
    set_kernel_calls_for_test(&blocked);
    const char* call = pidfd_supervision_fault(error);
    check(call != nullptr && std::string(call) == "waitid(P_PIDFD)" && error == EINVAL, "the gate must name waitid");
    set_kernel_calls_for_test(nullptr);
    check(pidfd_supervision_fault(error) == nullptr, "this host must support pidfd supervision (kernel >= 5.4)");
}

// 3. A slow kernel must not hold stop() hostage after SIGKILL.
void stop_is_bounded_after_sigkill(const Fixture& fixture) {
    const KernelCalls slow{slow_waitid, real_signal};
    set_kernel_calls_for_test(&slow);
    ReadyPipe ready;
    HelperProcess process;
    spawn_ok(process, fixture, {"--child", "ignore-term"}, ready.handoff());
    ready.wait();
    capture::clear();
    struct rlimit previous{};
    check(getrlimit(RLIMIT_NOFILE, &previous) == 0, "read the descriptor limit");
    const struct rlimit exhausted{0, previous.rlim_max};
    check(setrlimit(RLIMIT_NOFILE, &exhausted) == 0, "deny new descriptors during retirement");
    g_slow_kernel = true;
    g_refuse_thread = true;   // teardown must work even when a new worker cannot be created
    const auto start = std::chrono::steady_clock::now();
    process.stop();
    const double took = seconds_since(start);
    g_refuse_thread = false;
    check(setrlimit(RLIMIT_NOFILE, &previous) == 0, "restore the descriptor limit");
    check(took < 0.9, "stop() must hand the reap to a background thread instead of blocking on the kernel");
    check(process.pid() == -1, "the object must let go of a helper it handed to the reaper");
    usleep(1700 * 1000);   // the detached reaper finishes inside the slow window
    g_slow_kernel = false;
    capture::require("ended by signal 9", 1);
    set_kernel_calls_for_test(nullptr);
}

void thread_exhaustion_refuses_launch(const Fixture& fixture) {
    capture::clear();
    HelperProcess process;
    std::string error;
    g_refuse_thread = true;
    const bool launched = process.spawn(fixture.vetted, {"--child", "exit"}, {}, error);
    g_refuse_thread = false;
    check(!launched && error.find("reserve a reaper") != std::string::npos, "no reaper must refuse the launch");
    check(no_children(), "thread exhaustion must be handled before any child exists");
    capture::require("cannot reserve a reaper", 1);
}

void wait_error_retains_the_child(const Fixture& fixture, const KernelCalls& denied) {
    UniqueFd observer;
    capture::clear();
    {
        HelperProcess process;
        spawn_ok(process, fixture, {"--child", "exit"});
        observer.reset(fcntl(process.pidfd(), F_DUPFD_CLOEXEC, 0));
        check(observer.valid(), "keep an independent status observation descriptor");
        pollfd watch{observer.get(), POLLIN, 0};
        check(poll(&watch, 1, 5000) == 1, "child must have exited");
        set_kernel_calls_for_test(&denied);
        check(!process.reap(false), "a readable pidfd does not prove that the child was collected");
        const auto start = std::chrono::steady_clock::now();
        process.stop();
        check(seconds_since(start) < 0.9, "an unexpected wait error must not block the caller");
    }
    siginfo_t status{};
    const auto observe = [&] {
        return ::waitid(static_cast<idtype_t>(kWaitPidfd), static_cast<id_t>(observer.get()),
                        &status, WEXITED | WNOWAIT | WNOHANG);
    };
    check(observe() == 0 && status.si_pid != 0, "the kernel must still hold the exit status during refusal");
    set_kernel_calls_for_test(nullptr);
    for (int i = 0; i < 200 && capture::count("exited with code 0") == 0; ++i) usleep(10 * 1000);
    capture::require("exited with code 0", 1);
    check(observe() < 0 && errno == ECHILD, "the worker must consume the status when waitid recovers");
}

// 6. Setuid/setgid helpers and world-writable parent directories are refused.
void unsafe_helpers_are_refused(const char* self) {
    for (const mode_t bits : {mode_t(04755), mode_t(02755)}) {
        ScratchFile helper;
        helper.copy_of(self);
        helper.mode(bits);
        VettedHelper vetted;
        capture::clear();
        check(vet_helper(helper.path, vetted) != nullptr, "a setuid/setgid helper must be refused");
        capture::require("setuid or setgid", 1);
    }
    std::string dir = std::string(getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp") + "/seabraid_dir_XXXXXX";
    check(mkdtemp(dir.data()) != nullptr, "make a scratch directory");
    const std::string path = dir + "/helper";
    std::ifstream in(self, std::ios::binary);
    std::ofstream(path, std::ios::binary) << in.rdbuf();
    chmod(path.c_str(), 0755);
    chmod(dir.c_str(), 0777);   // world-writable, no sticky bit
    VettedHelper vetted;
    capture::clear();
    const bool refused = vet_helper(path, vetted) != nullptr;
    chmod(dir.c_str(), 01777);   // sticky, like /tmp: acceptable
    const bool sticky_ok = vet_helper(path, vetted) == nullptr;
    unlink(path.c_str());
    rmdir(dir.c_str());
    check(refused, "a helper in a world-writable directory must be refused");
    capture::require("world-writable without the sticky bit", 1);
    check(sticky_ok, "a sticky world-writable directory (like /tmp) is acceptable");
}

// 7. The helper sees allow-listed variables only; debug switches need the option.
void child_environment_is_gated(const Fixture& fixture) {
    for (const auto& entry : probe::kEnvCases) check(setenv(entry.name, entry.value, 1) == 0, "set probe variable");
    for (const bool debug : {false, true}) {
        HelperProcess process;
        spawn_ok(process, fixture, {"--child", "env", debug ? "1" : "0"}, {}, debug);
        check(process.reap(true), "collect the environment probe");
        const int code = process.last_exit().code;
        if (code != 0) std::fprintf(stderr, "environment probe failed with check %d (debug=%d)\n", code, debug);
        check(!process.last_exit().signaled && code == 0, "the child environment must match the allow-list policy");
    }
    for (const auto& entry : probe::kEnvCases) unsetenv(entry.name);
}

// 8. The helper's first act is no_new_privs (security review 2026-11-17); a scripted child proves the call sets it.
void helper_forbids_new_privileges(const Fixture& fixture) {
    HelperProcess process;
    spawn_ok(process, fixture, {"--child", "nnp"});
    check(process.reap(true), "collect the no_new_privs probe");
    const int code = process.last_exit().code;
    if (code != 0) std::fprintf(stderr, "no_new_privs probe failed with check %d\n", code);
    check(!process.last_exit().signaled && code == 0, "forbid_new_privileges must set PR_SET_NO_NEW_PRIVS");
}

// 1b. A host that reaps with waitpid(-1) must never leave a signalable stranger.
void host_reaping_is_handled(const Fixture& fixture) {
    std::atomic<bool> run{true};
    std::thread reaper([&] { while (run) { int status; if (waitpid(-1, &status, WNOHANG) <= 0) std::this_thread::yield(); } });
    for (int round = 0; round < 40; ++round) {
        HelperProcess process;
        std::string error;
        if (process.spawn(fixture.vetted, {"--child", "exit"}, FdHandoff{}, error)) { process.stop(); }
        else check(error.find("reaped") != std::string::npos, error.c_str());
        check(process.pid() == -1, "a stopped or refused helper must leave no pid behind");
    }
    run = false;
    reaper.join();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 && std::string(argv[1]) == "--child") return probe::child(argv[2], argc > 3 ? argv[3] : "");
    capture::start();
    const Fixture fixture(argv[0]);
    launch_refused_without_pidfd_signalling(fixture);
    ignored_sigchld_refuses_the_spawn(fixture);
    same_source_and_target_is_refused(fixture);
    gate_names_the_refused_call();
    thread_exhaustion_refuses_launch(fixture);
    wait_error_retains_the_child(fixture, {einval_waitid, real_signal});
    wait_error_retains_the_child(fixture, {eintr_waitid, real_signal});
    teardown_never_signals_by_number(fixture);
    stop_is_bounded_after_sigkill(fixture);
    unsafe_helpers_are_refused(argv[0]);
    child_environment_is_gated(fixture);
    helper_forbids_new_privileges(fixture);
    host_reaping_is_handled(fixture);
    std::puts("Strata helper hardening fixture passed");
}
