// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// CPU-only check of descriptor inheritance and generation replacement. No launcher or GPU is required.
#include "strata/seabraid/channel.hpp"
#include "strata/seabraid/peer_watch.hpp"

#include <cstdio>
#include <spawn.h>
#include <string_view>
#include <sys/wait.h>

extern char** environ;

namespace {
constexpr int kHandoffFd = 198;
constexpr seabraid::ChannelLimits kLimits{64, 32, 4, 8, 256, 0, 0};
constexpr uint32_t kWindow = 4;
constexpr unsigned kCrashExit = 17;

bool connect(seabraid::Channel& ch, int fd, std::string& err) {
    return ch.connect_private(fd, kLimits.hidden, kLimits.cap, kLimits.layers, kLimits.experts, kWindow, err) ==
           seabraid::Attach::kAttached;
}

int helper(bool crash) {
    alarm(8);   // Bound failures even if the parent never publishes its request.
    seabraid::Channel ch;
    std::string err;
    if (!ch.create_private(kHandoffFd, kLimits, kWindow, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    close(kHandoffFd);   // The mapping alone keeps the bytes alive within this process.
    auto* h = ch.header();
    seabraid::announce_identity(*h);
    h->state.store(seabraid::kHelperUp, std::memory_order_release);
    if (!seabraid::Channel::wait([&] { return h->req_seq.load(std::memory_order_acquire) == 1; }, 5)) return 2;
    if (crash) _exit(kCrashExit);   // Exercise loss of the process without running channel destructors.
    ch.out()[0] = ch.x()[0] + 1;
    h->resp_seq.store(1, std::memory_order_release);
    return 0;
}

pid_t start(char* program, char* mode, int fd, bool handoff = true) {
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) return -1;
    const int action = handoff ? posix_spawn_file_actions_adddup2(&actions, fd, kHandoffFd) : 0;
    std::string source = std::to_string(fd);
    char* args[] = {program, mode, handoff ? nullptr : source.data(), nullptr};
    pid_t pid = -1;
    const int spawn = action == 0 ? posix_spawn(&pid, program, &actions, nullptr, args, environ) : action;
    posix_spawn_file_actions_destroy(&actions);
    return spawn == 0 ? pid : -1;
}

bool exited(pid_t pid, int expected) {
    if (pid <= 0) return false;
    int status = 0;
    pid_t reaped;
    do {
        reaped = waitpid(pid, &status, 0);
    } while (reaped < 0 && errno == EINTR);
    return reaped == pid && WIFEXITED(status) && WEXITSTATUS(status) == expected;
}

bool await_helper(seabraid::Channel& ch, int fd, std::string& err) {
    if (!seabraid::Channel::wait([&] { return ch.valid() || connect(ch, fd, err); }, 3)) return false;
    return seabraid::Channel::wait([&] {
        return ch.header()->state.load(std::memory_order_acquire) == seabraid::kHelperUp;
    }, 3);
}

bool run_generation(char* program, seabraid::Channel& ch, bool crash) {
    std::string err;
    const int fd = seabraid::Channel::private_fd(err);
    if (fd < 0) { std::fprintf(stderr, "%s\n", err.c_str()); return false; }
    const bool cloexec = (fcntl(fd, F_GETFD) & FD_CLOEXEC) != 0;
    char observer_mode[] = "observer";
    const bool isolated = exited(start(program, observer_mode, fd, false), 0);
    char crash_mode[] = "crash", serve_mode[] = "serve";
    const pid_t pid = start(program, crash ? crash_mode : serve_mode, fd);
    const bool attached = pid > 0 && await_helper(ch, fd, err);
    const bool sealed = attached && ftruncate(fd, (off_t) ch.bytes() - 1) == -1 && errno == EPERM;
    close(fd);
    if (attached) {
        // Fresh counters prove no stale request/response state crossed the generation.
        if (ch.header()->req_seq.load() != 0 || ch.header()->resp_seq.load() != 0) {
            std::fprintf(stderr, "generation inherited request counters\n");
            exited(pid, crash ? kCrashExit : 0);
            return false;
        }
        ch.x()[0] = 42;
        ch.header()->req_seq.store(1, std::memory_order_release);
    } else {
        std::fprintf(stderr, "helper did not attach: %s\n", err.c_str());
    }
    const bool reaped = exited(pid, crash ? kCrashExit : 0);
    if (!cloexec || !isolated || !attached || !sealed || !reaped) return false;
    const auto response = ch.header()->resp_seq.load(std::memory_order_acquire);
    return crash ? response == 0 : response == 1 && ch.out()[0] == 43;
}
bool rejects_unsealed(const seabraid::Channel& reference) {
    std::string err;
    const int fd = seabraid::Channel::private_fd(err);
    if (fd < 0) return false;
    // Keep a valid header and extent, so seal validation is the only reason to reject this fixture.
    const bool ready = ftruncate(fd, (off_t) reference.bytes()) == 0 &&
                       pwrite(fd, reference.header(), sizeof(seabraid::Header), 0) == (ssize_t) sizeof(seabraid::Header);
    seabraid::Channel reader;
    // kNotReady is the structured "creator not done yet" answer callers poll on; the text is only for the log.
    const bool rejected = ready && reader.connect_private(fd, kLimits.hidden, kLimits.cap, kLimits.layers, kLimits.experts,
                                                          kWindow, err) == seabraid::Attach::kNotReady &&
                          err.find("sealed extent") != std::string::npos;
    close(fd);
    return rejected;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2) {
        const std::string_view mode = argv[1];
        return helper(mode == "crash");
    }
    if (argc == 3 && std::string_view(argv[1]) == "observer") {
        // The original descriptor and designated handoff slot must both be absent in an unrelated exec.
        const auto closed = [](int fd) { return fcntl(fd, F_GETFD) == -1 && errno == EBADF; };
        return closed(std::stoi(argv[2])) && closed(kHandoffFd) ? 0 : 1;
    }
    seabraid::Channel abandoned, replacement;
    if (!run_generation(argv[0], abandoned, true) || !run_generation(argv[0], replacement, false)) {
        std::fprintf(stderr, "private handoff or crash recovery failed\n");
        return 1;
    }
    if (!rejects_unsealed(replacement) || abandoned.header()->resp_seq.load(std::memory_order_acquire) != 0) return 1;
    std::puts("private handoff and fresh-generation recovery passed");
}
