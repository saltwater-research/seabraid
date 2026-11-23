// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// OWNER: how a helper process says WHICH process it is, and how the engine learns that process is gone.  The helper half
//   (announce_identity) and the engine half (PeerWatch) are one contract in one file, the same split as channel_residency.hpp.
// CONTRACT: the helper calls announce_identity(header) BEFORE the release store of kHelperUp; the engine reads helper_pid and
//   helper_start only after an acquire load shows kHelperUp, kLoaded or kFailed, then PeerWatch::start(pid, start, err).
//   lost() turns true once and stays true.  Nothing here ever signals a process (F1): the thread sets one atomic and the
//   ENGINE thread acts on it.  Wire v10 carries the identity (Header::helper_start).
// SIZE BUDGET: <=200 lines.
// FAILURE CONTRACT: start() refuses with a text naming the cause; a refusal is the caller's "this channel's helper is dead", never
//   an engine exit.  A failure to read stat in the thread is "no news" (the pidfd stays the authority); only a zombie/dead state, a
//   readable pidfd or ESRCH on the stat descriptor mean lost, so a busy or swapping helper is never declared dead.
// CHECKER: adapters/strata/peer_watch_test.cpp (W1 stat grammar, W2 start refusals, W3 detection timing, W4 publication order).
#include "strata/seabraid/channel_layout.hpp"
#include "strata/seabraid/pidfd.hpp"
#include "strata/seabraid/proc_stat.hpp"

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>

#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>

namespace seabraid {

/// pread of a /proc/<pid>/stat descriptor into `line` (NUL-terminated).  false = nothing readable (errno kept for the caller).
inline bool read_stat_fd(int fd, char (&line)[1024]) noexcept {
    ssize_t got;
    do { got = ::pread(fd, line, sizeof(line) - 1, 0); } while (got < 0 && errno == EINTR);
    if (got == 0) errno = ENODATA;
    if (got <= 0) return false;
    line[got] = '\0';
    return true;
}

/// (pid, starttime) of a live process by number; false when /proc/<pid>/stat is unreadable or malformed.
inline bool read_starttime(int32_t pid, uint64_t& start) noexcept {
    char path[40], line[1024];
    std::snprintf(path, sizeof(path), "/proc/%d/stat", static_cast<int>(pid));
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = read_stat_fd(fd, line) && procstat::starttime_of(line, start);
    ::close(fd);
    return ok;
}

/// Helper side: write this process's identity into the channel.  WHY before the release store of kHelperUp and not after: the
/// engine reads the identity only after its acquire load of the state, so an identity written afterwards could be read as zero.
/// A start time that cannot be read is published as 0, which PeerWatch::start refuses by name (never a silent match).
inline void announce_identity(Header& h) noexcept {
    uint64_t start = 0;
    h.helper_pid = static_cast<int32_t>(::getpid());
    if (!read_starttime(h.helper_pid, start)) start = 0;
    h.helper_start = start;
}

/// Engine side: one thread, two descriptors, one atomic.
class PeerWatch {
public:
    PeerWatch() = default;
    PeerWatch(const PeerWatch&) = delete;
    PeerWatch& operator=(const PeerWatch&) = delete;
    ~PeerWatch() { stop(); }

    /// pidfd_open FIRST, then the starttime compare: with the open first, a pid recycled afterwards cannot fool us, because the
    /// pidfd keeps naming the process it opened (comparing first would leave a reuse window between the compare and the open).
    /// A zombie whose identity matches is "gone": it is over, and watching it would only report a loss that already happened.
    bool start(int32_t pid, uint64_t start, std::string& err) {
        if (thread_.joinable()) { err = "peer watch already started"; return false; }
        if (pid <= 0 || start == 0) { err = "no identity published (pid " + std::to_string(pid) + ")"; return false; }
        pidfd_ = sys_pidfd_open(static_cast<pid_t>(pid));
        if (pidfd_ < 0) { err = "not visible (dead or another pid namespace)"; return false; }
        if (!open_and_compare(pid, start, err) || !launch(err)) { release(); return false; }
        return true;
    }

    bool lost() const noexcept { return lost_.load(std::memory_order_acquire); }
    /// The signal that ended the helper, from field 52 of its stat while it was a zombie; 0 for a normal exit or an unseen status.
    int exit_signal() const noexcept { return signal_.load(std::memory_order_acquire); }

private:
    bool open_and_compare(int32_t pid, uint64_t start, std::string& err) {
        char path[40], line[1024];
        std::snprintf(path, sizeof(path), "/proc/%d/stat", static_cast<int>(pid));
        statfd_ = ::open(path, O_RDONLY | O_CLOEXEC);
        uint64_t now = 0;
        char state = '?';
        if (statfd_ < 0 || !read_stat_fd(statfd_, line) || !procstat::state_of(line, state) || !procstat::starttime_of(line, now)) {
            err = "not visible (dead or another pid namespace)";
            return false;
        }
        if (now != start) { err = "pid reused"; return false; }
        if (procstat::dead_state(state)) { err = "gone"; return false; }
        return true;
    }

    bool launch(std::string& err) {
        wake_ = ::eventfd(0, EFD_CLOEXEC);
        if (wake_ < 0) { err = "cannot create the watch wakeup"; return false; }
        try {
            thread_ = std::thread([this] { run(); });
        } catch (const std::exception& ex) {
            err = std::string("cannot start the helper watch thread: ") + ex.what();
            return false;
        }
        return true;
    }

    // WHY 100 ms and both descriptors: the pidfd fires only when the LAST thread of the helper is gone, which a 22 GB GPU
    // teardown stretches to ~0.6 s, while the main thread shows 'Z' in stat at once; polling the stat descriptor too bounds the
    // detection at one period whatever the driver does, and the same wake is the only way to see an idle death with no request.
    void run() {
        pollfd fds[2] = {{pidfd_, POLLIN, 0}, {wake_, POLLIN, 0}};
        while (true) {
            const int ready = ::poll(fds, 2, kPeriodMs);
            if (ready < 0 && errno != EINTR) return;
            if (stop_.load(std::memory_order_acquire) || fds[1].revents != 0) return;   // the destructor
            if ((ready > 0 && (fds[0].revents & POLLIN)) || stat_says_dead()) { mark_lost(); return; }
        }
    }

    // One pread per wake.  Z or X: dead, and the exit status is readable while the zombie is unreaped.  ESRCH: reaped, so dead.
    // Any other failure is "no news" and the pidfd decides.
    bool stat_says_dead() {
        char line[1024], state = '?';
        if (!read_stat_fd(statfd_, line)) return errno == ESRCH;
        if (!procstat::state_of(line, state) || !procstat::dead_state(state)) return false;
        note_signal(line);
        return true;
    }

    void note_signal(const char* line) {
        long status = 0;
        if (!procstat::exit_status_of(line, status)) return;
        const procstat::WaitStatus decoded = procstat::decode_wait_status(status);
        if (decoded.signaled) signal_.store(decoded.code, std::memory_order_release);
    }

    // The pidfd can fire before the stat read saw the zombie; one last read picks up the status for the log line.
    void mark_lost() {
        char line[1024];
        if (read_stat_fd(statfd_, line)) note_signal(line);
        lost_.store(true, std::memory_order_release);   // after the signal, so a reader that sees lost() sees the signal too
    }

    void stop() {
        if (thread_.joinable()) {
            const uint64_t one = 1;
            stop_.store(true, std::memory_order_release);
            if (::write(wake_, &one, sizeof(one)) < 0) { /* the flag above ends the thread within one period instead */ }
            thread_.join();
        }
        release();
    }

    void release() {
        for (int* fd : {&pidfd_, &statfd_, &wake_}) {
            if (*fd >= 0) ::close(*fd);
            *fd = -1;
        }
    }

    static constexpr int kPeriodMs = 100;
    int pidfd_ = -1, statfd_ = -1, wake_ = -1;
    std::atomic<bool> lost_{false}, stop_{false};
    std::atomic<int> signal_{0};
    std::thread thread_;
};

}  // namespace seabraid
