// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Host fixture for the engine's watch of a helper process (wire/strata/seabraid/peer_watch.hpp, proc_stat.hpp, pidfd.hpp).
// No GPU, no Strata engine.  Children are real forked processes that end themselves on a byte from a pipe (_exit or raise(SIGKILL)),
// so this test never signals a pid.
//   W1 the /proc/<pid>/stat grammar     W2 every refusal of PeerWatch::start    W3 detection timing and the destructor's join
//   W4 publication order: after the acquire of kHelperUp the identity is complete (10,000 iterations, plus a control that proves
//      the detector sees a misordered publisher)
// Guards: a ')' inside comm shifts every later field (a wrong starttime); a start that skips the starttime compare accepts a
// recycled pid; a watch that polls only the pidfd reports a 22 GB card's death 0.6 s late; an identity written after kHelperUp
// is read as zero.
#include "test_support.hpp"

#include "strata/seabraid/channel.hpp"
#include "strata/seabraid/peer_watch.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include <cstring>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using seabraid::strata_adapter::testing::check;
using namespace seabraid;
using Clock = std::chrono::steady_clock;

double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// ---- a child that ends on command.  'x' = _exit(3), 'k' = raise(SIGKILL), end of file = _exit(0); alarm(30) is its own fuse.
struct Child {
    pid_t pid = -1;
    int control = -1;   // parent's write end
    void command(char c) { check(::write(control, &c, 1) == 1, "send a command to the child"); }
    void finish() {     // close the pipe (the child exits) and reap it
        if (control >= 0) ::close(control);
        control = -1;
        if (pid > 0) ::waitpid(pid, nullptr, 0);
        pid = -1;
    }
};

Child spawn() {
    int fds[2];
    check(::pipe(fds) == 0, "pipe");
    const pid_t pid = ::fork();
    check(pid >= 0, "fork");
    if (pid == 0) {
        ::close(fds[1]);
        ::alarm(30);
        char c = 0;
        const ssize_t got = ::read(fds[0], &c, 1);
        if (got == 1 && c == 'k') ::raise(SIGKILL);
        ::_exit(got == 1 && c == 'x' ? 3 : 0);
    }
    ::close(fds[0]);
    return {pid, fds[1]};
}

uint64_t identity_of(const Child& c) {
    uint64_t start = 0;
    check(read_starttime(c.pid, start) && start != 0, "read the child's starttime");
    return start;
}

bool wait_zombie(pid_t pid) {
    char path[40], line[1024];
    std::snprintf(path, sizeof(path), "/proc/%d/stat", (int) pid);
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    check(fd >= 0, "open the child's stat");
    const auto t0 = Clock::now();
    bool dead = false;
    char state = '?';
    while (!dead && since(t0) < 2.0) {
        dead = read_stat_fd(fd, line) && procstat::state_of(line, state) && procstat::dead_state(state);
        if (!dead) ::usleep(5000);
    }
    ::close(fd);
    return dead;
}

bool wait_lost(const PeerWatch& w, double seconds) {
    const auto t0 = Clock::now();
    while (!w.lost() && since(t0) < seconds) ::usleep(5000);
    return w.lost();
}

// ---- W1: the stat grammar
// A stat line with `comm` as field 2, `state`, `starttime` as field 22 and `exit_code` as field 52; every other field is 0.
std::string stat_line(const char* comm, char state, unsigned long long starttime, long exit_code) {
    std::string line = "4321 (" + std::string(comm) + ") " + state;
    for (int field = 4; field <= 52; ++field)
        line += " " + std::to_string(field == 22 ? (long long) starttime : field == 52 ? exit_code : 0);
    return line;
}

void w1_comm() {
    for (const char* comm : {"helper", "a b", "x) y", "we) ird (x)", ") 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 ("}) {
        const std::string line = stat_line(comm, 'S', 1234567, 0);
        char state = '?';
        uint64_t start = 0;
        check(procstat::state_of(line.c_str(), state) && state == 'S', "W1: the state is read after the LAST ')'");
        check(procstat::starttime_of(line.c_str(), start) && start == 1234567, "W1: field 22 survives a ')' and spaces inside comm");
    }
}

void w1_states_and_status() {
    char state = '?';
    for (const char z : {'Z', 'X'}) {
        const std::string line = stat_line("h", z, 7, 9);
        check(procstat::state_of(line.c_str(), state) && procstat::dead_state(state), "W1: Z and X are dead states");
    }
    for (const char s : {'R', 'S', 'D', 'T', 't', 'I'}) check(!procstat::dead_state(s), "W1: a live state is not a dead one");
    long status = 0;
    check(procstat::exit_status_of(stat_line("h", 'Z', 7, 9).c_str(), status) && status == 9, "W1: field 52 is the wait status");
    check(procstat::decode_wait_status(9).signaled && procstat::decode_wait_status(9).code == 9, "W1: status 9 is SIGKILL");
    check(!procstat::decode_wait_status(0x300).signaled && procstat::decode_wait_status(0x300).code == 3, "W1: status 0x300 is exit code 3");
}

// malformed lines answer false and leave their outputs alone
void w1_malformed() {
    char state = '?';
    uint64_t start = 99;
    long status = 0;
    for (const char* bad : {"", "garbage", "1 (x)", "1 (x)S 0", "1 (x) "}) check(!procstat::state_of(bad, state), "W1: no state in a malformed line");
    check(!procstat::starttime_of("1 (x) S 0 0 0", start) && start == 99, "W1: a line cut before field 22 has no starttime");
    std::string text = "1 (x) S";
    for (int field = 4; field < 22; ++field) text += " 0";   // fields 4..21, then a field 22 that is not a number
    check(!procstat::starttime_of((text + " abc").c_str(), start), "W1: a non-numeric starttime is refused");
    check(!procstat::exit_status_of("1 (x) Z 0 0 0", status), "W1: a kernel without field 52 has no status");
}

// round trip against an independent reader (sscanf over the same file) and the identity the helper publishes
void w1_round_trip() {
    char raw[1024];
    FILE* f = std::fopen("/proc/self/stat", "r");
    check(f != nullptr && std::fgets(raw, sizeof(raw), f) != nullptr, "read /proc/self/stat");
    std::fclose(f);
    unsigned long long expect = 0;
    const char* after = std::strrchr(raw, ')');
    check(after != nullptr && std::sscanf(after + 2, "%*c %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %llu", &expect) == 1,
          "sscanf reads field 22");
    uint64_t mine = 0;
    check(read_starttime(::getpid(), mine) && mine == expect && mine != 0, "W1: read_starttime(self) equals the independent read");
    static Header h{};
    announce_identity(h);
    check(h.helper_pid == (int32_t) ::getpid() && h.helper_start == mine, "W1: announce_identity publishes (pid, starttime) of this process");
}

void w1_grammar() {
    w1_comm();
    w1_states_and_status();
    w1_malformed();
    w1_round_trip();
}

// ---- W2: start's refusals
bool refused(int32_t pid, uint64_t start, const char* why) {
    PeerWatch w;
    std::string err;
    if (w.start(pid, start, err)) return false;
    if (err.find(why) == std::string::npos) std::fprintf(stderr, "refusal text was \"%s\", wanted \"%s\"\n", err.c_str(), why);
    return err.find(why) != std::string::npos;
}

void w2_start() {
    Child live = spawn();
    const uint64_t start = identity_of(live);
    {
        PeerWatch w;
        std::string err;
        check(w.start(live.pid, start, err), ("W2: a live child with its own identity is accepted: " + err).c_str());
        check(!w.lost(), "W2: and is not lost");
        check(!w.start(live.pid, start, err), "W2: a second start is refused");
    }
    check(refused(live.pid, start + 1, "pid reused"), "W2: a live pid with the wrong starttime is 'pid reused'");
    check(refused(::getpid(), 1, "pid reused"), "W2: our own pid with a wrong start is refused");
    check(refused(live.pid, 0, "no identity"), "W2: a zero start (an unpublished identity) is refused by name");
    check(refused(0, start, "no identity"), "W2: pid 0 is refused by name");
    {
        PeerWatch w;
        std::string err;
        check(w.start(::getpid(), [] { uint64_t s = 0; read_starttime(::getpid(), s); return s; }(), err), "W2: our own pid with its real start is accepted");
    }
    live.command('x');   // ends, and stays a zombie until reaped: its identity still matches, but it is over
    check(wait_zombie(live.pid), "W2: the child became a zombie");
    check(refused(live.pid, start, "gone"), "W2: a matching identity that is already a zombie is 'gone'");
    const pid_t dead = live.pid;
    live.finish();
    check(refused(dead, start, "not visible"), "W2: the identity of a reaped child is 'not visible'");
}

// ---- W3: detection timing
void w3_timing() {
    for (const char how : {'x', 'k'}) {
        Child child = spawn();
        PeerWatch w;
        std::string err;
        check(w.start(child.pid, identity_of(child), err), ("W3: start: " + err).c_str());
        ::usleep(300000);
        check(!w.lost(), "W3: an idle live child is not lost after 0.3 s");
        const auto t0 = Clock::now();
        child.command(how);
        check(wait_lost(w, 2.0), "W3: lost() within 2 s of the child's exit");
        std::printf("W3: %s detected after %.3f s\n", how == 'k' ? "SIGKILL" : "_exit(3)", since(t0));
        check(w.exit_signal() == (how == 'k' ? 9 : 0), "W3: exit_signal() is the killing signal, 0 for a plain exit");
        child.finish();
    }
    Child idle = spawn();
    auto w = std::make_unique<PeerWatch>();
    std::string err;
    check(w->start(idle.pid, identity_of(idle), err), "W3: start (idle)");
    const auto quiet = Clock::now();
    while (since(quiet) < 1.0) { check(!w->lost(), "W3: no false positive over 1 s"); ::usleep(20000); }
    const auto t0 = Clock::now();
    w.reset();
    check(since(t0) < 0.2, "W3: the destructor joins in under 200 ms");
    idle.finish();
}

// ---- W4: publication order.  The writer clears the identity, then publishes it the way a helper does; the reader acquires
// kHelperUp and reads the identity at once.  `misordered` is the CONTROL: it stores kHelperUp first, and the detector must see that.
int ordering_failures(Channel& channel, int iterations, bool misordered, uint64_t want_start) {
    Header* h = channel.header();
    std::atomic<int> go{0}, ack{0};
    std::atomic<int> bad{0};
    std::thread reader([&] {
        for (int i = 1; i <= iterations; ++i) {
            while (go.load(std::memory_order_acquire) != i) {}
            while (h->state.load(std::memory_order_acquire) != kHelperUp) {}
            if (h->helper_pid != (int32_t) ::getpid() || h->helper_start != want_start) bad.fetch_add(1);
            ack.store(i, std::memory_order_release);
        }
    });
    for (int i = 1; i <= iterations; ++i) {
        h->helper_pid = 0;
        h->helper_start = 0;
        h->state.store(kNone, std::memory_order_release);
        go.store(i, std::memory_order_release);
        if (misordered) {
            h->state.store(kHelperUp, std::memory_order_release);
            ::usleep(50);
            announce_identity(*h);
        } else {
            announce_identity(*h);
            h->state.store(kHelperUp, std::memory_order_release);
        }
        while (ack.load(std::memory_order_acquire) != i) {}
    }
    reader.join();
    return bad.load();
}

void w4_ordering() {
    Channel channel;
    std::string err;
    const std::string name = "/seabraid_pw_" + std::to_string((long long) ::getpid());
    check(channel.create(name, 64, 32, 4, 8, 4, 256, err), ("W4: create: " + err).c_str());
    uint64_t start = 0;
    check(read_starttime(::getpid(), start), "W4: own starttime");
    check(ordering_failures(channel, 200, true, start) > 0, "W4 control: a publisher that stores kHelperUp first IS caught");
    const int bad = ordering_failures(channel, 10000, false, start);
    std::printf("W4: %d incomplete identities in 10000 publications\n", bad);
    check(bad == 0, "W4: after the acquire of kHelperUp the identity is complete");
}

}  // namespace

int main() {
    w1_grammar();
    w2_start();
    w3_timing();
    w4_ordering();
    std::puts("peer watch: all checks passed");
    return 0;
}
