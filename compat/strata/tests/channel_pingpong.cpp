// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid phase-1 gate: round-trip latency of one layer request over the shared-memory channel between two
// processes (no GPU work), with a decode-sized payload: 1 token in (2560 floats), 10 rows of 2560 floats out.
#include "strata/seabraid/channel.hpp"
#include "strata/seabraid/peer_watch.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <sys/wait.h>
#include <vector>
int main() {
    const int64_t H = 2560, CAP = 160, L = 48, E = 512; const int iters = 20000;
    std::string err;
    pid_t pid = fork();
    if (pid == 0) {   // helper
        seabraid::Channel ch;
        if (!ch.create("/seabraid_pingpong", H, CAP, L, E, 16, 4096, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        auto* h = ch.header(); seabraid::announce_identity(*h); h->state.store(seabraid::kHelperUp);
        uint64_t seen = 0;
        for (int i = 0; i < iters; ++i) {
            uint64_t seq = 0;
            seabraid::Channel::wait([&] { seq = h->req_seq.load(std::memory_order_acquire); return seq != seen; }, 10);
            seen = seq;
            const float* x = ch.x(); float* out = ch.out();
            for (int r = 0; r < 10; ++r) std::memcpy(out + r * H, x, H * sizeof(float));   // 10 owned rows back
            h->ok = 1; h->resp_seq.store(seq, std::memory_order_release);
        }
        usleep(200000); return 0;
    }
    seabraid::Channel ch; bool ok = false;
    for (int t = 0; t < 100 && !ok; ++t) { ok = ch.connect("/seabraid_pingpong", H, CAP, L, E, 16, err); if (!ok) usleep(20000); }
    if (!ok) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    auto* h = ch.header();
    seabraid::Channel::wait([&] { return h->state.load() == seabraid::kHelperUp; }, 5);
    std::vector<float> x(16 * H, 1.0f), out(CAP * H);
    std::vector<double> us(iters);
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        std::memcpy(ch.x(), x.data(), 1 * H * sizeof(float));          // decode: 1 token
        h->req_seq.store(i + 1, std::memory_order_release);
        seabraid::Channel::wait([&] { return h->resp_seq.load(std::memory_order_acquire) == (uint64_t) i + 1; }, 10);
        for (int r = 0; r < 10; ++r) std::memcpy(out.data() + r * H, ch.out() + r * H, H * sizeof(float));
        us[i] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    }
    waitpid(pid, nullptr, 0);
    std::vector<double> s(us.begin() + 1000, us.end()); std::sort(s.begin(), s.end());
    std::printf("round trip (1 token in, 10 rows out), n=%zu: p50 %.2f us  p90 %.2f us  p99 %.2f us  max %.1f us\n",
                s.size(), s[s.size() / 2], s[s.size() * 9 / 10], s[s.size() * 99 / 100], s.back());
    std::printf("per token over 48 layers at p50: %.1f us\n", 48 * s[s.size() / 2]);
    return 0;
}
