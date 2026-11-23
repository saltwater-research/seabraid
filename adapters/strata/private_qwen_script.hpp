// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Test-only: the scripted CPU helper the private Qwen fixtures exec themselves as.
// OWNER: this header holds the peer's script and the shared geometry constants.
//   WHY a header: private_qwen_test.cpp reached the 500-line limit, and the fixture
//   (script) and its cases (private_qwen_test.cpp) are two concepts. Include once.
#pragma once

#include "private_qwen.hpp"

#include "strata/seabraid/channel.hpp"
#include "strata/seabraid/peer_watch.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/window_limits.hpp"

#include "helper_options.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <fcntl.h>
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using seabraid::strata_adapter::PrivateQwenPlan;
constexpr auto& kGeometry = strata::core::kCompiledGeometry;
constexpr auto kWindow = strata::kernels::kWindowMaxT;
constexpr seabraid::ChannelLimits kLimits{kGeometry.n_embd, kWindow * kGeometry.n_expert_used,
                                        kGeometry.n_layers, kGeometry.n_expert, 0, 0, 0};
constexpr uint32_t kLayer = 1;
constexpr uint32_t kExpert = 3;
const PrivateQwenPlan kPlan{{{kLayer, kExpert}}, 1, false};

// The fixture owns exactly one child at a time: after teardown nothing may be left to wait for.
// WHY inline: a TU that never calls it (the stale-epoch fixture) must not trip -Wunused-function.
inline void check_reaped() {
    seabraid::strata_adapter::testing::check(waitpid(-1, nullptr, WNOHANG) == -1 && errno == ECHILD,
                                             "controller must reap its child");
}

bool receive_plan(seabraid::Channel& channel) {
    auto* h = channel.header();
    const auto planned = [&] { return h->state.load(std::memory_order_acquire) == seabraid::kPlanReady; };
    return seabraid::Channel::wait(planned, 3) && h->plan_count == 1 &&
           h->plan_slots == 1 && h->plan_auto == 0 &&
           channel.plan()[0] == kLayer && channel.plan()[1] == kExpert &&
           h->req_seq.load() == 0 && h->resp_seq.load() == 0;
}

// The reply to request `sequence`. WHY any row count up to the window: the edge fixtures send a full window and
// compacted batches; a one-row request is answered exactly as before. Row t carries its own input, so a
// mis-scattered compacted row shows up as the wrong value, not just a wrong position.
bool serve_rows(seabraid::Channel& channel, uint64_t sequence) {
    auto* h = channel.header();
    if (h->layer != kLayer || h->n_tok < 1 || h->n_tok > kWindow || h->k != kGeometry.n_expert_used ||
        h->out_mode != seabraid::kOutPerToken) return false;
    for (int64_t t = 0; t < h->n_tok; ++t)
        std::fill_n(channel.out() + size_t(t) * kGeometry.n_embd, kGeometry.n_embd,
                    channel.x()[size_t(t) * kGeometry.n_embd] * channel.weights()[size_t(t) * kGeometry.n_expert_used] + 4.0f);
    h->out_rows = static_cast<int32_t>(h->n_tok);
    h->ok = 1;
    h->resp_seq.store(sequence, std::memory_order_release);
    return true;
}

bool answer(seabraid::Channel& channel, const std::string& mode, uint64_t sequence) {
    auto* h = channel.header();
    const auto published = [&] { return h->req_seq.load(std::memory_order_acquire) == sequence; };
    if (!seabraid::Channel::wait(published, 3)) return false;
    if (mode == "die-on-request") raise(SIGKILL);   // an OOM kill: no state change on the wire
    if (mode == "main-exits-on-request") {
        // A SIGKILLed GPU helper as the controller sees it while the kernel frees
        // device memory: main thread gone, process (and its pidfd) not yet.
        std::thread([] { sleep(3); _exit(0); }).detach();
        pthread_exit(nullptr);
    }
    if (mode == "stuck-request") {
        raise(SIGSTOP);   // SIGTERM cannot complete close.
        return false;
    }
    return serve_rows(channel, sequence);
}

// "serve-many" answers request after request (the edge fixtures drive several batches through one helper);
// every other mode answers exactly one and then idles, and "parent-loss" answers none.
bool answer_requests(seabraid::Channel& channel, const std::string& mode) {
    if (mode == "parent-loss") return true;
    for (uint64_t sequence = 1;; ++sequence) {
        if (!answer(channel, mode, sequence)) return false;
        if (mode != "serve-many") return true;
    }
}

bool open_channel(seabraid::Channel& channel, const seabraid::QwenHelperOptions& args) {
    auto limits = kLimits;
    if (args.pack == "bad-geometry") ++limits.hidden;
    std::string error;
    if (!args.private_channel() || !args.keep_descriptor_private(error) ||
        !args.create_channel(channel, limits, kWindow, error)) return false;
    return fcntl(args.shm_fd, F_GETFD) == -1 && errno == EBADF;
}

int script(const seabraid::QwenHelperOptions& args) {
    alarm(8);   // A broken parent must not leave an idle fixture running.
    // The controller exports one allow-listed and one non-allow-listed variable
    // (see main); every scripted launch therefore also verifies the spawn
    // environment filter, and a leak shows up as a startup failure.
    if (std::getenv("UNLISTED_TEST_SECRET")) return 21;
    if (!std::getenv("LC_NUMERIC")) return 22;
    if (args.pack == "startup-exit") return 17;
    seabraid::Channel channel;
    if (!open_channel(channel, args)) return 1;
    auto* h = channel.header();
    seabraid::announce_identity(*h);
    h->state.store(seabraid::kHelperUp, std::memory_order_release);
    if (!receive_plan(channel)) return 3;
    if (args.pack == "load-exit") return 19;
    if (args.pack == "reject-plan") {
        // Hostile text: an escape sequence, then 'X' to the last byte with no NUL (the controller must bound and strip it).
        std::memset(h->msg, 'X', sizeof(h->msg));
        std::memcpy(h->msg, "fixture refused expert\x1b[31m", 27);
        h->state.store(seabraid::kFailed, std::memory_order_release);
    } else {
        channel.held()[kLayer * kGeometry.n_expert + kExpert] = 1;
        h->state.store(seabraid::kLoaded, std::memory_order_release);
        if (!answer_requests(channel, args.pack)) return 4;
    }
    pause();   // Remain a dedicated worker until the controller stops it.
    return 7;
}

}  // namespace
