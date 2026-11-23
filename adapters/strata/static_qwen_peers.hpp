// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Test-only: an in-process helper peer (the scripted helper's side of one private channel)
// and the geometry constants the static-provider fixtures share.
// OWNER: this header; static_qwen_test.cpp and session_v2_test.cpp include it. Include once per TU.
#pragma once

#include "static_qwen.hpp"

#include "strata/seabraid/channel.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include <unistd.h>

namespace {

using seabraid::strata_adapter::StaticQwenProvider;
constexpr auto& kGeometry = strata::core::kCompiledGeometry;
constexpr auto kWindow = strata::kernels::kWindowMaxT;
constexpr seabraid::ChannelLimits kLimits{kGeometry.n_embd, kWindow * kGeometry.n_expert_used,
                                        kGeometry.n_layers, kGeometry.n_expert, 0, 0, 0};
constexpr uint32_t kLayer = 1;
constexpr uint32_t kTokens = 2;
constexpr uint32_t kDeadlineMs = 1000;
constexpr size_t kValues = kTokens * kGeometry.n_embd;
constexpr size_t kPicks = kTokens * kGeometry.n_expert_used;

void check(bool ok, const char* message) {
    if (ok) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

struct Peers {
    seabraid::Channel helper;
    std::unique_ptr<seabraid::Channel> main = std::make_unique<seabraid::Channel>();

    explicit Peers(seabraid::ChannelLimits limits = kLimits) {
        std::string error;
        const int fd = seabraid::Channel::private_fd(error);
        check(fd >= 0, "create private descriptor");
        check(helper.create_private(fd, limits, kWindow, error), "create helper mapping");
        check(main->connect_private(fd, limits.hidden, limits.cap, limits.layers, limits.experts, kWindow, error) ==
                  seabraid::Attach::kAttached, "attach main mapping");
        close(fd);
        auto* held = helper.held() + kLayer * limits.experts;
        std::fill_n(held, kGeometry.n_expert_used, uint8_t{1});
        helper.header()->state.store(seabraid::kLoaded, std::memory_order_release);
    }

    std::unique_ptr<StaticQwenProvider> bind(uint32_t deadline = kDeadlineMs) {
        std::string error;
        return StaticQwenProvider::bind(std::move(main), kLayer, deadline, error);
    }
};

}  // namespace
