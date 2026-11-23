// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "helper_process.hpp"
#include "static_qwen.hpp"

#include <vector>

namespace seabraid { class Channel; }

namespace seabraid::strata_adapter {

struct PrivateQwenLaunch {
    std::string helper_path;
    std::string expert_pack;
    std::string native_gguf;
    int device = 0;
    uint32_t startup_timeout_ms = 60000;
    uint32_t load_timeout_ms = 1800000;
    // Opens the session with SEABRAID_PROVIDER_COMPACT_ROWS: the helper gets only rows with a real
    // pick. Default OFF until measured; the one source of this option (no environment variable).
    bool compact_rows = false;
};

struct QwenExpertPick {
    int32_t layer;
    int32_t expert;
};

struct PrivateQwenPlan {
    std::vector<QwenExpertPick> ranked;
    int32_t slots = 0;
    bool auto_size = false;
};

// Owns one dedicated child and one sealed v7 generation. Startup fails closed
// and reaps the child on failure. Replacing a worker always creates new wire
// storage and publishes a new plan; each provider retains its own mapping.
// The helper must be an absolute, owner-only executable (spawn_policy.hpp) and
// receives an allow-listed environment; every failure is logged through diag.hpp.
class PrivateQwenWorker {
public:
    ~PrivateQwenWorker();
    PrivateQwenWorker(const PrivateQwenWorker&) = delete;
    PrivateQwenWorker& operator=(const PrivateQwenWorker&) = delete;

    static std::unique_ptr<PrivateQwenWorker> start(
        const PrivateQwenLaunch& launch, const PrivateQwenPlan& plan, uint32_t layer,
        uint32_t decode_timeout_ms, std::string& error);

    // Borrow for one Seabraid session. Finish seabraid_close before destroying the
    // worker, so the callback and child cannot race with teardown.
    StaticQwenProvider& adapter() noexcept { return *adapter_; }

private:
    PrivateQwenWorker() = default;
    bool spawn(const PrivateQwenLaunch& launch, const VettedHelper& helper, int fd, std::string& error);
    bool attach_and_wait(seabraid::Channel& channel, int fd, uint32_t timeout_ms,
                         std::string& error);
    bool wait_for_plan(seabraid::Channel& channel, uint32_t timeout_ms, std::string& error);

    // Declared before adapter_: the provider borrows process_'s pidfd, so it is destroyed first.
    HelperProcess process_;
    std::unique_ptr<StaticQwenProvider> adapter_;
};

}  // namespace seabraid::strata_adapter
