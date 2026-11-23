// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "private_qwen.hpp"

#include "diag.hpp"
#include "helper_process.hpp"
#include "spawn_policy.hpp"
#include "unique_fd.hpp"

#include "strata/seabraid/channel.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>


namespace seabraid::strata_adapter {
namespace {

using diag::kError;
using diag::kInfo;

constexpr auto& kGeometry = ::strata::core::kCompiledGeometry;
constexpr auto kWindow = ::strata::kernels::kWindowMaxT;
constexpr auto kCap = kWindow * kGeometry.n_expert_used;
constexpr auto kPairs = kGeometry.n_layers * kGeometry.n_expert;

template <typename Poll>
bool wait_cold(uint32_t timeout_ms, Poll ready) {
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + milliseconds(timeout_ms);
    // Startup and process exit can take seconds, so avoid the decode slot's busy polling.
    do {
        if (ready()) return true;
        std::this_thread::sleep_for(milliseconds(1));
    } while (steady_clock::now() < deadline);
    return ready();
}

// WHY one place: every startup failure sets the caller's error string and
// writes exactly one log record, so no branch can do only one of the two.
void fail(std::string& error, int32_t status, const char* message) {
    error = message;
    diag::logf(kError, status, "private Qwen: %s", message);
}

// Returns nullptr for a usable launch, else a static reason.
const char* launch_fault(const PrivateQwenLaunch& launch) noexcept {
    const auto is_path = [](const std::string& value) { return value.find('\0') == std::string::npos; };
    if (launch.helper_path.empty() || launch.expert_pack.empty()) return "helper or pack path is empty";
    // Strata's synthetic device IDs would connect to a second helper by name.
    if (launch.device < 0 || launch.device >= seabraid::kDeviceBase) return "device ordinal out of range";
    if (launch.startup_timeout_ms == 0 || launch.load_timeout_ms == 0) return "startup or load timeout is zero";
    if (!is_path(launch.helper_path) || !is_path(launch.expert_pack) || !is_path(launch.native_gguf))
        return "path contains a NUL byte";
    return nullptr;
}

const char* plan_fault(const PrivateQwenPlan& plan) noexcept {
    if (plan.ranked.empty() || plan.ranked.size() > kPairs) return "plan pick count out of range";
    if (plan.slots <= 0 || plan.slots > kPairs) return "plan slot count out of range";
    const bool inside = std::all_of(plan.ranked.begin(), plan.ranked.end(), [](QwenExpertPick pick) {
        return pick.layer >= 0 && pick.layer < kGeometry.n_layers &&
               pick.expert >= 0 && pick.expert < kGeometry.n_expert;
    });
    return inside ? nullptr : "plan names a layer or expert outside the geometry";
}

const char* start_fault(const PrivateQwenLaunch& launch, const PrivateQwenPlan& plan, uint32_t layer) noexcept {
    if (const char* fault = launch_fault(launch)) return fault;
    if (const char* fault = plan_fault(plan)) return fault;
    return layer < kGeometry.n_layers ? nullptr : "layer outside the geometry";
}

void publish_plan(seabraid::Channel& channel, const PrivateQwenPlan& plan) {
    auto* pairs = channel.plan();
    for (const auto& pick : plan.ranked) {
        *pairs++ = pick.layer;
        *pairs++ = pick.expert;
    }
    auto* header = channel.header();
    header->plan_count = plan.ranked.size();
    header->plan_slots = plan.slots;
    header->plan_auto = plan.auto_size;
    header->state.store(seabraid::kPlanReady, std::memory_order_release);
}

bool attach(seabraid::Channel& channel, int fd, std::string& error) {
    const auto status = channel.connect_private(fd, kGeometry.n_embd, kCap, kGeometry.n_layers,
                                                kGeometry.n_expert, kWindow, error);
    if (status == seabraid::Attach::kAttached) return true;
    // WHY the enum, not the error text: v7 seals the extent before publishing magic, and Channel reports "the creator is
    // still initialising" as kNotReady, so a reworded diagnostic can never turn a retry into a failure or the reverse.
    if (status == seabraid::Attach::kNotReady) error.clear();
    return false;
}

// Identity is the pid the helper wrote plus the engine it was built for.
const char* identity_fault(const seabraid::Header& header, pid_t pid) noexcept {
    if (header.state.load(std::memory_order_acquire) != seabraid::kHelperUp) return "helper reported an unexpected state";
    if (header.helper_pid != pid) return "helper pid does not match the spawned child";
    if (header.engine != seabraid::kEngineQwen) return "helper engine is not Qwen";
    return nullptr;
}

const char* plan_state_fault(const seabraid::Header& header) noexcept {
    const auto state = header.state.load(std::memory_order_acquire);
    if (state == seabraid::kFailed) return "helper rejected its plan";
    return state == seabraid::kLoaded ? nullptr : "unexpected helper state while loading its plan";
}

}  // namespace

std::unique_ptr<PrivateQwenWorker> PrivateQwenWorker::start(
    const PrivateQwenLaunch& launch, const PrivateQwenPlan& plan, uint32_t layer,
    uint32_t decode_timeout_ms, std::string& error) {
    error.clear();
    if (const char* fault = start_fault(launch, plan, layer)) {
        fail(error, SEABRAID_INVALID, fault);
        return nullptr;
    }
    // Vet before creating the channel so a refused helper costs nothing.
    VettedHelper helper;
    if (const char* fault = vet_helper(launch.helper_path, helper)) {
        error = fault;   // vet_helper already logged the refusal
        return nullptr;
    }
    UniqueFd descriptor(seabraid::Channel::private_fd(error));
    if (!descriptor.valid()) {
        diag::logf(kError, SEABRAID_PROVIDER, "private Qwen: cannot create the private channel descriptor");
        return nullptr;
    }
    auto channel = std::make_unique<seabraid::Channel>();
    auto worker = std::unique_ptr<PrivateQwenWorker>(new PrivateQwenWorker);
    if (!worker->spawn(launch, helper, descriptor.get(), error) ||
        !worker->attach_and_wait(*channel, descriptor.get(), launch.startup_timeout_ms, error)) return nullptr;
    descriptor.reset();
    publish_plan(*channel, plan);
    if (!worker->wait_for_plan(*channel, launch.load_timeout_ms, error)) return nullptr;
    worker->adapter_ = StaticQwenProvider::bind(std::move(channel), layer, decode_timeout_ms, error);
    if (!worker->adapter_) return nullptr;
    // The provider polls this pidfd, so a dead helper retires it in milliseconds
    // instead of at the decode deadline. process_ outlives adapter_ (member order).
    worker->adapter_->watch_helper(worker->process_.watch());
    diag::logf(kInfo, SEABRAID_OK, "private Qwen helper loaded: %zu picks, %d slots", plan.ranked.size(), plan.slots);
    return worker;
}

bool PrivateQwenWorker::spawn(const PrivateQwenLaunch& launch, const VettedHelper& helper, int fd,
                              std::string& error) {
    // Keep the parent capability CLOEXEC throughout. Duplicate into a different
    // slot only in the child, avoiding dependence on same-fd dup2 semantics.
    const int child_fd = fd == 3 ? 4 : 3;
    const std::vector<std::string> args = {
        "--shm-fd", std::to_string(child_fd), "--pack", launch.expert_pack,
        "--layers", std::to_string(kGeometry.n_layers), "--experts", std::to_string(kGeometry.n_expert),
        "--device", std::to_string(launch.device), "--native", launch.native_gguf,
        "--controller-pid", std::to_string(::getpid()),
    };
    return process_.spawn(helper, args, FdHandoff{fd, child_fd}, error);
}

bool PrivateQwenWorker::attach_and_wait(seabraid::Channel& channel, int fd, uint32_t timeout_ms,
                                        std::string& error) {
    const auto ready = [&] {
        if (process_.reap(false)) { fail(error, SEABRAID_PROVIDER, "helper exited during startup"); return true; }
        if (!channel.valid() && !attach(channel, fd, error)) {
            // Channel text carries geometry numbers only, never paths.
            if (!error.empty()) diag::logf(kError, SEABRAID_PROVIDER, "private Qwen: channel attach refused: %s", error.c_str());
            return !error.empty();
        }
        const auto* header = channel.header();
        if (header->state.load(std::memory_order_acquire) == seabraid::kNone) return false;
        if (const char* fault = identity_fault(*header, process_.pid())) fail(error, SEABRAID_PROVIDER, fault);
        return true;
    };
    if (!wait_cold(timeout_ms, ready)) fail(error, SEABRAID_TIMEOUT, "helper startup timed out");
    return error.empty();
}

bool PrivateQwenWorker::wait_for_plan(seabraid::Channel& channel, uint32_t timeout_ms,
                                     std::string& error) {
    const auto ready = [&] {
        if (process_.reap(false)) { fail(error, SEABRAID_PROVIDER, "helper exited while loading its plan"); return true; }
        const auto* header = channel.header();
        if (header->state.load(std::memory_order_acquire) == seabraid::kPlanReady) return false;
        if (const char* fault = plan_state_fault(*header)) {
            fail(error, SEABRAID_PROVIDER, fault);
            // The helper's own message can name files, so only the caller's
            // error string carries it; the log record does not.
            if (header->state.load(std::memory_order_acquire) == seabraid::kFailed)
                error += ": " + seabraid::bounded_text(header->msg, sizeof header->msg);   // bounded, control bytes stripped
        }
        return true;
    };
    if (!wait_cold(timeout_ms, ready)) fail(error, SEABRAID_TIMEOUT, "helper plan timed out");
    return error.empty();
}

// WHY an explicit stop(): the provider must not outlive the process it polls,
// and stop() is the one place that logs the teardown sequence.
PrivateQwenWorker::~PrivateQwenWorker() { process_.stop(); }

}  // namespace seabraid::strata_adapter
