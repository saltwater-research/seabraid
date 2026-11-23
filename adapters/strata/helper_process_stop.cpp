// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// HelperProcess teardown: reap, signal, stop. Split from helper_process.cpp
// (spawn) to keep each file in its size budget; the class stays the one owner.
#include "helper_process.hpp"

#include "helper_kernel.hpp"

#include "diag.hpp"

#include "seabraid/seabraid.h"

#include <chrono>
#include <cerrno>
#include <csignal>
#include <future>
#include <memory>
#include <system_error>
#include <thread>

#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

namespace seabraid::strata_adapter {
namespace {

using diag::kError;
using diag::kInfo;
using diag::kWarn;

constexpr uint32_t kStopGraceMs = 300;
// How long stop() waits in line for a helper that is already dying before it
// hands the reap to a background thread (see stop_dying).
constexpr int kDyingSyncMs = 20;

// One record per collected child, from whichever thread collected it.
HelperExit record_exit(const HelperExit& exit) noexcept {
    if (!exit.known) diag::logf(kWarn, SEABRAID_PROVIDER, "private Qwen helper is no longer waitable");
    else if (exit.signaled) diag::logf(kInfo, SEABRAID_OK, "private Qwen helper ended by signal %d", exit.code);
    else diag::logf(kInfo, SEABRAID_OK, "private Qwen helper exited with code %d", exit.code);
    return exit;
}

// Readability proves exit, but only waitid success or ECHILD discharges ownership.
bool collect(int pidfd, bool block, HelperExit& out) noexcept {
    siginfo_t info{};
    const int result = kernel_waitid_pidfd(pidfd, &info, WEXITED | (block ? 0 : WNOHANG));
    if (result == 0 && info.si_pid == 0) return false;
    if (result == 0) {
        out = {true, info.si_code != CLD_EXITED, info.si_status, false};
        return true;
    }
    out = {};
    if (errno == ECHILD) return true;
    if (errno != EINTR) note_waitid_fallback(errno);
    return false;
}

bool readable_within(int pidfd, int timeout_ms) noexcept {
    pollfd watch{pidfd, POLLIN, 0};
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + milliseconds(timeout_ms);
    auto remaining = milliseconds(timeout_ms);
    while (remaining.count() > 0) {
        const int ready = ::poll(&watch, 1, static_cast<int>(remaining.count()));
        if (ready >= 0 || errno != EINTR) return ready > 0;
        remaining = ceil<milliseconds>(deadline - steady_clock::now());
    }
    return false;
}

// All shutdown resources are acquired before launch, including the worker's shared
// ownership of the pidfd. A retired helper can outlive its HelperProcess object.
enum class ReapTask { cancel, collect, retry_signal };

void finish_retirement(int fd, ReapTask task) noexcept {
    if (task == ReapTask::cancel) return;
    bool signalled = task == ReapTask::collect;
    HelperExit exit;
    for (;;) {
        if (!signalled) signalled = kernel_send_signal(fd, SIGKILL) == 0 || errno == ESRCH;
        if (collect(fd, signalled, exit)) break;
        // A denied wait or signal must retain the child without busy-looping.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    record_exit(exit);
}
}  // namespace

struct HelperProcess::Retirement {
    std::shared_ptr<UniqueFd> fd = std::make_shared<UniqueFd>();
    std::promise<ReapTask> command;
    std::thread worker{[owned = fd, next = command.get_future()]() mutable {
        const auto task = next.get();
        finish_retirement(owned->get(), task);
    }};

    void cancel() noexcept {
        if (!worker.joinable()) return;
        command.set_value(ReapTask::cancel);
        worker.join();  // this task never waits on the child
    }
    ~Retirement() { cancel(); }
};

HelperProcess::HelperProcess() = default;
HelperProcess::~HelperProcess() { stop(); }

int HelperProcess::reserve_retirement() noexcept {
    try {
        retirement_ = std::make_unique<Retirement>();
        return 0;
    } catch (const std::system_error& failure) {  // allow-error-path: spawn logs this returned errno before creating a child
        return failure.code().value();
    } catch (const std::bad_alloc&) {  // allow-error-path: spawn logs ENOMEM before creating a child
        return ENOMEM;
    }
}

void HelperProcess::release_retirement() noexcept { retirement_.reset(); }
UniqueFd& HelperProcess::child_fd() noexcept { return *retirement_->fd; }
int HelperProcess::pidfd() const noexcept { return retirement_ ? retirement_->fd->get() : -1; }

void HelperProcess::defer_retirement(bool retry_signal) noexcept {
    retirement_->command.set_value(retry_signal ? ReapTask::retry_signal : ReapTask::collect);
    retirement_->worker.detach();
    diag::logf(kInfo, SEABRAID_OK, "private Qwen teardown: collection handed to the reserved reaper");
    exit_ = {};
    pid_ = -1;
}

bool HelperProcess::reap(bool block) noexcept {
    if (pid_ < 0) return true;
    HelperExit exit;
    while (!collect(pidfd(), block, exit)) {
        if (!block) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    retirement_->cancel();
    exit_ = record_exit(exit);
    // A reaped (or no longer waitable) child is no longer ours to signal.
    pid_ = -1;
    return true;
}

// WHY not block: a SIGKILLed GPU helper's main thread is gone at once, but the
// kernel spends ~0.6 s (measured 2026-11-16, 22 GB on an RTX 3090) freeing its
// device memory before the process can be reaped. It can no longer touch the
// channel, so the engine's error path must not wait for that; nor is a SIGTERM
// grace useful for a process whose main thread has already left. stop() reuses this
// after its own SIGKILL for the same reason: no teardown path may wait unboundedly.
void HelperProcess::stop_dying() noexcept {
    // A no-op for a group already exiting; ends threads a lone main-thread exit left.
    const bool signalled = signal(SIGKILL) || errno == ESRCH;
    if (!signalled)
        diag::logf(kError, SEABRAID_PROVIDER, "private Qwen teardown: SIGKILL failed (errno %d)", errno);
    if (readable_within(pidfd(), kDyingSyncMs) && reap(false)) return;
    defer_retirement(!signalled);
}

bool HelperProcess::signal(int number) noexcept {
    if (pid_ < 0) return false;
    // WHY the pidfd and never kill(pid_) (F1, audit 2026-11-19): a pid can be recycled by the time a fallback fires,
    // and a signal to a stranger's process cannot be undone, while a failure here is reported by the caller (with
    // errno) and spawn's gate already refused hosts that block pidfd_send_signal.  ESRCH (already gone) is expected.
    return kernel_send_signal(pidfd(), number) == 0;
}

void HelperProcess::stop() noexcept {
    if (reap(false)) return;
    HelperExit early;
    if (helper_exited(watch(), early)) {
        stop_dying();
        return;
    }
    diag::logf(kInfo, SEABRAID_OK, "private Qwen teardown: sending SIGTERM");
    if (!signal(SIGTERM))
        diag::logf(kWarn, SEABRAID_PROVIDER, "private Qwen teardown: SIGTERM failed (errno %d)", errno);
    readable_within(pidfd(), static_cast<int>(kStopGraceMs));
    if (reap(false)) return;
    diag::logf(kWarn, SEABRAID_TIMEOUT, "private Qwen teardown: helper ignored SIGTERM for %u ms, escalating to SIGKILL",
               kStopGraceMs);
    stop_dying();   // SIGKILL, then the bounded reap
}

}  // namespace seabraid::strata_adapter
