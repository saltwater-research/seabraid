// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The fixture execs itself as a CPU peer using the real helper's CLI and
// channel setup. It tests lifecycle and request isolation without a model.
#include "private_qwen.hpp"
#include "private_qwen_script.hpp"
#include "helper_options.hpp"

#include "diag_capture.hpp"
#include "spawn_policy.hpp"
#include "test_support.hpp"

#include "strata/core/geometry.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using seabraid::strata_adapter::PrivateQwenLaunch;
using seabraid::strata_adapter::PrivateQwenPlan;
using seabraid::strata_adapter::PrivateQwenWorker;
using seabraid::strata_adapter::StaticQwenProvider;
namespace capture = seabraid::strata_adapter::capture;

using seabraid::strata_adapter::testing::check;
using seabraid::strata_adapter::testing::ScratchFile;

std::unique_ptr<PrivateQwenWorker> start(const char* program, const char* mode, uint32_t deadline,
                                         std::string& error) {
    const PrivateQwenLaunch launch{program, mode, "", 0, 2000, 2000};
    return PrivateQwenWorker::start(launch, kPlan, kLayer, deadline, error);
}

bool child_is_stopped() {
    int status = 0;
    const auto stopped = [&] { return waitpid(-1, &status, WNOHANG | WUNTRACED) > 0; };
    return seabraid::Channel::wait(stopped, 1) && WIFSTOPPED(status);
}

void check_reply(StaticQwenProvider& adapter, int32_t status, float expected) {
    const auto config = adapter.config();
    const auto provider = adapter.provider();
    std::vector<float> input(config.hidden, 4.0f), output(config.hidden, -21.0f);
    const std::vector<int32_t> ids(config.top_k, kExpert);
    const std::vector<float> weights(config.top_k, 0.25f);
    const seabraid_batch_v1 batch{SEABRAID_ABI_VERSION, 1, input.data(), input.size(), ids.data(), ids.size(), weights.data(), weights.size()};
    check(provider.execute(provider.context, &config, &batch, output.data(), output.size()) == status,
          "unexpected private provider status");
    check(std::all_of(output.begin(), output.end(), [&](float row) { return row == expected; }),
          "unexpected private provider output");
}

// `logged` is the text the one specific ERROR record must contain.
void check_failure(const char* program, const char* mode, const char* expected, const char* logged) {
    capture::clear();
    std::string error;
    check(!start(program, mode, 1500, error), "expected private startup failure");
    check(error.find(expected) != std::string::npos, error.c_str());
    check(error.find('\x1b') == std::string::npos && error.size() < 600, "helper text must be bounded and escape-free");
    check_reaped();
    capture::require(logged, 1, SEABRAID_PROVIDER);
}

void replaces_failed_worker(const char* program) {
    capture::clear();
    std::string error;
    auto worker = start(program, "stuck-request", 50, error);
    check(bool(worker), error.c_str());
    check_reply(worker->adapter(), SEABRAID_PROVIDER, -21.0f);
    check_reply(worker->adapter(), SEABRAID_PROVIDER, -21.0f);
    // The reason names the cause (not just "retired") and is logged once, not per call.
    capture::require("retired at layer 1: reply timed out", 1, SEABRAID_PROVIDER);
    check(child_is_stopped(), "fixture child must require SIGKILL before close");
    worker.reset();
    check_reaped();
    capture::require("teardown: sending SIGTERM", 1);
    capture::require("escalating to SIGKILL", 1, SEABRAID_TIMEOUT);
    capture::require("ended by signal 9", 1);

    auto fresh = start(program, "serve", 1500, error);
    check(bool(fresh), error.c_str());
    check_reply(fresh->adapter(), SEABRAID_OK, 5.0f);
    fresh.reset();
    check_reaped();
}

// A helper that dies mid-request must retire the lease in milliseconds, not at
// the decode deadline (5 s here): the channel state of a killed helper never changes.
void dead_helper_retires_without_waiting_for_the_deadline(const char* program) {
    capture::clear();
    std::string error;
    auto worker = start(program, "die-on-request", 5000, error);
    check(bool(worker), error.c_str());
    const auto began = std::chrono::steady_clock::now();
    check_reply(worker->adapter(), SEABRAID_PROVIDER, -21.0f);
    const auto took = std::chrono::steady_clock::now() - began;
    check(took < std::chrono::seconds(1), "a killed helper must fail the layer in under a second");
    check(worker->adapter().retired(), "a killed helper retires the provider");
    check_reply(worker->adapter(), SEABRAID_PROVIDER, -21.0f);
    capture::require("retired at layer 1: helper exited (signal 9)", 1, SEABRAID_PROVIDER);
    capture::require("reply timed out", 0);
    worker.reset();
    check_reaped();
    capture::require("ended by signal 9", 1);   // reaping is a separate, informational event
}

// The pidfd fires only when the last thread exits; the lease must retire when the
// main thread does (measured live: ~0.6 s earlier for a 22 GB CUDA helper).
void exited_main_thread_retires_before_the_process_ends(const char* program) {
    capture::clear();
    std::string error;
    auto worker = start(program, "main-exits-on-request", 5000, error);
    check(bool(worker), error.c_str());
    const auto began = std::chrono::steady_clock::now();
    check_reply(worker->adapter(), SEABRAID_PROVIDER, -21.0f);
    const auto took = std::chrono::steady_clock::now() - began;
    check(took < std::chrono::seconds(1), "an exited main thread must fail the layer in under a second");
    capture::require("retired at layer 1: helper exited (status 0), process still exiting", 1, SEABRAID_PROVIDER);
    // Teardown of a helper whose main thread is gone skips the SIGTERM grace.
    const auto stopping = std::chrono::steady_clock::now();
    worker.reset();
    check(std::chrono::steady_clock::now() - stopping < std::chrono::milliseconds(200),
          "teardown of an exiting helper must not wait out the SIGTERM grace");
    check_reaped();
    capture::require("ended by signal 9", 1);
    capture::require("ignored SIGTERM", 0);
}

void helper_exits_with_controller(const char* program) {
    // Adopt the orphan so the fixture can verify and reap it without racing PID reuse.
    check(prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0, "enable fixture subreaper");
    int ready[2];
    check(pipe2(ready, O_CLOEXEC) == 0, "open fixture readiness pipe");
    const pid_t controller = fork();
    check(controller >= 0, "fork a test controller");
    if (controller == 0) {
        close(ready[0]);
        std::string error;
        auto worker = start(program, "parent-loss", 1500, error);
        const char loaded = 'y';
        if (!worker || write(ready[1], &loaded, 1) != 1) _exit(1);
        alarm(6);
        pause();
        _exit(2);
    }
    close(ready[1]);
    char loaded = 0;
    const auto got = read(ready[0], &loaded, 1);
    close(ready[0]);
    check(got == 1 && loaded == 'y', "test controller must load the helper");
    check(kill(controller, SIGKILL) == 0, "kill the controller without destructors");
    int status = 0;
    check(waitpid(controller, &status, 0) == controller && WIFSIGNALED(status),
          "test controller must be killed");
    const auto reaped = [&] { return waitpid(-1, &status, WNOHANG) > 0; };
    check(seabraid::Channel::wait(reaped, 2), "orphaned helper must exit before fixture timeout");
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "pidfd watcher must end the helper");
    check_reaped();
}

void refuses_helper(const char* path, const char* logged) {
    capture::clear();
    std::string error;
    const PrivateQwenLaunch launch{path, "serve", "", 0, 2000, 2000};
    check(!PrivateQwenWorker::start(launch, kPlan, kLayer, 1500, error), "an untrusted helper must not start");
    check(error.find(logged) != std::string::npos, error.c_str());
    capture::require(logged, 1, SEABRAID_INVALID);
    check_reaped();   // refused before any spawn
}

void rejects_untrusted_helper_paths(const char* program) {
    refuses_helper("seabraid-helper", "helper path is not absolute");
    refuses_helper("./private_qwen_test", "helper path is not absolute");
    refuses_helper("/nonexistent-seabraid-dir/helper", "helper path cannot be resolved");
    refuses_helper("/", "helper is not a regular file");
    ScratchFile file;
    file.mode(0775);
    refuses_helper(file.path.c_str(), "helper is group- or world-writable");
    file.mode(0757);
    refuses_helper(file.path.c_str(), "helper is group- or world-writable");
    file.mode(0644);
    refuses_helper(file.path.c_str(), "helper is not executable");
    // The accepted shape resolves to the vetted file itself; the real fixture
    // launches below prove a trusted helper still starts.
    file.mode(0755);
    seabraid::strata_adapter::VettedHelper vetted;
    check(seabraid::strata_adapter::vet_helper(file.path, vetted) == nullptr, "an owner-only executable is accepted");
    check(vetted.fd.valid(), "a vetted helper carries the descriptor that was checked");
    char* real = realpath(program, nullptr);
    check(real != nullptr, "resolve the fixture path");
    std::string link = file.path + ".lnk";
    seabraid::strata_adapter::VettedHelper linked;
    check(symlink(real, link.c_str()) == 0, "create a helper symlink");
    check(seabraid::strata_adapter::vet_helper(link, linked) == nullptr && linked.path == real,
          "a symlink resolves to the file that was vetted");
    unlink(link.c_str());
    std::free(real);
}

void filters_helper_environment() {
    char path[] = "PATH=/bin", secret[] = "AWS_SECRET_ACCESS_KEY=unit-test", locale[] = "LC_ALL=C";
    char cuda[] = "CUDA_VISIBLE_DEVICES=", seabraid[] = "SEABRAID_X=1", gpu[] = "GPU_MAX_HEAP_SIZE=100";
    char lookalike[] = "SEABRAIDX=1", prefix[] = "PATHOLOGICAL=1", bare[] = "HOME", ssh[] = "SSH_AUTH_SOCK=/s";
    char language[] = "LANGUAGE=en", strata_shm[] = "SEABRAID_STRATA_SHM=/x", strata[] = "STRATA_X=1";
    char* env[] = {path, secret, locale, cuda, seabraid, gpu, lookalike, prefix, bare, ssh, language, strata_shm, strata, nullptr};
    const auto kept = seabraid::strata_adapter::allowlisted_environment(env);
    const std::vector<std::string> expected{path, locale, cuda, gpu, strata_shm};
    check(kept == expected, "only allow-listed names survive; look-alikes and bare names do not");
}

void rejects_bad_numeric_options() {
    struct Case { std::vector<const char*> args; const char* error; };
    const Case cases[] = {
        {{"--layers", "abc"}, "--layers requires an integer from 1 to 1024"},
        {{"--layers", "12x"}, "--layers requires"},
        {{"--layers", ""}, "--layers requires"},
        {{"--layers", "0"}, "--layers requires"},
        {{"--layers", "1025"}, "--layers requires"},
        {{"--layers", "99999999999999999999"}, "--layers requires"},
        {{"--experts", "31"}, "--experts requires an integer from 32 to 512"},
        {{"--experts", "513"}, "--experts requires"},
        {{"--experts", "-1"}, "--experts requires"},
        {{"--device", "-1"}, "--device requires"},
        {{"--device", "1000"}, "--device requires"},
        {{"--prefill-tokens", "-1"}, "--prefill-tokens requires"},
        {{"--prefill-tokens", "9999999999"}, "--prefill-tokens requires"},
        {{"--shm-fd", "2"}, "--shm-fd requires"},
        {{"--layers"}, "--layers requires a value"},
    };
    for (const auto& c : cases) {
        std::vector<char*> argv{const_cast<char*>("seabraid-helper"), const_cast<char*>("--pack"), const_cast<char*>("any")};
        for (const char* arg : c.args) argv.push_back(const_cast<char*>(arg));
        seabraid::QwenHelperOptions args;
        std::string error;
        check(!args.parse(static_cast<int>(argv.size()), argv.data(), error), "a bad numeric option must be refused");
        check(error.find(c.error) != std::string::npos, error.c_str());
    }
    char* good[] = {const_cast<char*>("seabraid-helper"), const_cast<char*>("--pack"), const_cast<char*>("any"),
                    const_cast<char*>("--layers"), const_cast<char*>("48"), const_cast<char*>("--experts"),
                    const_cast<char*>("512"), const_cast<char*>("--device"), const_cast<char*>("0"),
                    const_cast<char*>("--prefill-tokens"), const_cast<char*>("0")};
    seabraid::QwenHelperOptions args;
    std::string error;
    check(args.parse(11, good, error) && args.layers == 48 && args.experts == 512, error.c_str());
}

void rejects_ambiguous_transport() {
    char program[] = "seabraid-helper";
    char fd_option[] = "--shm-fd", fd[] = "6";
    char name_option[] = "--shm", name[] = "/seabraid0", pack_option[] = "--pack", pack[] = "any";
    char* argv[] = {program, fd_option, fd, name_option, name, pack_option, pack};
    seabraid::QwenHelperOptions args;
    std::string error;
    check(!args.parse(7, argv, error) && error.find("mutually exclusive") != std::string::npos,
          "helper must refuse two transports");
    char* named_argv[] = {program, name_option, name, pack_option, pack};
    seabraid::QwenHelperOptions named_args;
    error.clear();
    check(named_args.parse(5, named_argv, error) && !named_args.private_channel() && named_args.shm == name,
          "helper must still accept named mode");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        seabraid::QwenHelperOptions args;
        std::string error;
        if (!args.parse(argc, argv, error) || !args.watch_controller(error)) return 1;
        return script(args);
    }
    capture::start();
    // One allow-listed and one non-allow-listed variable; the scripted helper
    // exits early if it sees the secret or misses the pass-through.
    setenv("UNLISTED_TEST_SECRET", "must-not-reach-helper", 1);
    setenv("LC_NUMERIC", "C", 1);
    rejects_ambiguous_transport();
    rejects_bad_numeric_options();
    filters_helper_environment();
    // The launcher requires an absolute, owner-only helper, so every scripted
    // launch below runs a 0755 copy of this fixture rather than the build output.
    ScratchFile trusted;
    trusted.copy_of(argv[0]);
    const std::string program = trusted.path;
    rejects_untrusted_helper_paths(program.c_str());
    check_failure(program.c_str(), "startup-exit", "exited during startup", "helper exited during startup");
    check_failure(program.c_str(), "bad-geometry", "geometry mismatch", "channel attach refused");
    check_failure(program.c_str(), "load-exit", "exited while loading", "helper exited while loading");
    check_failure(program.c_str(), "reject-plan", "fixture refused expert", "helper rejected its plan");
    // The helper's own message may name files, so the log record withholds it.
    capture::require("fixture refused expert", 0);
    argv[0] = const_cast<char*>(program.c_str());
    replaces_failed_worker(argv[0]);
    dead_helper_retires_without_waiting_for_the_deadline(argv[0]);
    exited_main_thread_retires_before_the_process_ends(argv[0]);
    helper_exits_with_controller(argv[0]);
    std::puts("Strata v7 private Qwen helper lifecycle fixture passed");
}
