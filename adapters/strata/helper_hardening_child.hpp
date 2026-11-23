// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Child-side behavior for helper_hardening_test. Parent and child share the probe's
// environment cases and ready descriptor so both ends exercise the same protocol.
#pragma once

#include "helper_options.hpp"

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <string>

#include <sys/prctl.h>
#include <unistd.h>

namespace seabraid::strata_adapter::testing::hardening_child {

inline constexpr int kReadyTarget = 20;   // the descriptor on which a child reports "ready"

enum class Inherit { never, always, debug };
struct EnvCase { const char* name; const char* value; Inherit policy; };
inline constexpr EnvCase kEnvCases[] = {
    {"STRATA_API_KEY", "synthetic-test-value", Inherit::never},
    {"UNLISTED_TEST_SECRET", "synthetic-test-value", Inherit::never},
    {"SEABRAID_STRATA_CARD_BENCH_TOKEN", "synthetic-test-value", Inherit::never},
    {"CUDA_ACCESS_TOKEN", "synthetic-test-value", Inherit::never},
    {"SEABRAID_STRATA_PTRACER", "1", Inherit::never},
    {"LD_PRELOAD", "/nonexistent/preload.so", Inherit::never},
    {"STRATA_FILL_AHEAD", "1", Inherit::always},
    {"CUDA_VISIBLE_DEVICES", "0", Inherit::always},
    {"CUDA_DEVICE_ORDER", "PCI_BUS_ID", Inherit::always},
    {"HIP_VISIBLE_DEVICES", "2", Inherit::always},
    {"ROCR_VISIBLE_DEVICES", "1", Inherit::always},
    {"HSA_SVM_FOR_PAGED_MEM", "0", Inherit::always},
    {"GPU_PINNED_MIN_XFER_SIZE", "1048576", Inherit::always},
    {"SEABRAID_STRATA_SHM", "x", Inherit::always},
    {"STRATA_DUMP_MOE_INPUT", "/tmp/never", Inherit::debug},
    {"SEABRAID_STRATA_CARD_BENCH", "1", Inherit::debug},
};

inline int env_probe(bool debug) {
    int code = 20;
    for (const auto& entry : kEnvCases) {
        ++code;
        const bool expected = entry.policy == Inherit::always || (debug && entry.policy == Inherit::debug);
        const char* value = std::getenv(entry.name);
        if ((value != nullptr) != expected) return code;
        if (value && std::strcmp(value, entry.value) != 0) return code;
    }
    return 0;
}

// 0 only if the flag was clear on entry, the helper's own first call set it, and it reads back set.
inline int no_new_privs_probe() {
    if (prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 0) return 41;
    std::string error;
    if (!seabraid::forbid_new_privileges(error)) return 42;
    return prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) == 1 ? 0 : 43;
}

inline int child(const std::string& mode, const char* argument) {
    alarm(10);   // a broken parent must not leave this child behind
    if (mode == "nnp") return no_new_privs_probe();
    if (mode == "env") return env_probe(std::atoi(argument) != 0);
    if (mode == "exit") return 0;
    if (mode == "ignore-term") signal(SIGTERM, SIG_IGN);
    if (mode == "ignore-term" && write(kReadyTarget, "r", 1) != 1) return 30;
    for (;;) pause();
}

}  // namespace seabraid::strata_adapter::testing::hardening_child
