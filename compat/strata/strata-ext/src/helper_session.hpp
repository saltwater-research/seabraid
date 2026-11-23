// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid-helper's serving half: what startup (helper.cpp: admit, load, open the channel) hands to the serve loop
// (helper_serve.cpp: plans, blob reads, tier rounds, prefill, layer requests).
// WHY two files: startup runs once and fails with an exit code; serving runs until stopped and fails into the channel
// (report()).  Keeping them apart keeps each function at CCN <= 10 and each file well under 500 lines.
#pragma once

#include "file_fetch.hpp"
#include "helper_options.hpp"
#include "strata/seabraid/channel.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/cpu/expert.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace seabraid::helper {

/// Routing width of the compiled model: the private channel's picks per token and every decode request's `k`.
inline constexpr int32_t kRoutedPicks = strata::core::kCompiledGeometry.n_expert_used;

/// Everything the serve loop reads or writes, borrowed from startup (args, source, channel) or owned here.
struct Session {
    Session(const QwenHelperOptions& a, strata::core::FileExpertSource& s, Channel& c, FileTable& f, std::string e)
        : args(a), src(s), ch(c), files(f), h(c.header()), layers(a.layers), experts(a.experts), err(std::move(e)) {}
    const QwenHelperOptions& args;
    strata::core::FileExpertSource& src;
    Channel& ch;
    FileTable& files;   ///< v9: the files a kTierSwapFile reads from, opened at launch (kFileIoOff when this helper serves none)
    Header* h;
    const int64_t layers, experts;
    std::string err;   ///< the last failure text (startup's, then each request's), as main() always kept it
    strata::core::RemoteExperts re;
    // The reducer for kOutPerToken requests (seabraid phase 1): the SAME RemoteExpertOpt the main engine uses for an
    // in-process helper GPU, attached to this process's RemoteExperts - one definition of "sum of w * expert(x) per
    // token", not a copy.  Attached before open() and rebuilt per plan because open() sizes its metadata from it.
    std::unique_ptr<strata::core::RemoteExpertOpt> opt;
    uint64_t served = 0;      ///< layer requests answered (the exit log line)
    uint64_t seen = 0, blob_seen = 0;   ///< the last request / blob-read sequence this plan answered
};

/// SIGINT/SIGTERM set the stop flag the serve loop polls.  Loud on failure: a helper that cannot be stopped by
/// SIGTERM would be killed uncleanly and leave its shm behind.
void install_stop_handlers();

/// Serve plans until stopped.  A new plan may arrive at ANY time - an engine that was killed (no kShutdown) and
/// restarted sends a fresh one - so the per-plan loop also watches for kPlanReady and reloads.
void serve_plans(Session& s);

}  // namespace seabraid::helper
