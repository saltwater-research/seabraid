// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid expert helper: serves Strata's RemoteExperts tier from a GPU of the OTHER vendor, in its own
// process, over the shared-memory channel (adapters/strata/wire/strata/seabraid/channel.hpp).  Built with whichever
// backend this GPU needs (CUDA for NVIDIA, HIP for AMD); the main engine is built with the other.
//
//   seabraid-helper --shm /seabraid0 --pack DIR --native SHARD1.gguf --layers 48 --experts 512 [--device 0]
// A dedicated Linux controller passes --shm-fd FD and --controller-pid PID.
//
// The helper loads nothing until the main engine sends its plan (the ranked experts its own caches do not hold),
// then fills its VRAM tier from the pack, reports what it holds, and computes those experts' rows per layer.
// This file is startup (admit, load, open the channel, announce); FAILURE: each step that fails logs once and
// exits (2 = bad command line, 1 = cannot start).  Serving is helper_serve.cpp.
#include "file_staging.hpp"
#include "helper_session.hpp"
#include "log_text.hpp"
#include "strata/seabraid/peer_watch.hpp"
#include "pack_extents.hpp"
#include "card_bench_run.hpp"
#include "strata/core/device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
using seabraid::helper::kRoutedPicks;
constexpr int kBadUsage = 2, kCannotStart = 1;

// One log line for a startup failure, then the exit code.  `what` is a fixed prefix; `err` may hold a path.
int fail(int code, const char* what, const std::string& err) {
    std::fprintf(stderr, "seabraid-helper: %s%s\n", what, seabraid::logtext::scrub_paths(err).c_str());
    return code;
}

// WHY fail-open-loud: a kernel that refuses no_new_privs must not stop the helper from serving, but running without that floor is
// logged once so an operator can see it.
void forbid_new_privileges_loudly() {
    std::string error;
    if (!seabraid::forbid_new_privileges(error))
        std::fprintf(stderr, "seabraid-helper: %s; continuing without no_new_privs\n", error.c_str());
}

// WHY not fatal: the default only chooses a faster staging mode, and an explicit STRATA_REMOTE_ZEROCOPY already wins.
void default_zerocopy_off() {
    if (setenv("STRATA_REMOTE_ZEROCOPY", "0", 0) != 0)
        std::fprintf(stderr, "seabraid-helper: setenv STRATA_REMOTE_ZEROCOPY failed (errno %d); staging stays at the engine default\n", errno);
}

// WHY not fatal: the summary is a label for logs and the channel; the card already passed preflight.
void device_label(int device, std::string& name, std::string& detail) {
    if (!strata::core::device_summary(device, name, detail))
        std::fprintf(stderr, "seabraid-helper: device %d has no summary; reporting it unnamed\n", device);
}

// Admit the command line and size the request region.  0 = admitted.
int admit(int argc, char** argv, seabraid::QwenHelperOptions& args, int64_t& cap, std::string& err) {
    if (!args.parse(argc, argv, err) || !args.keep_descriptor_private(err) || !args.watch_controller(err))
        return fail(kBadUsage, "", err);
    // The named engine and the helper's RemoteExperts buffers use a top-10
    // extent. A dedicated channel can use the compiled model's smaller width.
    constexpr int64_t legacy_picks = 10;
    if (args.private_channel() && kRoutedPicks > legacy_picks) {
        std::fprintf(stderr, "seabraid-helper: private decode exceeds RemoteExperts pick capacity\n"); return kBadUsage;
    }
    cap = strata::kernels::kWindowMaxT * (args.private_channel() ? kRoutedPicks : legacy_picks);
    return 0;
}

// Open the pack and the card.  0 = ready; `free_gib` and `name` describe the card.
int load_engine(const seabraid::QwenHelperOptions& args, strata::core::FileExpertSource& src, double& free_gib,
                std::string& name, std::string& err) {
    // RemoteExperts refuses the primary device; this process has no primary, so name an ordinal it never uses.
    strata::core::set_primary_device(args.device == 0 ? 99 : 0);   // returns void: nothing to check
    if (!strata::kernels::cpu::expert_layout_load(args.pack, args.layers, args.experts, err))
        return fail(kCannotStart, "expert layout: ", err);
    if (!args.native.empty()) src.set_gguf(args.native);
    if (!src.open(args.pack, args.layers, args.experts, err)) return fail(kCannotStart, "expert source: ", err);
    if (!strata::core::RemoteExperts::preflight(args.device, free_gib, err)) return fail(kCannotStart, "", err);
    std::string detail;
    device_label(args.device, name, detail);
    return 0;
}

// Create (named) or initialize (private) the channel.  0 = created.
int open_channel(const seabraid::QwenHelperOptions& args, int64_t cap, seabraid::Channel& ch, std::string& err) {
    int64_t blob_max = 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (int64_t l = 0; l < args.layers; ++l) blob_max = std::max<int64_t>(blob_max, (int64_t) lay.blob_bytes(l));
    // D4 (opt-in): a prefill region sized by TOKENS (never by pairs) lets the engine hand this card the experts it holds while a
    // prompt is read (engine side: STRATA_PREFILL_REMOTE_COMPUTE=1).  --prefill-tokens 0 (default) = no region, no host RAM.
    if (args.prefill_tokens < 0) { std::fprintf(stderr, "seabraid-helper: --prefill-tokens must be >= 0\n"); return kBadUsage; }
    const seabraid::ChannelLimits limits{strata::kernels::cpu::H, cap, args.layers, args.experts, blob_max,
                                        args.prefill_tokens, args.prefill_tokens ? kRoutedPicks : 0};
    if (!args.create_channel(ch, limits, strata::kernels::kWindowMaxT, err)) return fail(kCannotStart, "", err);
    return 0;
}

// v9: every cell of the file table is bound to the helper's OWN FileRef for it (its pack layout + its own expert-sums.v1, fresh for the
// descriptors it opened), so a file op is checked against the op's cell and never against a sum the wire supplied.  0 = bound or file
// moves off (auto: one warning); an explicit --file-io without a usable manifest is a launch failure, like a table that cannot open.
int bind_cells(const seabraid::QwenHelperOptions& args, const strata::core::FileExpertSource& src, seabraid::FileTable& files) {
    if (files.io() == seabraid::kFileIoOff) return 0;
    std::vector<uint32_t> blob_bytes;
    for (int64_t l = 0; l < args.layers; ++l) blob_bytes.push_back((uint32_t) strata::kernels::cpu::expert_layout().blob_bytes(l));
    std::string why;
    const std::string path = args.pack + "/" + seabraid::sums::kFileName;
    if (files.bind_manifest(path, args.layers, args.experts, blob_bytes, seabraid::PackExtents{src}, why)) return 0;
    if (args.file_io != seabraid::kFileIoAuto) return fail(kCannotStart, "file moves: ", why);
    std::fprintf(stderr, "seabraid-helper: file moves are off: %s\n", seabraid::logtext::scrub_paths(why).c_str());
    files.close_all();   // announces file_io 0: main never sends a file op to this helper
    return 0;
}

// v9: the file table a kTierSwapFile reads from: the files the pack loader above just opened, opened again here read-only and
// never again (file_fetch.hpp).  0 = ready; with --file-io auto a table that cannot be opened is a warning and file_io 0, with an
// explicit direct or buffered it is a launch failure.
int open_file_table(const seabraid::QwenHelperOptions& args, const strata::core::FileExpertSource& src, const seabraid::Channel& ch,
                    seabraid::helper::FileStaging& staging, seabraid::FileTable& files, std::string& err) {
    if (args.file_io == seabraid::kFileIoOff) return 0;
    const int64_t blob_max = ch.limits().blob_max;
    if (!staging.alloc(args.device, blob_max)) {
        if (args.file_io != seabraid::kFileIoAuto) return fail(kCannotStart, "file staging: ", "cannot allocate " + std::to_string(blob_max) + " bytes");
        std::fprintf(stderr, "seabraid-helper: file moves are off: cannot allocate the staging blob\n");
        return 0;
    }
    std::string warn;
    if (!files.open_all(src.file_paths(), args.file_io, staging.data(), blob_max, err, warn)) return fail(kCannotStart, "file table: ", err);
    if (!warn.empty()) std::fprintf(stderr, "seabraid-helper: %s\n", seabraid::logtext::scrub_paths(warn).c_str());
    if (const int code = bind_cells(args, src, files)) return code;
    if (files.io() != seabraid::kFileIoOff)
        std::fprintf(stderr, "seabraid-helper: file moves: %s, %d files, %s staging\n", files.io() == seabraid::kFileIoDirect ? "direct" : "buffered",
                     files.count(), staging.pinned() ? "pinned" : "pageable");
    return 0;
}

// Publish the card to main (kHelperUp), after the optional startup benchmark.
void announce(const seabraid::QwenHelperOptions& args, strata::core::FileExpertSource& src, seabraid::Channel& ch,
              seabraid::FileTable& files, double free_gib, const std::string& name) {
    seabraid::Header* h = ch.header();
    h->free_gib = free_gib;
    files.announce(*h);   // file_io / file_count / file_table_id, before kHelperUp: main reads them once it sees the card
    // D8 startup benchmark: opt-in, BEFORE kHelperUp so the engine's plan can use the result; it runs here, in the
    // helper, because only this process has the card's runtime and the real RemoteExperts path.  Failure is
    // logged and reported as "unmeasured" (nan) - it never stops the helper from serving.
    if (seabraid::card_bench_enabled()) {
        seabraid::CardBench cb;
        seabraid::run_card_bench(args.device, src, args.layers, args.experts, (uint64_t) (free_gib * 1073741824.0), name, cb);
        std::fprintf(stderr, "%s\n", seabraid::format_line(cb).c_str());
    }
    std::snprintf(h->helper_name, sizeof(h->helper_name), "%s", name.c_str());
    seabraid::announce_identity(*h);   // (pid, start time) before the release below: the engine watches this exact process (peer_watch.hpp)
    h->state.store(seabraid::kHelperUp, std::memory_order_release);
    std::fprintf(stderr, "seabraid-helper: %s (device %d), %.2f GiB free; channel %s ready, waiting for the plan\n",
                 name.c_str(), args.device, free_gib, args.private_channel() ? "(private)" : args.shm.c_str());
}
}  // namespace

int main(int argc, char** argv) {
    forbid_new_privileges_loudly();   // first: before argv parsing or the GPU runtime, so nothing this process execs can gain privilege
    seabraid::QwenHelperOptions args;
    std::string err;   // one error text for the whole run, as before the split (Session keeps it)
    int64_t cap = 0;
    if (const int code = admit(argc, argv, args, cap, err)) return code;
    seabraid::helper::install_stop_handlers();
    // The helper's input/output already sit in host memory (the channel), so zero-copy kernels read and write
    // across PCIe; staging through VRAM measured 77.7 -> 65.3 us per layer request (RTX 3060, 2026-11-06,
    // nsys: the down/quantize kernels were PCIe-bound).  An explicit STRATA_REMOTE_ZEROCOPY wins.
    default_zerocopy_off();
    strata::core::FileExpertSource src;
    double free_gib = 0;
    std::string name;
    if (const int code = load_engine(args, src, free_gib, name, err)) return code;
    seabraid::Channel ch;
    if (const int code = open_channel(args, cap, ch, err)) return code;
    seabraid::helper::FileStaging staging;   // declared first: destroyed after the table that reads into it
    seabraid::FileTable files;
    if (const int code = open_file_table(args, src, ch, staging, files, err)) return code;
    announce(args, src, ch, files, free_gib, name);
    seabraid::helper::Session session(args, src, ch, files, std::move(err));
    seabraid::helper::serve_plans(session);
    std::fprintf(stderr, "seabraid-helper: stopping after %llu layer requests (%lld expert rows)\n",
                 (unsigned long long) session.served, (long long) session.re.computed());
    return 0;
}
