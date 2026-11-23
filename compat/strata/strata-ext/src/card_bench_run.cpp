// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid D8 GPU half.  DESIGN GATE (the host half's gate is in card_bench.hpp):
//   Owner:    the measurement constants (batch sizes, iterations, time cap) live here, next to the only code that
//             knows what the helper's real path accepts (begin(): k == 10, n_tok <= kWindowMaxT).
//   Contract: card_bench_enabled(), run_card_bench().  Depends on RemoteExperts; nothing depends on this but helper.cpp.
//   Size:     <= 300 lines.
//   Failure:  fail-open-loudly - every error path leaves the regime NaN with a note; nothing here can stop the helper.
//   Checker:  compiled for CUDA sm_86 in D8; the numbers are timed on real cards.
//   Finish:   one SEABRAID-CARD-BENCH line per card on stderr with decode and prefill rates, bounded wall time.
//
// What is timed and why it is honest: a throwaway RemoteExperts is opened on the first 10 experts of one layer (k is
// fixed at 10, and the 10 must be distinct), then begin()+finish() runs on synthetic activations that route EVERY
// token to ALL 10 experts.  So each expert sees exactly n_tok rows per call - the batch size that sets per-row cost -
// and rows/s = (n_tok * 10) / median(begin+finish).  It is the same host grouping, staging, quantize and grouped GEMV
// the serving loop runs, minus only the shared-memory channel hop (which is the same for every card).
//
// Limit: begin() rejects n_tok > kWindowMaxT (8/12/16), so the "prefill" regime is the LARGEST batch the helper can
// serve through this path (kWindowMaxT rows per expert, 10 x kWindowMaxT per call), not thousands of rows.  Bigger
// prompt batches never reach a helper (the main engine reads cached blobs instead), so this is the regime that matters.
#include "card_bench_run.hpp"

#include "strata/seabraid/channel.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <utility>
#include <vector>

namespace seabraid {
namespace {
constexpr int kK = 10;                 // RemoteExperts::begin() demands k == 10
constexpr int kDecodeTokens = 4;       // ~3-4 rows per expert call: a decode step (T~4)
constexpr int kWarmup = 20;
constexpr int kMinIters = 30;          // fewer usable samples than this -> unmeasured
constexpr double kCapSeconds = 1.5;    // per regime; the whole bench is bounded by 2 x this plus the expert load

int env_int(const char* name, int dflt, int lo, int hi) {
    const char* v = std::getenv(name);
    const int x = v ? std::atoi(v) : dflt;
    return std::min(hi, std::max(lo, x));
}
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char* vendor_name() {
#if defined(STRATA_USE_HIP)
    return "hip";
#elif defined(STRATA_GPU_METAL)
    return "metal";
#else
    return "cuda";
#endif
}

// Time `n_tok` tokens x 10 experts.  `fixed` calls after the warm-up, stopping early only at the wall cap.
RegimeStat time_regime(strata::core::RemoteExperts& re, int64_t layer, int n_tok, int fixed, const float* x,
                       const int32_t* ids, std::vector<float>& out, std::string& note, const char* tag) {
    std::string err;
    std::vector<double> secs;
    secs.reserve((size_t) fixed);
    const int64_t want = (int64_t) n_tok * kK;
    const double start = now_s();
    for (int i = 0; i < kWarmup + fixed; ++i) {
        const int64_t before = re.computed();
        const double t0 = now_s();
        const bool ok = re.begin(layer, x, ids, n_tok, kK, nullptr, nullptr, err) && re.finish(out.data(), err);
        const double dt = now_s() - t0;
        if (!ok) { note += std::string(tag) + ": " + err + "; "; return make_stat((int) want, n_tok, {}, kMinIters); }
        // Every row must have been computed on this card; otherwise the rate would describe skipped work.
        if (re.computed() - before != want) {
            note += std::string(tag) + ": expected " + std::to_string(want) + " rows computed, got " +
                    std::to_string(re.computed() - before) + "; ";
            return make_stat((int) want, n_tok, {}, kMinIters);
        }
        if (i >= kWarmup) secs.push_back(dt);
        if (now_s() - start > kCapSeconds && (int) secs.size() >= kMinIters) break;
        if (now_s() - start > 2 * kCapSeconds) break;   // hard stop even if slow: min_iters then decides
    }
    return make_stat((int) want, n_tok, secs, kMinIters);
}

// The layer to time: a middle one (layer 0 is often dense or special, and rows/s differs with the layer's quant type),
// moved forward past layers with no expert blob; may return `layers` when none is left.
int64_t choose_layer(int64_t layers) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    int64_t layer = env_int("SEABRAID_STRATA_CARD_BENCH_LAYER", (int) (layers / 2), 0, (int) std::max<int64_t>(0, layers - 1));
    while (layer < layers && lay.blob_bytes(layer) == 0) ++layer;
    return layer;
}

// Fixed-LCG activations (the same every run, so runs are comparable) and ids routing EVERY token to all of `pool`.
void fill_inputs(std::vector<float>& x, std::vector<int32_t>& ids, const std::vector<int32_t>& pool, int n_tok) {
    uint32_t s = 12345u;
    for (float& f : x) { s = s * 1664525u + 1013904223u; f = ((float) (s >> 8) / 16777216.0f - 0.5f); }
    for (int t = 0; t < n_tok; ++t) for (int j = 0; j < kK; ++j) ids[(size_t) t * kK + j] = pool[(size_t) j];
}
}  // namespace

bool card_bench_enabled() {
    const char* v = std::getenv("SEABRAID_STRATA_CARD_BENCH");
    return v && v[0] == '1' && v[1] == '\0';
}

void run_card_bench(int device, strata::core::ExpertSource& src, int64_t layers, int64_t experts,
                    uint64_t free_vram_bytes, const std::string& device_name, CardBench& out) {
    out = CardBench{};
    out.name = device_name;
    out.vendor = vendor_name();
    out.kind = TierKind::Helper;
    out.free_vram_bytes = free_vram_bytes;
    try {
        const int64_t layer = choose_layer(layers);
        out.layer = (int) layer;
        if (layer >= layers || experts < kK) { out.note = "no layer with " + std::to_string(kK) + " experts to time"; return; }

        std::vector<std::pair<int32_t, int32_t>> ranked;
        std::vector<int32_t> pool;
        for (int e = 0; e < kK; ++e) { ranked.emplace_back((int32_t) layer, e); pool.push_back(e); }
        strata::core::ExpertCache no_primary;
        std::vector<uint8_t> claimed((size_t) layers * (size_t) experts, 0);
        strata::core::RemoteExperts re;
        std::string err;
        if (!re.open(device, kK, layers, experts, ranked, no_primary, src, claimed, err)) { out.note = "open: " + err; return; }

        const int dec_tok = kDecodeTokens, pre_tok = strata::kernels::kWindowMaxT;
        const int64_t H = strata::kernels::cpu::H;
        std::vector<float> x((size_t) pre_tok * (size_t) H), result((size_t) pre_tok * kK * (size_t) H);
        std::vector<int32_t> ids((size_t) pre_tok * kK);
        fill_inputs(x, ids, pool, pre_tok);

        const int dec_iters = env_int("SEABRAID_STRATA_CARD_BENCH_ITERS", 200, kMinIters, 5000);
        out.decode = time_regime(re, layer, dec_tok, dec_iters, x.data(), ids.data(), result, out.note, "decode");
        out.prefill = time_regime(re, layer, pre_tok, std::max(kMinIters, dec_iters / 2), x.data(), ids.data(), result, out.note, "prefill");
        re.close();
    } catch (const std::exception& e) {
        out.note += std::string("exception: ") + e.what();   // regimes measured before the throw are kept
    } catch (...) {
        out.note += "unknown exception";
    }
}

}  // namespace seabraid
