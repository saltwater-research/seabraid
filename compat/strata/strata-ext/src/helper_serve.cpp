// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid-helper, serving: load each plan main sends, then answer blob reads, tier rounds, prefill chunks and layer
// requests until main shuts down, restarts with a new plan, or the process is stopped.  FAILURE: every request is
// answered; a failure is written into the channel (and logged once) through report(), never left unanswered.
#include "helper_session.hpp"
#include "log_text.hpp"
#include "strata/seabraid/pf_service.hpp"
#include "strata/seabraid/tier_round.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/window_limits.hpp"

#include <chrono>
#include <csignal>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace seabraid::helper {
namespace {
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

// WHY one function: every error written into the channel is also logged, so the log and the channel never disagree
// and no shm-only error path can exist.  The text must already be free of paths and identifiers (callers pass fixed
// strings or library messages), because stderr ends up in journals.
void report(char* dst, size_t n, const std::string& text) {
    std::snprintf(dst, n, "%s", text.c_str());
    std::fprintf(stderr, "seabraid-helper: %s\n", seabraid::logtext::scrub_paths(text).c_str());
}

// ---- plan loading -------------------------------------------------------------------------------------------------

void fail_plan(Session& s, const std::string& why) {
    report(s.h->msg, sizeof(s.h->msg), why);
    s.h->state.store(kFailed, std::memory_order_release);
}

bool plan_counts_fit(const Session& s, int64_t count, int64_t slots) {
    const int64_t pairs = s.ch.limits().pairs();
    return !(count <= 0 || count > pairs || slots <= 0 || slots > pairs || slots > INT32_MAX);
}

// The plan's (layer, expert) pairs, in main's rank order; false at the first pair outside the channel.
bool read_plan(const Session& s, int64_t count, std::vector<std::pair<int32_t, int32_t>>& ranked) {
    ranked.reserve((size_t) count);
    const int32_t* plan = s.ch.plan();
    for (int64_t i = 0; i < count; ++i) {
        const int32_t l = plan[2 * i], e = plan[2 * i + 1];
        if (l < 0 || l >= s.layers || e < 0 || e >= s.experts) break;
        ranked.emplace_back(l, e);
    }
    return ranked.size() == (size_t) count;
}

// Fill the VRAM tier from the pack and mark what it really holds in held[].
bool open_plan(Session& s, int64_t slots, const std::vector<std::pair<int32_t, int32_t>>& ranked) {
    strata::core::ExpertCache no_primary;   // unopened: holds nothing, so the plan is taken as sent
    std::vector<uint8_t> claimed((size_t) s.layers * (size_t) s.experts, 0);
    uint8_t* held = s.ch.held();
    std::memset(held, 0, (size_t) s.layers * (size_t) s.experts);
    if (!s.re.open(s.args.device, (int) slots, s.layers, s.experts, ranked, no_primary, s.src, claimed, s.err, s.h->plan_auto != 0) ||
        !s.opt->init(s.err))
        return false;
    for (const auto& pr : ranked)
        if (s.re.holds(pr.first, pr.second)) held[(size_t) pr.first * (size_t) s.experts + (size_t) pr.second] = 1;
    return true;
}

// One plan: validate it, (re)build the tier, publish residency and kLoaded.  false = kFailed was published.
bool load_plan(Session& s) {
    const int64_t count = s.h->plan_count, slots = s.h->plan_slots;
    if (!plan_counts_fit(s, count, slots)) { fail_plan(s, "plan counts exceed the channel or the slot allocator"); return false; }
    s.opt.reset();   // (detaches itself from `re` and waits for its stream)
    s.re.close();
    s.opt = std::make_unique<strata::core::RemoteExpertOpt>();
    s.opt->attach(s.re);
    std::vector<std::pair<int32_t, int32_t>> ranked;
    if (!read_plan(s, count, ranked)) { fail_plan(s, "plan names a layer or expert outside the channel"); return false; }
    const auto t0 = Clock::now();
    if (!open_plan(s, slots, ranked)) { fail_plan(s, s.err); return false; }   // wait for another plan
    s.h->resident = s.re.resident();
    s.h->gib = s.re.gib();
    s.seen = s.h->req_seq.load(std::memory_order_acquire);
    s.h->resp_seq.store(s.seen, std::memory_order_release);
    s.blob_seen = s.h->blob_req_seq.load(std::memory_order_acquire);
    s.h->blob_resp_seq.store(s.blob_seen, std::memory_order_release);
    s.h->state.store(kLoaded, std::memory_order_release);
    std::fprintf(stderr, "seabraid-helper: holding %lld experts, %.2f GiB (filled in %.1f s); serving\n",
                 (long long) s.re.resident(), s.re.gib(), std::chrono::duration<double>(Clock::now() - t0).count());
    return true;
}

// ---- requests -----------------------------------------------------------------------------------------------------

struct Pending { uint64_t seq = 0, bseq = 0, tseq = 0, pseq = 0; uint32_t st = 0; };

Pending wait_for_work(const Session& s) {
    Pending p;
    Channel::wait([&] {
        p.seq = s.h->req_seq.load(std::memory_order_acquire);
        p.bseq = s.h->blob_req_seq.load(std::memory_order_acquire);
        p.st = s.h->state.load(std::memory_order_relaxed);
        p.tseq = s.ch.tier_pending();
        p.pseq = s.ch.pf_pending();   // D4: 0 unless main published a prefill request
        return p.seq != s.seen || p.bseq != s.blob_seen || p.tseq != 0 || p.pseq != 0 || g_stop || p.st != kLoaded;
    }, 0);
    return p;
}

// The prompt path asks for cached blobs: copy them from VRAM into the channel.
void serve_blobs(Session& s, uint64_t bseq) {
    s.blob_seen = bseq;
    const int nb = s.h->blob_n;
    s.h->blob_ok = 0;
    if (nb > 0 && nb <= kBlobBatch && s.ch.limits().blob_max > 0) {
        uint8_t* dst[kBlobBatch];
        for (int i = 0; i < nb; ++i) dst[i] = s.ch.blob(i);
        s.h->blob_ok = s.re.copy_cached_blobs(s.h->blob_layer, s.h->blob_expert, dst, (size_t) nb, s.h->blob_hit, s.err);
    } else {
        std::memset(s.h->blob_hit, 0, sizeof(s.h->blob_hit));
    }
    s.h->blob_resp_seq.store(bseq, std::memory_order_release);
}

// A kTierSwapBlob's bytes: the channel's blob slot, offered only for the cell the op brings in, so tier_apply's own size and
// same-layer checks run unchanged on them (DESIGN 6.3).
class ChannelBlobSource final : public strata::core::ExpertSource {
public:
    ChannelBlobSource(int64_t layer, int64_t expert, const uint8_t* bytes) : layer_(layer), expert_(expert), bytes_(bytes) {}
    const uint8_t* blob(int64_t layer, int64_t expert) override { return layer == layer_ && expert == expert_ ? bytes_ : nullptr; }
private:
    int64_t layer_, expert_;
    const uint8_t* bytes_;
};

// The RoundCache tier_round.hpp runs a round against: this helper's RemoteExperts and its pack.
struct HelperRoundCache {
    Session& s;
    bool apply(const TierWireOp& op, const uint8_t* blob, std::string& err) {
        if (blob == nullptr) return s.re.tier_apply(op, s.src, err);
        ChannelBlobSource from_channel(op.layer_new, op.expert_new, blob);
        return s.re.tier_apply(op, from_channel, err);
    }
    bool sync(std::string& err) { return s.re.tier_sync(err); }
    bool suspect(int32_t l, int32_t e) const { return s.re.tier_suspect(l, e); }
    bool verify(int32_t l, int32_t e, const uint8_t* blob, std::string& err) { return s.re.tier_verify(l, e, blob, err); }
    bool holds(int32_t l, int32_t e) const { return s.re.holds(l, e); }
    int64_t blob_bytes(int32_t l) const { return (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(l); }
    void totals(Header& h) { h.resident = s.re.resident(); h.gib = s.re.gib(); }
};

// v4 live re-tiering.  The main engine fences every layer request around this round, so a request cannot overlap it;
// servicing it first anyway keeps that assumption from being load-bearing in two places.  The sequence (apply, ONE sync
// before the ack, read-back of blob-written slots, publish held[], ack) is tier_round.hpp's; a kTierSwapFile's bytes are first
// fetched by s.files (file_fetch.hpp) into its staging blob, and from there it is a blob swap.
void serve_tier_round(Session& s, uint64_t tseq) {
    HelperRoundCache cache{s};
    seabraid::serve_tier_channel(s.ch, tseq, cache, [&](int op, bool first, const std::string& why) {
        if (first) report(s.h->tier_err, sizeof(s.h->tier_err), why);
        else std::fprintf(stderr, "seabraid-helper: tier op %d failed after the first: %s\n", op, seabraid::logtext::scrub_paths(why).c_str());
    }, s.files);
}

bool layer_request_valid(const Session& s, int32_t mode, int64_t layer, int64_t nt, int64_t k) {
    return (mode == kOutPerToken || mode == kOutPerPick) && k == kRoutedPicks &&
           s.ch.limits().valid_decode(layer, nt, k, strata::kernels::kWindowMaxT);
}

// finish() left the weighted per-token sums in the opt's host buffer; only n_tok x H floats cross to the channel.
void finish_reduce(Session& s, bool ok, int64_t nt) {
    if (ok) std::memcpy(s.ch.out(), s.opt->sum_rows(), (size_t) nt * (size_t) strata::kernels::cpu::H * sizeof(float));
    s.opt->end();
}

// The echo is the helper's statement of what it did; main treats a disagreement with its ask as an error.
void publish_layer_answer(Session& s, bool ok, bool per_token, int64_t nt, int64_t k) {
    s.h->out_mode = per_token ? kOutPerToken : kOutPerPick;
    s.h->out_rows = ok ? (int32_t) (per_token ? nt : nt * k) : 0;
    s.h->ok = ok ? 1 : 0;
    if (!ok) report(s.h->err, sizeof(s.h->err), s.err);
}

void serve_layer(Session& s, uint64_t seq) {
    const auto s0 = Clock::now();
    // The request's out_mode is main's ask; the echo below is the helper's statement of what it did.  Main treats a
    // disagreement as an error, so an answer is never silently the wrong shape.
    const int32_t mode = s.h->out_mode;
    const bool per_token = mode == kOutPerToken;
    const int64_t layer = s.h->layer, nt = s.h->n_tok, k = s.h->k;
    bool ok = layer_request_valid(s, mode, layer, nt, k);
    if (!ok) s.err = "seabraid-helper: invalid layer, routing width, window or output mode";
    const bool reduce = ok && per_token;
    if (reduce) s.opt->begin(s.ch.weights(), 0, (int) nt);   // main wrote one router weight per pick
    ok = ok && s.re.begin(layer, s.ch.x(), s.ch.ids(), nt, k, nullptr, nullptr, s.err);
    const auto s1 = Clock::now();
    ok = ok && s.re.finish(s.ch.out(), s.err);
    if (reduce) finish_reduce(s, ok, nt);
    publish_layer_answer(s, ok, per_token, nt, k);
    const auto s2 = Clock::now();
    s.h->svc_begin_ns.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count(), std::memory_order_relaxed);
    s.h->svc_ns.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(s2 - s0).count(), std::memory_order_relaxed);
    s.h->svc_n.fetch_add(1, std::memory_order_relaxed);
    s.h->resp_seq.store(seq, std::memory_order_release);
    ++s.served;
}

// true = this plan is over: stopped, shut down by main (back to kHelperUp for the next engine), or a restarted engine
// published a new plan (reload).
bool plan_over(Session& s, uint32_t st) {
    if (g_stop || st == kShutdown) {
        if (st == kShutdown) s.h->state.store(kHelperUp);
        return true;
    }
    return st == kPlanReady;
}

void serve_requests(Session& s) {
    while (!g_stop) {
        const Pending p = wait_for_work(s);
        if (p.bseq != s.blob_seen) serve_blobs(s, p.bseq);
        if (p.tseq != 0) { serve_tier_round(s, p.tseq); continue; }   // a tier round is not a layer request
        if (p.pseq != 0) {   // D4: compute the experts this card holds for a prompt chunk's layer; ALWAYS answered
            service_prefill(s.ch, p.pseq, [&](std::string& e) { return s.re.prefill_serve(s.ch, e); });
            continue;
        }
        if (plan_over(s, p.st)) break;
        if (p.seq == s.seen) continue;
        s.seen = p.seq;
        serve_layer(s, p.seq);
    }
}
}  // namespace

void install_stop_handlers() {
    if (std::signal(SIGINT, on_signal) == SIG_ERR || std::signal(SIGTERM, on_signal) == SIG_ERR)
        std::fprintf(stderr, "seabraid-helper: cannot install signal handlers (errno %d); stop it with SIGKILL and remove its shm\n", errno);
}

void serve_plans(Session& s) {
    while (!g_stop) {
        if (!Channel::wait([&] { return g_stop || s.h->state.load(std::memory_order_acquire) == kPlanReady; }, 0) || g_stop)
            break;
        if (load_plan(s)) serve_requests(s);
    }
}

}  // namespace seabraid::helper
