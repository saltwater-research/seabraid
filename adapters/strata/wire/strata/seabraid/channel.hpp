// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid channel, the operations: create/connect/map a channel whose bytes channel_layout.hpp defines, the doorbells,
// and the tier-residency publish/check pair.  Owned by adapters/strata/wire (see channel_layout.hpp for why); every
// party includes it as "strata/seabraid/channel.hpp" from that one directory.  WHY two files: the layout alone is the
// contract's frozen half (tests/channel_v6_golden.txt pins its sizes); keeping it apart keeps each under the 500-line cap.  The residency pair is in channel_residency.hpp (included at the end).
#include "strata/seabraid/channel_layout.hpp"

namespace seabraid {

/// What an attach to a channel the other side is still building came to.  A STRUCTURED result, so a caller never has to
/// match the English text of an error: kNotReady = the creator has not finished yet (poll again; err is set but is not a fault);
/// kRefused = a final no, err says why (log it, give up).
enum class Attach { kAttached, kNotReady, kRefused };

class Channel {
public:
    Channel() = default;
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;
    ~Channel() { unmap(); }
    /// Helper side: creates (or recreates) the region.
    /// `pf_tokens` (v7, D4): the prefill region's capacity in tokens per request, `pf_k` its pick slots per token; 0 = none.
    /// The region is only ftruncate'd: /dev/shm commits a page when it is first written, so a helper that never serves a
    /// prompt costs no RAM for it.
    bool create(const std::string& name, int64_t hidden, int64_t cap, int64_t n_layers, int64_t n_expert,
                uint32_t window_max_t, int64_t blob_max, std::string& err, int64_t pf_tokens = 0, int32_t pf_k = 0) {
        if (base_ != nullptr) { err = "seabraid channel already mapped"; return false; }
        const ChannelLimits limits{hidden, cap, n_layers, n_expert, blob_max, pf_tokens, pf_tokens ? pf_k : 0};
        Layout l{};
        PrefillRegion pr{};
        if (!limits.layout(l, pr)) { err = "seabraid channel: invalid geometry"; return false; }
        shm_unlink(name.c_str());
        const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd < 0) { err = "shm_open(" + name + ") failed: " + std::strerror(errno); return false; }
        if (ftruncate(fd, (off_t) (l.total + pr.bytes)) != 0) {
            err = std::string("ftruncate: ") + std::strerror(errno);
            ::close(fd); shm_unlink(name.c_str()); return false;
        }
        if (!map(fd, l.total + pr.bytes, err)) { shm_unlink(name.c_str()); return false; }
        name_ = name; owner_ = true;
        initialize(limits, l, pr, window_max_t);
        return true;
    }
    /// Main side: opens an existing region (the helper made it) and checks its geometry.
    bool connect(const std::string& name, int64_t hidden, int64_t cap, int64_t n_layers, int64_t n_expert,
                 uint32_t window_max_t, std::string& err) {
        if (base_ != nullptr) { err = "seabraid channel already mapped"; return false; }
        const int fd = shm_open(name.c_str(), O_RDWR, 0600);
        if (fd < 0) { err = "seabraid helper channel " + name + " not found (start seabraid-helper first)"; return false; }
        return attach(fd, hidden, cap, n_layers, n_expert, window_max_t, err) == Attach::kAttached;
    }

#if defined(__linux__)
    // The launcher owns the original descriptor and explicitly passes only this capability to each peer.
    static int private_fd(std::string& err) {
        const int fd = memfd_create("seabraid", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        if (fd < 0) { err = std::string("memfd_create: ") + std::strerror(errno); return -1; }
        if (fchmod(fd, S_IRUSR | S_IWUSR) == 0) return fd;
        err = std::string("fchmod: ") + std::strerror(errno);
        ::close(fd);
        return -1;
    }

    // A private generation starts from a new, empty object. Seal the extent before publishing its header.
    bool create_private(int inherited_fd, const ChannelLimits& limits, uint32_t window, std::string& err) {
        if (base_ != nullptr) { err = "seabraid channel already mapped"; return false; }
        Layout l{};
        PrefillRegion pf{};
        if (!limits.layout(l, pf)) { err = "seabraid channel: invalid geometry"; return false; }
        if (!fresh_sealable(inherited_fd)) { err = "seabraid channel requires a fresh sealable descriptor"; return false; }
        const auto bytes = l.total + pf.bytes;
        if (ftruncate(inherited_fd, (off_t) bytes) != 0 || fcntl(inherited_fd, F_ADD_SEALS, kSizeSeals) != 0) {
            err = std::string("seabraid channel could not size and seal the descriptor: ") + std::strerror(errno);
            return false;
        }
        const int fd = duplicate(inherited_fd, err);
        if (fd < 0 || !map(fd, bytes, err)) return false;
        initialize(limits, l, pf, window);
        return true;
    }

    /// Private-descriptor attach.  kNotReady = the creator has not sealed or published yet (retry); kRefused = final.
    Attach connect_private(int inherited_fd, int64_t hidden, int64_t cap, int64_t layers, int64_t experts,
                           uint32_t window, std::string& err) {
        if (base_ != nullptr) { err = "seabraid channel already mapped"; return Attach::kRefused; }
        const int seals = fcntl(inherited_fd, F_GET_SEALS);
        if (seals < 0 || (seals & kSizeSeals) != kSizeSeals) {
            err = "seabraid private channel has no sealed extent"; return Attach::kNotReady;
        }
        const int fd = duplicate(inherited_fd, err);
        return fd >= 0 ? attach(fd, hidden, cap, layers, experts, window, err) : Attach::kRefused;
    }
#endif

private:
    // The object behind `fd` has a plausible mapping size and is private to this user.  false = refused, fd closed.
    static bool admissible_object(int fd, struct stat& st, std::string& err) {
        if (fstat(fd, &st) != 0 || st.st_size < (off_t) sizeof(Header) || (uint64_t) st.st_size > kMappingLimit) {
            err = "seabraid channel has an invalid mapping size"; ::close(fd); return false;
        }
        // Refuse a broader mapping scope before attaching.
        if (!private_owner(st)) {
            err = "seabraid channel must be a regular object owned by this user with no group or other access";
            ::close(fd); return false;
        }
        return true;
    }

    // The negotiation preamble: kNotReady while the creator has not published, kRefused on a foreign or mismatched build.
    static Attach preamble_status(const Header* h, std::string& err) {
        const uint64_t magic = __atomic_load_n(&h->magic, __ATOMIC_ACQUIRE);
        if (magic == 0) { err = "seabraid helper channel is still starting"; return Attach::kNotReady; }
        if (magic != kMagic || h->version != kVersion) {
            err = "seabraid channel: wrong magic/version (helper v" + std::to_string(h->version) + ", main v" +
                  std::to_string(kVersion) + ")";
            return Attach::kRefused;
        }
        // The preamble is at fixed offsets, so these two reads are valid even when the rest of the struct
        // disagrees.  A size or op-capacity mismatch means two builds compiled a different Header - most
        // likely nvcc+gcc versus clang padding `alignas(64)` differently - and the failure mode without this
        // check is reading shifted fields and computing wrong answers, not crashing.
        if (h->hdr_bytes != (uint32_t) sizeof(Header) || h->tier_max_ops != (uint32_t) kTierMaxOps) {
            err = "seabraid channel ABI mismatch: helper sizeof(Header)=" + std::to_string(h->hdr_bytes) +
                  " tier_max_ops=" + std::to_string(h->tier_max_ops) + "; main sizeof(Header)=" +
                  std::to_string(sizeof(Header)) + " tier_max_ops=" + std::to_string(kTierMaxOps) +
                  " (the two processes were built with incompatible toolchains or flags)";
            return Attach::kRefused;
        }
        return Attach::kAttached;
    }

    // The creator's geometry equals what this side expects (0 layers/experts = accept the creator's).
    static bool geometry_matches(const Header* h, const ChannelLimits& limits, int64_t hidden, int64_t cap,
                                 int64_t n_layers, int64_t n_expert, uint32_t window_max_t, std::string& err) {
        if (limits.hidden != hidden || limits.cap != cap || (n_layers > 0 && limits.layers != n_layers) ||
            (n_expert > 0 && limits.experts != n_expert) || h->window_max_t != window_max_t) {
            err = "seabraid channel geometry mismatch: helper H=" + std::to_string(h->hidden) + " cap=" +
                  std::to_string(h->cap) + " layers=" + std::to_string(h->n_layers) + " experts=" +
                  std::to_string(h->n_expert) + " window=" + std::to_string(h->window_max_t) + "; main H=" +
                  std::to_string(hidden) + " cap=" + std::to_string(cap) + " window=" + std::to_string(window_max_t);
            return false;
        }
        return true;
    }

    Attach attach(int fd, int64_t hidden, int64_t cap, int64_t n_layers, int64_t n_expert,
                  uint32_t window_max_t, std::string& err) {
        struct stat st{};
        if (!admissible_object(fd, st, err)) return Attach::kRefused;
        if (!map(fd, (size_t) st.st_size, err)) return Attach::kRefused;
        Header* h = header();
        const Attach preamble = preamble_status(h, err);
        if (preamble != Attach::kAttached) { unmap(); return preamble; }
        const ChannelLimits limits{h->hidden, h->cap, h->n_layers, h->n_expert, h->blob_max, h->pf_tokens, h->pf_k};
        if (!geometry_matches(h, limits, hidden, cap, n_layers, n_expert, window_max_t, err)) {
            unmap(); return Attach::kRefused;
        }
        if (!limits.layout(layout_, pf_) || h->max_pairs != limits.pairs()) {
            err = "seabraid channel: invalid geometry"; unmap(); return Attach::kRefused;
        }
        if (layout_.total + pf_.bytes > size_) { err = "seabraid channel truncated"; unmap(); return Attach::kRefused; }
        // Once attached, offsets and bounds come from this snapshot throughout the mapping's lifetime.
        limits_ = limits;
        return Attach::kAttached;
    }

public:
    Header* header() const { return (Header*) base_; }
    int32_t* plan() const { return (int32_t*) (base_ + layout_.plan); }
    uint8_t* held() const { return base_ + layout_.held; }
    int32_t* ids() const { return (int32_t*) (base_ + layout_.ids); }
    float* x() const { return (float*) (base_ + layout_.x); }
    float* out() const { return (float*) (base_ + layout_.out); }
    uint8_t* blob(int i) const { return base_ + layout_.blobs + (size_t) i * (size_t) limits_.blob_max; }
    TierWireOp* tier_ops() const { return (TierWireOp*) (base_ + layout_.tier_ops); }
    float* weights() const { return (float*) (base_ + layout_.weights); }
    LayerSpec* layer_spec() const { return (LayerSpec*) (base_ + layout_.spec); }
    // v7 (D4) prefill region: token-major [token][H] rows and sums, [token][k] picks (-1 = not the helper's) and weights
    void* pf_rows() const { return base_ + layout_.total + pf_.x16; }   // tokens x hidden, in header()->pf_row_bytes' row type
    int32_t* pf_pick() const { return (int32_t*) (base_ + layout_.total + pf_.pick); }
    float* pf_wt() const { return (float*) (base_ + layout_.total + pf_.wt); }
    uint16_t* pf_out16() const { return (uint16_t*) (base_ + layout_.total + pf_.out16); }
    const PrefillRegion& prefill_region() const { return pf_; }
    const ChannelLimits& limits() const { return limits_; }
    // v7 request pair (raw uint64 + __atomic builtins, as tier_seq/tier_resp, for the same cross-toolchain reason)
    uint64_t pf_publish() const {   // main, after writing the region and pf_layer/pf_ntok
        Header* h = header();
        const uint64_t seq = __atomic_load_n(&h->pf_req_seq, __ATOMIC_RELAXED) + 1;
        __atomic_store_n(&h->pf_req_seq, seq, __ATOMIC_RELEASE);
        return seq;
    }
    bool pf_answered(uint64_t seq) const { return __atomic_load_n(&header()->pf_resp_seq, __ATOMIC_ACQUIRE) >= seq; }
    uint64_t pf_pending() const {   // helper: the request to serve, or 0
        Header* h = header();
        const uint64_t seq = __atomic_load_n(&h->pf_req_seq, __ATOMIC_ACQUIRE);
        return seq > __atomic_load_n(&h->pf_resp_seq, __ATOMIC_RELAXED) ? seq : 0;
    }
    void pf_answer(uint64_t seq) const { __atomic_store_n(&header()->pf_resp_seq, seq, __ATOMIC_RELEASE); }
    char* spec_path(int i) const {
        return (char*) (base_ + layout_.spec + (size_t) limits_.layers * sizeof(LayerSpec) + (size_t) i * kSpecPathMax);
    }
    bool valid() const { return base_ != nullptr; }
    void* region() const { return base_; }
    size_t bytes() const { return size_; }

    // ---- v4 tier channel.  Only these four functions touch tier_seq / tier_resp, and only through the
    // __atomic builtins, so the publication contract lives in one place.
    //
    // ORDERING CONTRACT.  Main writes the op block, then release-stores tier_seq.  The helper acquire-loads
    // tier_seq, which makes those writes visible, applies the ops, writes tier_res / tier_ok / held[], then
    // release-stores tier_resp.  Main acquire-loads tier_resp and only then reads tier_res and held[].
    //
    // The helper must not ack until its GPU work for the ops is COMPLETE, not merely enqueued: a CPU
    // release-store says nothing about an outstanding device copy.  `held[]` must stay unchanged from the ack
    // until main has finished copying it.
    //
    // Main must also guarantee no other producer publishes a layer request between its last drained response
    // and `tier_submit` - the lock-step req_seq/resp_seq pair bounds PUBLISHED requests, not future ones, so
    // admission control is the caller's job, and today it is the ONLY guard: `epoch` is bumped, but nothing
    // stamps or compares `req_epoch`, so a request that slips through is not rejected by the helper.

    /// Main: publish a round of ops.  `n` must be <= kTierMaxOps.  Returns the sequence to wait for.
    uint64_t tier_submit(const TierWireOp* ops, int n) const {
        Header* h = header();
        if (n > kTierMaxOps) n = kTierMaxOps;
        for (int i = 0; i < n; ++i) { tier_ops()[i] = ops[i]; h->tier_res[i] = kTierPending; }
        h->tier_n = n;
        h->tier_ok = 0;
        h->tier_err[0] = '\0';
        const uint64_t seq = __atomic_load_n(&h->tier_seq, __ATOMIC_RELAXED) + 1;
        __atomic_store_n(&h->tier_seq, seq, __ATOMIC_RELEASE);
        return seq;
    }
    /// Main: true once the helper has applied the round named by `seq`.  Read tier_res / held[] only after this.
    bool tier_done(uint64_t seq) const {
        return __atomic_load_n(&header()->tier_resp, __ATOMIC_ACQUIRE) >= seq;
    }
    /// Helper: the sequence of a round that has not been acked yet, or 0 when there is nothing to do.
    uint64_t tier_pending() const {
        Header* h = header();
        const uint64_t seq = __atomic_load_n(&h->tier_seq, __ATOMIC_ACQUIRE);
        return seq > __atomic_load_n(&h->tier_resp, __ATOMIC_RELAXED) ? seq : 0;
    }
    /// Helper: ack a round.  Call ONLY after every op is applied, held[] is updated, and the GPU work is done.
    void tier_ack(uint64_t seq, bool ok) const {
        Header* h = header();
        h->tier_ok = ok ? 1 : 0;
        __atomic_store_n(&h->tier_resp, seq, __ATOMIC_RELEASE);
    }
    /// Main: the current epoch.  Meant to be stamped into `req_epoch` so the helper could reject a request from an
    /// older epoch; neither side does that yet (see the admission note above tier_submit).
    uint64_t epoch() const { return __atomic_load_n(&header()->epoch, __ATOMIC_ACQUIRE); }
    void bump_epoch() const {
        Header* h = header();
        __atomic_store_n(&h->epoch, __atomic_load_n(&h->epoch, __ATOMIC_RELAXED) + 1, __ATOMIC_RELEASE);
    }

    /// Wait until `pred()` or `timeout_s` passes (0 = forever): spin hot for the first 2 ms (a decode layer's request
    /// arrives within ~0.2 ms), then sleep 20 us between checks so an idle engine does not burn a core.
    template <class P> static bool wait(P pred, double timeout_s) {
        uint64_t spins = 0;
        struct timespec t0{}; clock_gettime(CLOCK_MONOTONIC, &t0);
        bool hot = true;
        while (!pred()) {
            ++spins;
            if (hot) {
                SEABRAID_STRATA_PAUSE();
                if ((spins & 1023) != 0) continue;
            } else {
                struct timespec nap{0, 20000}; nanosleep(&nap, nullptr);
            }
            struct timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t);
            const double dt = (double) (t.tv_sec - t0.tv_sec) + 1e-9 * (double) (t.tv_nsec - t0.tv_nsec);
            if (dt > 0.002) hot = false;
            if (timeout_s > 0 && dt > timeout_s) return false;
        }
        return true;
    }

    void unmap() {
        if (base_) munmap(base_, size_);
        base_ = nullptr; size_ = 0;
        if (owner_ && !name_.empty()) shm_unlink(name_.c_str());
        owner_ = false;
    }

private:
    static bool private_owner(const struct stat& st) {
        return S_ISREG(st.st_mode) && st.st_uid == geteuid() && (st.st_mode & 077) == 0;
    }
#if defined(__linux__)
    static constexpr int kSizeSeals = F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;

    static int duplicate(int inherited_fd, std::string& err) {
        const int fd = fcntl(inherited_fd, F_DUPFD_CLOEXEC, 0);
        if (fd < 0) err = std::string("seabraid private descriptor: ") + std::strerror(errno);
        return fd;
    }

    // A new generation starts from an empty, unsealed, private object: never one a previous peer already sized.
    static bool fresh_sealable(int fd) {
        const int seals = fcntl(fd, F_GET_SEALS);
        struct stat st{};
        return !(seals < 0 || (seals & kSizeSeals) != 0 || fstat(fd, &st) != 0 || !private_owner(st) || st.st_size != 0);
    }
#endif

    void initialize(const ChannelLimits& limits, const Layout& l, const PrefillRegion& pr, uint32_t window_max_t) {
        std::memset(base_, 0, sizeof(Header));
        Header* h = header();
        h->version = kVersion; h->window_max_t = window_max_t;
        h->hidden = limits.hidden; h->cap = limits.cap; h->n_layers = limits.layers; h->n_expert = limits.experts;
        h->max_pairs = limits.pairs();
        h->blob_max = limits.blob_max;
        h->state.store(kNone); h->req_seq.store(0); h->resp_seq.store(0);
        h->blob_req_seq.store(0); h->blob_resp_seq.store(0);
        h->svc_n.store(0); h->svc_ns.store(0); h->svc_begin_ns.store(0);
        // v4: the creator initialises these once, before any peer can attach (magic is published last).
        h->hdr_bytes = (uint32_t) sizeof(Header);
        h->tier_max_ops = (uint32_t) kTierMaxOps;
        h->tier_seq = 0; h->tier_resp = 0; h->epoch = 0; h->req_epoch = 0;
        h->tier_n = 0; h->tier_ok = 1; h->out_rows = 0; h->out_mode = kOutPerPick;
        h->engine = kEngineQwen; h->spec_layers = 0; h->spec_files = 0; h->helper_pid = 0; h->helper_start = 0;
        h->pf_tokens = limits.pf_tokens; h->pf_k = limits.pf_k; h->pf_format = pr.bytes ? kPfFp16 : kPfNone;
        h->pf_layer = 0; h->pf_ntok = 0; h->pf_ok = 1; h->pf_picks = 0; h->pf_row_bytes = kPfRowsF16; h->pf_err[0] = '\0';
        h->pf_req_seq = 0; h->pf_resp_seq = 0; h->pf_svc_ns = 0; h->pf_svc_n = 0;
        __atomic_store_n(&h->magic, kMagic, __ATOMIC_RELEASE);
        layout_ = l; pf_ = pr; limits_ = limits;
    }

    bool map(int fd, size_t bytes, std::string& err) {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) { err = std::string("mmap: ") + std::strerror(errno); return false; }
        base_ = (uint8_t*) p; size_ = bytes;
        return true;
    }
    uint8_t* base_ = nullptr;
    size_t size_ = 0;
    Layout layout_{};
    PrefillRegion pf_{};
    ChannelLimits limits_{};
    std::string name_;
    bool owner_ = false;
};

} // namespace seabraid

#include "strata/seabraid/channel_residency.hpp"
