// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// OWNER: adapters/strata/wire is the ONE owner of the helper wire contract (v9): this layout header, channel.hpp (the
// operations on it), expert_spec.hpp (the v6 spec region), channel_names.hpp (the SEABRAID_STRATA_SHM channel list),
// helper_options.hpp (the helper's command line), file_ref.hpp (the FileRef rules), expert_sums.hpp (the expert-sums.v1 manifest)
// and peer_watch.hpp (the helper's identity and the engine's watch of it; proc_stat.hpp and pidfd.hpp are its two parts).
// WHY here and not in the Strata tree: four parties speak this wire (Strata's engine, compat's seabraid-helper, this
// adapter and compat's tests), and the dependency order is runtime <- adapters <- compat <- patched Strata.  The lowest
// of them owns it; compat's patches put this directory on Strata's include path instead of creating a copy, so the
// adapter builds without any compat-patched tree.  This file = the bytes (constants, Header, regions, Layout).
//
// seabraid: the shared-memory channel between a main engine of one GPU vendor (e.g. HIP on an R9700)
// and an expert-helper process of another (e.g. CUDA on an RTX 3060).  Host memory only: no GPU runtime is
// referenced here, so both builds include the same header and agree on the same bytes.
//
// Protocol (one helper, one main):
//   helper: create -> state kHelperUp (free_gib set) -> wait kPlanReady -> load the planned experts
//           -> write held[] + resident/gib -> state kLoaded (or kFailed + msg) -> serve requests until kShutdown
//   main:   connect (wait for kHelperUp) -> write plan[] -> state kPlanReady -> wait kLoaded
//           per layer: write layer/n_tok/k/ids/x, req_seq++ ; later spin until resp_seq == req_seq, read out rows
//   v7 (D4, prompt reading): per prefill layer and token range main writes the PREFILL REGION (fp16 rows, one pick id
//           and router weight per (token, k)), pf_req_seq++ ; the helper computes the picks it holds and writes one fp16
//           weighted sum per token, then pf_resp_seq = pf_req_seq.  See PrefillRegion below.
//   v8 (the expert pyramid): a tier op of kind kTierSwapBlob carries the new expert's bytes in blobs[blob_slot], written by
//           main, so a RAM -> helper move is literally RAM's bytes; the helper verifies its write by read-back before it acks.
//   v9 (file-reference moves): a tier op of kind kTierSwapFile carries a 72-byte FileRef (extents of the model file, a size, an
//           XXH3-64) in blobs[blob_slot]; the helper reads those bytes itself from files it opened at launch, checks the sum,
//           then applies them as a blob swap.  The wire never carries a path.  Header gains file fields (appended).
//   v10 (a lost helper closes its card): the helper publishes WHICH process it is, (helper_pid, helper_start), before kHelperUp,
//           and the engine watches that process (peer_watch.hpp) instead of probing a bare pid that may be recycled.
//   v11 (the tag): the magic text changes to SEABRD1 (the 7 bytes plus NUL still fill one u64) and nothing else; every offset, every
//           size and every Layout total is the v10 one.  kVersion is checked exactly, so a v10 peer is refused by its version and
//           an older helper binary that has no SEABRD1 tag cannot attach: the clean break the name change wants.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "strata/seabraid/expert_spec.hpp"

#if defined(__x86_64__)
// inline asm, not <immintrin.h>: this header reaches .cu files, and nvcc 12.4 + gcc 13 cannot parse the AMX intrinsics
#define SEABRAID_STRATA_PAUSE() __asm__ __volatile__("pause")
#else
#define SEABRAID_STRATA_PAUSE() ((void) 0)
#endif

namespace seabraid {

inline constexpr uint64_t kMagic = 0x31445242414553ull;   // "SEABRD1"
inline constexpr uint32_t kVersion = 11;                   // v11: the magic tag is SEABRD1, layout unchanged; v10: Header::helper_start (the helper's identity, appended; Layout unchanged); v9: kTierSwapFile + FileRef in a blob slot + Header file fields (appended; Layout unchanged); v8: kTierSwapBlob + TierWireOp::blob_slot (layout unchanged); v7: the prefill region (D4); v6: engine spec + per-token weighted output (dsv41); v5 = the neutral STRXHLP tag on the v4 layout
inline constexpr int kBlobBatch = 16;                      // blobs per cache-read request (prompt path)
inline constexpr int kDeviceBase = 1000;                   // remote_dev values >= this mean "the seabraid helper"
inline constexpr int kTierMaxOps = 64;                     // tier ops per round; the policy's throttle must be <= this

enum State : uint32_t { kNone = 0, kHelperUp = 1, kPlanReady = 2, kLoaded = 3, kFailed = 4, kShutdown = 5 };

/// Text the helper process wrote into shared memory, copied out safely.  The helper is another process and this engine
/// must not trust it: it may omit the NUL, so at most cap-1 bytes are read, memchr-bounded, never past the field; and
/// every byte outside printable ASCII (plus \t) is dropped, so helper text cannot inject terminal escape sequences
/// into our logs or error strings.  Helper text is ASCII by contract, so dropping non-ASCII loses nothing real.
inline std::string bounded_text(const char* field, size_t cap) {
    std::string out;
    if (field == nullptr || cap == 0) return out;
    const size_t limit = cap - 1;
    const void* nul = std::memchr(field, 0, limit);
    const size_t len = nul != nullptr ? static_cast<size_t>(static_cast<const char*>(nul) - field) : limit;
    out.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        const unsigned char c = static_cast<unsigned char>(field[i]);
        if (c == 0x09 || (c >= 0x20 && c < 0x7f)) out.push_back(static_cast<char>(c));
    }
    return out;
}

inline constexpr uint8_t kTierLoad = 1, kTierSwap = 2;
/// v9: kTierSwapBlob's cells and rules, but the bytes are the HELPER's read of the model file named by the FileRef in
/// blob(blob_slot).  WHY a new kind and not a flag on v8's: a v8 helper must never be sent kind 4, and kVersion is checked exactly.
inline constexpr uint8_t kTierSwapFile = 4;
/// v8: kTierSwap's fields, but the new expert's bytes are main's, in blob(blob_slot), instead of the helper's own pack read.
/// WHY the blob slots and not a new payload region: the channel's rule is that main-supplied bytes stage through the
/// existing `blobs` slots (they are idle during a fenced round on the serial serve path), so Header and shm size stay as v7.
inline constexpr uint8_t kTierSwapBlob = 3;
/// v4: one re-tiering op.  This is the ONLY struct here whose layout crosses the process line as data, so it is
/// explicitly padded and size-asserted: the helper is built by nvcc + gcc and the main engine by clang, and two
/// toolchains silently disagreeing on padding would shift every field instead of failing.  The POLICY layer's
/// own `TierOp` (tier_plan.hpp) stays in-process and is deliberately NOT this type.
///
/// THERE IS NO EVICT, and that is not an omission.  `ExpertCache` has no eviction and no free list, by design -
/// "it never evicts, because eviction policy is a measured question" (expert_cache.hpp:158) - and both `admit()`
/// branches only ever increment their counters.  An op that merely marked an expert non-resident would strand
/// its slot forever, shrinking the helper's usable cache every round.  So capacity is changed only by:
///   kTierLoad - admit into UNCLAIMED capacity and fill.  Fails when the cache is full.  Uses layer_new/expert_new.
///   kTierSwap - fill expert_new's blob into the slot held by expert_old, then `replace()`.  Uses all four fields.
///   kTierSwapBlob (v8) - kTierSwap with the bytes taken from blob(blob_slot); same same-layer and slot-size checks.
///   kTierSwapFile (v9) - kTierSwapBlob whose blob_slot holds a FileRef instead of the bytes; same cells, same checks.
///
/// CROSS-LAYER SWAP (layer_old != layer_new) IS SIZE-CONSTRAINED, and getting this wrong corrupts GPU memory
/// rather than failing.  The helper's cache is opened with `open_sized` on the native path
/// (remote_experts.cpp:219), so each slot is sized to the blob of the LAYER that first claimed it - and blob
/// sizes differ per layer because the gate/up quant type does (1510400 / 1305600 / 1177600 bytes in this pack).
/// Writing a big layer's expert into a small layer's slot overruns into the next slot's weights: silent
/// corruption, visible much later as nondeterministic quality loss.  THIS IS WHY `replace()` IS SAME-LAYER:
/// blob_bytes is per layer, so same-layer is exactly the condition under which the destination always fits.
/// The helper therefore checks the real per-slot capacity on every swap and fails the op when it does not fit.
struct TierWireOp {
    uint8_t kind;                 // kTierLoad | kTierSwap | kTierSwapBlob | kTierSwapFile
    uint8_t helper;               // always 0: k helpers are k channels (SEABRAID_STRATA_SHM list), one helper per channel
    uint16_t blob_slot;           // v8 (was pad_): kTierSwapBlob's / v9 kTierSwapFile's blob index, < kBlobBatch; 0 on every other kind
    int32_t layer_old;            // kTierSwap: the resident being displaced.  kTierLoad: -1.
    int32_t expert_old;           // kTierSwap: ditto.                        kTierLoad: -1.
    int32_t layer_new;            // the expert being brought in
    int32_t expert_new;
};
// 20 bytes, alignof 4, no tail padding: every member is <= 4-byte aligned, so padding to 24 would assert a
// guarantee the struct does not actually need.
static_assert(sizeof(TierWireOp) == 20, "TierWireOp must be 20 bytes in both builds");
static_assert(alignof(TierWireOp) == 4, "TierWireOp must be 4-aligned in both builds");
static_assert(offsetof(TierWireOp, blob_slot) == 2, "v8 blob_slot reuses v7's pad_ bytes, offset unchanged");

/// v8: the one rule for blob_slot, shared by the main side (before it submits) and the helper (before it applies).
inline constexpr bool blob_slot_valid(const TierWireOp& op) {
    return (op.kind == kTierSwapBlob || op.kind == kTierSwapFile) ? op.blob_slot < kBlobBatch : op.blob_slot == 0;
}

/// Per-op result, echoed by the helper so main can call the policy's note_failed() for exactly the ops that failed.
/// kTierBadSum (v9): refused before any slot write because the bytes read do not match FileRef::sum.  WHY not kTierRejected:
/// main must tell "use the byte path" from "this helper's files disagree with main's", which turns file moves off for it.
enum TierRes : uint8_t { kTierPending = 0, kTierApplied = 1, kTierFailed = 2, kTierRejected = 3, kTierBadSum = 4 };

/// v9: how a helper serves kTierSwapFile (Header::file_io).  kFileIoAuto is a command-line value only, never on the wire.
enum FileIo : int32_t { kFileIoOff = 0, kFileIoBuffered = 1, kFileIoDirect = 2, kFileIoGds = 3, kFileIoAuto = -1 };
inline constexpr int32_t kFileTableMax = 64;   // Header::file_count above this is a bad helper, not a big pack

inline constexpr int kFileRefMaxExtents = 3;   // = kRoles (expert_spec.hpp): a GGUF-in-place expert is three role slices
inline constexpr uint32_t kSumXxh3_64 = 1;     // XXH3_64bits, seed 0, over the extents' bytes concatenated in order

/// v9: where bytes live, as integers only.  WHY extents from main and not (layer, expert) resolved by the helper: main owns
/// where an expert's bytes sit (the v6 spec rule), so a helper stays a byte mover; the sum makes main's addressing checkable.
struct FileExtent { uint32_t file; uint32_t length; uint64_t offset; };   // file = index into the helper's own file table
struct FileRef {
    uint32_t ref_bytes;     // sizeof(FileRef) as main compiled it; the helper refuses any other value
    uint32_t sum_kind;      // kSumXxh3_64; anything else is refused
    uint64_t sum;           // of the extents' bytes concatenated in order (= the blob the slot must hold)
    uint32_t blob_bytes;    // must equal the sum of lengths AND the helper's own blob_bytes(layer_new)
    uint32_t n_extents;     // 1..kFileRefMaxExtents
    FileExtent extent[kFileRefMaxExtents];
};
static_assert(sizeof(FileExtent) == 16 && alignof(FileExtent) == 8, "FileExtent must be 16 bytes in both builds");
static_assert(sizeof(FileRef) == 72 && alignof(FileRef) == 8, "FileRef must be 72 bytes in both builds");
static_assert(kFileRefMaxExtents == kRoles, "an expert has kRoles slices; the extent table must hold all of them");

struct alignas(64) Header {
    // ---- negotiation preamble.  FIXED offsets: a peer that disagrees about everything after this can still
    // parse these five and produce a real error instead of reading shifted fields.  Never reorder or insert.
    uint64_t magic;               // 0
    uint32_t version;             // 8
    uint32_t hdr_bytes;           // 12: sizeof(Header) as the CREATOR compiled it
    uint32_t window_max_t;        // 16: both sides must agree (STRATA_VERIFY_MAX_T)
    uint32_t tier_max_ops;        // 20: v4 tier-op capacity
    // ---- end preamble
    int64_t hidden, cap, n_layers, n_expert, max_pairs;
    std::atomic<uint32_t> state;
    int32_t plan_auto;
    double free_gib;              // helper VRAM free before loading
    int64_t plan_count, plan_slots;
    int64_t resident;             // experts the helper actually holds
    double gib;
    char msg[512];
    char helper_name[128];
    alignas(64) std::atomic<uint64_t> req_seq;
    alignas(64) std::atomic<uint64_t> resp_seq;
    alignas(64) int64_t layer, n_tok, k;
    int32_t ok;
    char err[256];
    // v2: cache reads for the main engine's prompt path (RemoteExperts::copy_cached_blobs across the process line)
    alignas(64) std::atomic<uint64_t> blob_req_seq;
    alignas(64) std::atomic<uint64_t> blob_resp_seq;
    int64_t blob_max;             // bytes per blob slot (the largest blob of any layer)
    int32_t blob_n, blob_ok;
    int32_t blob_layer[kBlobBatch], blob_expert[kBlobBatch];
    uint8_t blob_hit[kBlobBatch];
    // v3: the helper's own service time per layer request (seen -> answered), and the part of it spent in begin()
    // (staging + launches) - so the main engine can split its wait into helper work and everything else
    alignas(64) std::atomic<uint64_t> svc_n;
    std::atomic<uint64_t> svc_ns, svc_begin_ns;
    // ---- v4 (D7, live re-tiering).
    // These five counters are RAW uint64 touched only through __atomic_* builtins, never std::atomic: the two
    // sides are built by different toolchains and std::atomic<uint64_t>'s REPRESENTATION is not guaranteed to
    // match uint64_t's.  Raw fields drop that library-layout assumption and keep only the platform contract.
    // (The v3 fields above stay as they are - churning a working protocol is a worse risk than the assumption.)
    alignas(64) uint64_t tier_seq;    // main: release-store after writing the op block
    alignas(64) uint64_t tier_resp;   // helper: release-store ONLY after every op is applied and its work is done
    alignas(64) uint64_t epoch;       // main: bumped on each committed tier change
    alignas(64) uint64_t req_epoch;   // the epoch a layer request was published under
    int32_t tier_n;                   // ops in this round (<= tier_max_ops)
    int32_t tier_ok;                  // 1 = all applied; 0 = at least one did not (see tier_res)
    uint8_t tier_res[kTierMaxOps];    // per-op TierRes, so main can note_failed() exactly the ops that failed
    char tier_err[256];
    // Per-request echo, checked by main BEFORE it reads `out`: a residual disagreement becomes a loud error
    // rather than a silently missing expert contribution.
    int32_t out_rows;                 // rows the helper actually wrote to `out`
    int32_t out_mode;                 // OutMode (expert_spec.hpp): kOutPerPick (Qwen) | kOutPerToken (dsv41)
    // ---- v6 (engine-neutral helpers).  Appended, never inserted: the fields above keep their offsets.
    int32_t engine;                   // Engine: the executor this helper serves.  Written by the helper; dsv41 main
                                      // refuses a helper of another engine, Qwen main does not read it
    int32_t spec_layers;              // LayerSpec entries main wrote into the spec region (0 = no spec: Qwen)
    int32_t spec_files;               // paths main wrote into the spec region's path table
    int32_t helper_pid;               // the helper's process id: main fails fast if it dies (a crashed helper never sets kFailed)
    // ---- v7 (D4: the expert process computes its own experts while a prompt is read).  Appended after v6; the region they
    // describe is appended after Layout::of's total (PrefillRegion), so every v6 offset stays where it was.
    int64_t pf_tokens;                // creator: the region's capacity in TOKENS per request (0 = no prefill region)
    int32_t pf_k;                     // creator: pick slots per token the region was sized for (the routing width)
    int32_t pf_format;                // creator: PrefillFormat of the rows and sums (kPfFp16)
    int32_t pf_layer, pf_ntok;        // main: this request's layer and token count (1 .. pf_tokens)
    int32_t pf_ok;                    // helper: 1 = every asked pick computed and every sum finite
    int32_t pf_picks;                 // helper echo: the picks it computed (main compares it with the picks it asked for)
    int32_t pf_row_bytes;             // main: this request's row element size, kPfRowsF16 or kPfRowsF32 (rows are declared per request)
    char pf_err[256];
    alignas(64) uint64_t pf_req_seq;  // main: release-store after writing the region (raw + __atomic, like the v4 counters)
    alignas(64) uint64_t pf_resp_seq; // helper: release-store after out16 and the echo are written
    uint64_t pf_svc_ns, pf_svc_n;     // helper: cumulative service time and requests (the cost model's measured rate)
    // ---- v9 (file-reference moves).  Appended after v7; every earlier offset is unchanged and the header stays under a page.
    int32_t file_io;                  // helper: FileIo it serves kTierSwapFile with (kFileIoOff = main must send bytes)
    int32_t file_count;               // helper: entries in its file table (<= kFileTableMax)
    uint64_t file_table_id;           // helper: XXH3-64 of the table's file sizes (LE u64, table order); main compares it with its own
    uint64_t file_direct_bytes, file_buffered_bytes, file_reads;   // helper: raw __atomic counters, like the v4 ones
    // ---- v10 (a lost helper closes its card).  Appended after v9; every earlier offset is unchanged.
    uint64_t helper_start;            // helper: field 22 of /proc/self/stat (start time in clock ticks), written with helper_pid BEFORE kHelperUp.
                                      // (helper_pid, helper_start) names one process for the life of the boot; peer_watch.hpp checks it
};

static_assert(std::is_standard_layout<Header>::value, "Header must be standard-layout for offsetof to be valid");
static_assert(offsetof(Header, magic) == 0, "preamble: magic must stay at 0");
static_assert(offsetof(Header, version) == 8, "preamble: version must stay at 8");
static_assert(offsetof(Header, hdr_bytes) == 12, "preamble: hdr_bytes must stay at 12");
static_assert(offsetof(Header, window_max_t) == 16, "preamble: window_max_t must stay at 16");
static_assert(offsetof(Header, tier_max_ops) == 20, "preamble: tier_max_ops must stay at 20");
static_assert(sizeof(uint64_t) == 8, "the v4 counters assume 8-byte uint64_t");
static_assert(__atomic_always_lock_free(8, nullptr), "the v4 counters must be lock-free across processes");
static_assert(offsetof(Header, pf_tokens) > offsetof(Header, helper_pid), "v7 fields are appended after v6's");
static_assert(offsetof(Header, file_io) > offsetof(Header, pf_svc_n), "v9 fields are appended after v7's");
static_assert(offsetof(Header, helper_start) > offsetof(Header, file_reads), "v10 fields are appended after v9's");
static_assert(sizeof(Header) <= 4096, "the header must stay under one page: Layout::of starts the first region at 4 KiB");

/// v7 (D4) wire format.  The SUMS are FP16 (a rounding of 2^-12 of a weighted sum: ~2e-4 nrmse per layer, measured by
/// STRATA_D4_CHECK; the same two bytes as BF16 with 8x its mantissa, RMS-normed activations are well inside its range, and
/// Strata's own second-GPU prompt path ships its sums as FP16).  The ROWS are declared PER REQUEST (Header::pf_row_bytes):
/// 2 = fp16, 4 = f32.  WHY BOTH: the expert card must compute from the activations card 1 would have used for the same experts.
/// An engine path that quantizes the FP16 image of its rows (MMQ: gather_rows16(mixed_h)) is served exactly by fp16 rows - f32
/// would hand the expert card a DIFFERENT activation than the lend path computes from.  An engine path that quantizes the
/// f32 rows (the fused int8 kernels: quantize_act_native(mixed)) is served exactly only by f32 rows: fp16 rows re-round every
/// activation and flip ~2% of the int8 codes, ~0.5% nrmse on the handed-over sums in every layer (measured, STRATA_D4_CHECK E3;
/// order-A KL against the lend path 0.057 with fp16 rows, 0.028 with f32).  The region has room for f32 rows (the larger).
enum PrefillFormat : int32_t { kPfNone = 0, kPfFp16 = 1 };
inline constexpr int32_t kPfRowsF16 = 2, kPfRowsF32 = 4;   ///< Header::pf_row_bytes: bytes per row element of a request

/// v7 (D4): the prefill region, appended at Layout::total.  SIZED BY TOKENS, NEVER BY PAIRS (review 2026-11-06): every byte
/// here is host RAM, and host-RAM pressure is what evicts the 32 GB AMD GPU's GPU queues (KFD userptr).  Per token: its fp16 row
/// (H x 2 B fp16 or H x 4 B f32, per request), one slot per routed pick (K x (int32 id + f32 weight)) and its fp16 weighted
/// sum (H x 2 B) - a fixed 4H + 8K bytes on the wire with fp16 rows (10,320 B at H 2560 / K 10), 6H + 8K with f32 rows (15,440 B).  A region holding one H-float row per PICK would be 2K = 20x larger (3.4 GB for a 16K
/// chunk).  A chunk longer than `tokens` is sent as several requests (RemoteExperts::prefill_*).
// Keep offsets representable for pointer arithmetic and ftruncate on either peer.
inline constexpr size_t kMappingLimit = std::numeric_limits<off_t>::max() < std::numeric_limits<std::ptrdiff_t>::max()
                                           ? std::numeric_limits<off_t>::max()
                                           : std::numeric_limits<std::ptrdiff_t>::max();

class PageLayout {
public:
    constexpr bool add(size_t& offset, uint64_t count, uint64_t width, size_t element = 1) {
        constexpr size_t page_mask = 4095;
        offset = end_;
        const size_t room = (kMappingLimit & ~page_mask) - end_;
        if (width > room / element || (width && count > room / (width * element))) return false;
        end_ = (end_ + count * width * element + page_mask) & ~page_mask;
        return true;
    }
    constexpr size_t bytes() const { return end_; }
private:
    size_t end_ = 0;
};

struct PrefillRegion {
    size_t x16, pick, wt, out16, bytes;   // offsets from the region's start; `bytes` = the whole region (page multiple)
    static size_t token_bytes(int64_t hidden, int64_t k, int row_bytes = kPfRowsF16) { return (size_t) ((row_bytes + 2) * hidden + 8 * k); }
    static constexpr PrefillRegion of(int64_t hidden, int64_t k, int64_t tokens) {
        if (tokens <= 0 || k <= 0 || hidden <= 0) return {};
        PrefillRegion r{};
        PageLayout pages;
        // Allocate the larger f32 input shape so both supported row formats fit.
        if (!pages.add(r.x16, tokens, hidden, kPfRowsF32) ||
            !pages.add(r.pick, tokens, k, sizeof(int32_t)) ||
            !pages.add(r.wt, tokens, k, sizeof(float)) ||
            !pages.add(r.out16, tokens, hidden, sizeof(uint16_t))) return {};
        r.bytes = pages.bytes();
        return r;
    }
};

struct Layout {
    size_t plan, held, ids, x, out, blobs, tier_ops, weights, spec, total;
    static constexpr Layout of(int64_t hidden, int64_t cap, int64_t n_layers, int64_t n_expert, int64_t blob_max) {
        if (hidden <= 0 || cap <= 0 || n_layers <= 0 || n_expert <= 0 || blob_max < 0) return {};
        // The spec has an unpadded layer table followed by its fixed path table.
        if ((uint64_t) n_layers > (kMappingLimit - spec_region_bytes(0)) / sizeof(LayerSpec)) return {};
        Layout l{};
        PageLayout pages;
        size_t header_offset = 0;
        if (!pages.add(header_offset, 1, sizeof(Header)) ||
            !pages.add(l.plan, n_layers, n_expert, 2 * sizeof(int32_t)) ||
            !pages.add(l.held, n_layers, n_expert) ||
            !pages.add(l.ids, cap, 1, sizeof(int32_t)) ||
            !pages.add(l.x, cap, hidden, sizeof(float)) ||
            !pages.add(l.out, cap, hidden, sizeof(float)) ||
            !pages.add(l.blobs, kBlobBatch, blob_max) ||
            !pages.add(l.tier_ops, kTierMaxOps, sizeof(TierWireOp)) ||
            !pages.add(l.weights, cap, 1, sizeof(float)) ||
            !pages.add(l.spec, 1, spec_region_bytes(n_layers))) return {};
        l.total = pages.bytes();
        return l;
    }
};

struct ChannelLimits {
    int64_t hidden, cap, layers, experts, blob_max, pf_tokens;
    int32_t pf_k;

    constexpr bool layout(Layout& layout, PrefillRegion& prefill) const {
        layout = Layout::of(hidden, cap, layers, experts, blob_max);
        // The response echoes the total pick count in one int32_t.
        if (pf_tokens == 0 && pf_k != 0) return false;
        if (pf_tokens < 0 || (pf_tokens > 0 && (pf_k <= 0 || pf_tokens > INT32_MAX / pf_k))) return false;
        prefill = PrefillRegion::of(hidden, pf_k, pf_tokens);
        return layout.total && (pf_tokens == 0 || prefill.bytes) && prefill.bytes <= kMappingLimit - layout.total;
    }

    // layout() already proved the checked plan array fits.
    constexpr int64_t pairs() const { return layers * experts; }

    constexpr bool valid_decode(int64_t layer, int64_t tokens, int64_t picks, int64_t window) const {
        // Bound the product before any consumer computes it or echoes it as int32_t.
        return layer >= 0 && layer < layers && tokens > 0 && tokens <= window &&
               picks > 0 && tokens <= cap / picks && tokens <= INT32_MAX / picks;
    }
};

} // namespace seabraid
