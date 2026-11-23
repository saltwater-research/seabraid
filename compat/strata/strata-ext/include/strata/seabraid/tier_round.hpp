// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid helper side of ONE tier round: how a round of wire ops becomes per-op results, held[] and an ack, including the
// blob swap (wire v8 kTierSwapBlob: the new expert's bytes ride a channel blob slot, the helper writes them into the slot
// of the expert they displace and READS THEM BACK before it acks) and the file swap (wire v9 kTierSwapFile: the blob slot holds
// a FileRef, the helper FETCHES the bytes from the model file into its own staging blob, checks their sum, and from there it is
// a blob swap: same write, same sync, same read-back).
//
// DESIGN GATE
//   Owner    : this header owns "a round on the helper": op validation, the kTierSwapBlob -> kTierSwap view, the order
//              apply -> sync -> read-back -> publish -> ack, and the per-op result mapping.  The wire (adapters/strata/wire)
//              owns the bytes of the ops; the helper's RemoteExperts owns the slots; this file owns only the sequence.
//   Contract : op_refusal, blob_size_refusal, as_swap, run_tier_round, serve_tier_channel.  The cache and the file fetcher are
//              template parameters (the RoundCache and FileFetch members below), so the sequence runs in host tests with no Strata
//              and no GPU; helper_serve.cpp passes adapters over RemoteExperts and the FileTable.  Imports the wire only.
//   Size     : <= 290 lines.
//   Failure  : an op refused before any slot is touched is kTierRejected (main fails open); a file whose bytes are not the
//              manifest's is kTierBadSum, also before any slot is touched; an op whose copy started and did not complete or did
//              not read back is kTierFailed and its slot is quarantined (main closes the card).  A round is ALWAYS acked, and
//              never before the read-back.  The first failure's text goes to `report` as `first`.
//   Checker  : tests/blob_swap_test.cpp (the sequence, every refusal, a read-back mismatch), tests/channel_blob_swap_test.cpp (the
//              same on a real shared-memory channel, plus a peer rewriting an op or the Header geometry mid-round) and
//              tests/channel_file_swap_test.cpp (the file swap, bad sum, bounds, round shape, another cell's ref).
//
// RoundCache (what the helper's cache must offer; a host test fakes it):
//   bool apply(const TierWireOp& op, const uint8_t* blob, std::string& err)   op is kTierLoad or kTierSwap; `blob` non-null
//        = the bytes come from the channel (not from the pack), and they stay untouched until the round is acked
//   bool sync(std::string& err)                          drain the copies, publish the deferred swaps
//   bool suspect(int32_t layer, int32_t expert) const    the cell is quarantined (a copy into it did not complete)
//   bool verify(int32_t layer, int32_t expert, const uint8_t* blob, std::string& err)   read the slot back and compare;
//        a mismatch quarantines the cell
//   bool holds(int32_t layer, int32_t expert) const      the cell is resident and not quarantined
//   int64_t blob_bytes(int32_t layer) const              the byte size of one expert of this layer
//   void totals(Header& h)                               refresh the header's resident / gib after the round
//
// FileFetch (what a kTierSwapFile needs; tests fake it or pass the real FileTable):
//   TierRes fetch(int32_t layer, int32_t expert, const FileRef& wire, int64_t want_blob_bytes, std::string& why)   the bytes of
//        the op's OWN destination cell (layer_new, expert_new): kTierApplied = staging() holds that expert's blob;
//        kTierRejected / kTierBadSum = nothing was written anywhere, `why` says why.  The cell is passed so the fetcher can check
//        the wire's FileRef against what it knows of THAT cell; a ref is never trusted on its own.
//   const uint8_t* staging() const                       the assembled blob; untouched until the round is acked (one file op per round)
#include <cstdint>
#include <cstring>
#include <string>

#include "strata/seabraid/channel.hpp"
#include "strata/seabraid/file_ref.hpp"

namespace seabraid {

/// The swap the cache already knows, for a blob swap: same cells, kind kTierSwap, no slot.  Every other op is unchanged.
/// WHY: publish_tier_residency and tier_residency_consistent judge a displacement by `kind == kTierSwap`; the plain view keeps
/// them right for a blob swap without teaching the shared wire helpers about it.
inline TierWireOp as_swap(const TierWireOp& op) {
    TierWireOp p = op;
    if (p.kind == kTierSwapBlob || p.kind == kTierSwapFile) { p.kind = kTierSwap; p.blob_slot = 0; }
    return p;
}

/// A file swap's FileRef is read through the channel's blob slot; a helper that serves no file moves refuses the op with this.
struct NoFileFetch {
    TierRes fetch(int32_t, int32_t, const FileRef&, int64_t, std::string& why) { why = "this helper serves no file moves"; return kTierRejected; }
    const uint8_t* staging() const { return nullptr; }
};

/// Why an op is refused before any slot is touched ("" = fine).  Shape only; the size of the blob is blob_size_refusal's.
inline bool blob_cells_in_model(const TierWireOp& op, int64_t layers, int64_t experts) {
    const bool layer_ok = op.layer_new >= 0 && op.layer_new < layers;
    const bool new_ok = op.expert_new >= 0 && op.expert_new < experts;
    return layer_ok && new_ok && op.expert_old >= 0 && op.expert_old < experts;
}

inline bool carries_blob(const TierWireOp& op) { return op.kind == kTierSwapBlob || op.kind == kTierSwapFile; }

inline std::string op_refusal(const TierWireOp& op, int64_t layers, int64_t experts, int blob_batch = kBlobBatch) {
    if (!carries_blob(op)) return op.blob_slot == 0 ? std::string() : "blob_slot is " + std::to_string(op.blob_slot) + " on an op that carries no blob";
    const std::string what = op.kind == kTierSwapFile ? "a file swap" : "a blob swap";   // the same rules, named for the op they refuse
    if (op.blob_slot >= blob_batch) return "blob slot " + std::to_string(op.blob_slot) + " is outside the channel's " + std::to_string(blob_batch);
    if (op.layer_old != op.layer_new) return what + " is same-layer (layer " + std::to_string(op.layer_old) + " -> " + std::to_string(op.layer_new) + ")";
    if (!blob_cells_in_model(op, layers, experts)) return what + " names a layer or expert outside the model";
    return op.expert_old == op.expert_new ? what + " of an expert with itself" : std::string();
}

/// The blob must fit the channel's slot; `need` is the layer's blob size.
inline std::string blob_size_refusal(int32_t layer, int64_t need, int64_t blob_max) {
    if (need > 0 && need <= blob_max) return std::string();
    return "layer " + std::to_string(layer) + " has blobs of " + std::to_string(need) + " bytes but the channel's blob slots hold " + std::to_string(blob_max);
}

namespace detail {
/// The cell an op's copy lands in: a SWAP writes the displaced expert's slot, a LOAD the one admit() just mapped.
template <class Cache>
bool written_cell_suspect(const TierWireOp& plain, const Cache& cache) {
    return plain.kind == kTierSwap ? cache.suspect(plain.layer_old, plain.expert_old) : cache.suspect(plain.layer_new, plain.expert_new);
}

/// kTierApplied, or kTierFailed when a copy started into a slot that is now quarantined, else kTierRejected (untouched).
template <class Cache>
uint8_t result_of(bool applied, const TierWireOp& plain, const Cache& cache) {
    if (applied) return (uint8_t) kTierApplied;
    return written_cell_suspect(plain, cache) ? (uint8_t) kTierFailed : (uint8_t) kTierRejected;
}

/// Where the round's inputs and outputs live: plain pointers, so the sequence needs no Channel.
struct RoundIO {
    const TierWireOp* ops; int n; uint8_t* res;
    uint8_t* held; int64_t layers, experts, blob_max;
};

/// What one op needs before the cache is asked to apply it: its bytes (null for a pack-read swap) or why it cannot go, and the
/// result to report when it cannot (kTierRejected, or kTierBadSum for a file whose sum is not the manifest's).  Nothing in here
/// writes a slot, so every refusal in this struct is a refusal BEFORE any slot write.
struct Prepared { const uint8_t* bytes = nullptr; uint8_t res = (uint8_t) kTierRejected; std::string why; bool ok = false; };

/// A file swap's bytes: the FileRef copied out of its blob slot ONCE (the slot is shared, so untrusted), then fetched and summed.
template <class Cache, class BlobAt, class Fetch>
void prepare_file(Cache& c, const RoundIO& io, const TierWireOp& op, BlobAt& blob_at, Fetch& fetch, Prepared& p) {
    FileRef ref;
    p.ok = file_ref_from(blob_at(op.blob_slot), io.blob_max, ref);
    if (!p.ok) { p.why = "a channel blob slot cannot hold a FileRef"; return; }
    const TierRes r = fetch.fetch(op.layer_new, op.expert_new, ref, c.blob_bytes(op.layer_new), p.why);
    p.ok = r == kTierApplied;
    p.res = (uint8_t) r;
    p.bytes = fetch.staging();
}

/// "" or why an op is refused because its destination expert is QUARANTINED here.
/// WHY refuse and not "clear the quarantine after a good read-back": a quarantined expert still owns a torn slot, and the
/// read-back that would absolve it only runs for cells held() says are resident, which a quarantined one never is; refusing
/// before the write keeps "applied" = "written AND read back" with no new state, and main fails open (kTierRejected).
template <class Cache>
std::string quarantine_refusal(const Cache& c, const RoundIO& io, const TierWireOp& op) {
    const bool in_model = op.layer_new >= 0 && op.layer_new < io.layers && op.expert_new >= 0 && op.expert_new < io.experts;
    if (!in_model || !c.suspect(op.layer_new, op.expert_new)) return std::string();
    return "expert " + std::to_string(op.expert_new) + " of layer " + std::to_string(op.layer_new) + " is quarantined here";
}

template <class Cache, class BlobAt, class Fetch>
Prepared prepare_op(Cache& c, const RoundIO& io, const TierWireOp& op, BlobAt& blob_at, Fetch& fetch) {
    Prepared p;
    p.why = op_refusal(op, io.layers, io.experts);
    if (p.why.empty() && carries_blob(op)) p.why = blob_size_refusal(op.layer_new, c.blob_bytes(op.layer_new), io.blob_max);
    if (p.why.empty()) p.why = quarantine_refusal(c, io, op);
    if (!p.why.empty()) return p;
    p.ok = true;
    if (op.kind == kTierSwapBlob) p.bytes = blob_at(op.blob_slot);
    if (op.kind == kTierSwapFile) prepare_file(c, io, op, blob_at, fetch, p);
    return p;
}

template <class Cache, class BlobAt, class Report, class Fetch>
bool apply_ops(Cache& c, const RoundIO& io, TierWireOp* plain, BlobAt& blob_at, Report& report, Fetch& fetch) {
    bool all_ok = true;
    for (int i = 0; i < io.n; ++i) {
        plain[i] = as_swap(io.ops[i]);
        Prepared p = prepare_op(c, io, io.ops[i], blob_at, fetch);
        const bool ok = p.ok && c.apply(plain[i], p.bytes, p.why);
        io.res[i] = p.ok ? result_of(ok, plain[i], c) : p.res;
        if (!ok) { report(i, all_ok, p.why); all_ok = false; }
    }
    return all_ok;
}

/// The bytes a read-back compares a blob-written slot with: the channel's blob slot, or for a file swap the helper's own staging.
template <class BlobAt, class Fetch>
const uint8_t* expected_bytes(const TierWireOp& op, BlobAt& blob_at, Fetch& fetch) {
    return op.kind == kTierSwapFile ? fetch.staging() : blob_at(op.blob_slot);
}

/// After the sync: every applied blob or file op's slot is read back against the bytes it was written from.  An op whose cell a
/// later op of the same round displaced is not read (its slot now holds the later expert; that op reads itself).
template <class Cache, class BlobAt, class Report, class Fetch>
bool verify_blob_ops(Cache& c, const RoundIO& io, BlobAt& blob_at, Report& report, Fetch& fetch, bool all_ok) {
    for (int i = 0; i < io.n; ++i) {
        const TierWireOp& op = io.ops[i];
        if (!carries_blob(op) || io.res[i] != (uint8_t) kTierApplied || !c.holds(op.layer_new, op.expert_new)) continue;
        std::string why;
        if (c.verify(op.layer_new, op.expert_new, expected_bytes(op, blob_at, fetch), why)) continue;
        io.res[i] = (uint8_t) kTierFailed;   // the bytes did not land: main must close this card, not route to the slot
        report(i, all_ok, std::string("read-back of the ") + (op.kind == kTierSwapFile ? "file" : "blob") + " swap failed: " + why);
        all_ok = false;
    }
    return all_ok;
}

/// "" or why the round cannot run: two ops naming one destination cell.  WHY whole-round: the second op's displaced slot and the
/// first's read-back both lose their bytes to whichever op lands last, so no per-op answer is true; refusing the round before any
/// slot write keeps held[] the truth (the same shape rule as tier_round_shape_ok, which only covers file swaps).
inline std::string duplicate_destination(const TierWireOp* ops, int n, int64_t layers, int64_t experts) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)   // an op op_refusal rejects writes nothing, so it cannot collide
            if (op_refusal(ops[i], layers, experts).empty() && op_refusal(ops[j], layers, experts).empty() &&
                ops[i].layer_new == ops[j].layer_new && ops[i].expert_new == ops[j].expert_new)
                return "ops " + std::to_string(i) + " and " + std::to_string(j) + " name one destination cell (layer " +
                       std::to_string(ops[i].layer_new) + " expert " + std::to_string(ops[i].expert_new) + ")";
    return std::string();
}

/// A copy that did not complete leaves its slot torn: tier_sync has quarantined it, so an op reported applied is FAILED.
template <class Cache>
void fail_unsynced(Cache& c, const TierWireOp* plain, int n, uint8_t* res) {
    for (int i = 0; i < n; ++i)
        if (res[i] == (uint8_t) kTierApplied && written_cell_suspect(plain[i], c)) res[i] = (uint8_t) kTierFailed;
}
}  // namespace detail

/// One round, WITHOUT the ack: apply every op, sync once, read every blob-written slot back, publish held[].
/// `blob_at(slot)` is the channel's blob slot; `report(op, first, why)` gets every failure (op -1 = the round itself); `fetch` is
/// the FileFetch a kTierSwapFile reads its bytes with.  Returns whether every op applied and read back.
template <class Cache, class BlobAt, class Report, class Fetch>
bool run_tier_round(Cache& c, const detail::RoundIO& io, BlobAt&& blob_at, Report&& report, Fetch&& fetch) {
    const std::string dup = detail::duplicate_destination(io.ops, io.n, io.layers, io.experts);
    if (!dup.empty()) {   // before any slot write: held[] already is the truth
        for (int i = 0; i < io.n; ++i) io.res[i] = (uint8_t) kTierRejected;
        report(-1, true, dup);
        return false;
    }
    TierWireOp plain[kTierMaxOps];
    bool all_ok = detail::apply_ops(c, io, plain, blob_at, report, fetch);
    std::string serr;
    if (!c.sync(serr)) {   // BEFORE the ack: an ack is a CPU store and says nothing about a copy still in flight
        report(-1, all_ok, serr);
        all_ok = false;
        detail::fail_unsynced(c, plain, io.n, io.res);
    }
    all_ok = detail::verify_blob_ops(c, io, blob_at, report, fetch, all_ok);
    publish_tier_residency(io.held, io.layers, io.experts, plain, io.n, [&](int32_t l, int32_t e) { return c.holds(l, e); });
    return all_ok;
}

/// A helper with no file table: a kTierSwapFile is refused.
template <class Cache, class BlobAt, class Report>
bool run_tier_round(Cache& c, const detail::RoundIO& io, BlobAt&& blob_at, Report&& report) {
    NoFileFetch none;
    return run_tier_round(c, io, blob_at, report, none);
}

/// The helper's whole answer to the round `tseq`: validate the count and the round's shape, run it, refresh the totals, ack.
/// ALWAYS acks.
/// WHY the ops are copied ONCE into helper memory and only the copy is validated and used: the op array is shared memory a peer can
/// rewrite at any moment, so a check on it says nothing about a later read (an op turned into a plain swap after its write skips its
/// read-back; a second op turned into a file op after the shape check shares the one staging buffer).  WHY the geometry is
/// ch.limits() and not the Header's n_layers / n_expert: limits() is this process's copy of the layout it MAPPED, so held[] indexing
/// stays inside the mapping whatever a peer writes into the Header.
template <class Cache, class Report, class Fetch>
void serve_tier_channel(Channel& ch, uint64_t tseq, Cache& c, Report&& report, Fetch&& fetch) {
    Header* h = ch.header();
    const int n = __atomic_load_n(&h->tier_n, __ATOMIC_RELAXED);   // one read: the bound below holds for the copy that follows
    if (n < 1 || n > kTierMaxOps) {
        report(-1, true, "tier count is outside the channel's op capacity");
        ch.tier_ack(tseq, false);
        return;
    }
    TierWireOp ops[kTierMaxOps];
    std::memcpy(ops, ch.tier_ops(), (size_t) n * sizeof(TierWireOp));
    h->tier_err[0] = '\0';
    std::string shape;
    if (!tier_round_shape_ok(ops, n, shape)) {   // one staging buffer: a file swap rides alone, refused before anything is read
        for (int i = 0; i < n; ++i) h->tier_res[i] = (uint8_t) kTierRejected;
        report(-1, true, shape);
        ch.tier_ack(tseq, false);
        return;
    }
    const ChannelLimits& lim = ch.limits();
    const detail::RoundIO io{ops, n, h->tier_res, ch.held(), lim.layers, lim.experts, lim.blob_max};
    const bool ok = run_tier_round(c, io, [&](int slot) { return (const uint8_t*) ch.blob(slot); }, report, fetch);
    c.totals(*h);
    ch.tier_ack(tseq, ok);   // after the read-back and the publish: main copies held[] the moment it sees this
}

template <class Cache, class Report>
void serve_tier_channel(Channel& ch, uint64_t tseq, Cache& c, Report&& report) {
    NoFileFetch none;
    serve_tier_channel(ch, tseq, c, report, none);
}

}  // namespace seabraid
