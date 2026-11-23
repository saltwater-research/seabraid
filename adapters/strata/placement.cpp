// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "placement.hpp"

#include "diag.hpp"
#include "diag_seabraid.hpp"

#include "strata/core/ownership_map.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>

namespace seabraid::strata_adapter {
namespace {

using diag::kError;
using strata::core::IoStatus;

struct CallbackContext {
    const PlacementRungs& rungs;
    std::string& first_error;
    uint64_t& stage_us;
    uint64_t& replace_us;
    std::vector<trace::CallSample>* calls;   // nullptr while the trace is off: no allocation, no record
    uint32_t& file_asked;
    uint32_t& file_acked;
};

// Adds the elapsed microseconds to `total` when the provider call ends, including when it throws.
class PhaseTimer {
public:
    // `calls` (may be null) also gets one CallSample: the trace's per-move stage and write times.
    PhaseTimer(uint64_t& total, std::vector<trace::CallSample>* calls, trace::CallSample sample) noexcept
        : total_(total), calls_(calls), sample_(sample), start_(std::chrono::steady_clock::now()) {}
    ~PhaseTimer() {
        const auto us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_).count());
        total_ += us;
        if (!calls_) return;
        sample_.us = us;
        try {
            calls_->push_back(sample_);
        } catch (...) {  // allow-error-path: a sample lost to out-of-memory only blanks one move's timing
            return;
        }
    }
    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;
private:
    uint64_t& total_;
    std::vector<trace::CallSample>* calls_;
    trace::CallSample sample_;
    std::chrono::steady_clock::time_point start_;
};

// Identifies one provider call in log text; ids and counts only.
struct Site {
    const char* op;
    uint8_t rung;
    uint32_t layer;
    uint32_t expert;
};

int32_t to_seabraid(IoStatus result) noexcept {
    switch (result) {
    case IoStatus::kOk: return SEABRAID_PLACEMENT_IO_ACK;
    case IoStatus::kRefused: return SEABRAID_PLACEMENT_IO_REFUSED;
    case IoStatus::kFailed: return SEABRAID_PLACEMENT_IO_FAILED;
    case IoStatus::kTimeout: return SEABRAID_PLACEMENT_IO_TIMEOUT;
    }
    return SEABRAID_PLACEMENT_IO_TIMEOUT;
}

// Keeps the first failure for the caller; the record itself is already logged
// by the caller, so a failed copy here is the only thing this function logs.
void record_first(CallbackContext& context, const char* message) noexcept {
    try {
        if (context.first_error.empty()) context.first_error.assign(message);
    } catch (...) {
        diag::logf(kError, SEABRAID_INTERNAL, "placement: could not record the first error (allocation failed)");
    }
}

int32_t report(CallbackContext& context, IoStatus result, std::string& error, const Site& site) noexcept {
    if (result != IoStatus::kOk) {
        // The provider's own text may name files, so only codes are logged.
        diag::logf(kError, SEABRAID_PROVIDER, "placement %s failed: rung %u layer %u expert %u, io status %d",
                   site.op, site.rung, site.layer, site.expert, static_cast<int>(result));
        // swap cannot allocate, so keeping the first error cannot fail here.
        if (context.first_error.empty()) context.first_error.swap(error);
    }
    return to_seabraid(result);
}

// A foreign exception gives no evidence that the provider drained, so the
// callback reports timeout and the runtime quarantines the card.
int32_t threw(CallbackContext& context, const Site& site) noexcept {
    diag::logf(kError, SEABRAID_PROVIDER,
               "placement %s: provider threw (rung %u layer %u expert %u); reported as timeout",
               site.op, site.rung, site.layer, site.expert);
    record_first(context, "placement provider threw an exception");
    return SEABRAID_PLACEMENT_IO_TIMEOUT;
}

int32_t out_of_range(CallbackContext& context) noexcept {
    // Refused (not timeout): no provider was touched, so nothing is uncertain.
    record_first(context, "placement rung is out of range");
    return SEABRAID_PLACEMENT_IO_REFUSED;
}

// The legacy RungIO interface owns its vectors. Copy into Rust's staging
// buffer only after verifying the length so an ACK cannot expose a short slot.
int32_t stage(void* opaque, uint8_t rung, uint32_t layer, uint32_t expert,
              uint8_t* output, size_t capacity, size_t* written) noexcept {
    auto& context = *static_cast<CallbackContext*>(opaque);
    auto* card = checked_rung(context.rungs, rung);
    if (!card) return out_of_range(context);
    const Site site{"stage", rung, layer, expert};
    try {
        std::vector<uint8_t> bytes;
        std::string error;
        const PhaseTimer timer(context.stage_us, context.calls, {false, rung, layer, expert, 0});
        const auto result = card->stage(layer, expert, bytes, error);
        if (result == IoStatus::kOk && bytes.size() == capacity) {
            std::memcpy(output, bytes.data(), capacity);
            *written = capacity;
        } else if (result == IoStatus::kOk) {
            // The runtime rejects an ACK without a written length; say why.
            diag::logf(kError, SEABRAID_PROVIDER, "placement stage returned %zu bytes for a %zu byte slot: rung %u layer %u expert %u",
                       bytes.size(), capacity, rung, layer, expert);
        }
        return report(context, result, error, site);
    } catch (...) {  // allow-error-path: threw() logs the failure through diag::logf
        return threw(context, site);
    }
}

int32_t replace(void* opaque, uint8_t rung, uint32_t layer, uint32_t old_expert,
                uint32_t new_expert, const uint8_t* input, size_t size) noexcept {
    auto& context = *static_cast<CallbackContext*>(opaque);
    auto* card = checked_rung(context.rungs, rung);
    if (!card) return out_of_range(context);
    const Site site{"replace", rung, layer, new_expert};
    // WHY a separate phase: the copy runs before any provider I/O, so a failure here
    // (bad_alloc, length_error) proves nothing was touched. Reporting it as TIMEOUT
    // would make the core quarantine a healthy card; REFUSED leaves it in service.
    std::vector<uint8_t> bytes;
    try {
        bytes.assign(input, input + size);
    } catch (...) {  // allow-error-path: logged on the next line, mapped to REFUSED
        diag::logf(kError, SEABRAID_INTERNAL, "placement replace: cannot copy %zu bytes before the provider call "
                   "(rung %u layer %u expert %u); refused", size, rung, layer, new_expert);
        record_first(context, "placement replace could not allocate its input copy");
        return SEABRAID_PLACEMENT_IO_REFUSED;
    }
    try {
        std::string error;
        const PhaseTimer timer(context.replace_us, context.calls, {true, rung, layer, new_expert, 0});
        const auto result = card->replace(layer, old_expert, new_expert, bytes, error);
        return report(context, result, error, site);
    } catch (...) {  // allow-error-path: threw() logs the failure through diag::logf
        return threw(context, site);
    }
}

// One sample of the trace, built after the call ended: the by-reference ask knows only then whether it wrote anything.
void note_call(CallbackContext& context, const trace::CallSample& sample) noexcept {
    if (!context.calls) return;
    try {
        context.calls->push_back(sample);
    } catch (...) {  // allow-error-path: a sample lost to out-of-memory only blanks one move's timing
        return;
    }
}

// FILE MOVES (provider_v2): the library asks `rung` to load `new_expert` from the model file itself.  REFUSED means "use bytes": the
// rung wrote nothing and the library stages the expert and calls replace() in the same swap, so it is neither logged as a failure nor
// kept as the round's first error; every other answer is reported as replace's is.  Time and trace: a REFUSED ask counts as staging
// work (the swap goes on), an answered one as the write.
int32_t replace_from_fallback(void* opaque, uint8_t rung, uint32_t layer, uint32_t old_expert, uint32_t new_expert) noexcept {
    auto& context = *static_cast<CallbackContext*>(opaque);
    auto* card = checked_rung(context.rungs, rung);
    if (!card) return out_of_range(context);
    const Site site{"replace_from_fallback", rung, layer, new_expert};
    try {
        std::string error;
        const auto start = std::chrono::steady_clock::now();
        const auto result = card->replace_from_fallback(layer, old_expert, new_expert, error);
        const auto us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count());
        const bool wrote = result != IoStatus::kRefused;
        ++context.file_asked;
        if (result == IoStatus::kOk) ++context.file_acked;
        (wrote ? context.replace_us : context.stage_us) += us;
        note_call(context, {wrote, rung, layer, new_expert, us, result == IoStatus::kOk});
        if (!wrote) return SEABRAID_PLACEMENT_IO_REFUSED;
        return report(context, result, error, site);
    } catch (...) {  // allow-error-path: threw() logs the failure through diag::logf
        return threw(context, site);
    }
}

// Returns nullptr when the rung table matches the runtime's snapshot.
const char* shape_fault(const PlacementRungs& rungs, const seabraid_placement_snapshot_v1& shape) {
    constexpr auto int32_max = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
    if (shape.layers > int32_max || shape.experts > int32_max) return "snapshot geometry exceeds int32";
    if (rungs.layer_slot_bytes.size() != shape.layers) return "slot-size table does not match the layer count";
    if (rungs.cards.size() != shape.fallback_rung) return "card table does not match the fallback rung";
    const bool bound = std::all_of(rungs.cards.begin(), rungs.cards.end(), [](auto* card) { return card != nullptr; });
    return bound ? nullptr : "a card binding is null";
}

// WHY two entry points: with no rung that can load from the model file (mask 0) the call is exactly the v1 call, so a run without
// file moves is bit-for-bit what it was; only a mask that names a rung opens provider_v2 and apply_v2.
int32_t apply_with_provider(seabraid_placement* placement, uint64_t ticket, const PlacementRungs& rungs, CallbackContext& context,
                            PlacementRound& round, seabraid_placement_applied_v1& applied) noexcept {
    const uint64_t mask = from_fallback_mask(rungs);
    if (mask == 0) {
        const seabraid_placement_provider_v1 provider{
            SEABRAID_ABI_VERSION, &context, rungs.layer_slot_bytes.data(), rungs.layer_slot_bytes.size(), stage, replace
        };
        return seabraid_placement_apply(placement, ticket, &provider, round.outcomes.data(), round.outcomes.size(), &applied);
    }
    const seabraid_placement_provider_v2 provider{
        SEABRAID_PLACEMENT_PROVIDER_V2_VERSION, sizeof(seabraid_placement_provider_v2), &context, rungs.layer_slot_bytes.data(),
        rungs.layer_slot_bytes.size(), stage, replace, mask, replace_from_fallback
    };
    return seabraid_placement_apply_v2(placement, ticket, &provider, round.outcomes.data(), round.outcomes.size(), &applied);
}

}  // namespace

// Takes the executor snapshot; the caller-set header (version and size) is part of the ABI call.
int32_t take_snapshot(seabraid_placement* placement, PlacementRound& round) noexcept {
    round.snapshot = seabraid_placement_snapshot_v1{};
    round.snapshot.abi_version = SEABRAID_ABI_VERSION;
    round.snapshot.struct_size = sizeof round.snapshot;
    return seabraid_placement_snapshot(placement, round.executors.data(), round.executors.size(), &round.snapshot);
}

uint64_t from_fallback_mask(const PlacementRungs& rungs) noexcept {
    uint64_t mask = 0;
    for (size_t rung = 0; rung < rungs.cards.size() && rung < SEABRAID_PLACEMENT_MAX_CARDS; ++rung)
        if (rungs.cards[rung] != nullptr && rungs.cards[rung]->reads_files()) mask |= uint64_t{1} << rung;
    return mask;
}

strata::core::RungIO* checked_rung(const PlacementRungs& rungs, uint8_t rung) noexcept {
    // WHY fail closed: the runtime chooses `rung`; trusting it would turn a
    // runtime bug into an out-of-bounds read of the card table.
    if (rung < rungs.cards.size()) return rungs.cards[rung];
    if (rung == rungs.cards.size() && rungs.fallback != nullptr) return rungs.fallback;
    diag::logf(kError, SEABRAID_INVALID, "placement refused: rung %u is outside the %zu bound cards%s", rung,
               rungs.cards.size(), rungs.fallback ? " and the fallback" : "");
    return nullptr;
}

int32_t apply_placement_at_fence(seabraid_placement* placement,
                                 const seabraid_placement_ticket_v1& ticket,
                                 const PlacementRungs& rungs,
                                 PlacementRound& round) noexcept {
    install_seabraid_log_sink();
    // Validate the routing table's size before any irreversible provider I/O.
    auto status = take_snapshot(placement, round);
    if (status != SEABRAID_OK) {
        diag::logf(kError, status, "placement snapshot failed: %s", diag::status_name(status));
        return status;
    }
    if (const char* fault = shape_fault(rungs, round.snapshot)) {
        diag::logf(kError, SEABRAID_INVALID, "placement refused: %s", fault);
        return SEABRAID_INVALID;
    }
    // WHY the cell count and not half of it: a multi-hop climb of ONE expert is one promotion per hop, so on a tiny layout
    // (1 layer x 5 experts, 3 hops) the count legitimately passes half the cells.  Every promotion moves at least one
    // distinct cell, so more promotions than cells is the only provably corrupt ticket; the library enforces the real plan.
    if (ticket.promotion_count > round.executors.size()) {
        diag::logf(kError, SEABRAID_INVALID, "placement refused: %llu promotions exceed the %zu executors",
                   static_cast<unsigned long long>(ticket.promotion_count), round.executors.size());
        return SEABRAID_INVALID;
    }

    try {
        // Element 0 carries the header; the library strides by its struct_size.
        round.outcomes.assign(ticket.promotion_count, seabraid_placement_outcome_v1{});
        if (!round.outcomes.empty()) {
            round.outcomes[0].abi_version = SEABRAID_ABI_VERSION;
            round.outcomes[0].struct_size = sizeof(seabraid_placement_outcome_v1);
        }
    } catch (...) {
        diag::logf(kError, SEABRAID_INTERNAL, "placement: outcome table allocation failed for %llu promotions",
                   static_cast<unsigned long long>(ticket.promotion_count));
        return SEABRAID_INTERNAL;
    }
    round.first_io_error.clear();
    round.stage_us = round.replace_us = 0;
    round.file_asked = round.file_acked = 0;
    round.calls.clear();
    CallbackContext context{rungs, round.first_io_error, round.stage_us, round.replace_us,
                            trace::on() ? &round.calls : nullptr, round.file_asked, round.file_acked};
    seabraid_placement_applied_v1 applied{};
    applied.abi_version = SEABRAID_ABI_VERSION;
    applied.struct_size = sizeof applied;
    status = apply_with_provider(placement, ticket.ticket, rungs, context, round, applied);
    if (status != SEABRAID_OK) {
        diag::logf(kError, status, "placement apply failed: %s", diag::status_name(status));
        return status;
    }
    round.outcomes.resize(applied.outcome_count);
    // Publish the full executor map, including fallback routing after a
    // quarantine. The engine holds the same fence through this export.
    status = take_snapshot(placement, round);
    if (status != SEABRAID_OK) diag::logf(kError, status, "placement republish failed: %s", diag::status_name(status));
    return status;
}

}  // namespace seabraid::strata_adapter
