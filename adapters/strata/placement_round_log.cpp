// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Round bookkeeping (see placement_round.hpp). OWNER: the field names of the round line, which an operator and the
// qualification runner parse. FAILURE CONTRACT: pure counting plus one diag record; nothing here can fail or throw.
#include "placement_round.hpp"

#include "diag.hpp"

namespace seabraid::strata_adapter {
namespace {

// The owner a cell has once the host's own lead move is recorded. WHY: Strata's lead<->RAM adapt only touches lead
// and fallback cells, so a cell that lost its lead slot falls to the fallback and one that gained it goes to the lead.
uint8_t adopted_owner(uint8_t current, bool lead, uint8_t fallback) noexcept {
    if (lead) return 0;
    return current == 0 ? fallback : current;
}

}  // namespace

// A moved cell that a helper held is a conflict: the helper's residency changed by a route nobody owns, so the
// table cannot be trusted for that rung.
Adoption scan_adoption(const std::vector<uint8_t>& published, const std::vector<uint8_t>& lead, uint8_t fallback,
                       std::vector<uint8_t>& next) noexcept {
    Adoption result;
    for (size_t i = 0; i < published.size(); ++i) {
        const uint8_t owner = adopted_owner(published[i], lead[i] != 0, fallback);
        next[i] = owner;
        if (owner == published[i]) continue;
        if (published[i] != 0 && published[i] != fallback) result.conflict = published[i];
        else ++result.moved;
    }
    return result;
}

// Cells the table gives the lead that the host's lead does not hold, plus the reverse. A size mismatch counts all.
size_t lead_mismatches(const std::vector<uint8_t>& executors, const std::vector<uint8_t>& lead) noexcept {
    if (lead.size() != executors.size()) return executors.size();
    size_t bad = 0;
    for (size_t i = 0; i < executors.size(); ++i) bad += (executors[i] == 0) != (lead[i] != 0);
    return bad;
}

HelperCheck check_helper(const std::vector<uint8_t>& executors, const std::vector<uint8_t>& held,
                         uint8_t rung) noexcept {
    HelperCheck result;
    if (held.size() != executors.size()) {
        result.missing = executors.size();
        return result;
    }
    for (size_t i = 0; i < executors.size(); ++i) {
        const bool owned = executors[i] == rung;
        result.missing += owned && !held[i];
        result.spares += !owned && held[i];
    }
    return result;
}

void tally_outcomes(const std::vector<seabraid_placement_outcome_v1>& outcomes, RoundStats& stats) noexcept {
    for (const auto& outcome : outcomes) {
        switch (outcome.kind) {
        case SEABRAID_PLACEMENT_COMMITTED: ++stats.committed; break;
        case SEABRAID_PLACEMENT_REJECTED: ++stats.rejected; break;
        case SEABRAID_PLACEMENT_RESTORED: ++stats.restored; break;
        case SEABRAID_PLACEMENT_SKIPPED: ++stats.skipped; break;
        case SEABRAID_PLACEMENT_QUARANTINED: ++stats.quarantined; break;   // the mask stays the authoritative record
        default: break;
        }
    }
}

double helper_share(const uint8_t* executors, const float* heat, size_t cells, uint8_t fallback) noexcept {
    double helper = 0.0, total = 0.0;
    for (size_t i = 0; i < cells; ++i) {
        total += heat[i];
        if (executors[i] != 0 && executors[i] < fallback) helper += heat[i];
    }
    return total > 0.0 ? helper / total : 0.0;
}

// WHY one record: the diag buffer holds 255 characters, so the names are short and the fields an operator reads
// first come first. A cut tail loses a timing, never an identity field. Typical length is about 250 characters.
void log_round(const RoundStats& s) noexcept {
    const auto u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
    diag::logf(diag::kInfo, 0,
               "placement round=%llu epoch=%llu planned=%u committed=%u rejected=%u restored=%u qmask=0x%llx "
               "adopted=%u spares=%u defer=%u helper_share=%.3f table_hash=%016llx plan_us=%llu fence_us=%llu "
               "stage_us=%llu replace_us=%llu publish_us=%llu follow_us=%llu",
               u(s.round), u(s.epoch), s.planned, s.committed, s.rejected, s.restored, u(s.qmask), s.adopted,
               s.spares, s.deferred, s.helper_share, u(s.table_hash), u(s.plan_us), u(s.fence_us), u(s.stage_us),
               u(s.replace_us), u(s.publish_us), u(s.follow_us));
    // WHY a second record: the round line is already ~250 of diag's 255 characters, and these two are rare.
    if (s.skipped != 0 || s.qual)
        diag::logf(diag::kInfo, 0, "placement round=%llu skipped=%u qual=%d", u(s.round), s.skipped, s.qual ? 1 : 0);
    // Third record, only on a round that asked for file moves (--retier-file-moves): fr = asked, fr_ok = loaded from the file,
    // fr_bytes = asked - ok (refused, so those moved as bytes).
    if (s.file_asked != 0)
        diag::logf(diag::kInfo, 0, "placement round=%llu fr=%u fr_ok=%u fr_bytes=%u", u(s.round), s.file_asked, s.file_acked,
                   s.file_asked - s.file_acked);
}

namespace {

// Whether both writes of a hop (the promoted expert up, the displaced one down) were read back by their writer.
// WHY these rules: rung 0 is the lead, which always runs verify_slot; a helper reads back only with verify_writes; the
// fallback's own copy has no read-back, so any hop that touches it is not fully verified.
bool hop_verified(uint8_t upper, uint8_t lower, uint8_t fallback, bool verify_writes) noexcept {
    const auto read_back = [&](uint8_t rung) { return rung == 0 || (rung != fallback && verify_writes); };
    return read_back(upper) && read_back(lower);
}

// The trace record for outcome `index`; its time is the matching PairTiming when the driver measured one.
void trace_outcome(const seabraid_placement_outcome_v1& o, size_t index, size_t bytes, uint8_t fallback,
                   const MoveContext& context) noexcept {
    if (!trace::on()) return;
    trace::MoveFields fields;
    fields.round = context.round;
    fields.epoch = o.committed_epoch != 0 ? o.committed_epoch : context.epoch;
    fields.outcome = &o;
    fields.fallback = fallback;
    fields.bytes = bytes;
    fields.verified = hop_verified(o.upper_rung, static_cast<uint8_t>(o.upper_rung + 1u), fallback, context.verify_writes);
    if (context.timing != nullptr && index < context.timing->size()) fields.timing = (*context.timing)[index];
    trace::move(fields);
}

size_t slot_size(const std::vector<size_t>& slot_bytes, uint32_t layer) noexcept {
    return layer < slot_bytes.size() ? slot_bytes[layer] : 0;
}

}  // namespace

void log_moves(const std::vector<seabraid_placement_outcome_v1>& outcomes, const std::vector<size_t>& slot_bytes,
               uint8_t fallback, const MoveContext& context) noexcept {
    for (size_t i = 0; i < outcomes.size(); ++i) {
        const auto& o = outcomes[i];
        if (o.kind != SEABRAID_PLACEMENT_COMMITTED) continue;
        const unsigned lower = o.upper_rung + 1u, upper = o.upper_rung;
        const size_t bytes = slot_size(slot_bytes, o.layer);
        diag::logf(diag::kInfo, 0, "placement move %u/%u %u->%u src=%s bytes=%zu verified=%d", o.layer, o.promote, lower,
                   upper, lower == fallback ? "fallback" : "card", bytes,
                   hop_verified(o.upper_rung, static_cast<uint8_t>(lower), fallback, context.verify_writes) ? 1 : 0);
        // WHY a separate line: the move line above is parsed by two tools, so the pushed-down expert gets its own.
        diag::logf(diag::kInfo, 0, "placement evict %u/%u %u->%u dst=%s", o.layer, o.displace, upper, lower,
                   lower == fallback ? "fallback" : "card");
        trace_outcome(o, i, bytes, fallback, context);
    }
}

void log_failures(const std::vector<seabraid_placement_outcome_v1>& outcomes, const std::vector<size_t>& slot_bytes,
                  uint8_t fallback, const MoveContext& context) noexcept {
    for (size_t i = 0; i < outcomes.size(); ++i) {
        const auto& o = outcomes[i];
        if (o.kind == SEABRAID_PLACEMENT_COMMITTED) continue;
        // ONE line per outcome; the restore half (when the undo itself failed) is a tail, so a cut loses only it.
        diag::logf(diag::kInfo, 0, "placement fail %u/%u %u->%u kind=%s stage=%s result=%s restore=%s/%s", o.layer,
                   o.promote, o.upper_rung + 1u, static_cast<unsigned>(o.upper_rung), trace::kind_name(o.kind),
                   trace::phase_name(o.initial.phase), trace::issue_name(o.initial.error),
                   trace::phase_name(o.restore.phase), trace::issue_name(o.restore.error));
        trace_outcome(o, i, slot_size(slot_bytes, o.layer), fallback, context);
    }
}

bool ram_follows(const seabraid_placement_outcome_v1& outcome) noexcept {
    return outcome.kind == SEABRAID_PLACEMENT_COMMITTED && outcome.upper_rung == 0;
}

}  // namespace seabraid::strata_adapter
