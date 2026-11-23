// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "private_helper_rung.hpp"

#include "diag.hpp"

#include "seabraid/placement.h"

#include <algorithm>
#include <memory>
#include <string>

namespace seabraid::strata_adapter {
namespace {

using ::strata::core::IoStatus;

IoStatus from_code(int32_t code) noexcept {
    switch (code) {
    case SEABRAID_PLACEMENT_IO_ACK: return IoStatus::kOk;
    case SEABRAID_PLACEMENT_IO_REFUSED: return IoStatus::kRefused;
    case SEABRAID_PLACEMENT_IO_FAILED: return IoStatus::kFailed;
    default: return IoStatus::kTimeout;   // WHY: an unknown code means unknown state, the quarantining answer
    }
}

}  // namespace

std::unique_ptr<PrivateHelperRung> PrivateHelperRung::open(QwenDecodeTier& tier, CopyBlob copy_blob,
                                                           size_t layers, std::vector<size_t> slot_bytes,
                                                           uint32_t op_timeout_ms, std::string& err) {
    if (layers == 0 || slot_bytes.size() != layers || std::find(slot_bytes.begin(), slot_bytes.end(), size_t{0}) != slot_bytes.end()) {
        err = "the private helper needs one nonzero slot size for each layer";
        diag::logf(diag::kError, SEABRAID_INVALID, "private helper rung refused: %zu slot sizes for %zu layers, or a zero entry", slot_bytes.size(), layers);
        return nullptr;
    }
    // WHY new and not make_unique: the constructor is private so only this checked path builds a rung.
    return std::unique_ptr<PrivateHelperRung>(new PrivateHelperRung(tier, std::move(copy_blob), std::move(slot_bytes), op_timeout_ms));
}

IoStatus PrivateHelperRung::stage(int32_t layer, int32_t expert, std::vector<uint8_t>& out, std::string& err) {
    // WHY kRefused for every failure: a read touches no slot, so nothing needs restoring.
    // WHY the two logs: a false return and a throw are different faults, each logged once.
    try {
        if (layer < 0 || static_cast<size_t>(layer) >= slot_bytes_.size()) {
            diag::logf(diag::kWarn, SEABRAID_INVALID, "private helper stage refused at layer %d: no slot size for the layer", layer);
            out.clear();
            err = "the engine could not stage the expert blob";
            return IoStatus::kRefused;
        }
        const size_t bytes = slot_bytes_[static_cast<size_t>(layer)];   // this layer's blob size, never another layer's
        out.assign(bytes, 0);
        if (copy_blob_ && copy_blob_(layer, expert, out.data(), bytes)) return IoStatus::kOk;
        diag::logf(diag::kWarn, SEABRAID_PROVIDER, "private helper stage refused at layer %d expert %d: copy failed", layer, expert);
    } catch (...) {
        diag::logf(diag::kError, SEABRAID_INTERNAL, "private helper stage threw at layer %d expert %d", layer, expert);
    }
    out.clear();
    err = "the engine could not stage the expert blob";
    return IoStatus::kRefused;
}

IoStatus PrivateHelperRung::replace(int32_t layer, int32_t old_e, int32_t new_e, const std::vector<uint8_t>& bytes,
                                    std::string& err) {
    // WHY refuse: there is no in-place wire form, so a torn slot could not be told from a good one;
    // the map quarantines the rung instead of rewriting it (README:327-329).
    if (old_e == new_e) {
        err = "an in-place slot rewrite has no wire form";
        diag::logf(diag::kWarn, SEABRAID_INVALID, "private helper replace refused at layer %d: old equals new expert %d", layer, old_e);
        return IoStatus::kRefused;
    }
    // WHY the layer cast needs no guard: a negative layer becomes a huge unsigned that the lease refuses.
    // WHY send the staged bytes (wire v8 kTierSwapBlob): they are the engine's verified copy, and the helper writes them
    // and reads its slot back before it acks, so the helper never trusts its own pack read for a placement move.
    const int32_t code = tier_.tier_swap(static_cast<uint32_t>(layer), old_e, new_e, op_timeout_ms_,
                                         TierBlob{bytes.data(), bytes.size()});
    if (code != SEABRAID_PLACEMENT_IO_ACK) err = "the private helper did not apply the swap";
    return from_code(code);
}

}  // namespace seabraid::strata_adapter
