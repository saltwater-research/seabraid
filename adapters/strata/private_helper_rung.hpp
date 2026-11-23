// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The private helper as a placement rung: Strata's RungIO over QwenDecodeTier.
// OWNER: how one slot of the private helper is staged and swapped; residency stays with
//   the tier (held_) and the lease. Slot sizes are per layer (blob sizes differ across layers). CONTRACT: engine thread, fence held, no layer pending.
//   Imports the tier and Strata's RungIO only, never RemoteExperts or engine internals.
// FAILURE: never throws. A read touches no slot, so any stage failure is kRefused; a swap
//   maps the lease's code one to one (ACK, REFUSED, FAILED, TIMEOUT), and every failure
//   branch logs once. CHECKER: private_qwen_test (old == new, mapping, stage).
#pragma once

#include "qwen_decode.hpp"

#include "strata/core/ownership_map.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace seabraid::strata_adapter {

class PrivateHelperRung final : public ::strata::core::RungIO {
public:
    // The engine's ExpertSource::copy_blob, bound: writes `bytes` bytes of (layer, expert) to dst.
    using CopyBlob = std::function<bool(int64_t layer, int64_t expert, uint8_t* dst, size_t bytes)>;

    // WHY a factory and not a constructor: the table's shape is the caller's input, so a bad one must come back as a
    // refusal with a reason (nullptr, `err` set), never a half-built rung. One entry per layer, each > 0: the engine's
    // expert blob size differs per layer (mixed per-layer quants), so one slot size cannot describe the helper.
    static std::unique_ptr<PrivateHelperRung> open(QwenDecodeTier& tier, CopyBlob copy_blob,
                                                   size_t layers, std::vector<size_t> slot_bytes,
                                                   uint32_t op_timeout_ms, std::string& err);

    ::strata::core::IoStatus stage(int32_t layer, int32_t expert, std::vector<uint8_t>& out,
                                   std::string& err) override;
    ::strata::core::IoStatus replace(int32_t layer, int32_t old_e, int32_t new_e,
                                     const std::vector<uint8_t>& bytes, std::string& err) override;

private:
    PrivateHelperRung(QwenDecodeTier& tier, CopyBlob copy_blob, std::vector<size_t> slot_bytes, uint32_t op_timeout_ms)
        : tier_(tier), copy_blob_(std::move(copy_blob)), slot_bytes_(std::move(slot_bytes)), op_timeout_ms_(op_timeout_ms) {}

    QwenDecodeTier& tier_;
    CopyBlob copy_blob_;
    std::vector<size_t> slot_bytes_;   // by layer; stage() sizes the staging buffer from the layer's own entry
    uint32_t op_timeout_ms_;
};

}  // namespace seabraid::strata_adapter
