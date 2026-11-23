// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Wire v8+ host fixture (version pin: v11): the SEABRAID_STRATA_SHM list parser (the engine's channel_name, compat 0017, picks entry `card`) and the
// blob_slot rule of kTierSwapBlob.
// No GPU, no Strata engine: these are the pure pieces of the wire that the engine (compat 0017) and the helper share.
#include "test_support.hpp"

#include "strata/seabraid/channel.hpp"
#include "strata/seabraid/channel_names.hpp"

#include <string>
#include <vector>

namespace {
using seabraid::strata_adapter::testing::check;

bool parses(const char* env, size_t n) {
    std::vector<std::string> out;
    std::string err;
    return seabraid::parse_channel_list(env, out, err) && out.size() == n && err.empty();
}

bool refused_naming(const char* env, const char* needle) {
    std::vector<std::string> out{"stale"};
    std::string err;
    return !seabraid::parse_channel_list(env, out, err) && out.empty() && err.find(needle) != std::string::npos;
}

void list_cases() {
    check(parses("/seabraid0", 1), "one entry parses");
    check(parses("/seabraid0,/seabraid1", 2), "two entries parse");
    check(parses("/a.b-c_9,/seabraid1,/x2", 3), "three entries with every allowed character parse");
    check(refused_naming("/seabraid0,/seabraid1,/seabraid2,/seabraid3", "more than 3"), "a fourth channel is refused");
    check(refused_naming("/seabraid0,/seabraid0", "twice"), "a duplicate is refused");
    check(refused_naming("seabraid0", "'seabraid0'"), "a missing slash is refused, naming the entry");
    check(refused_naming("/seabraid0,", "''"), "a trailing empty entry is refused");
    check(refused_naming("/a/b", "'/a/b'"), "a second slash is refused");
    check(refused_naming("/", "'/'"), "a bare slash is refused");
    check(refused_naming("/..", "'/..'"), "dot-dot is refused");
    check(refused_naming("/bad name", "'/bad name'"), "a space is refused");
    check(refused_naming("", "empty"), "an empty value is refused");
    check(refused_naming(nullptr, "empty"), "a null value is refused");
    const std::string long_name = "/" + std::string(64, 'a');
    check(refused_naming(long_name.c_str(), "is not /name"), "a 65-byte entry is refused");
    check(parses(("/" + std::string(63, 'a')).c_str(), 1), "a 64-byte entry parses");
}

void wire_cases() {
    static_assert(seabraid::kVersion == 11, "wire v11");
    static_assert(sizeof(seabraid::TierWireOp) == 20, "v8 to v11 keep the 20-byte op");
    seabraid::TierWireOp op{};
    op.kind = seabraid::kTierSwapBlob;
    op.blob_slot = 0;
    check(seabraid::blob_slot_valid(op), "blob swap with slot 0 is valid");
    op.blob_slot = seabraid::kBlobBatch - 1;
    check(seabraid::blob_slot_valid(op), "blob swap with the last slot is valid");
    op.blob_slot = seabraid::kBlobBatch;
    check(!seabraid::blob_slot_valid(op), "blob swap past kBlobBatch is invalid");
    op.kind = seabraid::kTierSwap;
    op.blob_slot = 1;
    check(!seabraid::blob_slot_valid(op), "a non-blob op must carry blob_slot 0");
    op.blob_slot = 0;
    check(seabraid::blob_slot_valid(op), "a plain swap with blob_slot 0 is valid");
    seabraid::TierWireOp blob{seabraid::kTierSwapBlob, 0, 0, 2, 5, 2, 9};
    check(seabraid::tier_op_displaces(blob), "a blob swap displaces its old cell");
    uint8_t held[3 * 10] = {};
    held[2 * 10 + 9] = 1;   // the helper published: new resident, old gone
    const uint8_t res[1] = {seabraid::kTierApplied};
    std::string why;
    check(seabraid::tier_residency_consistent(held, 3, 10, &blob, res, 1, why), "a blob swap replays as a swap");
    held[2 * 10 + 5] = 1;   // stale bitmap: the displaced cell still marked resident
    check(!seabraid::tier_residency_consistent(held, 3, 10, &blob, res, 1, why), "a stale displaced cell is caught");
}
}  // namespace

int main() {
    list_cases();
    wire_cases();
    std::puts("channel_names: ok");
    return 0;
}
