// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace seabraid {
// IEEE-754 binary32 with an all-ones exponent is Inf or NaN. WHY a bit test and
// not std::isfinite: a consumer built with -ffast-math may fold isfinite to true.
inline bool all_finite(const float* values, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, values + i, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) return false;
    }
    return true;
}
}  // namespace seabraid
