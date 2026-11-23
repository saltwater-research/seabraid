// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Where this process's pack keeps cell (layer, expert), as the wire's FileExtent: the one adapter from Strata's
// FileExpertSource::file_extents (patch 0018, the one owner of addressing) to FileTable::ExtentsOf.  Used by seabraid-sums (what it
// hashes) and seabraid-helper (what it expects a file op to name), so the two can never address a cell differently.
// WHY field by field: Strata's FileExtent and the wire's are two types with one layout, and neither side may include the other's.
#include "strata/seabraid/channel_layout.hpp"
#include "strata/core/expert_source.hpp"

namespace seabraid {

struct PackExtents {
    const strata::core::FileExpertSource& src;
    bool operator()(int64_t layer, int64_t expert, FileExtent* out, int& n) const {
        strata::core::FileExpertSource::FileExtent fe[kFileRefMaxExtents];
        if (!src.file_extents(layer, expert, fe, n)) return false;
        for (int i = 0; i < n && i < kFileRefMaxExtents; ++i) out[i] = {fe[i].file, fe[i].length, fe[i].offset};
        return true;
    }
};

}  // namespace seabraid
