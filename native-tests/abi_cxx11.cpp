// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The public headers must compile as strict C++11 (-std=c++11 -pedantic-errors): the C++ fixtures use C++17, so
// without this file nothing in the tree proves the documented C++11 floor. Compile-only in spirit: it needs no
// library, so it is not linked against Seabraid::runtime.
#include "seabraid/placement.h"
#include "seabraid/seabraid.h"

int main()
{
    seabraid_batch_v2 batch2 = seabraid_batch_v2();
    seabraid_provider_v3 provider3 = seabraid_provider_v3();
    seabraid_placement_outcome_v1 outcome = seabraid_placement_outcome_v1();
    batch2.abi_version = SEABRAID_BATCH_V2_VERSION;
    provider3.abi_version = SEABRAID_PROVIDER_V3_VERSION;
    outcome.abi_version = SEABRAID_ABI_VERSION;
    return (sizeof batch2 + sizeof provider3 + sizeof outcome == 0 || batch2.tokens != 0 || provider3.flags != 0) ? 1 : 0;
}
