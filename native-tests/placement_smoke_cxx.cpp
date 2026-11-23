// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// WHY a wrapper file: the placement fixture claims to build as C11 and as C++17 (docs/PLACEMENT-ABI.md), and CMake picks
// the language from the file extension. Including the .c file here compiles the same text as C++ under the same
// -Wall -Wextra -Werror, so the claim has a checker (CTest seabraid_placement_smoke_cxx) instead of being a comment.
#include "placement_smoke.c"
