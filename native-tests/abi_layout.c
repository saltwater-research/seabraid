// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Layout parity: every public struct matches src/layout/abi_layout.txt, checked by the C compiler.
 * OWNER: the table file. CMake generates abi_layout_asserts.h from it (one _Static_assert per row), so this
 * file holds no number of its own. FAILURE: a header edit that moves a field or changes a size stops the
 * build of this test. Rust (src/layout.rs) and Python (ctypes) check the same table. 64-bit targets only:
 * the table records their layout, and 32-bit pointers shrink several structs. */
#include "seabraid/placement.h"
#include "seabraid/seabraid.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#if UINTPTR_MAX == 0xffffffffffffffffu
#include "abi_layout_asserts.h"
#endif

int main(void)
{
    /* The one run-time check: this binary and the library agree on the epoch the headers declare. */
    if (seabraid_abi_epoch() != SEABRAID_ABI_EPOCH) {
        fprintf(stderr, "library epoch %u != header epoch %u\n", (unsigned)seabraid_abi_epoch(),
                (unsigned)SEABRAID_ABI_EPOCH);
        return 1;
    }
    puts("Seabraid ABI layout parity passed");
    return 0;
}
