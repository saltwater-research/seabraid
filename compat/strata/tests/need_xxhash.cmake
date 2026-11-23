# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Run (and fail) by the one test registered when seabraid.cmake found no xxhash.h: the file-move tests cannot be built without it.
message(FATAL_ERROR "the file-move tests were NOT built: Strata's vendored xxhash.h was not found.  Configure with "
        "-DSEABRAID_STRATA_XXHASH_DIR=<strata>/third_party/strata-llama/vendor/hash/xxhash (or -DSTRATA_SOURCE=<strata checkout>).")
