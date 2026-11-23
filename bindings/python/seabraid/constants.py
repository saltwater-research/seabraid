# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Numeric constants of the v1 C ABI, mirrored for Python.

OWNER   : include/seabraid/seabraid.h owns every value here; this file only restates them.
CONTRACT: plain integers and one IntFlag; imports stdlib only.
SIZE    : under 100 lines.
FAILURE : none at runtime (data only).
CHECKER : tests/test_header_parity.py parses seabraid.h and fails when a value drifts or the header gains an unmirrored name.
FINISH  : the parity test passes.
"""

from __future__ import annotations

from enum import IntFlag

ABI_VERSION = 1
# The layout epoch this binding was written against (SEABRAID_ABI_EPOCH); load_library refuses any other.
ABI_EPOCH = 1
MULTI_PROVIDER_VERSION = 2
PROVIDER_PARTIAL_ROUTES = 1 << 0
# Batch v2 / provider v3 declarations, used by seabraid_session_open and seabraid_session_submit.
BATCH_V2_VERSION = 2
PROVIDER_V3_VERSION = 3
PROVIDER_COMPACT_ROWS = 1 << 1
PROVIDER_ACCEPTS_F16 = 1 << 2
PROVIDER_ACCEPTS_BF16 = 1 << 3
PROVIDER_REFERENCE = 1 << 4
SKIP_EXPERT_ID = -1
# placement.h: seabraid_placement_policy_v2 (seabraid_placement_open_v2) and the outcome kind a skipped climb hop reports.
PLACEMENT_POLICY_V2_VERSION = 2
PLACEMENT_SKIPPED = 5
# placement.h: seabraid_placement_provider_v2 (seabraid_placement_apply_v2), the by-reference write.
PLACEMENT_PROVIDER_V2_VERSION = 2

OK = 0
INVALID = 1
BUSY_OR_STATE = 2
TIMEOUT = 3
PROVIDER = 4
INTERNAL = 5

OPERATION_ROUTED_MOE = 1
FORMAT_ROUTED_F32 = 1
FORMAT_PARTIAL_F32 = 2
FORMAT_ROUTED_F16 = 3
FORMAT_ROUTED_BF16 = 4

LOG_DEBUG = 0
LOG_INFO = 1
LOG_WARN = 2
LOG_ERROR = 3
LOG_OFF = 4


class SourceModality(IntFlag):
    """Origins of encoded activations already routed by the engine."""

    TEXT = 1 << 0
    IMAGE = 1 << 1
    AUDIO = 1 << 2
    VIDEO = 1 << 3
    OTHER = 1 << 4


ALL_SOURCE_MODALITIES = (
    SourceModality.TEXT | SourceModality.IMAGE | SourceModality.AUDIO
    | SourceModality.VIDEO | SourceModality.OTHER
)
