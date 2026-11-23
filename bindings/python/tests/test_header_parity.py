# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The C header is the ABI source of truth: Python constants must equal it, and new header names must be mirrored.

Checker for seabraid.constants and seabraid.errors.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from seabraid import constants as c
from seabraid import placement
from seabraid.errors import _STATUS_NAMES

HEADER = Path(__file__).resolve().parents[3] / "include" / "seabraid" / "seabraid.h"
# header name -> the Python value that must equal it. Enum names are listed with their enum so completeness is checked.
DEFINES = {
    "SEABRAID_ABI_EPOCH": c.ABI_EPOCH,
    "SEABRAID_ABI_VERSION": c.ABI_VERSION,
    "SEABRAID_MULTI_PROVIDER_VERSION": c.MULTI_PROVIDER_VERSION,
    "SEABRAID_PROVIDER_PARTIAL_ROUTES": c.PROVIDER_PARTIAL_ROUTES,
    "SEABRAID_SKIP_EXPERT_ID": c.SKIP_EXPERT_ID,
    "SEABRAID_BATCH_V2_VERSION": c.BATCH_V2_VERSION,
    "SEABRAID_PROVIDER_V3_VERSION": c.PROVIDER_V3_VERSION,
    "SEABRAID_PROVIDER_COMPACT_ROWS": c.PROVIDER_COMPACT_ROWS,
    "SEABRAID_PROVIDER_ACCEPTS_F16": c.PROVIDER_ACCEPTS_F16,
    "SEABRAID_PROVIDER_ACCEPTS_BF16": c.PROVIDER_ACCEPTS_BF16,
    "SEABRAID_PROVIDER_REFERENCE": c.PROVIDER_REFERENCE,
}
PLACEMENT_HEADER = HEADER.with_name("placement.h")
PLACEMENT_DEFINES = {"SEABRAID_PLACEMENT_MAX_CARDS": placement._MAX_CARDS,
                     "SEABRAID_PLACEMENT_MAX_CELLS": placement._MAX_CELLS,
                     # WHY a literal here: the Python binding does not expose seabraid_placement_apply, so it has no
                     # such constant of its own; this pins the number every other binding author reads in the header.
                     "SEABRAID_PLACEMENT_MAX_SLOT_BYTES": (1 << 32) - 1}
ENUMS = {
    "seabraid_status": {
        "SEABRAID_OK": c.OK, "SEABRAID_INVALID": c.INVALID, "SEABRAID_BUSY_OR_STATE": c.BUSY_OR_STATE,
        "SEABRAID_TIMEOUT": c.TIMEOUT, "SEABRAID_PROVIDER": c.PROVIDER, "SEABRAID_INTERNAL": c.INTERNAL,
    },
    "seabraid_source_modality": {
        "SEABRAID_SOURCE_TEXT": c.SourceModality.TEXT, "SEABRAID_SOURCE_IMAGE": c.SourceModality.IMAGE,
        "SEABRAID_SOURCE_AUDIO": c.SourceModality.AUDIO, "SEABRAID_SOURCE_VIDEO": c.SourceModality.VIDEO,
        "SEABRAID_SOURCE_OTHER": c.SourceModality.OTHER,
    },
    "seabraid_operation": {"SEABRAID_OPERATION_ROUTED_MOE": c.OPERATION_ROUTED_MOE},
    "seabraid_buffer_format": {
        "SEABRAID_FORMAT_ROUTED_F32": c.FORMAT_ROUTED_F32, "SEABRAID_FORMAT_PARTIAL_F32": c.FORMAT_PARTIAL_F32,
        "SEABRAID_FORMAT_ROUTED_F16": c.FORMAT_ROUTED_F16, "SEABRAID_FORMAT_ROUTED_BF16": c.FORMAT_ROUTED_BF16,
    },
}
LOG_LEVELS = {
    "SEABRAID_LOG_DEBUG": c.LOG_DEBUG, "SEABRAID_LOG_INFO": c.LOG_INFO, "SEABRAID_LOG_WARN": c.LOG_WARN,
    "SEABRAID_LOG_ERROR": c.LOG_ERROR, "SEABRAID_LOG_OFF": c.LOG_OFF,
}
_EXPRESSION = re.compile(r"^[0-9u<>()|\s-]+$")


def _value(expression: str) -> int:
    text = re.sub(r"(?<=\d)u\b", "", expression.strip())
    if not _EXPRESSION.match(text):
        raise AssertionError(f"unsupported header expression: {expression!r}")
    return eval(text, {"__builtins__": {}})  # noqa: S307 - characters restricted to digits and shift/or operators


def _enum_values(text: str, name: str) -> dict[str, int]:
    match = re.search(rf"enum\s+{name}\s*\{{(.*?)\}}", text, re.S)
    if match is None:
        return {}
    values, counter = {}, 0
    for item in (part.strip() for part in match.group(1).split(",")):
        if not item:
            continue
        key, _, expression = (piece.strip() for piece in item.partition("="))
        counter = _value(expression) if expression else counter
        values[key] = counter
        counter += 1
    return values


class HeaderParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = re.sub(r"/\*.*?\*/", "", HEADER.read_text(encoding="utf-8"), flags=re.S)

    def test_defines_match(self) -> None:
        for name, expected in DEFINES.items():
            with self.subTest(name):
                match = re.search(rf"#define\s+{name}\s+(.+)", self.text)
                self.assertIsNotNone(match, f"{name} is not in the header")
                self.assertEqual(_value(match.group(1)), expected)

    def test_placement_limits_match(self) -> None:
        text = re.sub(r"/\*.*?\*/", "", PLACEMENT_HEADER.read_text(encoding="utf-8"), flags=re.S)
        for name, expected in PLACEMENT_DEFINES.items():
            with self.subTest(name):
                match = re.search(rf"#define\s+{name}\s+(.+)", text)
                self.assertIsNotNone(match, f"{name} is not in the placement header")
                self.assertEqual(_value(match.group(1)), expected)

    def test_enums_match_and_are_complete(self) -> None:
        for enum, mirrored in ENUMS.items():
            with self.subTest(enum):
                header = _enum_values(self.text, enum)
                self.assertEqual(header, {name: int(value) for name, value in mirrored.items()})

    def test_log_levels_match(self) -> None:
        header = _enum_values(self.text, "seabraid_log_level")
        if not header:
            # The diagnostics ABI lands in the header from another change; skip loudly until it is there.
            self.skipTest("enum seabraid_log_level is not in this header yet")
        self.assertEqual(header, LOG_LEVELS)

    def test_status_names_cover_every_failure_status(self) -> None:
        failures = {value for name, value in _enum_values(self.text, "seabraid_status").items() if name != "SEABRAID_OK"}
        self.assertEqual(set(_STATUS_NAMES), failures)
