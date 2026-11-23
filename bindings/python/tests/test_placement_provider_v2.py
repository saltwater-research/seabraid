# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Python parity for placement provider v2: the constant equals placement.h and the ctypes struct equals the layout table.

Checker for seabraid.constants.PLACEMENT_PROVIDER_V2_VERSION and seabraid._abi._PlacementProviderV2 (the header is the
source of truth), and for the by-reference mask contract documented in the header comment.
"""

from __future__ import annotations

import ctypes as ct
import re
import unittest
from pathlib import Path

from seabraid import _abi
from seabraid import constants as c

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "include" / "seabraid" / "placement.h"
TABLE = ROOT / "src" / "layout" / "abi_layout.txt"


def _header_text() -> str:
    return re.sub(r"/\*.*?\*/", "", HEADER.read_text(encoding="utf-8"), flags=re.S)


@unittest.skipUnless(HEADER.is_file(), "include/seabraid/placement.h is not shipped with this install")
class PlacementProviderV2Parity(unittest.TestCase):
    def test_version_define_equals_the_python_constant(self) -> None:
        match = re.search(r"#define\s+SEABRAID_PLACEMENT_PROVIDER_V2_VERSION\s+(\d+)u?", _header_text())
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), c.PLACEMENT_PROVIDER_V2_VERSION)

    def test_the_header_declares_the_struct_and_the_entry_with_the_documented_fields(self) -> None:
        text = _header_text()
        body = re.search(r"typedef struct seabraid_placement_provider_v2 \{(.*?)\} seabraid_placement_provider_v2;", text, re.S)
        self.assertIsNotNone(body)
        names = re.findall(r"(\w+)\)?\s*(?:\([^)]*\))?;", body.group(1))
        for name in ("abi_version", "struct_size", "from_fallback_rungs", "replace_from_fallback"):
            self.assertIn(name, names)
        self.assertRegex(text, r"int32_t\s+seabraid_placement_apply_v2\s*\(")

    @unittest.skipUnless(TABLE.is_file() and ct.sizeof(ct.c_void_p) == 8, "the layout table records 64-bit layouts")
    def test_ctypes_struct_equals_the_layout_table(self) -> None:
        rows = dict(line.split() for line in TABLE.read_text(encoding="utf-8").splitlines()
                    if line.startswith("seabraid_placement_provider_v2"))
        cls = _abi._PlacementProviderV2
        self.assertEqual(ct.sizeof(cls), int(rows["seabraid_placement_provider_v2"]))
        for name, _ in cls._fields_:
            self.assertEqual(getattr(cls, name).offset, int(rows[f"seabraid_placement_provider_v2.{name}"]), name)
        self.assertEqual(len(rows), len(cls._fields_) + 1)
        self.assertEqual(ct.sizeof(cls), 64)

    def test_the_struct_round_trips_a_mask_and_the_by_reference_callback(self) -> None:
        calls: list[tuple[int, int, int, int]] = []

        def by_reference(_context, rung, layer, old, new):
            calls.append((rung, layer, old, new))
            return 0

        callback = _abi.PlacementReplaceFromFallback(by_reference)
        provider = _abi._PlacementProviderV2(
            abi_version=c.PLACEMENT_PROVIDER_V2_VERSION, struct_size=ct.sizeof(_abi._PlacementProviderV2),
            from_fallback_rungs=0b110, replace_from_fallback=callback,
        )
        self.assertEqual(provider.from_fallback_rungs, 6)
        self.assertEqual(provider.replace_from_fallback(None, 1, 2, 3, 4), 0)
        self.assertEqual(calls, [(1, 2, 3, 4)])


if __name__ == "__main__":
    unittest.main()
