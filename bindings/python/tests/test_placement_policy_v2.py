# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Python parity for placement policy v2: the constants equal placement.h and the ctypes struct equals the layout table.

Checker for seabraid.constants.PLACEMENT_* and seabraid._abi._PlacementPolicyV2 (the header is the source of truth).
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
class PlacementPolicyV2Parity(unittest.TestCase):
    def test_version_define_equals_the_python_constant(self) -> None:
        match = re.search(r"#define\s+SEABRAID_PLACEMENT_POLICY_V2_VERSION\s+(\d+)u?", _header_text())
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), c.PLACEMENT_POLICY_V2_VERSION)

    def test_outcome_kind_enum_equals_the_python_constant(self) -> None:
        body = re.search(r"enum\s+seabraid_placement_outcome_kind\s*\{(.*?)\}", _header_text(), re.S).group(1)
        kinds = {name: int(value) for name, value in re.findall(r"(SEABRAID_PLACEMENT_\w+)\s*=\s*(\d+)", body)}
        self.assertEqual(kinds["SEABRAID_PLACEMENT_SKIPPED"], c.PLACEMENT_SKIPPED)
        self.assertEqual(sorted(kinds.values()), [1, 2, 3, 4, 5], "append only: 1..4 are v1")

    @unittest.skipUnless(TABLE.is_file() and ct.sizeof(ct.c_void_p) == 8, "the layout table records 64-bit layouts")
    def test_ctypes_struct_equals_the_layout_table(self) -> None:
        rows = dict(line.split() for line in TABLE.read_text(encoding="utf-8").splitlines()
                    if line.startswith("seabraid_placement_policy_v2"))
        cls = _abi._PlacementPolicyV2
        self.assertEqual(ct.sizeof(cls), int(rows["seabraid_placement_policy_v2"]))
        for name, _ in cls._fields_:
            self.assertEqual(getattr(cls, name).offset, int(rows[f"seabraid_placement_policy_v2.{name}"]), name)
        self.assertEqual(len(rows), len(cls._fields_) + 1)
        self.assertEqual(ct.sizeof(cls), 40)


if __name__ == "__main__":
    unittest.main()
