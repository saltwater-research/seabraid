# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""ctypes struct layouts equal src/layout/abi_layout.txt, the one table the C compiler and rustc also check.

Checker for the struct declarations in seabraid._abi. 64-bit only: the table records 64-bit layouts.
"""

from __future__ import annotations

import ctypes as ct
import unittest
from pathlib import Path

from seabraid import _abi

TABLE = Path(__file__).resolve().parents[3] / "src" / "layout" / "abi_layout.txt"
MIRRORS = {
    "seabraid_capabilities_v1": _abi._CapabilitiesV1,
    "seabraid_session_status_v1": _abi._SessionStatusV1,
    "seabraid_config_v1": _abi._ConfigV1,
    "seabraid_batch_v1": _abi._BatchV1,
    "seabraid_provider_v2": _abi._ProviderV2,
    "seabraid_batch_v2": _abi._BatchV2,
    "seabraid_provider_v3": _abi._ProviderV3,
    "seabraid_placement_fill_v1": _abi._PlacementFillV1,
}
LIBRARY_FILLED = ("capabilities_v1", "status_v1", "ticket_v1", "outcome_v1", "applied_v1", "snapshot_v1")


def read_table() -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    sizes: dict[str, int] = {}
    offsets: dict[str, dict[str, int]] = {}
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, number = line.split()
        if "." in key:
            struct, field = key.split(".")
            offsets.setdefault(struct, {})[field] = int(number)
        else:
            sizes[key] = int(number)
    return sizes, offsets


@unittest.skipUnless(TABLE.is_file(), "src/layout/abi_layout.txt is not shipped with this install")
@unittest.skipUnless(ct.sizeof(ct.c_void_p) == 8, "the layout table records 64-bit layouts")
class LayoutTests(unittest.TestCase):
    def test_every_mirrored_struct_matches_the_table(self) -> None:
        sizes, offsets = read_table()
        for name, cls in MIRRORS.items():
            with self.subTest(name):
                self.assertEqual(ct.sizeof(cls), sizes[name])
                fields = [field[0] for field in cls._fields_]
                self.assertEqual(sorted(fields), sorted(offsets[name]), "field sets differ")
                for field in fields:
                    self.assertEqual(getattr(cls, field).offset, offsets[name][field], field)

    def test_library_filled_structs_lead_with_version_and_size(self) -> None:
        _, offsets = read_table()
        for name, fields in offsets.items():
            with self.subTest(name):
                if name.endswith(LIBRARY_FILLED):
                    self.assertEqual((fields["abi_version"], fields["struct_size"]), (0, 4))
