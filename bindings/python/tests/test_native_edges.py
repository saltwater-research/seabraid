# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""placement.fill and enable_logging against the real native library (the other tests of both use mocks).

Checker for seabraid.placement and seabraid.diagnostics marshalling: the struct layout, the callback type and the
status mapping only prove themselves when the engine itself reads and calls them. Needs SEABRAID_LIBRARY.
"""

from __future__ import annotations

import os
import unittest

from seabraid import SeabraidError, _abi, disable_logging, enable_logging, placement
from seabraid.constants import INVALID, LOG_WARN

LIBRARY = os.environ.get("SEABRAID_LIBRARY")


@unittest.skipUnless(LIBRARY, "set SEABRAID_LIBRARY for native integration")
class NativePlacementFillTests(unittest.TestCase):
    def test_a_2x3_fill_gives_the_hottest_cells_to_the_first_card_layer_major(self) -> None:
        # Heat by (layer, expert): [5, 1, 4 | 0.5, 3, 2]; two slots on card 0, one on card 1, the rest fall back (rung 2).
        owners = placement.fill(2, 3, [2, 1], [5.0, 1.0, 4.0, 0.5, 3.0, 2.0], library_path=LIBRARY)
        self.assertEqual(owners, [0, 2, 0, 2, 1, 2])

    def test_every_cell_falls_back_when_no_card_has_room(self) -> None:
        self.assertEqual(placement.fill(1, 3, [0], [1.0, 2.0, 3.0], library_path=LIBRARY), [1, 1, 1])

    def test_a_native_refusal_becomes_a_seabraid_error_with_the_native_text(self) -> None:
        with self.assertRaises(SeabraidError) as raised:
            placement.fill(1, 2, [1], [1.0, -1.0], library_path=LIBRARY)  # negative heat passes Python, native refuses
        self.assertEqual((raised.exception.operation, raised.exception.status),
                         ("seabraid_placement_fill", INVALID))
        self.assertTrue(raised.exception.detail)


@unittest.skipUnless(LIBRARY, "set SEABRAID_LIBRARY for native integration")
class NativeLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(disable_logging)

    def test_a_native_record_arrives_on_the_seabraid_logger_with_its_status(self) -> None:
        self.assertTrue(enable_logging("WARNING", library_path=LIBRARY))
        library = _abi.resolve_library(LIBRARY)
        with self.assertLogs("seabraid", "WARNING") as logs:
            self.assertEqual(library.seabraid_log(LOG_WARN, 4, b"native-edge-test record"), 0)
        record = next(r for r in logs.records if r.getMessage() == "native-edge-test record")
        self.assertEqual(record.seabraid_status, 4)
        self.assertEqual(record.levelname, "WARNING")

    def test_a_record_below_the_threshold_is_not_forwarded(self) -> None:
        self.assertTrue(enable_logging("ERROR", library_path=LIBRARY))
        library = _abi.resolve_library(LIBRARY)
        with self.assertNoLogs("seabraid", "DEBUG"):
            library.seabraid_log(LOG_WARN, 0, b"native-edge-test quiet")


if __name__ == "__main__":
    unittest.main()
