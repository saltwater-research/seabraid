# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Stage 1, PCI id database lookups, parsed against recorded output.

Split from ``test_detect_parsers.py`` to keep both files under the size cap.

Run:  python3 -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from install import _detect_pci as dpci  # noqa: E402


class TestPciIds(unittest.TestCase):
    """Parsed from the real pci.ids format, including the line that actually names the test system's card."""

    SAMPLE = (
        "# comment\n"
        "1002  Advanced Micro Devices, Inc. [AMD/ATI]\n"
        "\t7550  Navi 48\n"
        "\t7551  Navi 48 [Radeon AI PRO R9700]\n"
        "\t\t7551  a subsystem that repeats the device id\n"
        "\t\t9999  a subsystem id that is not a device at all\n"
        "10de  NVIDIA Corporation\n"
        "\t2504  GA106 [GeForce RTX 3060]\n"
    )

    def test_finds_the_reference_cards_name(self):
        self.assertEqual("Navi 48 [Radeon AI PRO R9700]",
                         dpci.pci_ids_name(self.SAMPLE, 0x1002, 0x7551))

    def test_a_subsystem_id_is_never_returned_as_a_device(self):
        """Subsystem entries sit under two tabs and their ids live in a different namespace -- 9999 below
        is a subsystem of 7551 and not a card at all.

        The first version of this test asserted that looking up 7551 did not return the SUBSYSTEM text,
        which could not fail: the device line precedes its own subsystems, so the function returns before
        ever reaching one.  An id that exists ONLY as a subsystem is the case that distinguishes them.
        """
        self.assertIsNone(dpci.pci_ids_name(self.SAMPLE, 0x1002, 0x9999))

    def test_a_device_still_wins_over_a_subsystem_repeating_its_id(self):
        self.assertEqual("Navi 48 [Radeon AI PRO R9700]", dpci.pci_ids_name(self.SAMPLE, 0x1002, 0x7551))

    def test_the_device_is_looked_up_under_its_own_vendor(self):
        self.assertEqual("GA106 [GeForce RTX 3060]", dpci.pci_ids_name(self.SAMPLE, 0x10de, 0x2504))
        self.assertIsNone(dpci.pci_ids_name(self.SAMPLE, 0x1002, 0x2504),
                          "a device id must not match under the wrong vendor")

    def test_an_unknown_id_is_none_not_a_guess(self):
        self.assertIsNone(dpci.pci_ids_name(self.SAMPLE, 0x1002, 0xffff))
        self.assertIsNone(dpci.pci_ids_name("", 0x1002, 0x7551))

    def test_a_missing_database_is_not_an_error(self):
        self.assertIsNone(dpci.pci_device_name(0x1002, 0x7551, paths=("/nonexistent/pci.ids",)))

if __name__ == "__main__":
    unittest.main()
