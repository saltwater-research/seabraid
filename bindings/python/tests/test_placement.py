# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid.placement.fill: marshalling and validation against a mocked native library.

Checker for seabraid.placement. The native fill function is another change, so only its declared contract is used here.
"""

from __future__ import annotations

import ctypes as ct
import os
import unittest
from unittest.mock import Mock, patch

from seabraid import SeabraidError, _abi, placement

BASE = ["seabraid_open", "seabraid_prepare", "seabraid_submit", "seabraid_collect", "seabraid_close"]


class PlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.library = Mock(spec=BASE + ["seabraid_placement_fill"])
        self.library.seabraid_placement_fill.side_effect = self._fill
        self.seen: dict[str, object] = {}
        patcher = patch.object(_abi, "_active", self.library)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = patch.dict(os.environ)
        env.start()
        os.environ.pop("SEABRAID_LIBRARY", None)
        self.addCleanup(env.stop)

    def _fill(self, request: object, owners: ct.Array, count: int) -> int:
        view = ct.cast(request, ct.POINTER(_abi._PlacementFillV1)).contents
        self.seen = {
            "head": (view.abi_version, view.layers, view.experts, view.cards, view.slot_count, view.heat_count),
            "slots": view.slots[: view.slot_count], "heat": view.heat[: view.heat_count], "count": count,
        }
        for index in range(count):
            owners[index] = index % (view.cards + 1)
        return 0

    def test_request_is_marshalled_and_owners_returned_layer_major(self) -> None:
        owners = placement.fill(2, 3, [4, 2], [0.5, 0.25, 0.0, 9.0, 1.0, 2.0])
        self.assertEqual(owners, [0, 1, 2, 0, 1, 2])
        self.assertEqual(self.seen["head"], (1, 2, 3, 2, 2, 6))
        self.assertEqual((self.seen["slots"], self.seen["heat"], self.seen["count"]),
                         ([4, 2], [0.5, 0.25, 0.0, 9.0, 1.0, 2.0], 6))

    def test_shapes_are_validated_before_any_native_call(self) -> None:
        bad = {
            "layers": (0, 3, [1], [0.0] * 3), "heat length": (1, 3, [1], [0.0] * 2),
            "no cards": (1, 3, [], [0.0] * 3), "256 cards": (1, 3, [1] * 256, [0.0] * 3),
            "negative slots": (1, 3, [-1], [0.0] * 3), "nan heat": (1, 1, [1], [float("nan")]),
            "inf heat": (1, 1, [1], [float("inf")]),
        }
        for name, args in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                placement.fill(*args)
        self.library.seabraid_placement_fill.assert_not_called()

    def test_more_than_the_native_cell_limit_is_refused_before_heat_is_read(self) -> None:
        """R7 LOW 13: Python did not check layers * experts <= 2^26 (SEABRAID_PLACEMENT_MAX_CELLS) like native."""

        class Untouchable:
            def __len__(self) -> int:
                return (1 << 26) + 1

            def __iter__(self):
                raise AssertionError("heat was read although the request is over the cell limit")

        with self.assertRaisesRegex(ValueError, "at most 67108864"):
            placement.fill(1, (1 << 26) + 1, [1], Untouchable())
        with self.assertRaisesRegex(ValueError, "at most 67108864"):
            placement.fill(1 << 13, (1 << 13) + 1, [1], Untouchable())  # the product, not either factor
        self.library.seabraid_placement_fill.assert_not_called()

    def test_the_cell_limit_is_inclusive(self) -> None:
        with patch.object(placement, "_MAX_CELLS", 6):
            self.assertEqual(len(placement.fill(2, 3, [4], [0.0] * 6)), 6)
            with self.assertRaises(ValueError):
                placement.fill(2, 4, [4], [0.0] * 8)

    def test_missing_function_and_missing_library_are_clear_errors(self) -> None:
        del self.library.seabraid_placement_fill
        with self.assertRaisesRegex(RuntimeError, "does not expose placement fill"):
            placement.fill(1, 1, [1], [0.0])
        with patch.object(_abi, "_active", None), self.assertRaisesRegex(RuntimeError, "needs a native Seabraid library"):
            placement.fill(1, 1, [1], [0.0])

    def test_native_status_raises_seabraid_error(self) -> None:
        self.library.seabraid_placement_fill.side_effect = None
        self.library.seabraid_placement_fill.return_value = 1
        with self.assertRaisesRegex(SeabraidError, r"^seabraid_placement_fill: invalid argument \(status 1\)$") as raised:
            placement.fill(1, 1, [1], [0.0])
        self.assertEqual(raised.exception.status, 1)
