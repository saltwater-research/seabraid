# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The public import surface, including the names the compat add-on installer imports from seabraid.ranking.

Checker for seabraid/__init__.py and seabraid/ranking.py's import contract.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import seabraid
from seabraid import _abi, ranking


class PublicApiTests(unittest.TestCase):
    def test_exported_names_exist(self) -> None:
        for name in seabraid.__all__:
            with self.subTest(name):
                self.assertTrue(hasattr(seabraid, name))

    def test_bundled_library_is_public_and_returns_a_path_or_none(self) -> None:
        self.assertIn("bundled_library", seabraid.__all__)
        self.assertIsInstance(seabraid.bundled_library(), (str, type(None)))

    def test_version_is_a_nonempty_string(self) -> None:
        self.assertRegex(seabraid.__version__, r"^\d")

    def test_installer_import_contract_is_unchanged(self) -> None:
        from seabraid.ranking import LEAD_NOT_RUN, Slot, admit, lead_class, rank, rank_note  # noqa: F401
        for name in ("Terms", "layer_cost", "bandwidth_gbs"):
            self.assertTrue(hasattr(ranking, name), name)

    def test_the_types_a_provider_callable_receives_are_public(self) -> None:
        """R7 NIT: Shape and BatchInfo are documented as the callable's arguments, so they must be importable."""
        self.assertEqual(seabraid.Shape(2, 3, 2, 2).hidden, 2)
        self.assertEqual(seabraid.BatchInfo(1, 0).placement_epoch, 0)
        self.assertIn("Shape", seabraid.__all__)
        self.assertIn("BatchInfo", seabraid.__all__)

    def test_one_owner_for_the_library_environment_rule(self) -> None:
        """R7 NIT: the SEABRAID_LIBRARY lookup was written in runtime.py and in _abi.py; only _abi may read it."""
        package = Path(seabraid.__file__).parent
        readers = [p.name for p in package.glob("*.py") if 'environ.get("SEABRAID_LIBRARY")' in p.read_text(encoding="utf-8")]
        self.assertEqual(readers, ["_abi.py"])
        with patch.dict(os.environ, {"SEABRAID_LIBRARY": "from-env"}):
            self.assertEqual(_abi.configured_path(None), "from-env")
            self.assertEqual(_abi.configured_path("explicit"), "explicit")
        with patch.dict(os.environ, {"SEABRAID_LIBRARY": ""}):
            self.assertIsNone(_abi.configured_path(None))

    def test_old_flat_modules_are_gone(self) -> None:
        for name in ("seabraid_runtime", "_seabraid_abi"):
            with self.subTest(name):
                self.assertIsNone(importlib.util.find_spec(name))
