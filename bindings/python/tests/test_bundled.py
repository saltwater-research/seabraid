# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for seabraid/_bundled.py: None without a shipped library, the path when the platform's file is present."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import seabraid
from seabraid import _bundled


class BundledLibraryTests(unittest.TestCase):
    def test_none_when_the_package_carries_no_library(self) -> None:
        with TemporaryDirectory() as root, patch.object(_bundled, "_PACKAGE_DIR", Path(root)):
            self.assertIsNone(_bundled.bundled_library())

    def test_path_when_the_platform_file_exists(self) -> None:
        with TemporaryDirectory() as root, patch.object(_bundled, "_PACKAGE_DIR", Path(root)):
            lib_dir = Path(root) / "_lib"
            lib_dir.mkdir()
            (lib_dir / _bundled._library_name()).write_bytes(b"not a real library")
            self.assertEqual(_bundled.bundled_library(), str(lib_dir / _bundled._library_name()))

    def test_name_matches_the_platform(self) -> None:
        expected = {"win32": "seabraid.dll", "darwin": "libseabraid.dylib"}.get(sys.platform, "libseabraid.so")
        self.assertEqual(_bundled._library_name(), expected)

    def test_exported_from_the_package(self) -> None:
        self.assertIs(seabraid.bundled_library, _bundled.bundled_library)
        self.assertIn("bundled_library", seabraid.__all__)
