# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Pins the two parameters that let ``_probe`` be the single owner of the probe
helpers.  Each guards a behaviour that two modules used to keep in private
copies, so a regression here is the drift coming back:

* ``strip`` -- ``_detect_host`` feeds raw file text to line-anchored parsers.
* ``cwd`` / ``timeout`` -- ``_detect_toolkit`` compiles in a throwaway directory
  and gives that compile 90s where a ``--version`` query gets 20s.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from install import _detect_host as dh  # noqa: E402
from install import _detect_toolkit as dt  # noqa: E402
from install import _probe  # noqa: E402


class TestProbeRead(unittest.TestCase):
    def _file(self, tmp: str, text: str) -> str:
        path = os.path.join(tmp, "f")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        return path

    def test_default_strips(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_probe._read(self._file(tmp, "  x \n")), "x")

    def test_strip_false_keeps_the_file_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_probe._read(self._file(tmp, "  x \n"), strip=False), "  x \n")

    def test_unreadable_is_none_either_way(self):
        gone = os.path.join(tempfile.gettempdir(), "seabraid-no-such-file-xyz")
        self.assertIsNone(_probe._read(gone))
        self.assertIsNone(_probe._read(gone, strip=False))

    def test_why_the_host_needs_raw_text(self):
        """parse_os_release anchors on ^PRETTY_NAME, so stripping a leading space
        changes the answer -- the drift is real, not cosmetic."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._file(tmp, ' PRETTY_NAME="Odd OS"\n')
            self.assertIsNone(dh.parse_os_release(_probe._read(path, strip=False)))
            self.assertEqual(dh.parse_os_release(_probe._read(path)), "Odd OS")


class TestHostReadsParserInputRaw(unittest.TestCase):
    def test_parser_fed_files_are_read_with_strip_false(self):
        seen: dict[str, bool] = {}

        def recorder(path, strip=True):
            seen[path] = strip
            return ""

        real, dh._read = dh._read, recorder
        try:
            dh.probe_filesystems([])
            dh.probe_kernel()
            try:
                dh.probe_ram()
            except Exception:  # noqa: BLE001 -- "" is refused; only the read matters here
                pass
        finally:
            dh._read = real
        for path in ("/proc/meminfo", "/proc/mounts", "/etc/os-release"):
            self.assertIn(path, seen)
            self.assertIs(seen[path], False, path + " must be read raw")


class TestProbeRun(unittest.TestCase):
    def test_cwd_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _probe._run(
                [sys.executable, "-c", "import os; print(os.getcwd())"], cwd=tmp
            )
            self.assertEqual(os.path.realpath(out.strip()), os.path.realpath(tmp))

    def test_nonzero_exit_is_none(self):
        self.assertIsNone(_probe._run([sys.executable, "-c", "raise SystemExit(3)"]))


class TestToolkitCompileBudget(unittest.TestCase):
    def test_compile_gets_90s_in_a_throwaway_directory(self):
        calls: list[dict] = []

        def recorder(cmd, timeout=20.0, cwd=None):
            calls.append({"timeout": timeout, "cwd": cwd})
            return ""

        real, dt._run = dt._run, recorder
        try:
            cc, tried = dt.verify_host_compiler("nvcc", ["gcc-13"])
        finally:
            dt._run = real
        self.assertEqual((cc, tried), ("gcc-13", ("gcc-13",)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["timeout"], 90.0)
        self.assertIsNotNone(calls[0]["cwd"])


if __name__ == "__main__":
    unittest.main()
