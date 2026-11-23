# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""examples/quickstart.py runs end to end, pure and (with SEABRAID_LIBRARY) native, with the same code.

Checker for the quickstart contract: Runtime(reference=True) is the one call that works on both paths, so a documented
example can never again fail on the native path (a bare Runtime() with the variable set is refused by design).
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QUICKSTART = ROOT / "examples" / "quickstart.py"
PACKAGE_PARENT = ROOT / "bindings" / "python"
LIBRARY = os.environ.get("SEABRAID_LIBRARY")


def run_quickstart(library: str | None) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if key != "SEABRAID_LIBRARY"}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PACKAGE_PARENT), env.get("PYTHONPATH")]))
    if library is not None:
        env["SEABRAID_LIBRARY"] = library
    return subprocess.run([sys.executable, "-B", str(QUICKSTART)], env=env, capture_output=True, text=True,
                          timeout=120, check=False)


@unittest.skipUnless(QUICKSTART.is_file(), "examples/ is not shipped with this install")
class QuickstartTests(unittest.TestCase):
    def check(self, done: subprocess.CompletedProcess, provider: str) -> None:
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(f"provider={provider} result=[2.5, 5.0]", done.stdout)
        self.assertIn("stale epoch refused: status 2", done.stdout)
        self.assertIn("card order: RX 7900 XTX > RTX 3060", done.stdout)
        self.assertNotIn("Warning", done.stderr)  # reference=True is the explicit request: no notice on either path

    def test_pure_python_without_a_native_library(self) -> None:
        self.check(run_quickstart(None), "reference")

    @unittest.skipUnless(LIBRARY, "set SEABRAID_LIBRARY for the native run")
    def test_native_with_seabraid_library_set(self) -> None:
        self.check(run_quickstart(LIBRARY), "native")


if __name__ == "__main__":
    unittest.main()
