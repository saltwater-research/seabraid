# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""find_prebuilt's return type: a NamedTuple that keeps the older positional and tuple call shapes working.

Split from test_setup_cli.py to keep each file under the 500-line limit.
Run:  python3 -m pytest tests/install/test_setup_prebuilt.py -q
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from install import seabraid_setup as CLI  # noqa: E402


def touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("x")


class TestPrebuiltIsANamedTupleOnPurpose(unittest.TestCase):
    """find_prebuilt's return must carry NAMES without breaking the three older call shapes.

    The bench stage reads it positionally (`find_prebuilt(work)[1]`), _finish unpacks it, and a
    sibling test compares it to a plain tuple. The names were added for the bench stage's benefit,
    so this pins that adding them cost nothing: swapping the NamedTuple for a dataclass satisfies
    the named access alone and silently breaks the other three.
    """

    def _one(self, tmp):
        d = os.path.join(tmp, "strata", "build-nvidia", "seabraid")
        os.makedirs(d)
        touch(os.path.join(tmp, "strata", "build-nvidia", "strata"))
        touch(os.path.join(d, "seabraid-helper"))
        return CLI.find_prebuilt(tmp)

    def test_named_access_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = self._one(tmp)
        self.assertTrue(got.engine.endswith("strata"))
        self.assertIn("nvidia", got.helper_exes)

    def test_positional_access_still_works_for_the_bench_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = self._one(tmp)
        self.assertEqual(got.engine, got[0])
        self.assertEqual(got.helper_exes, got[1])

    def test_tuple_unpacking_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, helpers = self._one(tmp)
        self.assertTrue(engine.endswith("strata"))
        self.assertIn("nvidia", helpers)

    def test_it_still_compares_equal_to_a_plain_tuple(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(("", {}), CLI.find_prebuilt(tmp))


if __name__ == "__main__":
    unittest.main()
