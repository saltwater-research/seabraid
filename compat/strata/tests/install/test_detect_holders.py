# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Stage 1, GPU holders: who occupies a card's VRAM, parsed against recorded output.

Split from ``test_detect_parsers.py`` to keep both files under the size cap; the fixtures and the
provenance note in that file's docstring apply here too.

Run:  python3 -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
sys.path.insert(0, REPO)

from install import _detect_holders as dhold  # noqa: E402
from install import _detect_gpu as dg  # noqa: E402
from install import detect  # noqa: E402

GIB = 1024 ** 3


def fixture(name: str) -> str:
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as fh:
        return fh.read()


class TestHolderTuple(unittest.TestCase):
    def test_the_biggest_holder_comes_first_because_the_report_truncates(self):
        got = dhold._holder_tuple({7: (100, "small"), 9: (900, "big"), 8: (500, "mid")})
        self.assertEqual(("big", "mid", "small"), tuple(n for _, _, n in got))


class TestComputeApps(unittest.TestCase):
    """``--query-compute-apps=gpu_uuid,pid,used_memory,process_name``, in the shape smi prints it."""

    SAMPLE = (
        "GPU-aaaa, 1234, 512 MiB, python3\n"
        "GPU-aaaa, 1235, 2048 MiB, /usr/bin/strata --arg, with-comma\n"
        "GPU-bbbb, 9999, 64 MiB, seabraid-helper\n"
        "GPU-bbbb, 9998, [N/A], memory-not-reported\n"
    )

    def test_each_process_is_attributed_to_its_own_card(self):
        got = dhold.parse_compute_apps(self.SAMPLE)
        self.assertEqual({1234, 1235}, set(got["GPU-aaaa"]))
        self.assertEqual({9999}, set(got["GPU-bbbb"]))

    def test_mib_becomes_bytes(self):
        self.assertEqual(512 * 1024 * 1024, dhold.parse_compute_apps(self.SAMPLE)["GPU-aaaa"][1234][0])

    def test_a_process_name_containing_a_comma_is_not_truncated(self):
        """The format is CSV and a command line may hold commas, so only the first three fields are
        positional -- splitting on every comma would silently rename the process."""
        self.assertEqual("/usr/bin/strata --arg, with-comma",
                         dhold.parse_compute_apps(self.SAMPLE)["GPU-aaaa"][1235][1])

    def test_a_row_with_no_memory_figure_is_dropped_not_counted_as_zero(self):
        self.assertNotIn(9998, dhold.parse_compute_apps(self.SAMPLE)["GPU-bbbb"])

    def test_no_processes_is_empty_and_not_an_error(self):
        self.assertEqual({}, dhold.parse_compute_apps(""))
        self.assertEqual({}, dhold.parse_compute_apps("\n\n"))


class TestNvidiaHoldersAreNotShared(unittest.TestCase):
    """Two NVIDIA cards, one busy: the idle card must not inherit the busy card's process.

    This is the discriminating case.  The first version of this probe attached the machine-wide
    compute-apps list to every row, so an idle card reported 'in use by' another card's process --
    the same wrong-card attribution that already made the AMD card answer for the NVIDIA helper.
    The second card here is the recorded card with only its index and UUID changed.
    """

    APPS = "GPU-bbbb, 4242, 9000 MiB, soak_load.py\n"

    @staticmethod
    def _two_cards():
        line = fixture("sample-nvidia-smi.csv").strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        second = list(parts)
        second[0], second[-1] = "1", "GPU-bbbb"
        return parts[-1], line + "\n" + ", ".join(second) + "\n"

    def _rows(self):
        recorded_uuid, cards = self._two_cards()
        self.assertTrue(recorded_uuid.startswith("GPU-"), "the UUID must be the last field")

        def run(argv):
            return self.APPS if any("compute-apps" in a for a in argv) else cards

        # Both modules, because each function resolves _run in ITS OWN globals: the card list is
        # read in _detect_gpu and the holder list in _detect_holders.  Patching only one leaves the
        # other calling the real nvidia-smi, which on a machine with no card silently returns no
        # holders -- a green-looking test that proves nothing about attribution.
        real = dg._run, dhold._run
        dg._run = dhold._run = run
        try:
            return recorded_uuid, dg.nvidia_gpus()
        finally:
            dg._run, dhold._run = real

    def test_only_the_busy_card_reports_a_holder(self):
        recorded_uuid, rows = self._rows()
        by_uuid = {r["uuid"]: r for r in rows}
        self.assertEqual(2, len(by_uuid))
        self.assertEqual((), by_uuid[recorded_uuid]["holders"],
                         "the idle card must claim no holder")
        self.assertEqual(1, len(by_uuid["GPU-bbbb"]["holders"]))
        self.assertEqual("soak_load.py", by_uuid["GPU-bbbb"]["holders"][0][2])


class TestKfdHolders(unittest.TestCase):
    """AMD has no --query-compute-apps; the KFD states usage per (pid, gpu_id) and is the only source."""

    PIDS = (("1234", 16290, "805306368"), ("5678", 99999, "123456"), ("4321", 16290, "0"))

    def _tree(self, tmp):
        for pid, gpu, used in self.PIDS:
            d = os.path.join(tmp, pid)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "vram_%d" % gpu), "w", encoding="utf-8") as fh:
                fh.write(used + "\n")
        return tmp

    def test_only_processes_on_the_asked_for_card_are_returned(self):
        """5678 holds memory on gpu_id 99999.  Reading the directory instead of the per-gpu file
        would report it against this card -- the same wrong-card attribution, one layer down."""
        with tempfile.TemporaryDirectory() as tmp:
            got = dhold.kfd_holders(16290, sys_kfd_proc=self._tree(tmp))
        self.assertEqual({1234}, set(got))
        self.assertEqual(805306368, got[1234][0])

    def test_a_pid_with_the_card_open_but_holding_nothing_is_not_a_holder(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = dhold.kfd_holders(16290, sys_kfd_proc=self._tree(tmp))
        self.assertNotIn(4321, got)

    def test_a_missing_kfd_is_empty_and_not_an_error(self):
        self.assertEqual({}, dhold.kfd_holders(16290, sys_kfd_proc="/nonexistent/kfd/proc"))


class TestHoldersInTheReport(unittest.TestCase):
    @staticmethod
    def _gpu(holders=()):
        return detect.Gpu(0, "nvidia", "RTX 3060", "sm_86", 12 * GIB, 1 * GIB, None,
                          "0000:03:00.0", 4, 16, 4, 16, "595", "nvidia-smi", "GPU-aaaa", holders)

    def test_the_report_names_who_holds_the_card(self):
        text = "\n".join(detect._gpu_lines(self._gpu(((4242, 9 * GIB, "soak_load.py"),))))
        self.assertIn("in use by", text)
        self.assertIn("soak_load.py(4242)", text)

    def test_a_card_nobody_holds_says_nothing(self):
        self.assertNotIn("in use by", "\n".join(detect._gpu_lines(self._gpu())))

if __name__ == "__main__":
    unittest.main()
