# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""_capture against REAL child processes (python itself): stop at the marker, report exit, report timeout."""
from __future__ import annotations

import sys
import time

from install._capture import capture

PY = sys.executable


def test_stops_a_child_that_prints_the_marker_and_then_keeps_serving():
    t0 = time.monotonic()
    got = capture([PY, "-c", "import time,sys;print('MARK x',flush=True);time.sleep(60)"], None, {}, "MARK", 30)
    assert got.found and got.rc is None and not got.timed_out and "MARK x" in got.text
    assert time.monotonic() - t0 < 25


def test_reports_the_exit_code_when_the_child_dies_without_the_marker():
    got = capture([PY, "-c", "import sys;print('boom');sys.exit(3)"], None, {}, "MARK", 30)
    assert not got.found and got.rc == 3 and not got.timed_out and "boom" in got.text


def test_times_out_a_silent_child():
    got = capture([PY, "-c", "import time;time.sleep(60)"], None, {}, "MARK", 0.5)
    assert not got.found and got.timed_out and got.rc is None


def test_a_missing_executable_is_reported_not_raised():
    got = capture(["/no/such/exe-seabraid"], None, {}, "MARK", 1)
    assert not got.found and "cannot start" in got.text


def test_the_stopped_child_is_really_dead(tmp_path):
    # A child that keeps appending to a file; if capture() left it running the file would still grow after it returned.
    beat = tmp_path / "beat.txt"
    code = ("import time,sys; print('MARK', flush=True)" + chr(10) + "while True:" + chr(10)
            + "    open(sys.argv[1], 'a').write('x'); time.sleep(0.02)")
    got = capture([PY, "-c", code, str(beat)], None, {}, "MARK", 30)
    assert got.found
    before = beat.stat().st_size if beat.exists() else 0
    time.sleep(0.5)
    after = beat.stat().st_size if beat.exists() else 0
    assert before == after, "the child kept running after capture() returned"
