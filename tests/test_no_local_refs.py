# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Runs checks/check_no_local_refs.py over the tree and proves each shape with a planted positive and a clean negative."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "checks"))
_spec = importlib.util.spec_from_file_location("check_no_local_refs", os.path.join(REPO, "checks", "check_no_local_refs.py"))
chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)

# (planted positive, expected kind, clean negative that must not fire); every shape appears exactly once.
CASES = [
    ("root=/home/alice/x", "home-path", "root=/home/user/x"),
    ("cd /Users/alice", "home-path", "cd /Users/<name>"),
    ("C:\\Users\\example\\x", "home-path", "C:\\Program Files\\x"),
    ("/media/store/models", "data-mount", "/dev/shm/x"),
    ("/mnt/store/x", "data-mount", "mount at /mnt"),
    ("ssh host1234 ls", "ssh-alias", "ssh host1 ls"),
    ("ip 192.168.77.7", "private-ip", "ip 8.8.8.8"),
    ("ip 10.1.2.3", "private-ip", "version 110.1.2.3"),
    ("ip 172.16.0.9", "private-ip", "ip 172.32.0.9"),
    ("ip 100.64.0.1", "private-ip", "ip 100.200.0.1"),
    ("run on 2026-01-02 14:30 here", "time-of-day", "run on 2026-01-02 here"),
    ("run on 2026-01-02T9:05", "time-of-day", "port 8080:80"),
    ("status (2026-01-02 evening)", "time-of-day", "status (2026-01-02 a)"),
    ("run-20260102-143015", "run-id-time", "run-20260102"),
    ("CUDA_VISIBLE_DEVICES=GPU-9c0ffee0-1234-5678", "gpu-uuid", "CUDA_VISIBLE_DEVICES=0 GPU-bound"),
    ("# Linux first (owner, 2026-11-19): jobs", "dated-owner", "# the owner table (OWNER: src/placement.rs, owner: checks/x.py)"),
    ("compat's helper TUs compile (GPU link: box job)", "internal-vocab", "the box below holds the options; a sandbox job"),
    ("CI's strata job and the box run)", "internal-vocab", "the independent audit of 2026-11-19; the owner table; on the box of tools"),
    ("between the desktop source archive and the graph", "internal-vocab", "a desktop's source archive and the graph"),
]


def test_tracked_tree_has_no_local_references():
    assert chk.scan(REPO) == []


@pytest.mark.parametrize("bad,kind,good", CASES)
def test_each_shape_fires_on_positive_and_not_on_negative(bad, kind, good):
    assert kind in [k for _, k in chk.line_hits(bad)], bad
    assert kind not in [k for _, k in chk.line_hits(good)], good


def test_phrases_that_are_legitimate_shipped_text_are_not_shapes():
    # "independent audit" is the audit's cited name in README/RESEARCH-LOG/DECISIONS; "owner" is design and placement vocabulary.
    for ok in ("recorded by the independent audit of 2026-11-19", "OWNER: this module owns it", "(owner: checks/check_pins.py)", "the owner table"):
        assert chk.line_hits(ok) == [], ok


def test_legal_entity_name_is_not_a_shape():
    assert chk.line_hits("Copyright 2026 Saltwater Research LLC") == []


def test_checker_fails_on_planted_files(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "clean.md").write_text("a clean line, RTX 3090, gfx1201, (c) 2026 Saltwater Research LLC\n")
    for i, (bad, _, _) in enumerate(CASES[:4]):
        (tmp_path / f"bad{i}.md").write_text(bad + "\n")
    git("add", ".")
    hits = chk.scan(str(tmp_path))
    assert {h.split(":")[0] for h in hits} == {f"bad{i}.md" for i in range(4)}
    assert all(h.count(":") >= 3 for h in hits)


def test_unused_allowlist_entry_is_an_error(tmp_path):
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "local_refs_allow.txt").write_text("nothing/*\tproject\n")
    (tmp_path / "a.md").write_text("fine\n")
    assert any("unused allowlist entry" in h for h in chk.scan(str(tmp_path)))
