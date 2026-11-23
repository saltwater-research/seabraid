# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The sdist stage keeps setuptools contents while normalizing archive metadata."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

from conftest import REPO

sys.path.insert(0, str(Path(REPO) / "scripts" / "release"))
import rel_build  # noqa: E402


def test_normalization_fixes_tar_and_gzip_metadata_and_keeps_payload_and_mode(tmp_path):
    archive = tmp_path / "sample.tar.gz"
    payload = b"package data"
    info = tarfile.TarInfo("sample/data.txt")
    info.size, info.mode, info.mtime, info.uid, info.gid = len(payload), 0o640, 123, 77, 88
    info.pax_headers = {"atime": "123.5", "ctime": "124.5", "custom": "kept"}
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as source:
        source.addfile(info, io.BytesIO(payload))

    rel_build.normalize_sdist(archive, "1700000000")

    assert archive.read_bytes()[4:8] == b"\0\0\0\0"
    with tarfile.open(archive, "r:gz") as normalized:
        member = normalized.getmember("sample/data.txt")
        assert normalized.extractfile(member).read() == payload
        assert (member.mode, member.mtime, member.uid, member.gid) == (0o640, 1700000000, 0, 0)
        assert member.pax_headers == {"custom": "kept"}
