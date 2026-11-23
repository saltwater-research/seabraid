# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Windows wheel tag from the DLL's PE machine type (audit F13), with synthetic headers; no cross build needed."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
from conftest import REPO

_spec = importlib.util.spec_from_file_location("build_wheel", os.path.join(REPO, "scripts", "build_wheel.py"))
build_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_wheel)


def pe_image(machine: int, offset: int = 0x80) -> bytes:
    image = bytearray(offset + 24)
    image[0:2] = b"MZ"
    image[0x3C:0x40] = offset.to_bytes(4, "little")
    image[offset:offset + 4] = b"PE\0\0"
    image[offset + 4:offset + 6] = machine.to_bytes(2, "little")
    return bytes(image)


@pytest.mark.parametrize("machine,tag", [(0x8664, "win_amd64"), (0xAA64, "win_arm64"), (0x14C, "win32")])
def test_the_tag_follows_the_dll_machine_type(machine, tag):
    assert build_wheel.windows_platform_tag(pe_image(machine)) == tag


@pytest.mark.parametrize("image", [b"", b"MZ", b"\x7fELF" + bytes(200), pe_image(0x8664)[:0x82], pe_image(0x1C0),
                                   pe_image(0x8664)[:0x80] + b"XX\0\0" + bytes(20)])
def test_a_non_pe_or_unlisted_machine_is_refused(image):
    with pytest.raises(SystemExit, match="PE|machine type"):
        build_wheel.windows_platform_tag(image)


def test_platform_tag_of_on_windows_reads_the_dll_not_the_interpreter(monkeypatch, tmp_path):
    dll = tmp_path / "seabraid.dll"
    dll.write_bytes(pe_image(0xAA64))  # an arm64 DLL built by an amd64 interpreter's cargo
    monkeypatch.setattr(sys, "platform", "win32")
    assert build_wheel.platform_tag_of(dll) == "win_arm64"
