# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The Linux wheel entry point tags the built ELF, including on a non-Linux test runner."""

import importlib.util
import struct
from pathlib import Path

import pytest
from conftest import REPO

_spec = importlib.util.spec_from_file_location("build_wheel_elf", Path(REPO) / "scripts" / "build_wheel.py")
build_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_wheel)


@pytest.fixture
def elf_library(tmp_path, monkeypatch):
    library = tmp_path / "libseabraid.so"
    monkeypatch.setattr(build_wheel, "_run", lambda command, **_: {"-d": "", "-V": "GLIBC_2.17"}[command[1]])
    return library


def elf_header(machine=62, elf_class=2, data=1, kind=3):
    header = bytearray(64)
    header[:7] = b"\x7fELF" + bytes((elf_class, data, 1))
    struct.pack_into("<HH", header, 16, kind, machine)
    return header


def test_linux_wheel_accepts_the_x86_64_elf(elf_library):
    elf_library.write_bytes(elf_header())
    assert build_wheel.linux_platform_tag(elf_library) == "manylinux_2_17_x86_64"


@pytest.mark.parametrize("image", [elf_header(machine=183), elf_header(elf_class=1), elf_header(data=2),
                                   elf_header(kind=2), b"\x7fELF", b"not an ELF file"])
def test_linux_wheel_rejects_an_incompatible_elf(image, elf_library):
    elf_library.write_bytes(image)
    with pytest.raises(SystemExit, match="ELF"):
        build_wheel.linux_platform_tag(elf_library)


@pytest.mark.parametrize(("dynamic", "versions", "reason"), [
    (" 0x1 (NEEDED) Shared library: [libstdc++.so.6]\n", "GLIBC_2.17", "libstdc"),
    ("", "", "no GLIBC"),
])
def test_linux_wheel_still_checks_dependencies_and_glibc(elf_library, monkeypatch, dynamic, versions, reason):
    elf_library.write_bytes(elf_header())
    monkeypatch.setattr(build_wheel, "_run", lambda command, **_: {"-d": dynamic, "-V": versions}[command[1]])
    with pytest.raises(SystemExit, match=reason):
        build_wheel.linux_platform_tag(elf_library)
