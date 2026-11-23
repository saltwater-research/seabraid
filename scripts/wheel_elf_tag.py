# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""ELF validation and the glibc floor for scripts/build_wheel.py's Linux x86-64 wheel.

The architecture comes from the library because Cargo can build for a target that differs from the host.
An invalid header or unrecognized glibc floor raises SystemExit. tests/test_build_wheel_elf.py checks the entry point.
"""

import re
import struct

# Each allowed dependency is available to a glibc-only x86-64 CDLL.
ALLOWED_NEEDED = {"libc.so.6", "libm.so.6", "libgcc_s.so.1", "libpthread.so.0", "libdl.so.2", "ld-linux-x86-64.so.2"}


def check_elf(header: bytes) -> None:
    """Require a 64-bit little-endian x86-64 shared-object header for the only supported Linux wheel tag."""
    if len(header) < 64 or header[:4] != b"\x7fELF":
        raise SystemExit("the built Linux library has no complete ELF64 header")
    elf_class, encoding = header[4:6]
    if (elf_class, encoding) != (2, 1):
        raise SystemExit("Linux wheels require a 64-bit little-endian ELF library")
    file_type, machine = struct.unpack_from("<HH", header, 16)
    if (file_type, machine) != (3, 62):  # ET_DYN, EM_X86_64
        raise SystemExit(f"Linux wheels require an x86-64 ELF shared object (ELF type {file_type}, machine {machine})")


def disallowed_needed(readelf_dynamic: str) -> list[str]:
    """DT_NEEDED entries of `readelf -d` output that are not in ALLOWED_NEEDED."""
    found = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", readelf_dynamic)
    return sorted(name for name in found if name not in ALLOWED_NEEDED)


def manylinux_tag(readelf_versions: str) -> str:
    """Use the highest required GLIBC_2.x version from readelf -V as the PEP 600 floor."""
    versions = {tuple(int(part) for part in found) for found in re.findall(r"GLIBC_(\d+)\.(\d+)", readelf_versions)}
    if not versions:
        raise SystemExit("no GLIBC_x.y version needed found in readelf -V output; refusing to guess a manylinux tag")
    major, minor = max(versions)
    if major != 2:
        raise SystemExit(f"unexpected glibc major version {major}; manylinux tags exist only for glibc 2.y")
    return f"manylinux_2_{minor}_x86_64"
