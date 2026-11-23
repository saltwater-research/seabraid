# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The wheel platform tag of a Windows DLL, read from its PE header (stdlib only).

OWNER   : the mapping from a PE COFF machine type to a wheel platform tag; scripts/build_wheel.py is its one consumer.
CONTRACT: windows_platform_tag(image: bytes) -> "win_amd64" | "win_arm64" | "win32".
FAILURE : SystemExit with the reason for bytes that are not a PE image, or a machine type with no tag here.
WHY     : with CARGO_BUILD_TARGET set for another architecture, setuptools' own tag is the host interpreter's, which
          would label an arm64 DLL win_amd64; the DLL itself is the only honest source.
CHECKER : tests/test_build_wheel_pe.py (synthetic headers, no cross build).
FINISH  : those tests pass.
"""

from __future__ import annotations

# IMAGE_FILE_HEADER.Machine -> wheel platform tag.
PE_MACHINE_TAGS = {0x8664: "win_amd64", 0xAA64: "win_arm64", 0x14C: "win32"}


def windows_platform_tag(image: bytes) -> str:
    offset = int.from_bytes(image[0x3C:0x40], "little") if image[:2] == b"MZ" else 0  # e_lfanew
    if not offset or image[offset:offset + 4] != b"PE\0\0" or len(image) < offset + 6:
        raise SystemExit("the built Windows library is not a PE image; refusing to guess a wheel tag")
    machine = int.from_bytes(image[offset + 4:offset + 6], "little")
    if machine not in PE_MACHINE_TAGS:
        raise SystemExit(f"the built Windows library has PE machine type {machine:#x}, which has no wheel tag here "
                         f"(known: {', '.join(f'{m:#x} {t}' for m, t in PE_MACHINE_TAGS.items())})")
    return PE_MACHINE_TAGS[machine]
