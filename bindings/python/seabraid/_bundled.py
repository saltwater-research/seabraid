# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Locate the native library that a platform wheel ships inside the package, if any.

OWNER   : where the bundled native library lives and what it is called per platform (seabraid/_lib/<name>).
CONTRACT: bundled_library() -> str | None: the absolute path when the file exists in the installed package, else None.
          Imports: stdlib only; it never loads the library and never touches the environment.
SIZE    : under 50 lines.
FAILURE : none; a pure sdist or pure wheel has no _lib directory and gets None, exactly as before wheels existed.
CHECKER : tests/test_bundled.py.
FINISH  : that test passes and a wheel built by scripts/build_wheel.py returns a loadable path after a clean install.

WHY this is a lookup and not a default: runtime.py still chooses the native runtime only when a path or SEABRAID_LIBRARY
is given, because native mode requires a provider and silently switching an existing program from the reference
provider to native would change its behaviour; pass library_path=seabraid.bundled_library() to opt in.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def _library_name() -> str:
    # The names cargo gives a cdylib on each platform; scripts/build_wheel.py copies the file under this same name.
    if sys.platform == "win32":
        return "seabraid.dll"
    if sys.platform == "darwin":
        return "libseabraid.dylib"
    return "libseabraid.so"


def bundled_library() -> str | None:
    """Return the path of the native library shipped in this installation, or None when the wheel carries none."""
    candidate = _PACKAGE_DIR / "_lib" / _library_name()
    return str(candidate) if candidate.is_file() else None
