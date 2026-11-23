# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Where the modules this harness REUSES live, and the sys.path entries that make them importable.

OWNER: this file is the one place that names the placement-qual runner directory and tools/kl.py; every other pay_*.py
imports it first (``import pay_paths  # noqa: F401``) instead of computing a relative path of its own.
WHY a path hook and not a copy: the engine lifetime (qual_engine), the recorded chain (qual_report), the HTTP client
(qual_work.send), the confidence bounds (qual_stats) and the KL gate (tools/kl.py) already exist and are owned by the
placement-qual track and tools/; a second copy would drift (the t quantile was corrected there once already).
FAILURE: importing this module cannot fail; a missing directory shows up as an ImportError in the importer.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLACEMENT_DIR = os.path.normpath(os.path.join(HERE, "..", "placement-qual"))
TOOLS_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "tools"))
KL_PY = os.path.join(TOOLS_DIR, "kl.py")

for _d in (HERE, PLACEMENT_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)
