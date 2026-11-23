# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Where the modules the overflow harness REUSES live, and the sys.path entries that make them importable.

OWNER: this file is the one place that names the payoff runner directory; every other ov_*.py imports it first
(``import ov_paths  # noqa: F401``).  payoff's own pay_paths then adds placement-qual and tools/kl.py, so the path of each
reused module has exactly one owner and is never computed twice.
WHY a path hook and not a copy: the engine lifetime (qual_engine), the request loop (pay_measure), the log formats
(pay_parse), the KL runner (pay_gate.run_kl_py), the recorded chain (qual_report) and the W7 row pooling (qual_decode) are
owned by the placement-qual and payoff tracks; a second copy would drift.
FAILURE: importing this module cannot fail; a missing directory shows up as an ImportError in the importer.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAYOFF_DIR = os.path.normpath(os.path.join(HERE, "..", "payoff"))
SEABRAID_STRATA_DIR = os.path.normpath(os.path.join(HERE, "..", ".."))  # compat/strata: lets ``from install import _ladder`` find the installer's flag owner

for _d in (PAYOFF_DIR, HERE, SEABRAID_STRATA_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import pay_paths  # noqa: E402,F401  (puts placement-qual on sys.path)

PLACEMENT_DIR = pay_paths.PLACEMENT_DIR
