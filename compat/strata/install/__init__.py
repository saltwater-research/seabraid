# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid installer stages.  Import direction: detect <- plan <- config <- build/services <- setup.

Card ranking (R1-R3) is owned by the runtime's Python package (`bindings/python/seabraid/ranking.py`) so engines and this
installer share one definition.  The installer runs from a checkout before anything is built or installed, so the
package directory is put on the import path here, once, instead of keeping a second copy of the rules.
"""
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "bindings" / "python"
if (_PACKAGE_ROOT / "seabraid" / "ranking.py").is_file() and str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))
