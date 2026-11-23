# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for an installed platform wheel: open the bundled native library and weave one batch.

Run with `python -I scripts/wheel_smoke.py` from the venv the wheel was installed into (CI does exactly this).
Exits non-zero with the reason when the installation has no bundled library or the weave result is wrong.
"""

import seabraid

library = seabraid.bundled_library()
if library is None:
    raise SystemExit("this installation has no bundled native library")
with seabraid.Runtime(library, reference=True) as runtime:
    prepared = runtime.prepare(hidden=2, experts=3, max_tokens=2, top_k=2)
    result = prepared.weave([1.0, 2.0], [0, 2], [0.25, 0.75])
if result != [2.5, 5.0]:
    raise SystemExit(f"unexpected weave result {result}")
print(f"ok seabraid {seabraid.version()} library={library} result={result}")
