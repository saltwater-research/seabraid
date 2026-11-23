# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""L9 (R8 review): each constant install/_plan_shm.py mirrors from C++ equals the C++ symbol its comment names.

WHY by symbol and not by line: a ``helper.cpp:77`` citation went stale the first time the file moved (it pointed at a
line that no longer held the constant).  The comments now name symbols; this reads those symbols from the wire headers
(adapters/strata/wire) and the helper (strata-ext/src/helper.cpp) and fails on the first disagreement.
"""
from __future__ import annotations

import re
from pathlib import Path

from install import _plan_shm as M

ADDON = Path(__file__).resolve().parents[2]
WIRE = ADDON.parents[1] / "adapters" / "strata" / "wire" / "strata" / "seabraid"


def _int(path: Path, pattern: str) -> int:
    m = re.search(pattern, path.read_text(encoding="utf-8"))
    assert m, f"{pattern!r} not found in {path.name}: the citation in _plan_shm.py names a symbol that moved"
    return int(m.group(1))


def test_channel_layout_constants():
    layout = WIRE / "channel_layout.hpp"
    assert M.KBLOBBATCH == _int(layout, r"constexpr int kBlobBatch = (\d+);")
    assert M.KTIERMAXOPS == _int(layout, r"constexpr int kTierMaxOps = (\d+);")
    assert M.TIER_WIRE_OP_BYTES == _int(layout, r"static_assert\(sizeof\(TierWireOp\) == (\d+)")


def test_spec_region_constants():
    spec = WIRE / "expert_spec.hpp"
    assert M.LAYER_SPEC_BYTES == _int(spec, r"static_assert\(sizeof\(LayerSpec\) == (\d+)")
    assert M.SPEC_MAX_FILES == _int(spec, r"kSpecMaxFiles = (\d+)")
    assert M.SPEC_PATH_MAX == _int(spec, r"kSpecPathMax = (\d+)")


def test_named_channel_pairs_per_token_is_the_helpers_legacy_width():
    helper = ADDON / "strata-ext" / "src" / "helper.cpp"
    assert M.CAP_PAIRS_PER_TOKEN == _int(helper, r"constexpr int64_t legacy_picks = (\d+);")
    assert "kWindowMaxT * (args.private_channel() ? kRoutedPicks : legacy_picks)" in helper.read_text(encoding="utf-8")
