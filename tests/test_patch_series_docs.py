# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The patch series the docs name is the series the patch files make.

OWNER: compat/strata/patches/strata/*.patch is the one source of the series; the docs below only quote it.  A new patch
file makes every doc that still names the old last number fail here, so the next patch cannot leave stale prose behind.
"""

from __future__ import annotations

import os
import re

from conftest import REPO

PATCHES = os.path.join(REPO, "compat", "strata", "patches", "strata")
DOCS = ("README.md", "compat/strata/MAP.md", "compat/strata/docs/STRATA-INTEGRATION.md", "compat/strata/docs/MIGRATION-INVENTORY.md")
# The prose form every doc uses for the series; a doc that rewords it away must reword this test, not silently escape it.
SERIES = re.compile(r"\b0001-0006 and 0010-(\d{4})\b")


def numbers() -> list[int]:
    return sorted(int(m.group(1)) for f in os.listdir(PATCHES) if (m := re.match(r"(\d{4})-.*\.patch$", f)))


def last_number(nums: list[int]) -> str:
    assert nums[:6] == [1, 2, 3, 4, 5, 6] and nums[6:] == list(range(10, nums[-1] + 1)), f"the series no longer reads 0001-0006 and 0010-NNNN: {nums}"
    return f"{nums[-1]:04d}"


def read(rel: str) -> str:
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        return fh.read()


def test_the_series_has_the_shape_the_prose_describes():
    assert last_number(numbers()) == "0020"


def test_every_doc_names_the_last_patch_number():
    want = last_number(numbers())
    for rel in DOCS:
        found = SERIES.findall(read(rel))
        assert found, f"{rel} no longer states the series as '0001-0006 and 0010-NNNN'"
        assert set(found) == {want}, f"{rel} names a last patch {sorted(set(found))}, the patch files end at {want}"


def test_the_readme_table_has_a_row_for_every_patch_after_0001():
    rows = re.findall(r"(?m)^\| (\d{4})(?:, (\d{4}))? \|", read("README.md"))
    listed = {n for pair in rows for n in pair if n}
    assert {f"{n:04d}" for n in numbers() if n > 1} <= listed


def test_the_doc_check_notices_a_stale_number():
    assert SERIES.findall("series 0001-0006 and 0010-0016 here") == ["0016"]
