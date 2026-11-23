# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""M4 (R8 review): the helper wire contract has ONE owner, adapters/strata/wire, and the edge is one-way.

The wire (channel_layout.hpp, channel.hpp, expert_spec.hpp under strata/seabraid/, and helper_options.hpp) is what
Strata's engine, compat's seabraid-helper and the Strata adapter all compile.  These checks fail on the next second
owner: a patch that recreates a wire header in the Strata tree, a copy of a wire header anywhere else in the
repository, an adapter build that again needs a compat-patched tree for the wire, or a compat build that stops taking
the wire from adapters/strata.  Both hardening consumers are checked here too (L17): one flag owner, two includers.
"""
from __future__ import annotations

import re
from pathlib import Path

ADDON = Path(__file__).resolve().parents[1]          # compat/strata
REPO = ADDON.parents[1]
WIRE = REPO / "adapters" / "strata" / "wire"
WIRE_HEADERS = ("strata/seabraid/channel_layout.hpp", "strata/seabraid/channel.hpp", "strata/seabraid/expert_spec.hpp",
                "helper_options.hpp")
STRATA_PATCHES = sorted((ADDON / "patches" / "strata").glob("*.patch")) + [REPO / "adapters" / "strata" / "strata-v7-seabraid.patch"]
HARDEN = "adapters/strata/cmake/seabraid_harden.cmake"
HARDEN_FLAGS = re.compile(r"-fstack-protector|_FORTIFY_SOURCE|_GLIBCXX_ASSERTIONS|-z,relro")


def _patched_paths(patch: Path) -> set[str]:
    return set(re.findall(r"^diff --git a/(\S+) b/", patch.read_text(encoding="utf-8", errors="replace"), re.M))


def test_every_wire_header_lives_in_the_wire_directory():
    missing = [h for h in WIRE_HEADERS if not (WIRE / h).is_file()]
    assert not missing, missing


def test_no_strata_patch_creates_or_edits_a_wire_header():
    wire_in_tree = {"include/" + h for h in WIRE_HEADERS if h.startswith("strata/")}
    offenders = {p.name: sorted(_patched_paths(p) & wire_in_tree) for p in STRATA_PATCHES}
    assert not {k: v for k, v in offenders.items() if v}, offenders


def test_no_second_copy_of_a_wire_header_in_the_repository():
    names = {Path(h).name for h in WIRE_HEADERS}
    copies = [p for p in REPO.rglob("*.hpp") if p.name in names and WIRE not in p.parents
              and ".git" not in p.parts and "target" not in p.parts and not any(part.startswith("build") for part in p.parts)]
    assert not copies, copies


def test_the_adapter_build_needs_no_compat_patched_tree_for_the_wire():
    cmake = (REPO / "adapters" / "strata" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert not re.search(r"STRATA_SOURCE\}/include/strata/seabraid", cmake), "the adapter must take the wire from wire/"
    assert '"${CMAKE_CURRENT_SOURCE_DIR}/wire"' in cmake


def test_compat_takes_the_wire_from_the_adapter_and_patch_0001_puts_it_on_stratas_path():
    shared = (ADDON / "strata-ext" / "seabraid.cmake").read_text(encoding="utf-8")
    assert "adapters/strata/wire" in shared
    first = next(p for p in STRATA_PATCHES if p.name.startswith("0001-")).read_text(encoding="utf-8")
    assert "+include_directories(\"${_seabraid_wire}\")" in first and "adapters/strata/wire" in first


def test_hardening_flags_have_one_owner_and_both_ends_of_the_channel_use_it():
    owner = REPO / HARDEN
    assert HARDEN_FLAGS.search(owner.read_text(encoding="utf-8"))
    adapter = (REPO / "adapters" / "strata" / "CMakeLists.txt").read_text(encoding="utf-8")
    helper = (ADDON / "strata-ext" / "CMakeLists.txt").read_text(encoding="utf-8")
    shared = (ADDON / "strata-ext" / "seabraid.cmake").read_text(encoding="utf-8")
    assert "cmake/seabraid_harden.cmake" in adapter and "adapters/strata/cmake/seabraid_harden.cmake" in shared
    assert "seabraid_harden(seabraid-helper EXE)" in helper
    scanned = [p for root in (REPO / "adapters" / "strata", ADDON) for p in root.rglob("*")
               if p.is_file() and (p.name == "CMakeLists.txt" or p.suffix == ".cmake")]
    second = [str(p.relative_to(REPO)) for p in scanned if p != owner and HARDEN_FLAGS.search(p.read_text(encoding="utf-8"))]
    assert not second, f"hardening flags defined outside {HARDEN}: {second}"
