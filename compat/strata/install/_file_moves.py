# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid installer: file-reference expert moves (opt-in) -- the flags, the refusals, and the per-pack checksum manifest.

OWNER     : this module owns WHICH flags file moves add (engine `--retier-file-moves 1`, helper `--file-io auto`), the rule that they
            need the ladder, and getting `<pack>/expert-sums.v1` built or verified by `seabraid-sums`.  The manifest FORMAT and its
            freshness test belong to the tool (adapters expert_sums.hpp / strata-ext seabraid_sums.cpp): this module never parses it.
CONTRACT  : refusal(ladder, args) -> reason or None;  ENGINE_ARGS;  HELPER_ARGS;  find_tool(strata_dir) -> path or "";
            ensure_manifest(pack, tool, run, dry_run, *, native, layers, experts) -> FRESH | BUILT | WOULD_BUILD, or raises FileMovesRefused.
            Build artifact locations come from build.py; neither module imports config or the setup stages.
FAILURE   : fail-fast.  The operator asked for file moves, so a manifest that cannot be had is a refusal naming the command to run
            by hand, never a silent downgrade.  (The engine, at start, separately turns file moves off with a WARN if the manifest
            it finds is missing or stale -- that is its own line of defence, not this stage's.)
CHECKER   : tests/install/test_file_moves.py.
SIZE      : ~90 lines.

WHY THE INSTALLER BUILDS THE MANIFEST: the experts file moves exist for are the ones main has never read, so no hash exists at
move time; hashing them then would be the very read the feature removes (FR DESIGN 2.5).  Once per pack, here, is the only
place it is free.
"""

from __future__ import annotations

import os
import shlex
from typing import Callable, Optional, Sequence

from .build import find_built_exes

MANIFEST = "expert-sums.v1"
TOOL = "seabraid-sums"
ENGINE_ARGS = ("--retier-file-moves", "1")
HELPER_ARGS = ("--file-io", "auto")
FRESH, BUILT, WOULD_BUILD = "fresh", "built", "would-build"


class FileMovesRefused(Exception):
    """File moves were asked for and cannot be set up; the message names the fix."""


def refusal(ladder: bool, args: Sequence[str]) -> Optional[str]:
    """Why file moves cannot be launched with these options, or None."""
    if not ladder:
        return ("--file-moves needs --ladder: file moves change HOW the pyramid's live moves read an expert, so without live "
                "moves there is nothing for them to act on")
    if any(a == "--retier-file-moves" or a.startswith("--retier-file-moves=") for a in args):
        return ("--file-moves sets --retier-file-moves 1 itself; the same flag given by hand would be a second owner of it "
                "(drop one of the two)")
    return None


def find_tool(strata_dir: str) -> str:
    """Find sums using the build owner's current and supported legacy output layouts."""
    return next(iter(find_built_exes(strata_dir, TOOL).values()), "")


def ensure_manifest(pack: str, tool: str, run: Callable[[Sequence[str]], int], dry_run: bool, *,
                    native: str | None, layers: int, experts: int) -> str:
    """Make `<pack>/expert-sums.v1` fresh.  `run(argv) -> exit status` is the one exec point (injected so tests never spawn).

    Use the helper's native source and geometry for both calls so verification checks the same expert data the helper reads.
    Fresh means the tool's verify run exits 0. Otherwise the manifest is
    built, and the file must then exist: a tool that exits 0 and wrote nothing is as much a refusal as one that failed.
    A dry run runs nothing and says what a real run would do."""
    if dry_run:
        return WOULD_BUILD
    if not tool:
        raise FileMovesRefused("--file-moves needs the %s tool and it is not built (looked in <work>/strata/build-*/seabraid-strata-ext/ "
                               "and supported legacy build outputs). "
                               "Build without --no-build, or build the Strata tree's %s target." % (TOOL, TOOL))
    argv = (tool, pack, "--layers", str(layers), "--experts", str(experts))
    if native:
        argv += ("--native", native)
    path = os.path.join(pack, MANIFEST)
    if os.path.exists(path) and run((*argv, "--verify")) == 0:
        return FRESH
    if run(argv) != 0:
        raise FileMovesRefused("%s could not build %s for %s. Run it by hand: %s"
                               % (TOOL, MANIFEST, pack, shlex.join(argv)))
    if not os.path.exists(path):
        raise FileMovesRefused("%s exited 0 but wrote no %s in %s. Run it by hand: %s" % (TOOL, MANIFEST, pack, shlex.join(argv)))
    return BUILT
