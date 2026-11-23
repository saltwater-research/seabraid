# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""What the setup flow needs from the operator and from a previous run: prompts, and prebuilt binaries.

OWNER     : this module owns how a question is asked (and when it is NOT asked: --yes) and where a prior build's
            binaries are looked for.  seabraid_setup.py owns the order of the stages; _setup_stages.py owns what the two
            long stages do.  Both import from here; this module imports neither (one-way edge).
CONTRACT  : ask_yes, ask_choice (raises PromptAborted), find_prebuilt -> Prebuilt, engine_for, Refused (a stage's
            "stop with this message and this exit status", turned into the process status by the caller).
FAILURE   : fail-fast; an unanswered question aborts, it never falls back to a default nobody chose.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, NamedTuple

from .build import find_built_exes


class Refused(Exception):
    """A stage cannot go on: ``message`` names the fix, ``status`` is the exit status (see seabraid_setup's CONTRACT).

    WHY an exception, not a returned int: the stages are split into small steps and each step has its own refusals;
    the one place that prints it and logs it (seabraid_setup._fail) stays where the logger is configured."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.message = message
        self.status = status


def ask_yes(question: str, yes: bool, default: bool = True) -> bool:
    """Ask, or take the recommendation under --yes.

    --yes answers questions; it never hides the report above them, so an
    unattended run still leaves the operator the whole reasoning in its log.
    """
    hint = "[Y/n]" if default else "[y/N]"
    if yes:
        print("%s %s yes" % (question, hint))
        return True
    try:
        reply = input("%s %s " % (question, hint)).strip().lower()
    except EOFError:
        print("\n(nothing to ask on -- pass --yes to accept, or run this in a terminal)")
        return False
    return default if not reply else reply.startswith("y")


class PromptAborted(Exception):
    """A question could not be answered: the input ended, or three answers in a row were not a choice."""


MAX_PROMPT_TRIES = 3


def ask_choice(question: str, choices: list, yes: bool) -> str:
    """The first choice is the recommended one, and is what --yes takes, and what an empty reply takes (the prompt says so).

    WHY it aborts instead of falling back to choice 1: an answer nobody gave (closed stdin) or one nobody understood (a typo,
    "7" of 3) would otherwise pick a model for the user and build it; the safe default is to stop and say why."""
    if len(choices) == 1:
        print("%s %s" % (question, choices[0]))
        return choices[0]
    for i, c in enumerate(choices, 1):
        print("  %d) %s%s" % (i, c, "   (recommended)" if i == 1 else ""))
    if yes:
        print("%s %s" % (question, choices[0]))
        return choices[0]
    for _ in range(MAX_PROMPT_TRIES):
        try:
            reply = input("%s [1-%d, default 1] " % (question, len(choices))).strip()
        except EOFError:
            raise PromptAborted("no answer to %r: the input ended. Pass --yes to take the recommended choice, or run this in a terminal."
                                % question) from None
        if not reply:
            return choices[0]
        if reply.isdigit() and 1 <= int(reply) <= len(choices):
            return choices[int(reply) - 1]
        print("  please answer with a number from 1 to %d (or press Enter for 1)" % len(choices))
    raise PromptAborted("no valid answer to %r after %d tries; nothing was changed." % (question, MAX_PROMPT_TRIES))


class Prebuilt(NamedTuple):
    """What a previous run left behind: the engine and one helper per vendor.

    A NamedTuple and deliberately NOT a dataclass.  Callers already unpack this
    (``engine, helpers = find_prebuilt(...)``), index it (``find_prebuilt(...)[1]``
    in the bench stage) and compare it to a plain tuple in tests; a NamedTuple adds
    the names while keeping all three working, where a frozen dataclass would break
    every one of them.  ``helper_exes`` is spelled as in BuildResult and _finish, so
    the same thing has the same name everywhere it is passed.
    """

    engine: str
    helper_exes: Dict[str, str]


def find_prebuilt(work_dir: str) -> Prebuilt:
    """The binaries a previous run left, as (engine, {vendor: helper}).

    build_all puts them at <work>/strata/build-<vendor>/strata and
    <work>/strata/build-<vendor>/seabraid-strata-ext/seabraid-helper, so that is where
    --no-build looks.  Anything missing comes back empty and is refused by the
    caller with the flag that fixes it -- never substituted.
    """
    root = os.path.join(work_dir, "strata")
    engine = next(iter(sorted(glob.glob(os.path.join(root, "build-*", "strata")))), "")
    helpers = find_built_exes(root, "seabraid-helper")
    return Prebuilt(engine, helpers)


def engine_for(work_dir: str, vendor: str) -> str:
    """The engine built for `vendor`'s backend, or "" -- never another vendor's.

    find_prebuilt's single engine is the first build-*/strata, which is whichever backend sorts first; a CUDA card started with
    the HIP binary exits 127.  A directory counts for a vendor when it is build-<vendor> or build-<vendor>-<suffix> (a separate
    main-engine build); the plain build-<vendor> wins when both hold an engine.
    """
    root = os.path.join(work_dir, "strata")
    hits = sorted(glob.glob(os.path.join(root, "build-%s" % vendor, "strata")) + glob.glob(os.path.join(root, "build-%s-*" % vendor, "strata")))
    return hits[0] if hits else ""
