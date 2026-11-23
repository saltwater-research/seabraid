# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid installer: the expert-pyramid ("ladder") flag set and the refusals that keep one publisher per rung.

OWNER     : this leaf owns WHICH engine flags and env switch moves on, and which option combinations are refused because two
            things would then publish the same slots.  config.py asks it and renders the answer; plan.py owns how many helpers
            there are (MAX_HELPERS) and config.py owns SEABRAID_STRATA_SHM.  PY DESIGN 5.8 / 7.
CONTRACT  : ladder_args(k, fallback) -> list;  rendered(k, ladder, fallback, extra) -> list;  LADDER_ENGINE_ENV;  refusal(k, ladder, args, env, rendered) -> reason or None.
            The engine's retier rules are NOT restated here: _retier_rules owns them (mirror of retier_loop.cpp).
            Pure functions over strings; imports only its sibling leaf _retier_rules (config <- _ladder <- _retier_rules).
FAILURE   : refusal() returns the reason text and config raises it as ConfigRefused: a launch line the engine would reject
            with exit 2 is stopped here at plan time, with the way out named.
CHECKER   : tests/install/test_ladder.py (one test per rule).
SIZE      : ~100 lines.

WHY OPT-IN: the audit measured live placement LOSING to static placement (75.66 vs 93.66 tok/s, an archived private run), so no
default turns it on; the flag set exists to be measured (PY DESIGN section 9) and the owner turns it on with --ladder.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from . import _retier_rules

# WHY these two: Strata's RAM<->NVMe adaptation must be on for the bottom of the pyramid (RAM keeps a copy by its own heat) and it
# needs batch == 0, which is why the server runs --parallel 1 (services' default).  Engine env, not server env: only the engine reads them.
LADDER_ENGINE_ENV: Mapping[str, str] = {"STRATA_RAM_ADAPT": "4", "STRATA_RAM_ADAPT2": "1"}


# WHY here: the engine's retier_check (retier_loop.cpp check_counts) exits 2 unless --adapt-every AND --adapt-swaps are > 0 (no routing
# counts exist otherwise), so the pyramid's flag set must carry them.  64 / 8 are the values the placement-qual arms and the S8 run
# measured with (scripts/placement-qual/arms.json repeats them as JSON, which cannot import this).  The user's own later value
# wins (extra_args come after this set and the engine takes the last one); refusal() stops a non-positive one.
ADAPT_EVERY, ADAPT_SWAPS = 64, 8


def ladder_args(k: int, fallback: bool = True) -> list:
    """The engine flags for a k-helper pyramid: k+1 hops reach the top from the fallback rung, 2 moves per hop of budget."""
    return ["--retier", "on", "--retier-promote", "seabraid", "--retier-fallback", "1" if fallback else "0",
            "--retier-hops", str(k + 1), "--retier-moves", str(2 * (k + 1)),
            "--adapt-every", str(ADAPT_EVERY), "--adapt-swaps", str(ADAPT_SWAPS)]


def rendered(k: int, ladder: bool, fallback: bool = True, extra: Sequence[str] = ()) -> list:
    """Every engine flag config generates ahead of the user's extra_args: the pyramid set plus `extra` (file moves), or nothing."""
    return ladder_args(k, fallback) + list(extra) if ladder else []


def _value(args: Sequence[str], flag: str) -> Optional[str]:
    """The last value given for `flag` ('--flag v' or '--flag=v'); '' when it is present without one; None when absent."""
    found = None
    for i, a in enumerate(args):
        if a == flag:
            found = args[i + 1] if i + 1 < len(args) else ""
        elif a.startswith(flag + "="):
            found = a[len(flag) + 1:]
    return found


def _adapt_refusal(args: Sequence[str]) -> Optional[str]:
    """The user's own --adapt-every / --adapt-swaps must be positive integers: the engine needs both > 0 or it exits 2."""
    for name in ("--adapt-every", "--adapt-swaps"):
        v = _value(args, name)
        if v is not None and not (v.lstrip("+").isdigit() and int(v) > 0):
            return "--ladder needs %s > 0 (the engine refuses --retier without routing counts); got %r" % (name, v)
    return None


def _second_owner(args: Sequence[str]) -> Optional[str]:
    """The second --retier / --retier-promote in the whole line (either spelling), or None: the ladder renders each exactly once."""
    for name in ("--retier", "--retier-promote"):
        hits = [a for a in args if a == name or a.startswith(name + "=")]
        if len(hits) > 1:
            i = max(j for j, a in enumerate(args) if a == hits[-1])
            return " ".join(args[i:i + 2]) if hits[-1] == name else hits[-1]   # `--retier-promote legacy`: name the value too
    return None


def refusal(k: int, ladder: bool, args: Sequence[str], env: Mapping[str, str], rendered: Sequence[str] = ()) -> Optional[str]:
    """Why this combination cannot be launched, or None.  k = helper count; `rendered` = the engine flags config generates ahead of
    `args` (the ladder set, file moves), so the engine's rules run over the line the engine will really get."""
    if k > 1 and env.get("STRATA_PREFILL_REMOTE_COMPUTE", "0") not in ("", "0"):
        return ("STRATA_PREFILL_REMOTE_COMPUTE (D4 prefill remote compute) runs on ONE helper and this plan has %d: Strata would "
                "pick the last process tier without saying so. Unset it" % k)
    if ladder:
        if k < 1:
            return "--ladder needs at least one helper: with only the main card there is no rung to move experts to"
        if _value(args, "--remote-expert-opt") is not None:
            return "--ladder never runs with --remote-expert-opt: it switches RAM<->NVMe adaptation off, and the pyramid needs it"
        dup = _second_owner(list(rendered) + list(args))
        if dup:
            return ("--ladder already sets --retier on --retier-promote seabraid; `%s` given as well would be a second owner of the flag "
                    "(drop it, or drop --ladder to set the retier flags yourself)" % dup)
        why = _adapt_refusal(args)
        if why:
            return why
    return _retier_rules.check(list(rendered) + list(args), k)
