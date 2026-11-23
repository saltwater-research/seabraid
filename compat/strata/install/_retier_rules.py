# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid installer: the engine's --retier* startup rules, mirrored so a launch line the engine would refuse is refused at plan time.

OWNER     : strata-ext/src/retier_loop.cpp owns the rules (retier_parse_flag, retier_check, retier_check_helpers).  This leaf is the
            installer's ONE mirror of them; _ladder and config ask it and never restate a rule.
CONTRACT  : check(args, helpers) -> the engine's own error text, or None.  `args` is the whole engine argument list, in order (the
            engine takes the last value of a flag); `helpers` is the SEABRAID_STRATA_SHM channel count.  Pure; imports nothing.
FAILURE   : returns the text (config raises it as ConfigRefused).  A flag it cannot read is refused, never guessed at.
CHECKER   : tests/data/retier_cases.tsv is read by BOTH this module (tests/install/test_retier_rules.py) and the engine's own code
            (tests/retier_rules_case_test.cpp, CTest seabraid_retier_rules_case_test); each asserts the same accept/refuse and the same
            text, so a rule changed on one side only fails the other side's test.
SIZE      : ~240 lines.

WHY THE ENGINE'S SPELLINGS, NOT OURS: the engine matches flag names exactly (no `--flag=value`), strtol/strtod-reads each value,
and clamps or truncates to int before it range-checks.  Each of those is copied, because a mirror that is kinder than the engine
passes a line the engine then rejects at start, which is the bug this file exists to stop.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

K_TIER_MAX_OPS = 64      # channel_layout.hpp kTierMaxOps
K_MAX_HOPS = 64          # retier_loop.hpp kMaxHops
# WHY ASCII classes spelled out: C strtol/strtod/isspace in the engine's C locale know only [0-9], [0-9a-fA-F] and " \t\n\v\f\r"; Python's
# \d and \s also match Unicode digits and spaces, and `$` matches before a trailing newline, so every pattern here is fullmatch + ASCII.
_WS = "[ \t\n\v\f\r]*"
_INT = re.compile(_WS + "[+-]?[0-9]+")
_ATOI = re.compile(_WS + "[+-]?[0-9]+")   # re.match: atoi reads the leading digits and ignores the rest
_DEC = "(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_HEX = "0[xX](?:[0-9a-fA-F]+\\.?[0-9a-fA-F]*|\\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?"
_DBL = re.compile(_WS + "([+-]?)(" + _HEX + "|" + _DEC + ")")
# LP64: the engine is Linux x86-64 where `long` is 64 bits and `int` 32; the long/int rules below are pinned to that, not to the
# machine the installer runs on (a Windows long is 32 bits and would give other answers).
# The engine's own defaults when --adapt-every / --adapt-swaps are not given: `int adapt_every = 4;` and `int adapt_swaps = 96;` in
# strata src/program/generate.cpp (Options, ~lines 1038 and 1080).  A ladder run that moved experts with NO --adapt-* flags relied on
# them; refusing a missing flag was stricter than the engine.  The engine reads them with std::atoi (leading digits, never an error).
# NOT mirrored (engine-only, named so nobody wonders): retier_check's "the routing counts have N cells, the model has M" needs the model's
# layer x expert count, which the flags cannot change and the engine sizes both sides from; and generate.cpp ~5239 sets adapt_swaps = 0 when STRATA_EXPERT_NIBBLE is on AND the pack's layers qualify; the installer
# cannot know the pack's quant layout, so that case is left to the engine's own startup refusal (it names both adapt flags).
ENGINE_ADAPT_DEFAULTS = {"--adapt-every": 4, "--adapt-swaps": 96}
_LONG_MIN, _LONG_MAX = -(2 ** 63), 2 ** 63 - 1


def _to_int(v: str) -> Optional[int]:
    """strtol base 10 over the whole string, errno clean (an out-of-range long is ERANGE, a read failure)."""
    n = int(v) if _INT.fullmatch(v) else None
    return n if n is not None and _LONG_MIN <= n <= _LONG_MAX else None


def _atoi(v: str) -> int:
    """glibc atoi = (int) strtol(v): leading digits, long-clamped on overflow (LONG_MAX is -1 as an int), unreadable is 0."""
    m = _ATOI.match(v)
    return _int32(max(_LONG_MIN, min(_LONG_MAX, int(m.group())))) if m else 0


def _to_float(v: str) -> Optional[float]:
    """strtod over the whole string with errno clean and a finite result (the engine's to_dbl).  ERANGE means overflow, and underflow
    to zero or to a subnormal; the subnormal case is refused conservatively (glibc spares exactly representable ones)."""
    m = _DBL.fullmatch(v)
    if not m:
        return None
    text, hexed = m.group(1) + m.group(2), m.group(2)[:2] in ("0x", "0X")
    try:
        f = float.fromhex(text) if hexed else float(text)
    except (ValueError, OverflowError):
        return None
    mantissa = re.split("[pP]" if hexed else "[eE]", m.group(2))[0]
    nonzero = any(ch in "123456789abcdefABCDEF" for ch in (mantissa[2:] if hexed else mantissa))
    if f - f != 0 or (nonzero and abs(f) < 2.2250738585072014e-308):   # inf/nan, or an underflow
        return None
    return f


def _int32(n: int) -> int:
    """The (int) cast the engine applies to a long: wraps, and the range check then runs on the wrapped value."""
    return (n + 2 ** 31) % 2 ** 32 - 2 ** 31


def _clamp(n: int, lo: int, hi: int) -> int:
    return lo if n < lo else hi if n > hi else n


# flag -> (field, conversion of the long).  The same rows as kLongFlags in retier_loop.cpp.
_LONG = {
    "--retier-every": ("every", _int32), "--retier-moves": ("moves", _int32), "--retier-cooldown": ("cooldown", _int32),
    "--retier-fallback": ("fallback", lambda n: _clamp(n, -1, 2)), "--retier-hops": ("hops", lambda n: _clamp(n, 0, 1000)),
    "--retier-file-moves": ("file_moves", lambda n: _clamp(n, -1, 2)),
}
_DBL_FLAGS = {"--retier-margin": "margin", "--retier-ceiling": "ceiling"}
_MODES = {"off", "on", "zero-op"}
_PROMOTE = {"legacy": "legacy", "on": "legacy", "seabraid": "seabraid", "off": "off"}   # `on` is the engine's spelling of legacy


def _mode(c: dict, v: str) -> Optional[str]:
    if v not in _MODES:
        return "--retier: expected off, on or zero-op, got '%s'" % v
    c["mode"] = v
    return None


def _promote(c: dict, v: str) -> Optional[str]:
    if v not in _PROMOTE:
        return "--retier-promote: expected legacy, seabraid or off (on = legacy), got '%s'" % v
    c["promote"] = _PROMOTE[v]
    return None


def _number(c: dict, a: str, v: str) -> Optional[str]:
    """A numeric flag (kLongFlags / kDblFlags): read it like strtol / strtod, convert like the row's setter, store it."""
    if a in _LONG:
        n = _to_int(v)
        if n is not None:
            c[_LONG[a][0]] = _LONG[a][1](n)
    else:
        n = _to_float(v)
        if n is not None:
            c[_DBL_FLAGS[a]] = n
    return None if n is not None else "%s: cannot read '%s'" % (a, v)


def _apply(c: dict, adapt: dict, a: str, v: str) -> Optional[str]:
    """One flag and its value, as retier_parse_flag reads them; the error text or None."""
    if a in adapt:
        adapt[a] = _atoi(v)
        return None
    if a in _SETTERS:
        return _SETTERS[a](c, v)
    if a in _LONG or a in _DBL_FLAGS:
        return _number(c, a, v)
    return "unknown argument: %s" % a   # includes every `--flag=value` spelling: the engine does not split on '='


_SETTERS = {"--retier": _mode, "--retier-promote": _promote}


def _parse(args: Sequence[str]):
    """(config, adapt_every, adapt_swaps, error): retier_parse_flag over `args`, the first error wins."""
    c = dict(mode="off", promote="off", every=128, moves=4, margin=1.25, cooldown=8, ceiling=0.08, fallback=0, hops=1, file_moves=0)
    adapt = dict(ENGINE_ADAPT_DEFAULTS)
    i = 0
    while i < len(args):
        a = args[i]
        i += 1
        if a not in adapt and not a.startswith("--retier"):
            continue
        if a not in adapt and a not in _SETTERS and a not in _LONG and a not in _DBL_FLAGS:
            return c, 0, 0, "unknown argument: %s" % a   # retier_parse_flag returns 0 without calling next(); even as the last argument
        if i >= len(args):
            return c, 0, 0, "%s needs a value" % a        # generate.cpp next(): "<flag> needs a value", exit 2
        err = _apply(c, adapt, a, args[i])
        i += 1
        if err:
            return c, 0, 0, err
    return c, adapt["--adapt-every"], adapt["--adapt-swaps"], None


def _pyramid(c: dict) -> Optional[str]:
    if c["fallback"] not in (0, 1):
        return "--retier-fallback must be 0 or 1"
    if not 1 <= c["hops"] <= K_MAX_HOPS:
        return "--retier-hops must be 1..%d" % K_MAX_HOPS
    if (c["fallback"] != 0 or c["hops"] > 1) and c["promote"] != "seabraid":
        return "--retier-fallback 1 and --retier-hops > 1 need --retier-promote seabraid (only its planner moves a cell between a card and the fallback, or climbs)"
    if c["hops"] > 1 and 2 * c["hops"] > c["moves"]:
        return ("--retier-hops %d needs --retier-moves >= %d (a hop is a LOAD and an EVICT), got --retier-moves %d"
                % (c["hops"], 2 * c["hops"], c["moves"]))
    return None


def _file_moves(c: dict) -> Optional[str]:
    if c["file_moves"] not in (0, 1):
        return "--retier-file-moves must be 0 or 1"
    if c["file_moves"] != 0 and c["promote"] != "seabraid":
        return "--retier-file-moves 1 needs --retier-promote seabraid (only its swap can take an expert's bytes from the model file)"
    return None


def _bounds(c: dict) -> Optional[str]:
    if c["every"] < 1:
        return "--retier-every must be >= 1"
    if not 1 <= c["moves"] <= K_TIER_MAX_OPS:
        return "--retier-moves must be 1..%d" % K_TIER_MAX_OPS
    if not c["margin"] >= 1.0:
        return "--retier-margin must be >= 1"
    if c["cooldown"] < 0:
        return "--retier-cooldown must be >= 0"
    if not 0.0 <= c["ceiling"] <= 1.0:
        return "--retier-ceiling must be in [0, 1]"
    return None


def _counts(every: int, swaps: int) -> Optional[str]:
    if every <= 0 or swaps <= 0:
        return ("--retier needs the adaptive tier's routing counts, which exist only with --adapt-every > 0 AND --adapt-swaps > 0 "
                "(got --adapt-every %d, --adapt-swaps %d; counts not allocated)" % (every, swaps))
    return None


def _off(c: dict) -> Optional[str]:
    return "--retier-promote legacy|seabraid needs --retier on: it is a mode of the retier round" if c["promote"] != "off" else None


def _zero_op(c: dict) -> Optional[str]:
    return "--retier-promote seabraid needs --retier on (zero-op sends no ops)" if c["promote"] == "seabraid" and c["mode"] != "on" else None


def _helper_open(helpers: int) -> Optional[str]:
    return "--retier needs a seabraid helper tier (SEABRAID_STRATA_SHM with --expert-cache-device1); none is open" if helpers < 1 else None


def _one_helper(c: dict, helpers: int) -> Optional[str]:
    """retier_check_helpers: only the seabraid publisher serves more than one helper."""
    if helpers <= 1 or c["promote"] == "seabraid":
        return None
    return ("--retier %s serves ONE helper, but SEABRAID_STRATA_SHM lists %d; use --retier-promote seabraid"
            % ("with --retier-promote legacy" if c["promote"] == "legacy" else "without --retier-promote seabraid", helpers))


def _check(c: dict, every: int, swaps: int, helpers: int) -> Optional[str]:
    """retier_check, then retier_check_helpers, in the engine's order."""
    err = _pyramid(c) or _file_moves(c)
    if err:
        return err
    if c["mode"] == "off":
        return _off(c)
    return _zero_op(c) or _counts(every, swaps) or _helper_open(helpers) or _bounds(c) or _one_helper(c, helpers)


def check(args: Sequence[str], helpers: int) -> Optional[str]:
    """The engine's refusal text for this whole argument list, or None when its startup checks would pass."""
    c, every, swaps, err = _parse(args)
    return err or _check(c, every, swaps, helpers)
