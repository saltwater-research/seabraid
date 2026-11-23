# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the release-gate tests: a throwaway copy of the tree, and a way to run a tool inside it."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "checks"))
import _common  # noqa: E402


def copy_tree(dest: str) -> str:
    """Copy every tracked and untracked-not-ignored file of the repo to ``dest`` (no .git, so checkers use the walk path).

    WHY the refusal: the checkers walk the copy through ``_common.tracked``, which reads only ``SHIPPED_TOP_LEVEL``
    entries, so a new top-level file or directory that is missing from that list would be copied and then silently not
    checked. Failing here names it (the same list has its own test against ``git ls-files``, which cannot see an
    untracked file).
    """
    unlisted = sorted({rel.split("/")[0] for rel in _common.tracked(REPO)} - _common.SHIPPED_TOP_LEVEL)
    if unlisted:
        pytest.fail(f"top-level entries missing from checks/_common.SHIPPED_TOP_LEVEL (the copy's walk would skip them): {unlisted}")
    for rel in _common.tracked(REPO):
        target = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(os.path.join(REPO, rel), target)
    return dest


def run_tool(root: str, rel_script: str, *args: str) -> tuple[int, str]:
    """Run ``root/rel_script`` with this interpreter and return (exit code, stdout + stderr)."""
    proc = subprocess.run([sys.executable, "-B", os.path.join(root, rel_script), *args], capture_output=True, text=True, cwd=root)
    return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture
def tree(tmp_path):
    """A private copy of the repo that a test may damage."""
    return copy_tree(str(tmp_path / "repo"))


_BLOCK_OPEN = re.compile(r"#\[(=*)\[")
# A CMake bracket argument with at least one '=' (``[=[ ... ]=]``). WHY not ``[[``: in sh ``[[ ... ]]`` is a test whose
# quotes are real, and a missing partner would hide the rest of the file.
_BRACKET_ARG = re.compile(r"\[(=+)\[")


def _skip_comment(text: str, i: int) -> int:
    """Index just past the comment that starts at the ``#`` at ``text[i]``: a ``#[=*[ ... ]=*]`` block if one opens
    here and closes later (it may span lines), else the rest of the line (the newline itself is kept)."""
    opened = _BLOCK_OPEN.match(text, i)
    if opened:
        close = text.find("]" + opened.group(1) + "]", opened.end())
        if close != -1:
            return close + len(opened.group(1)) + 2
    end = text.find("\n", i)
    return len(text) if end == -1 else end


# WHY a bound: a double-quoted string may span lines (CMake, sh), but a LONE quote in unquoted text must not swallow the
# rest of the file; the string is carried across a newline only when its closing quote follows within this many lines.
_STRING_SPAN_LINES = 100


def _string_closes_soon(text: str, start: int) -> bool:
    """Whether an unescaped double quote ends the string within ``_STRING_SPAN_LINES`` lines of ``text[start:]``."""
    lines = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            return True
        if ch == "\n":
            lines += 1
            if lines > _STRING_SPAN_LINES:
                return False
        i += 1
    return False


def _starts_comment(text: str, i: int) -> bool:
    """Whether the ``#`` at ``text[i]`` begins a comment, judged by the character before it (a '#' that is the first
    character of a word, in sh; a comment after ``)`` in CMake). ``${#x}`` (length in sh) is not one."""
    if i == 0:
        return True
    prev = text[i - 1]
    if prev.isspace() or prev in ");":
        return True
    return prev == "{" and (i < 2 or text[i - 2] != "$")


def _in_bracket_arg(text: str, i: int, mode: str, out: list[str]) -> tuple[int, str]:
    """One step inside a CMake bracket argument: quotes and backslashes are text, comments still go. -> (i, mode)."""
    if text.startswith(mode, i):
        out.append(mode)
        return i + len(mode), ""
    if text[i] == "#" and _starts_comment(text, i):
        return _skip_comment(text, i), mode
    out.append(text[i])
    return i + 1, mode


def _mode_after_newline(text: str, i: int, mode: str) -> str:
    """The quote state past the newline at ``text[i]``: only a double-quoted string that closes soon is carried over."""
    return mode if mode == '"' and _string_closes_soon(text, i + 1) else ""


def _escapes_next(text: str, i: int, mode: str) -> bool:
    """A backslash escapes the next character everywhere except in plain single quotes, and never across a newline."""
    return text[i] == "\\" and mode != "'" and i + 1 < len(text) and text[i + 1] != "\n"


def _open_in_code(text: str, i: int, out: list[str]) -> tuple[int, str, bool]:
    """What the character at ``text[i]`` starts while in code. -> (i, mode, done); done: this call already emitted or skipped."""
    ch = text[i]
    if ch == "[" and (opened := _BRACKET_ARG.match(text, i)) and text.find("]" + opened.group(1) + "]", opened.end()) != -1:
        out.append(opened.group(0))
        return opened.end(), "]" + opened.group(1) + "]", True
    if ch == "'" and out and out[-1] == "$":
        return i, "$'", False
    if ch in "\"'":
        return i, ch, False
    if ch == "#" and _starts_comment(text, i):
        return _skip_comment(text, i), "", True
    return i, "", False


def code_only(text: str) -> str:
    """The text without comments, so a needle test cannot be satisfied by a comment that merely mentions the command.

    One left-to-right pass with a quote state (code, double-quoted, single-quoted, sh ``$'...'``), so a ``#`` inside any
    quote is text. A backslash escapes the next character in code, in double quotes and in ``$'...'`` (a string ending in
    an escaped backslash ends at the quote after it) and is literal in plain single quotes (sh, TOML literal strings).
    A ``#`` starts a comment only in code, after whitespace, ``)`` (``set(a b)#c``), ``;`` (``a;# c``) or ``{`` (not
    ``${#x}``) or at the start of the text, so ``"^#define X"``, ``$#`` and ``a#b`` survive. A block comment
    ``#[[ ... ]]`` / ``#[=[ ... ]=]`` is recognised only at such a comment position, never inside a string. A CMake bracket argument ``[=[ ... ]=]`` (with
    at least one ``=``, closed later) is raw: quotes and backslashes in it are text, its comments still go.

    A double-quoted string may span lines (a CMake multi-line argument, an sh string) when its closing quote follows
    within ``_STRING_SPAN_LINES`` lines; the quote state otherwise resets at each newline. EXACT LIMITS: (1) a ``;``
    before a ``#`` inside an unquoted CMake argument (``a;#b``) is read as a comment start, so that text is dropped (a
    missed needle). (2)-(4) keep a comment that is really one, so a needle that occurs ONLY in such a comment is falsely
    found: (2) a single-quoted string that spans lines is not carried over, so the closing quote on a later line opens a
    string that hides a ``#`` after it; (3) a lone apostrophe in code hides the ``#`` after it on that line; (4) a lone
    double quote whose partner is more than ``_STRING_SPAN_LINES`` lines away is not carried over. A needle test must not
    rely on a needle that these shapes could host.
    """
    out: list[str] = []
    mode = ""  # "" = code, otherwise the quote that is open: '"', "'" or "$'"
    i = 0
    while i < len(text):
        ch = text[i]
        if mode[:1] == "]":
            i, mode = _in_bracket_arg(text, i, mode, out)
            continue
        if ch == "\n":
            mode = _mode_after_newline(text, i, mode)
        elif _escapes_next(text, i, mode):
            out.append(text[i : i + 2])
            i += 2
            continue
        elif mode:
            mode = "" if ch == mode[-1] else mode
        else:
            i, mode, done = _open_in_code(text, i, out)
            if done:
                continue
        out.append(ch)
        i += 1
    return "\n".join(line.rstrip() for line in "".join(out).splitlines() if line.strip())
