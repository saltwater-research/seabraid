# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Sabotage proof: every checker is shown to FAIL when a violation is planted in a throwaway copy of the tree.

A gate nobody has watched fail is not known to work.  Each case: prepare a copy so the checker is green, assert it
is green, plant ONE violation, assert the checker fails with the expected line.  Set SABOTAGE_REPORT=<file> to write
the table that lives in checks/SABOTAGE.md; test_sabotage_doc_lists_every_case keeps that document from drifting.
"""

from __future__ import annotations

import os
import re

import pytest
from conftest import REPO, run_tool

SPLIT = lambda *parts: "".join(parts)  # noqa: E731  (assemble shapes at run time so this file stays clean)


def put(root: str, rel: str, text: str, mode: str = "w") -> None:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def edit(root: str, rel: str, old: str, new: str) -> None:
    with open(os.path.join(root, rel), encoding="utf-8", newline="") as fh:
        text = fh.read()
    assert old in text, f"{rel}: setup text not found: {old!r}"
    put(root, rel, text.replace(old, new, 1))


def prep_spdx(root: str) -> None:
    code, out = run_tool(root, "scripts/add_spdx.py", "--root", root, "--write")
    assert code == 0, out


def prep_version(root: str) -> None:
    ver = open(os.path.join(root, "VERSION"), encoding="utf-8").read().strip()
    cargo = open(os.path.join(root, "Cargo.toml"), encoding="utf-8").read()
    put(root, "Cargo.toml", re.sub(r'(?m)^version = "[^"]+"', f'version = "{ver}"', cargo, count=1))
    lock = open(os.path.join(root, "Cargo.lock"), encoding="utf-8").read()
    put(root, "Cargo.lock", re.sub(r'(name = "seabraid"\nversion = ")[^"]+', rf"\g<1>{ver}", lock, count=1))
    cmake = open(os.path.join(root, "CMakeLists.txt"), encoding="utf-8").read()
    put(root, "CMakeLists.txt", re.sub(r"(project\([^)]*?VERSION )[0-9.]+", rf"\g<1>{ver.split('-')[0]}", cmake, count=1))


def prep_size(root: str) -> None:
    put(root, "checks/size_allow.json", "{}\n")


def prep_links(root: str) -> None:
    """Neutralise any link the copied tree already has broken, so the case proves ONE planted link, whatever the tree's own state."""
    code, out = run_tool(root, "checks/check_links.py", "--root", root)
    for rel, number, target in re.findall(r"(?m)^(\S+\.md):(\d+): broken link \((.*?)\): ", out):
        lines = open(os.path.join(root, rel), encoding="utf-8", newline="").read().split("\n")
        lines[int(number) - 1] = lines[int(number) - 1].replace(target, "#")
        put(root, rel, "\n".join(lines))


def plant_big_then_grow(root: str) -> None:
    big = "# Copyright 2026 Saltwater Research LLC\n# SPDX-License-Identifier: Apache-2.0\n" + "x = 1\n" * 499
    put(root, "src/big.py", big)
    put(root, "checks/size_allow.json", '{"src/big.py": {"ceiling": 501, "why": "planted exception for the sabotage case"}}\n')
    put(root, "src/big.py", big + "y = 2\n" * 20)


# WHY read, not literal: a version bump must not break the sabotage proof of a checker that compares to VERSION.
VERSION = open(os.path.join(REPO, "VERSION"), encoding="utf-8").read().strip()
PROBE = SPLIT("zz", "rig")  # a name that occurs nowhere in the tree; assembled so the whole-tree name gate never sees it here
HOME = SPLIT("/hom", "e/zed/proj")
GPU_UUID = SPLIT("GP", "U-9c0ffee0-1234-5678")  # made up; never a real card's UUID
LAN = SPLIT("19", "2.168.4.9")
DATED_OWNER = SPLIT("Linux first (own", "er, 2026-11-19):")
BOX_RUN = SPLIT("verified in the bo", "x run")
MARKER = SPLIT("REPLA", "CE-WITH")
# (id, checker, args, prep, plant(root), expected substring in the output)
CASES = [
    ("refs-home-path", "check_no_local_refs.py", [], None, lambda r: put(r, "README.md", f"\nbuilt in {HOME}\n", "a"), "README.md:"),
    ("refs-private-ip", "check_no_local_refs.py", [], None, lambda r: put(r, "src/lib.rs", f"\n// host {LAN}\n", "a"), "private-ip"),
    ("refs-gpu-uuid", "check_no_local_refs.py", [], None, lambda r: put(r, "README.md", f"\nCUDA_VISIBLE_DEVICES={GPU_UUID}\n", "a"), "gpu-uuid"),
    ("refs-dated-owner", "check_no_local_refs.py", [], None, lambda r: put(r, "README.md", f"\n{DATED_OWNER}\n", "a"), "dated-owner"),
    ("refs-internal-vocab", "check_no_local_refs.py", [], None, lambda r: put(r, "src/lib.rs", f"\n// {BOX_RUN}\n", "a"), "internal-vocab"),
    ("placeholder-marker", "check_no_placeholders.py", [], None, lambda r: put(r, "docs/leak.md", f"\nDOI: {MARKER}-THE-PAPER-DOI\n"), "docs/leak.md:2: unfilled placeholder"),
    ("compat-edge-import", "check_compat_edges.py", [], None, lambda r: put(r, "src/leak.py", "import compat.strata\n"), "src/leak.py:1:1: compat reference"),
    ("compat-edge-path-join", "check_compat_edges.py", [], None, lambda r: put(r, "adapters/ggml/leak.py", 'p = os.path.join(root, "compat", "strata")' + "\n"), "adapters/ggml/leak.py:1:"),
    ("compat-edge-allow-row-rots", "check_compat_edges.py", [], None, lambda r: put(r, "checks/compat_edges_allow.txt", "src/lib.rs\tno compat reference lives there any more\n", "a"), "matches no compat reference"),
    ("egress-rust", "check_no_egress.py", [], None, lambda r: put(r, "src/leak.rs", "use std::net::TcpStream;\n"), "src/leak.rs:1: egress"),
    ("egress-c", "check_no_egress.py", [], None, lambda r: put(r, "adapters/ggml/leak.c", "int s = socket(2, 1, 0);\n"), "adapters/ggml/leak.c:1: egress"),
    ("egress-python", "check_no_egress.py", [], None, lambda r: put(r, "bindings/python/seabraid/leak.py", "import urllib.request\n"), "leak.py:1: egress"),
    ("egress-cmake", "check_no_egress.py", [], None, lambda r: put(r, "adapters/ggml/CMakeLists.txt", "\nfile(DOWNLOAD x y)\n", "a"), "egress: file(DOWNLOAD"),
    (
        "egress-compat-non-loopback",
        "check_no_egress.py",
        [],
        None,
        lambda r: put(r, "compat/strata/tools/leak.py", 'import urllib.request\nurllib.request.urlopen("http://example.invalid/x")\n'),
        "compat/strata/tools/leak.py:2: egress",
    ),
    ("egress-allow-row-rots", "check_no_egress.py", [], None, lambda r: edit(r, "compat/strata/install/build.py", "Niko1221/Strata", "x/y"), "unused ALLOW row"),
    ("errors-cpp-catch", "check_error_paths.py", ["--strict-runtime", "--runtime-scope", "examples/"], None, lambda r: put(r, "examples/leak.cpp", "void f() { try { g(); } catch (...) { } }\n"), "examples/leak.cpp:1: catch-without-log"),
    ("errors-rust-discard", "check_error_paths.py", ["--strict-runtime", "--runtime-scope", "examples/"], None, lambda r: put(r, "examples/leak.rs", "fn f() { let _ = g(); }\n"), "examples/leak.rs:1: let-underscore-call"),
    ("errors-py-silent", "check_error_paths.py", ["--strict-runtime", "--runtime-scope", "examples/"], None, lambda r: put(r, "examples/leak.py", "try:\n    f()\nexcept Exception:\n    pass\n"), "examples/leak.py:3: silent-except"),
    ("errors-ratchet-new-file", "check_error_paths.py", [], None, lambda r: put(r, "compat/strata/tools/leak.py", "try:\n    f()\nexcept:\n    pass\n"), "compat/strata/tools/leak.py: 1 hit(s), baseline 0"),
    ("errors-ratchet-grows", "check_error_paths.py", [], None, lambda r: put(r, "compat/strata/tests/install/test_probe.py", "\ntry:\n    f()\nexcept:\n    pass\n", "a"), "test_probe.py: 2 hit(s), baseline 1"),
    ("spdx-missing", "check_spdx.py", [], prep_spdx, lambda r: put(r, "src/new.rs", "fn main() {}\n"), "src/new.rs:1: expected Apache-2.0, found missing"),
    ("spdx-header-wrong", "check_spdx.py", [], prep_spdx, lambda r: edit(r, "include/seabraid/seabraid.h", "Apache-2.0 OR MIT", "Apache-2.0"), "expected Apache-2.0 OR MIT, found Apache-2.0"),
    ("spdx-too-late", "check_spdx.py", [], prep_spdx, lambda r: put(r, "src/late.rs", "// a\n// b\n// c\n// d\n// e\n// SPDX-License-Identifier: Apache-2.0\n"), "src/late.rs:1: expected Apache-2.0, found missing"),
    ("size-501-lines", "check_size.py", [], prep_size, lambda r: put(r, "src/big.py", "x = 1\n" * 501), "src/big.py: 501 lines (limit 500)"),
    ("size-excepted-file-grows", "check_size.py", [], prep_size, plant_big_then_grow, "src/big.py: 521 lines (ceiling 501)"),
    ("version-cargo", "check_version.py", [], prep_version, lambda r: edit(r, "Cargo.toml", f'version = "{VERSION}"', 'version = "0.5.1"'), f"Cargo.toml: found 0.5.1, expected {VERSION}"),
    ("version-cargo-lock", "check_version.py", [], prep_version, lambda r: edit(r, "Cargo.lock", f'name = "seabraid"\nversion = "{VERSION}"', 'name = "seabraid"\nversion = "0.5.1"'), f"Cargo.lock: found 0.5.1, expected {VERSION}"),
    ("version-cmake", "check_version.py", [], prep_version, lambda r: edit(r, "CMakeLists.txt", "VERSION ${seabraid_version_numeric}", "VERSION 0.6.0"), f"CMakeLists.txt: found 0.6.0, expected {VERSION.split('-')[0]}"),
    ("version-pyproject-literal", "check_version.py", [], prep_version, lambda r: put(r, "pyproject.toml", '[project]\nname = "x"\nversion = "9.9.9"\n'), "pyproject.toml: found 9.9.9"),
    ("version-pyproject-dynamic-wrong-file", "check_version.py", [], prep_version, lambda r: put(r, "pyproject.toml", '[project]\ndynamic = ["version"]\n[tool.x.version]\nfile = "OTHER"\n'), "must read the VERSION file"),
    (
        "helper-tier-ram-rule-reverted",
        "check_helper_tier.py",
        [],
        None,
        lambda r: edit(r, "adapters/strata/strata-v7-seabraid.patch", "-            if (remote_experts[(size_t) r].seabraid() && !bl_exclude) continue;\n", ""),
        "0001-External-expert-helper-tier-an-out-of-process-helper.patch:690: seabraid() asks for the named-channel helper only",
    ),
    ("helper-tier-stale-row", "check_helper_tier.py", [], None, lambda r: put(r, "checks/helper_tier_allow.txt", "gone.seabraid() | a row whose call site was deleted\n", "a"), "fragment matches 0 call sites"),
    (
        "pid-probe-kill-readded",
        "check_pid_probes.py",
        [],
        None,
        lambda r: put(r, "adapters/strata/strata-v7-seabraid.patch", "+        if (::kill(helper_pid, 0) != 0) return false;\n", "a"),
        "strata-v7-seabraid.patch:",
    ),
    ("pid-probe-kill-in-repo-cpp", "check_pid_probes.py", [], None, lambda r: put(r, "adapters/strata/leak_probe.cpp", "bool alive(int helper_pid) { return ::kill(helper_pid, 0) == 0; }\n"), "adapters/strata/leak_probe.cpp:1: kill:"),
    ("pid-probe-second-pidfd-open", "check_pid_probes.py", [], None, lambda r: put(r, "adapters/strata/leak_pidfd.cpp", "int f(int p) { return (int) ::syscall(SYS_pidfd_open, p, 0); }\n"), "adapters/strata/leak_pidfd.cpp:1: pidfd_open:"),
    (
        "pid-probe-bare-wait",
        "check_pid_probes.py",
        [],
        None,
        lambda r: put(r, "adapters/strata/strata-v7-seabraid.patch", "diff --git a/src/core/remote_experts.cpp b/src/core/remote_experts.cpp\n+    if (!seabraid::Channel::wait([&] { return done(); }, 30)) return false;\n", "a"),
        "bare-wait: use the PeerWatch",
    ),
    ("pid-probe-stale-row", "check_pid_probes.py", [], None, lambda r: put(r, "checks/pid_probes_allow.txt", "gone.kill(1) | a row whose site was deleted from the tree\n", "a"), "fragment matches 0 sites"),
    ("abi-offset-moved", "check_abi_append.py", [], None, lambda r: edit(r, "src/layout/abi_layout.txt", "seabraid_batch_v1.tokens 4", "seabraid_batch_v1.tokens 8"), "seabraid_batch_v1.tokens: offset moved 4 -> 8"),
    ("abi-grows-without-struct-size", "check_abi_append.py", [], None, lambda r: edit(r, "src/layout/abi_layout.txt", "seabraid_config_v1 20", "seabraid_config_v1 24"), "seabraid_config_v1: size grew 20 -> 24"),
    ("abi-v2-offset-moved", "check_abi_append.py", [], None, lambda r: edit(r, "src/layout/abi_layout.txt", "seabraid_batch_v2.tokens 24", "seabraid_batch_v2.tokens 28"), "seabraid_batch_v2.tokens: offset moved 24 -> 28"),
    ("abi-epoch-bumped-without-snapshot", "check_abi_append.py", [], None, lambda r: edit(r, "include/seabraid/seabraid.h", "#define SEABRAID_ABI_EPOCH 1u", "#define SEABRAID_ABI_EPOCH 2u"), "no snapshot for epoch 2"),
    ("links-missing-file", "check_links.py", [], prep_links, lambda r: put(r, "README.md", "\n[gone](docs/NO-SUCH-DOC.md)\n", "a"), "broken link (docs/NO-SUCH-DOC.md): no such file or directory"),
    ("links-missing-anchor", "check_links.py", [], prep_links, lambda r: put(r, "README.md", "\n[gone](docs/ABI.md#no-such-heading)\n", "a"), "no heading or anchor '#no-such-heading' in docs/ABI.md"),
    ("names-in-content", "check_names.py", ["--forbid", PROBE], None, lambda r: put(r, "src/lib.rs", f"\n// {PROBE}_open()\n", "a"), f"forbidden name '{PROBE}'"),
    ("names-in-path", "check_names.py", ["--forbid", PROBE], None, lambda r: put(r, f"include/{PROBE}/x.h", "int x;\n"), f"include/{PROBE}/x.h: forbidden name '{PROBE}' in the path"),
    ("names-camel-prefix", "check_names.py", ["--forbid", PROBE], None, lambda r: put(r, "src/lib.rs", f"\nconst int k{PROBE.capitalize()} = 1;\n", "a"), f"forbidden name '{PROBE.capitalize()}'"),
    ("names-upper-suffix", "check_names.py", ["--forbid", PROBE], None, lambda r: put(r, "src/lib.rs", f"\nconst int {PROBE.upper()}X = 1;\n", "a"), f"forbidden name '{PROBE.upper()}'"),
    ("names-in-readme", "check_names.py", ["--forbid", PROBE], None, lambda r: put(r, "README.md", f"\n{PROBE.capitalize()} was the old name.\n", "a"), "README.md:"),
    ("names-allow-row-rots", "check_names.py", ["--forbid", PROBE], None, lambda r: put(r, "checks/names_allow.txt", "gone/*.txt\tkept for lineage, but the file is gone\n", "a"), "matches no file"),
]


DESC = {
    "refs-home-path": "a developer home-directory path appended to README.md",
    "refs-private-ip": "a private LAN address in a comment in src/lib.rs",
    "refs-gpu-uuid": "a GPU UUID (GPU-xxxxxxxx-xxxx-...) in a CUDA_VISIBLE_DEVICES line appended to README.md",
    "refs-dated-owner": "a dated person attribution in parentheses (the word owner, a comma, a year) appended to README.md",
    "refs-internal-vocab": "the maintainers' machine named as a place for a run (the word box before run) in a comment in src/lib.rs",
    "placeholder-marker": "docs/leak.md: a line with an unfilled fill-in marker, as .zenodo.json once had for the paper DOI",
    "compat-edge-import": "src/leak.py: `import compat.strata` (the core importing the add-on)",
    "compat-edge-path-join": "adapters/ggml/leak.py: a path join that names the compat directory",
    "compat-edge-allow-row-rots": "an allow row for a file that has no compat reference",
    "egress-rust": "src/leak.rs: `use std::net::TcpStream;`",
    "egress-c": "adapters/ggml/leak.c: a `socket(` call",
    "egress-python": "bindings/python/seabraid/leak.py: `import urllib.request`",
    "egress-cmake": "`file(DOWNLOAD ...)` appended to adapters/ggml/CMakeLists.txt",
    "egress-compat-non-loopback": "compat tool that opens http://example.invalid (not loopback, not allow-listed)",
    "egress-allow-row-rots": "the allow-listed Strata URL removed from build.py, so its ALLOW row matches nothing",
    "errors-cpp-catch": "examples/leak.cpp: an empty `catch (...)` block (--strict-runtime; examples/ is a runtime scope with no files today)",
    "errors-rust-discard": "examples/leak.rs: `let _ = g();` (--strict-runtime)",
    "errors-py-silent": "examples/leak.py: `except Exception: pass` (--strict-runtime)",
    "errors-ratchet-new-file": "a NEW compat file with a bare `except:` (ratchet baseline 0)",
    "errors-ratchet-grows": "a second swallowed error appended to a baselined compat file (ratchet grows 1 to 2)",
    "spdx-missing": "a new src/new.rs with no SPDX line",
    "spdx-header-wrong": "include/seabraid/seabraid.h downgraded from `Apache-2.0 OR MIT` to `Apache-2.0`",
    "spdx-too-late": "SPDX identifier on line 6 (outside the first 5 lines)",
    "size-501-lines": "src/big.py with 501 lines",
    "size-excepted-file-grows": "an excepted 501-line file grown to 521 lines",
    "version-cargo": "Cargo.toml version changed to 0.5.1",
    "version-cargo-lock": "Cargo.lock seabraid entry left at 0.5.1 while VERSION and Cargo.toml moved on",
    "version-cmake": "CMake project VERSION hard-coded to 0.6.0 instead of read from VERSION",
    "version-pyproject-literal": "pyproject.toml with literal version 9.9.9",
    "version-pyproject-dynamic-wrong-file": "pyproject.toml dynamic version reading a file other than VERSION",
    "helper-tier-ram-rule-reverted": "the 2026-11-17 bug put back: the v7 patch no longer replaces 0001's `seabraid()` RAM-copy rule",
    "helper-tier-stale-row": "an allow row whose `seabraid()` call site does not exist",
    "pid-probe-kill-readded": "the D4 bug put back: a `::kill(helper_pid, 0)` liveness probe added to the v7 patch (the sites 0013 had are explained rows; a new one is not)",
    "pid-probe-kill-in-repo-cpp": "adapters/strata/leak_probe.cpp: a `::kill(helper_pid, 0)` probe in repo C++",
    "pid-probe-second-pidfd-open": "adapters/strata/leak_pidfd.cpp: a second `SYS_pidfd_open` syscall outside wire pidfd.hpp",
    "pid-probe-bare-wait": "a `Channel::wait(` line added to src/core/remote_experts.cpp by the v7 patch that is not the single bl_wait wrapper",
    "pid-probe-stale-row": "an allow row whose pid-probe site does not exist",
    "abi-offset-moved": "src/layout/abi_layout.txt: seabraid_batch_v1.tokens moved from offset 4 to 8",
    "abi-grows-without-struct-size": "src/layout/abi_layout.txt: seabraid_config_v1 (no struct_size field) grown from 20 to 24 bytes",
    "abi-v2-offset-moved": "src/layout/abi_layout.txt: seabraid_batch_v2.tokens (a v2 row, absent from the old v1-only snapshot) moved from offset 24 to 28",
    "abi-epoch-bumped-without-snapshot": "SEABRAID_ABI_EPOCH raised to 2 in the header with no abi_layout.epoch2.txt (the old checker skipped silently)",
    "links-missing-file": "a relative link to docs/NO-SUCH-DOC.md appended to README.md",
    "links-missing-anchor": "a link to docs/ABI.md#no-such-heading (the file exists, the heading does not) appended to README.md",
    "names-camel-prefix": "`const int k<Probe> = 1;` appended to src/lib.rs (a letter before the name; the old rule let it through)",
    "names-upper-suffix": "`const int <PROBE>X = 1;` appended to src/lib.rs (a letter after the name)",
    "names-in-readme": "the name in README.md (outside the old runtime-only scan)",
    "names-allow-row-rots": "a checks/names_allow.txt row whose glob matches no file",
    "names-in-content": "`<probe>_open()` in a comment in src/lib.rs (forbidding a probe name that occurs nowhere else)",
    "names-in-path": "include/<probe>/x.h (name in a directory path)",
}


REPORT_HEADER = """# Sabotage record

Every checker in `checks/` has been watched failing.  `tests/test_sabotage.py` copies the tree, proves the checker is
green on the copy, plants ONE violation, and requires exit code 1 plus the output line below.  Regenerate this file
with `SABOTAGE_REPORT=checks/SABOTAGE.md python -m pytest tests/test_sabotage.py`; `test_sabotage_doc_lists_every_case`
fails if a case is missing from it.  Only the table between the marker comments is regenerated; the sections below it are hand-written and are kept.  Line numbers in the last column are those of the recording run.

"""


TABLE_BEGIN = "<!-- sabotage-table:begin -->"
TABLE_END = "<!-- sabotage-table:end -->"


def write_report(target: str, rows: list[tuple[str, str, str]]) -> None:
    """Regenerate ONLY the table between the two marker comments; every other line of the document is hand-written
    (the egress and complexity sections) and must survive. WHY refuse a document without markers: guessing where the
    table is would risk overwriting prose, and a missing marker means someone edited the document by hand."""
    table = "| case | checker | planted | failing output line |\n|---|---|---|---|\n"
    table += "".join(f"| {a} | {b} | {c} | {d} |\n" for a, b, c, d in rows)
    block = f"{TABLE_BEGIN}\n{table}{TABLE_END}\n"
    if not os.path.exists(target):
        text = REPORT_HEADER + block
    else:
        with open(target, encoding="utf-8", newline="") as fh:
            old = fh.read()
        if old.count(TABLE_BEGIN) != 1 or old.count(TABLE_END) != 1 or old.index(TABLE_BEGIN) > old.index(TABLE_END):
            raise ValueError(f"{target}: expected exactly one {TABLE_BEGIN} ... {TABLE_END} marker pair; add them around the table")
        head, rest = old.split(TABLE_BEGIN)
        tail = rest.split(TABLE_END, 1)[1]
        text = head + block + tail[1:]  # `block` already ends the marker line, so drop that line's own newline
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


@pytest.fixture(scope="module")
def recorded():
    rows: list[tuple[str, str, str]] = []
    yield rows
    target = os.environ.get("SABOTAGE_REPORT")
    if target:
        write_report(target, rows)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_checker_fails_when_violation_is_planted(case, tree, recorded):
    cid, script, args, prep, plant, expected = case
    if prep:
        prep(tree)
    green_code, green = run_tool(tree, f"checks/{script}", "--root", tree, *args)
    assert green_code == 0, f"{cid}: checker not green before planting:\n{green[-1500:]}"
    plant(tree)
    code, out = run_tool(tree, f"checks/{script}", "--root", tree, *args)
    assert code == 1, f"{cid}: checker did not fail on the planted violation:\n{out[-1500:]}"
    line = next((ln for ln in out.splitlines() if expected in ln), None)
    assert line, f"{cid}: expected {expected!r} in output:\n{out[-1500:]}"
    shown = line.replace("|", "\\|").replace(PROBE.upper(), "<PROBE>").replace(PROBE.capitalize(), "<Probe>").replace(PROBE, "<probe>")
    recorded.append((cid, script, DESC[cid], shown))


def test_every_case_is_described():
    assert {c[0] for c in CASES} == set(DESC)


def test_usage_error_without_forbid_is_not_green(tree):
    code, out = run_tool(tree, "checks/check_names.py", "--root", tree)
    assert code == 2 and "--forbid" in out


HAND_WRITTEN = "\n## check_no_egress.py evasions\n\nHand-written text that no run of the generator may drop.\n"
def test_report_regeneration_keeps_the_hand_written_sections(tmp_path):
    doc = tmp_path / "SABOTAGE.md"
    write_report(str(doc), [("a", "check_a.py", "first plant", "x:1: bad")])
    doc.write_text(doc.read_text(encoding="utf-8") + HAND_WRITTEN, encoding="utf-8", newline="\n")
    write_report(str(doc), [("b", "check_b.py", "second plant", "y:2: bad")])
    text = doc.read_text(encoding="utf-8")
    assert HAND_WRITTEN in text, "the generator overwrote a hand-written section"
    assert "| b | check_b.py |" in text and "| a | check_a.py |" not in text, "the table was not regenerated"


def test_report_regeneration_refuses_a_document_without_markers(tmp_path):
    doc = tmp_path / "SABOTAGE.md"
    original = "# Sabotage record\n\nhand-written only\n"
    doc.write_text(original, encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="marker"):
        write_report(str(doc), [("a", "check_a.py", "p", "x")])
    assert doc.read_text(encoding="utf-8") == original


def test_the_committed_doc_has_exactly_one_generated_table():
    doc = open(os.path.join(REPO, "checks", "SABOTAGE.md"), encoding="utf-8").read()
    assert doc.count(TABLE_BEGIN) == 1 and doc.count(TABLE_END) == 1 and doc.index(TABLE_BEGIN) < doc.index(TABLE_END)


def test_sabotage_doc_lists_every_case():
    doc = open(os.path.join(REPO, "checks", "SABOTAGE.md"), encoding="utf-8").read()
    missing = [c[0] for c in CASES if f"| {c[0]} |" not in doc]
    assert not missing, f"checks/SABOTAGE.md is missing cases: {missing} (regenerate with SABOTAGE_REPORT)"
