# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""scripts/add_spdx.py on temp trees only: placement, comment syntax, line endings, idempotence, and the whole repo copy."""

from __future__ import annotations

import compileall
import os

from conftest import REPO, run_tool

HDR = "Copyright 2026 Saltwater Research LLC"
SPDX = "SPDX-License-Identifier:"


def write(root, rel, data: bytes):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def read(root, rel) -> bytes:
    with open(os.path.join(root, rel), "rb") as fh:
        return fh.read()


def add(root, *flags):
    return run_tool(REPO, "scripts/add_spdx.py", "--root", str(root), *flags)


def test_slash_languages_use_double_slash_and_the_right_expression(tmp_path):
    root = str(tmp_path)
    for rel, data in {"src/a.rs": b"fn main() {}\n", "adapters/x/a.cpp": b"#include <x>\n", "include/seabraid/seabraid.h": b"#pragma once\n"}.items():
        write(root, rel, data)
    code, out = add(root, "--write")
    assert code == 0, out
    assert read(root, "src/a.rs").startswith(f"// {HDR}\n// {SPDX} Apache-2.0\n".encode())
    assert read(root, "adapters/x/a.cpp").startswith(f"// {HDR}\n// {SPDX} Apache-2.0\n".encode())
    assert read(root, "include/seabraid/seabraid.h").startswith(f"// {HDR}\n// {SPDX} Apache-2.0 OR MIT\n".encode())


def test_hash_languages_keep_shebang_and_encoding_line_first(tmp_path):
    root = str(tmp_path)
    for rel, data in {
        "tools/run.sh": b"#!/usr/bin/env bash\nset -e\n",
        "tools/a.py": b"#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n'''doc'''\n",
        "tools/b.py": b"# -*- coding: latin-1 -*-\nimport os\n",
        "tools/c.py": b"import os\n",
        "cmake/Pkg.cmake.in": b"@PACKAGE_INIT@\n",
        "CMakeLists.txt": b"project(x)\n",
    }.items():
        write(root, rel, data)
    code, out = add(root, "--write")
    assert code == 0, out
    assert read(root, "tools/run.sh").split(b"\n")[:3] == [b"#!/usr/bin/env bash", f"# {HDR}".encode(), f"# {SPDX} Apache-2.0".encode()]
    py = read(root, "tools/a.py").split(b"\n")
    assert py[:2] == [b"#!/usr/bin/env python3", b"# -*- coding: utf-8 -*-"]
    assert py[2:4] == [f"# {HDR}".encode(), f"# {SPDX} Apache-2.0".encode()]
    assert read(root, "tools/b.py").split(b"\n")[0] == b"# -*- coding: latin-1 -*-"
    for rel in ("tools/c.py", "cmake/Pkg.cmake.in", "CMakeLists.txt"):
        assert read(root, rel).startswith(f"# {HDR}\n".encode()), rel


def test_patches_and_fixtures_are_left_alone(tmp_path):
    root = str(tmp_path)
    files = {"a.patch": b"--- a\n+++ b\n", "compat/t/fixtures/f.py": b"x = 1\n"}
    for rel, data in files.items():
        write(root, rel, data)
    code, out = add(root, "--write")
    assert code == 0 and "0 file(s)" in out
    assert all(read(root, rel) == data for rel, data in files.items())


def test_idempotent_and_dry_run_writes_nothing(tmp_path):
    root = str(tmp_path)
    write(root, "src/a.rs", b"fn main() {}\n")
    code, out = add(root)
    assert code == 0 and "would update: src/a.rs" in out and read(root, "src/a.rs") == b"fn main() {}\n"
    add(root, "--write")
    once = read(root, "src/a.rs")
    code, out = add(root, "--write")
    assert code == 0 and "0 file(s)" in out and read(root, "src/a.rs") == once


def test_crlf_and_bom_are_preserved(tmp_path):
    root = str(tmp_path)
    write(root, "src/a.rs", b"\xef\xbb\xbffn main() {}\r\n")
    write(root, "src/b.rs", b"fn b() {}\r\nfn c() {}\r\n")
    add(root, "--write")
    assert read(root, "src/a.rs").startswith(b"\xef\xbb\xbf// " + HDR.encode() + b"\r\n")
    assert read(root, "src/b.rs") == f"// {HDR}\r\n// {SPDX} Apache-2.0\r\nfn b() {{}}\r\nfn c() {{}}\r\n".encode()
    assert b"\n" not in read(root, "src/b.rs").replace(b"\r\n", b"")  # no bare LF crept in


def test_wrong_expression_is_reported_not_rewritten(tmp_path):
    root = str(tmp_path)
    write(root, "include/seabraid/seabraid.h", f"// {SPDX} MIT\n".encode())
    code, out = add(root, "--write")
    assert code == 1 and "not rewritten" in out and read(root, "include/seabraid/seabraid.h") == f"// {SPDX} MIT\n".encode()


def test_undecodable_source_file_is_a_problem_not_a_skip(tmp_path):
    root = str(tmp_path)
    write(root, "src/bad.rs", b"\xff\xfe\x00bad")
    code, out = add(root, "--write")
    assert code == 1 and "cannot read" in out


def test_whole_repo_copy_becomes_clean_compiles_and_second_run_is_a_no_op(tree):
    code, out = run_tool(REPO, "scripts/add_spdx.py", "--root", tree, "--write")
    assert code == 0 and "problem" not in out.replace("0 problem(s)", "")
    code, out = run_tool(tree, "checks/check_spdx.py", "--root", tree)
    assert code == 0, out[-800:]
    code, out = run_tool(REPO, "scripts/add_spdx.py", "--root", tree, "--write")
    assert "0 file(s) updated" in out
    assert compileall.compile_dir(os.path.join(tree, "compat"), quiet=2, force=True, workers=1) and compileall.compile_dir(os.path.join(tree, "bindings"), quiet=2, force=True)
