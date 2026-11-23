# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Run every release checker on the CURRENT tree, and unit-test each checker's rules.

The sabotage proof (planting a violation in a copy) lives in test_sabotage.py.  The SPDX, version and strict-runtime
error-path gates run as ordinary tests.
"""

from __future__ import annotations

import os

import pytest
from conftest import REPO, run_tool

import check_abi_append as abi  # noqa: E402
import check_error_paths as errors  # noqa: E402  (checks/ is put on sys.path by conftest)
import check_no_egress as egress  # noqa: E402
import check_size as size  # noqa: E402
import check_spdx as spdx  # noqa: E402
import check_version as version  # noqa: E402

def run_checker(name: str, *args: str) -> tuple[int, str]:
    return run_tool(REPO, f"checks/{name}", *args)


@pytest.mark.parametrize("script", ["check_no_local_refs.py", "check_no_egress.py", "check_error_paths.py", "check_size.py", "check_abi_append.py", "check_compat_edges.py"])
def test_checker_is_green_on_the_current_tree(script):
    code, out = run_checker(script)
    assert code == 0, out[-2000:]


@pytest.mark.parametrize("script", ["check_spdx.py", "check_version.py", "check_links.py"])
def test_merge_dependent_checker_is_green_on_the_current_tree(script):
    code, out = run_checker(script)
    assert code == 0, out[-2000:]


def test_runtime_scopes_have_no_swallowed_errors():
    code, out = run_checker("check_error_paths.py", "--strict-runtime")
    assert code == 0, out[-2000:]


def test_egress_prints_every_allowed_compat_hit():
    code, out = run_checker("check_no_egress.py")
    assert code == 0 and "allowed: compat/strata/install/build.py" in out and "loopback" in out


def test_egress_runtime_scopes_have_zero_network_apis():
    bad, _ = egress.scan(REPO)
    assert bad == []


# --- error paths -----------------------------------------------------------------------------------------------------
def kinds(rel: str, text: str) -> list[str]:
    return [k for _, k in errors.file_hits(rel, text)]


def test_cpp_catch_rules():
    assert kinds("a.cpp", "void f(){ try{g();}catch(...){ } }") == ["catch-without-log"]
    assert kinds("a.cpp", 'void f(){ try{g();}catch(const E& e){ fprintf(stderr, "x %s", e.what()); } }') == []
    assert kinds("a.cpp", "void f(){ try{g();}catch(...){ throw; } }") == []
    assert kinds("a.cpp", 'void f(){ try{g();}catch(...){ /* "}" brace in a comment */ } }') == ["catch-without-log"]


def test_cpp_void_cast_of_a_call_is_flagged_but_unused_parameter_is_not():
    assert kinds("a.cpp", "void f(int x){ (void)g(x); }") == ["void-cast-of-call"]
    assert kinds("a.cpp", "void f(int x){ (void)x; }") == []


def test_python_handler_rules():
    assert kinds("a.py", "try:\n    f()\nexcept:\n    raise\n") == ["bare-except"]
    assert kinds("a.py", "try:\n    f()\nexcept OSError:\n    pass\n") == ["silent-except"]
    assert kinds("a.py", "for x in y:\n    try:\n        f()\n    except OSError:\n        continue\n") == ["silent-except"]
    assert kinds("a.py", "try:\n    f()\nexcept OSError as e:\n    log.error(e)\n") == []
    assert kinds("a.py", "def (:\n") == ["unparseable-python"]


def test_rust_rules():
    assert kinds("a.rs", "let r = x.map_err(|_| Error::Bad);") == ["map_err-discards-error"]
    assert kinds("a.rs", "    let _ = send(x);") == ["let-underscore-call"]
    assert kinds("a.rs", "let _guard = lock();\nlet _ = &x;\n// let _ = f();") == []


def test_allow_tag_exempts_and_is_listed(tmp_path):
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "a.cpp").write_text("void f(){ try{g();}catch(...){ } }  // allow-error-path: probe must not throw\n")
    viol, allowed, runtime, _ = errors.scan(str(tmp_path), ("examples/",))
    assert viol == {} and runtime == {} and allowed == ["allowed: examples/a.cpp:1: probe must not throw"]


# --- size, version, spdx units -----------------------------------------------------------------------------------------
def test_size_judge_rules():
    bad, notes = size.judge({"a.py": 501, "b.py": 450, "c.py": 520}, {"c.py": 520}, 500, {"b.py": 450})
    assert [b.split(":")[0] for b in bad] == ["a.py"] and any("excepted file c.py" in n for n in notes)
    bad, _ = size.judge({"b.py": 450}, {}, 500)  # R8 L6: over the 400-line split point with no ratchet row is a failure, not a note
    assert [b.split(":")[0] for b in bad] == ["b.py"] and "not in checks/size_ratchet.json" in bad[0]
    bad, _ = size.judge({"c.py": 400}, {"c.py": 520}, 500)
    assert bad and "stale" in bad[0]


@pytest.mark.parametrize(
    "entry, error",
    [
        ({"ceiling": 751, "why": "a flat table that reads worse split"}, "ceiling must be"),  # past HARD_MAX
        ({"ceiling": 500, "why": "a flat table that reads worse split"}, "ceiling must be"),  # not an exception
        ({"ceiling": 600, "why": "short"}, "needs a reason"),
        ({"ceiling": 600}, "needs a reason"),
        (600, "ceiling must be"),  # the old bare-count form carries no reason
    ],
)
def test_size_allow_entries_are_validated(tmp_path, entry, error):
    import json

    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "size_allow.json").write_text(json.dumps({"src/big.py": entry}), encoding="utf-8")
    ceilings, _, errors = size.load_allow(str(tmp_path))
    assert not ceilings and len(errors) == 1 and error in errors[0]


def test_size_allow_valid_entry_is_loaded(tmp_path):
    (tmp_path / "checks").mkdir()
    text = '{"src/big.py": {"ceiling": 750, "why": "one protocol state machine; splitting hides transitions"}}'
    (tmp_path / "checks" / "size_allow.json").write_text(text, encoding="utf-8")
    ceilings, reasons, errors = size.load_allow(str(tmp_path))
    assert ceilings == {"src/big.py": 750} and reasons["src/big.py"].startswith("one protocol") and not errors


def test_version_pyproject_dynamic_ok_and_literal():
    assert version.pyproject_problem('[project]\ndynamic = ["version"]\n[tool.setuptools.dynamic]\nversion = {file = "VERSION"}\n', "1.0.0") is None
    assert version.pyproject_problem('[project]\nversion = "1.0.0"\n', "1.0.0") is None
    assert version.pyproject_problem("[project]\nname = 'x'\n", "1.0.0")


def test_version_skips_missing_files_with_a_note(tmp_path):
    (tmp_path / "VERSION").write_text("1.2.3-alpha.1\n")
    bad, notes = version.check(str(tmp_path))
    assert bad == [] and sum(n.startswith("skip:") for n in notes) == 4


def test_version_fails_when_cargo_lock_lags_cargo_toml(tmp_path):
    # `cargo build --locked` refuses a stale lock entry, so a bump that skipped Cargo.lock must not pass this checker.
    (tmp_path / "VERSION").write_text("1.2.3-alpha.1\n")
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "seabraid"\nversion = "1.2.3-alpha.1"\n')
    lock = '# generated\nversion = 4\n\n[[package]]\nname = "seabraid"\nversion = "%s"\n'
    (tmp_path / "Cargo.lock").write_text(lock % "1.2.3-alpha.1")
    assert version.check(str(tmp_path))[0] == []
    (tmp_path / "Cargo.lock").write_text(lock % "1.2.2")
    assert version.check(str(tmp_path))[0] == ["Cargo.lock: found 1.2.2, expected 1.2.3-alpha.1"]
    (tmp_path / "Cargo.lock").write_text("version = 4\n")
    assert version.check(str(tmp_path))[0] == ["Cargo.lock: no version found, expected 1.2.3-alpha.1"]


def test_version_fails_when_cargo_lock_is_missing_next_to_cargo_toml(tmp_path):
    # Builds use `--locked`; a tree without the lock cannot build, so this is a failure, not a skip.
    (tmp_path / "VERSION").write_text("1.2.3-alpha.1\n")
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "seabraid"\nversion = "1.2.3-alpha.1"\n')
    bad, notes = version.check(str(tmp_path))
    assert bad == ["Cargo.lock: missing, but Cargo.toml exists and builds use --locked"]
    assert not any("Cargo.lock" in n for n in notes)


def test_spdx_expected_expression_per_path():
    assert spdx.expected("include/seabraid/seabraid.h") == "Apache-2.0 OR MIT"
    assert spdx.expected("adapters/ggml/seabraid_ggml.h") == "Apache-2.0"
    assert spdx.expected("src/lib.rs") == "Apache-2.0"
    assert spdx.found_expression("/* SPDX-License-Identifier: Apache-2.0 OR MIT */\n") == "Apache-2.0 OR MIT"
    assert spdx.found_expression("a\nb\nc\nd\ne\n// SPDX-License-Identifier: Apache-2.0\n") is None


def test_patch_files_and_fixtures_are_not_source():
    import _common

    assert not _common.is_source("compat/strata/patches/strata/0001-x.patch")
    assert not _common.is_source("compat/strata/tests/install/fixtures/x.py")
    assert _common.is_source("cmake/SeabraidConfig.cmake.in") and _common.is_source("native-tests/CMakeLists.txt")


def test_the_no_git_file_list_skips_any_venv_and_cmake_tree_by_marker(tmp_path):
    # The 2026-11-19 journey rerun: the README wheel recipe's .venv-wheel made five checkers scan site-packages.
    import _common

    for rel, text in (("src/a.c", "int a;"), (".venv-wheel/pyvenv.cfg", "x"), (".venv-wheel/lib/m.py", "x"),
                      ("myenv/pyvenv.cfg", "x"), ("myenv/bin/p", "x"), ("out/CMakeCache.txt", "x"), ("out/o.c", "x")):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    assert _common.tracked(str(tmp_path)) == ["src/a.c"]


def test_the_no_git_walk_scans_only_the_shipped_top_level_entries(tmp_path):
    # The 2026-11-19 review: an SDK installed into the tree (--prefix sdk) and a logs/ directory made run_all.py and 17 root
    # tests fail in an unpacked archive, because the walk read every directory that was not pruned by name.
    import _common

    for rel in ("src/a.c", "README.md", "sdk/include/seabraid/seabraid.h", "sdk/lib/cmake/Seabraid/SeabraidConfig.cmake", "logs/run.log",
                "notes.txt", "src/nested/b.c", "docs/sdk/x.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    assert _common.tracked(str(tmp_path)) == ["README.md", "docs/sdk/x.md", "src/a.c", "src/nested/b.c"]


def test_the_shipped_top_level_list_is_exactly_what_git_tracks_at_the_top_level():
    # One owner (_common.SHIPPED_TOP_LEVEL); this is its checker. A new top-level file or directory must be added there,
    # or the no-.git walk would silently skip it. Needs a git checkout: an unpacked archive has no list to compare with.
    import subprocess

    import _common

    if not os.path.exists(os.path.join(REPO, ".git")):
        pytest.skip("needs a git checkout (git ls-files is the reference list)")
    out = subprocess.run(["git", "-C", REPO, "ls-files", "-z"], capture_output=True, check=True).stdout.decode("utf-8")
    top = {p.split("/")[0] for p in out.split("\0") if p}
    assert top == set(_common.SHIPPED_TOP_LEVEL), (sorted(top - _common.SHIPPED_TOP_LEVEL), sorted(_common.SHIPPED_TOP_LEVEL - top))


def test_baselines_exist_and_are_valid_json():
    import json

    for name in ("error_paths_baseline.json", "size_allow.json"):
        with open(os.path.join(REPO, "checks", name), encoding="utf-8") as fh:
            assert isinstance(json.load(fh), dict)


# --- abi append ratchet ----------------------------------------------------------------------------------------------
def test_abi_growth_with_struct_size_and_an_appended_field_is_allowed():
    old = {"s": 16, "s.abi_version": 0, "s.struct_size": 4, "s.a": 8}
    new = {**old, "s": 24, "s.b": 16, "t": 8, "t.x": 0}
    assert abi.compare(old, new) == []


def test_abi_field_inside_the_old_struct_and_removed_rows_fail():
    old = {"s": 16, "s.struct_size": 4, "s.a": 8}
    assert len(abi.compare(old, {**old, "s": 24, "s.b": 12})) == 1
    assert abi.compare(old, {"s": 16, "s.struct_size": 4})[0].startswith("s.a: row removed")
    assert abi.compare(old, {**old, "s": 8})[0].startswith("s: size shrank")


def _abi_tree(tmp_path, epoch, snapshots):
    (tmp_path / "include" / "seabraid").mkdir(parents=True)
    (tmp_path / "include" / "seabraid" / "seabraid.h").write_text(f"#define SEABRAID_ABI_EPOCH {epoch}u\n")
    (tmp_path / "src" / "layout").mkdir(parents=True)
    (tmp_path / "src" / "layout" / "abi_layout.txt").write_text("s 8\ns.struct_size 4\ns.a 0\n")
    for n, body in snapshots.items():
        (tmp_path / "src" / "layout" / f"abi_layout.epoch{n}.txt").write_text(body)
    return str(tmp_path)


def test_abi_uses_the_snapshot_of_the_current_epoch_and_fails_without_one(tmp_path):
    root = _abi_tree(tmp_path, 2, {1: "s 8\ns.struct_size 4\ns.a 0\n"})
    bad, _ = abi.check(root)
    assert len(bad) == 1 and "no snapshot for epoch 2" in bad[0]
    (tmp_path / "src" / "layout" / "abi_layout.epoch2.txt").write_text("s 8\ns.struct_size 4\ns.a 0\n")
    assert abi.check(root)[0] == []


def test_abi_compares_against_the_newest_snapshot_not_every_older_one(tmp_path):
    # epoch 1 had a row that epoch 2 legitimately dropped; only the epoch-2 snapshot may judge the current table.
    root = _abi_tree(tmp_path, 2, {1: "s 8\ns.struct_size 4\ns.a 0\ngone 4\n", 2: "s 8\ns.struct_size 4\ns.a 0\n"})
    assert abi.check(root)[0] == []
    (tmp_path / "src" / "layout" / "abi_layout.epoch2.txt").write_text("s 8\ns.struct_size 4\ns.a 4\n")
    assert "offset moved 4 -> 0" in abi.check(root)[0][0]


# --- link gate -------------------------------------------------------------------------------------------------------
def test_link_slug_follows_github_rules():
    import check_links as links

    assert links.slug("Version 2 entry points") == "version-2-entry-points"
    assert links.slug("The `ffi` layer: A & B (v2)!") == "the-ffi-layer-a--b-v2"
    assert links.slug("See [the ABI](ABI.md) *now*") == "see-the-abi-now"
    assert links.anchors("# A\n## A\n### A\n```\n# not a heading\n```\n<a name=\"Old\"></a>\n") == {"a", "a-1", "a-2", "old"}


def _link_tree(tmp_path, readme: str):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ABI.md").write_text("# ABI\n## Epoch rules\n")
    (tmp_path / "docs" / "x.rs").write_text("// x\n")
    (tmp_path / "README.md").write_text(readme)
    return str(tmp_path)


def test_links_resolve_files_dirs_anchors_and_skip_urls_code_and_fences(tmp_path):
    import check_links as links

    scheme = "ht" + "tps:"  # assembled: the egress gate scans this file and a URL literal would be a hit
    root = _link_tree(
        tmp_path,
        f"# Top\n[a](docs/ABI.md#epoch-rules) [b](docs/) [c](#top) [d]({scheme}//x/nope.md#z) [e](docs/x.rs#L10)\n"
        "![i](<docs/ABI.md>) `[code](nope.md)`\n```\n[fenced](nope.md)\n```\n[r]: docs/ABI.md\n",
    )
    assert links.scan(root) == []


def test_links_report_missing_file_anchor_escape_and_reference_definitions(tmp_path):
    import check_links as links

    root = _link_tree(tmp_path, "[a](docs/Nope.md)\n[b](docs/ABI.md#nope)\n[c](#nope)\n[d](../outside.md)\n[r]: gone.md\n")
    hits = links.scan(root)
    assert [h.split(": broken link")[0] for h in hits] == ["README.md:1", "README.md:2", "README.md:3", "README.md:4", "README.md:5"]
    assert "leaves the repository" in hits[3]


def _scan_readme(tmp_path, readme: str, extra: dict[str, str] | None = None) -> list[str]:
    import check_links as links

    root = _link_tree(tmp_path, readme)
    for name, body in (extra or {}).items():
        (tmp_path / name).write_text(body)
    return links.scan(root)


def test_links_accept_parentheses_in_a_file_name(tmp_path):
    assert _scan_readme(tmp_path, "[p](docs/a_(b).md) [q](docs/a_(b).md#t)\n", {"docs/a_(b).md": "# T\n"}) == []


def test_links_ignore_a_query_string_on_the_path(tmp_path):
    assert _scan_readme(tmp_path, "[q](README.md?plain=1) [r](docs/ABI.md?x=1#abi)\n") == []


def test_links_slug_drops_emphasis_underscores_but_keeps_snake_case(tmp_path):
    import check_links as links

    assert links.slug("_Emph_ title") == "emph-title" and links.slug("__init__ rules") == "init-rules"
    assert links.slug("the snake_case name") == "the-snake_case-name"
    assert _scan_readme(tmp_path, "[e](#emph-title) [s](#a_b)\n\n## _Emph_ title\n\n## a_b\n") == []


def test_links_skip_a_fence_indented_inside_a_list_item(tmp_path):
    assert _scan_readme(tmp_path, "- item\n\n    ```\n    [x](nope.md)\n    ```\n") == []


def test_links_a_longer_fence_is_not_closed_by_a_shorter_one(tmp_path):
    readme = "````\n```\n[inside](nope.md)\n```\n````\n[after](nope2.md)\n"
    assert [h.split(": broken link")[0] for h in _scan_readme(tmp_path, readme)] == ["README.md:6"]  # only the real one


def test_links_check_html_href_and_src_attributes(tmp_path):
    hits = _scan_readme(tmp_path, '<a href="gone.md">x</a> <img src="gone.png"> <a href="docs/ABI.md">ok</a>\n')
    assert len(hits) == 2 and "gone.md" in hits[0] and "gone.png" in hits[1]


# --- name gate -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["kName", "NAMEX", "name_open", "libname", "-lname", "Name::x", "xNaMex"])
def test_name_regex_matches_the_bare_token_with_no_boundary(text):
    import _common

    assert _common.name_regex("name").search(text)


def test_name_gate_scans_every_tracked_file_and_honours_the_allow_list(tmp_path):
    import check_names

    probe = "zz" + "rig"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text(f"see k{probe.capitalize()}\n")
    (tmp_path / "CMakeLists.txt").write_text(f"project({probe.upper()}X)\n")
    hits = check_names.scan(str(tmp_path), [probe])
    assert {h.split(":")[0] for h in hits} == {"docs/a.md", "CMakeLists.txt"}
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "names_allow.txt").write_text("docs/a.md\tthe lineage note must keep the old name\nCMakeLists.txt\tx\n")
    hits = check_names.scan(str(tmp_path), [probe])
    assert len(hits) == 1 and "needs a why" in hits[0]


def test_name_gate_skips_whole_words_listed_in_excluded_words(monkeypatch, tmp_path):
    import _common
    import check_names

    probe = "zz" + "rig"
    monkeypatch.setitem(_common.EXCLUDED_WORDS, probe, (f"b{probe}",))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text(f"b{probe} but not k{probe}\n")
    hits = check_names.scan(str(tmp_path), [probe])
    assert len(hits) == 1 and hits[0].startswith("docs/a.txt:1:")


def test_the_tree_copy_refuses_a_top_level_entry_the_walk_would_skip(tmp_path, monkeypatch):
    # R6b #11: the copy used by the sabotage and rename tests is walked WITHOUT .git, which reads only SHIPPED_TOP_LEVEL;
    # a new untracked top-level file is copied and then silently not checked unless the copy refuses it by name.
    import conftest

    monkeypatch.setattr(conftest._common, "tracked", lambda root: ["README.md", "brand_new_dir/x.c"])
    with pytest.raises(pytest.fail.Exception, match="brand_new_dir"):
        conftest.copy_tree(str(tmp_path / "copy"))
    assert not (tmp_path / "copy").exists(), "nothing may be copied before the refusal"
