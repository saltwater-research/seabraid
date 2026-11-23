# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""File-reference moves, installer side (FR DESIGN 8/9): opt-in switch, the manifest built once per pack by seabraid-sums, the
flags it adds, and the refusals.  No GPU, no host, no real tool: the sums tool is an injected callable."""
from __future__ import annotations

import dataclasses
import os

import pytest
from test_config import host_paths, host_report
from test_services import mkplan

from install import _file_moves as F
from install import config as C


def make(n_help, opts):
    return C.make_config(host_report(), mkplan(n_help), host_paths(), opts, isdir=lambda p: False)


# ---- config: the switch, the flags -------------------------------------------------------------------------------------

def test_file_moves_is_off_by_default_and_adds_nothing():
    assert C.ConfigOptions().file_moves is False
    cfg = make(2, C.ConfigOptions(ladder=True))
    assert "--retier-file-moves" not in cfg.cfg_json["args"]
    assert all("--file-io" not in a for a in cfg.helper_args.values())


def test_file_moves_with_the_ladder_adds_the_engine_flag_and_file_io_auto_to_every_helper():
    cfg = make(2, C.ConfigOptions(ladder=True, file_moves=True))
    args = cfg.cfg_json["args"]
    assert args[args.index("--retier-file-moves") + 1] == "1"
    assert set(cfg.helper_args) == {0, 1}
    for h in cfg.helper_args.values():
        assert h[h.index("--file-io") + 1] == "auto"


def test_file_moves_without_the_ladder_is_refused_and_names_the_flag():
    with pytest.raises(C.ConfigRefused, match="--ladder"):
        make(2, C.ConfigOptions(file_moves=True))


def test_file_moves_beside_a_hand_given_retier_file_moves_is_refused_as_a_second_owner():
    with pytest.raises(C.ConfigRefused, match="--retier-file-moves"):
        make(2, C.ConfigOptions(ladder=True, file_moves=True, extra_args=("--retier-file-moves", "0")))


def test_options_stay_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        C.ConfigOptions().file_moves = True


# ---- the manifest ------------------------------------------------------------------------------------------------------

GEOMETRY = dict(native=None, layers=7, experts=13)
GEOMETRY_ARGS = ("--layers", "7", "--experts", "13")


class Tool:
    """A fake seabraid-sums: records argv, and behaves per its rcs (build, verify)."""

    def __init__(self, pack, build_rc=0, verify_rc=0, writes=True):
        self.pack, self.calls, self.build_rc, self.verify_rc, self.writes = pack, [], build_rc, verify_rc, writes

    def __call__(self, argv):
        self.calls.append(tuple(argv))
        if "--verify" in argv:
            return self.verify_rc
        if self.build_rc == 0 and self.writes:
            with open(os.path.join(self.pack, F.MANIFEST), "wb") as fh:
                fh.write(b"SBSUMS1\0")
        return self.build_rc


def test_a_missing_manifest_is_built_once_with_seabraid_sums(tmp_path):
    t = Tool(str(tmp_path))
    assert F.ensure_manifest(str(tmp_path), "/x/seabraid-sums", t, dry_run=False, **GEOMETRY) == F.BUILT
    assert t.calls == [("/x/seabraid-sums", str(tmp_path), *GEOMETRY_ARGS)]
    assert (tmp_path / F.MANIFEST).exists()


def test_a_fresh_manifest_is_verified_not_rebuilt(tmp_path):
    (tmp_path / F.MANIFEST).write_bytes(b"SBSUMS1\0")
    t = Tool(str(tmp_path))
    assert F.ensure_manifest(str(tmp_path), "/x/seabraid-sums", t, dry_run=False, **GEOMETRY) == F.FRESH
    assert t.calls == [("/x/seabraid-sums", str(tmp_path), *GEOMETRY_ARGS, "--verify")]


def test_a_stale_manifest_is_rebuilt(tmp_path):
    (tmp_path / F.MANIFEST).write_bytes(b"old")
    t = Tool(str(tmp_path), verify_rc=1)
    assert F.ensure_manifest(str(tmp_path), "/x/seabraid-sums", t, dry_run=False, **GEOMETRY) == F.BUILT
    assert [("--verify" in c) for c in t.calls] == [True, False]


def test_a_manifest_that_cannot_be_built_refuses_and_says_how_to_run_it_by_hand(tmp_path):
    with pytest.raises(F.FileMovesRefused, match="seabraid-sums"):
        F.ensure_manifest(str(tmp_path), "/x/seabraid-sums", Tool(str(tmp_path), build_rc=2), dry_run=False, **GEOMETRY)


def test_a_tool_that_claims_success_but_wrote_nothing_refuses(tmp_path):
    with pytest.raises(F.FileMovesRefused, match=F.MANIFEST):
        F.ensure_manifest(str(tmp_path), "/x/seabraid-sums", Tool(str(tmp_path), writes=False), dry_run=False, **GEOMETRY)


def test_no_tool_refuses_naming_the_build_that_makes_it(tmp_path):
    with pytest.raises(F.FileMovesRefused, match="not built"):
        F.ensure_manifest(str(tmp_path), "", Tool(str(tmp_path)), dry_run=False, **GEOMETRY)


def test_a_dry_run_runs_nothing_and_reports_what_it_would_do(tmp_path):
    t = Tool(str(tmp_path))
    assert F.ensure_manifest(str(tmp_path), "", t, dry_run=True, **GEOMETRY) == F.WOULD_BUILD
    assert t.calls == [] and not (tmp_path / F.MANIFEST).exists()


def test_native_manifest_reuses_the_helper_inputs_on_verify_and_rebuild(tmp_path, monkeypatch):
    from types import SimpleNamespace

    (tmp_path / F.MANIFEST).write_bytes(b"stale")
    tool = Tool(str(tmp_path), verify_rc=1)
    monkeypatch.setattr(F, "find_tool", lambda _: "/x/seabraid-sums")
    monkeypatch.setattr("subprocess.run", lambda argv: SimpleNamespace(returncode=tool(argv)))

    _stage(dict(ladder=True, dry_run=False), tmp_path)

    model = mkplan(1).model
    argv = ("/x/seabraid-sums", str(tmp_path), "--layers", str(model.layers), "--experts", str(model.experts),
            "--native", str(tmp_path / "first shard.gguf"))
    assert tool.calls == [(*argv, "--verify"), argv]


def test_find_tool_looks_under_the_strata_build_dirs(tmp_path):
    d = tmp_path / "build-nvidia" / "seabraid"
    d.mkdir(parents=True)
    (d / "seabraid-sums").write_text("x")
    assert F.find_tool(str(tmp_path)).endswith("seabraid-sums")
    assert F.find_tool(str(tmp_path / "nowhere")) == ""


# ---- the CLI stage -----------------------------------------------------------------------------------------------------

def _stage(args, tmp_path):
    from types import SimpleNamespace

    from install import _setup_stages as ST
    paths = SimpleNamespace(pack=str(tmp_path), native=str(tmp_path / "first shard.gguf"))
    ST._manifest_for(SimpleNamespace(**args), paths, mkplan(1).model, str(tmp_path))


def test_the_cli_flag_parses_and_defaults_off():
    from install import seabraid_setup as S
    assert S.build_parser().parse_args(["--file-moves"]).file_moves is True
    assert S.build_parser().parse_args([]).file_moves is False


def test_the_stage_refuses_file_moves_without_the_ladder_before_any_hashing(tmp_path):
    from install._setup_support import Refused
    with pytest.raises(Refused, match="--ladder") as e:
        _stage(dict(ladder=False, dry_run=False), tmp_path)
    assert e.value.status == 3


def test_the_stage_refuses_with_status_3_when_no_tool_is_built(tmp_path):
    from install._setup_support import Refused
    with pytest.raises(Refused, match="not built") as e:
        _stage(dict(ladder=True, dry_run=False), tmp_path)
    assert e.value.status == 3


def test_the_stage_dry_run_prints_and_runs_nothing(tmp_path, capsys):
    _stage(dict(ladder=True, dry_run=True), tmp_path)
    assert "would be built or verified" in capsys.readouterr().out


@pytest.mark.parametrize("subdir", ["seabraid-strata-ext", "seabraid", "", "bin"])
def test_sums_lookup_supports_current_and_legacy_build_outputs(tmp_path, subdir):
    directory = tmp_path / "build-nvidia" / subdir
    directory.mkdir(parents=True)
    tool = directory / F.TOOL
    tool.write_text("binary")
    assert F.find_tool(str(tmp_path)) == str(tool)


def test_sums_lookup_rejects_a_directory_named_like_the_tool(tmp_path):
    (tmp_path / "build-nvidia/seabraid-strata-ext" / F.TOOL).mkdir(parents=True)
    assert F.find_tool(str(tmp_path)) == ""
