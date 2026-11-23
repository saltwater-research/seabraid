# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid_setup hands the measurement to PLAN: --bench measures and the plan says so; the default never measures and says how to opt in."""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import pytest

from install import _bench as BN
from install import seabraid_setup as S
from install import plan as P

G = lambda x: int(x * (1 << 30))  # noqa: E731


@dataclass(frozen=True)
class Gpu:
    index: int
    vendor: str
    name: str
    arch: str
    vram_total_bytes: int
    vram_free_bytes: int
    gtt_total_bytes: int = G(30)
    pcie_gen_max: int = 4
    pcie_width_max: int = 16
    uuid: str = None
    holders: tuple = ()


@dataclass(frozen=True)
class Tk:
    vendor: str
    kind: str
    root: str = "/t"
    version: str = "1"
    compiler: str = "/t/cc"
    host_compiler: str = "/usr/bin/gcc"
    host_compiler_tried: tuple = ()
    lib_dirs: tuple = ()


@dataclass(frozen=True)
class Fs:
    mount: str
    free_bytes: int
    rotational: bool
    fstype: str = "ext4"


REPORT = SimpleNamespace(
    gpus=(Gpu(0, "amd", "R9700", "gfx1201", G(32), G(31.5)), Gpu(0, "nvidia", "RTX 3090", "sm_86", G(24), G(23.5))),
    toolkits=(Tk("amd", "rocm"), Tk("nvidia", "cuda")), filesystems=(Fs("/", G(500), False), Fs("/dev/shm", G(15), None, "tmpfs")),
    ram_total_bytes=G(64), ram_available_bytes=G(50), ttm_pages_limit_bytes=G(30), memlock_bytes=8 << 20, blockers=())
FACTS = P.ModelFacts(48, 512, G(5), G(3), 2560, 1510400)


def run_plan_only(extra, measure, report=REPORT):
    paths = SimpleNamespace(native="/m/m.gguf", ple_gguf="", pack="/p", expert_profile="/ep.bin")
    with mock.patch.object(S, "detect_host", return_value=report), mock.patch.object(S, "format_report", return_value=""), \
            mock.patch.object(S, "find_prebuilt", return_value=S.Prebuilt("", {"amd": "/h"})), \
            mock.patch("install._setup_model.list_packs", return_value=["x"]), \
            mock.patch("install._setup_model.discover", return_value=paths), mock.patch("install._setup_model.model_shards", return_value=[paths.native]), \
            mock.patch("install._setup_model.resolve_profile", return_value=paths), \
            mock.patch.object(P, "read_model_facts", return_value=FACTS), mock.patch.object(BN, "measure", measure):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = S.main(["--plan-only", "--yes", *extra])
    return rc, out.getvalue()


def test_a_blocked_card_does_not_hide_another_usable_card():
    from install.detect import Blocker
    report = SimpleNamespace(**{**vars(REPORT), "blockers": (Blocker("gpu:amd:0", "bad AMD", "repair"),)})
    rc, out = run_plan_only([], lambda *a, **k: None, report)
    assert rc == 0
    assert "RTX 3090" in out
    assert "bad AMD" in out

def test_context_override_reaches_the_real_planner():
    calls = []
    original = P.make_plan

    def record(*args, **kwargs):
        calls.append(kwargs["opts"].assumed_max_context)
        return original(*args, **kwargs)

    with mock.patch.object(P, "make_plan", side_effect=record):
        rc, _ = run_plan_only(["--max-context", "65536"], lambda *a, **k: None)
    assert rc == 0 and calls == [65536]

def test_plan_only_passes_the_measurement_to_the_plan_and_says_measured():
    seen = {}

    def measure(report, cands, exes, pack, native, layers, experts, runner):
        seen.update(cands=[(c.vendor, c.index) for c in cands], exes=exes, layers=layers, experts=experts, dry=runner.dry_run)
        return BN.BenchResult({("amd", 0): 100.0, ("nvidia", 0): 400.0}, "")
    rc, out = run_plan_only(["--bench"], measure)
    assert rc == 0 and "card 1 ranked by heuristic, cards 2..N by bench" in out and "main: R9700 by heuristic" in out
    assert "not measured: the lead-job bench was not run" in out and "measured lead job" not in out    # rows/s is not a lead claim
    assert seen == {"cands": [("amd", 0), ("nvidia", 0)], "exes": {"amd": "/h"}, "layers": 48, "experts": 512, "dry": False}


def test_default_never_measures_and_the_plan_says_how_to_opt_in():
    calls = []
    rc, out = run_plan_only([], lambda *a, **k: calls.append(a))
    assert calls == [], "the default run measured (loaded a card) without --bench"
    assert "card 1 ranked by heuristic, cards 2..N by heuristic" in out and "main: R9700 by heuristic" in out
    assert "not measured: pass --bench to measure (it briefly loads every card)" in out
    assert "not measured: not measured" not in out          # the prefix was once printed twice


def test_bench_flag_calls_measure():
    calls = []
    run_plan_only(["--bench"], lambda *a, **k: calls.append(a) or BN.skipped("x"))
    assert len(calls) == 1


def test_a_failed_measurement_falls_back_loudly():
    rc, out = run_plan_only(["--bench"], lambda *a, **k: BN.skipped("RTX 3090 (nvidia gpu0) timed out after 120s: loading"))
    assert "card 1 ranked by heuristic" in out and "not measured: RTX 3090 (nvidia gpu0) timed out" in out


def _lead_measure(*a, **k):
    return BN.BenchResult({("amd", 0): 270000.0, ("nvidia", 0): 226000.0}, "",
                          {1: {("amd", 0): 26.7, ("nvidia", 0): 24.0}, 4: {("amd", 0): 24.0, ("nvidia", 0): 26.0}}, "")


def test_lead_figures_reach_the_plan_per_streams_and_the_user_can_override_card_1():
    rc, out = run_plan_only(["--bench"], _lead_measure)
    assert "card 1 ranked by bench, cards 2..N by bench" in out and "main: RTX 3090 (measured lead job, 1 stream: 24.0 ms/window" in out
    _, four = run_plan_only(["--bench", "--streams", "4"], _lead_measure)
    assert "main: R9700 (measured lead job, 4 streams: 24.0 ms/window" in four
    _, forced = run_plan_only(["--bench", "--main-card", "amd:0"], _lead_measure)          # the bench says 3090; the user said R9700
    assert "card 1 ranked by user" in forced and "main: R9700 (chosen by the user" in forced


def test_a_malformed_main_card_is_refused_before_anything_is_measured():
    calls = []
    rc, _ = run_plan_only(["--bench", "--main-card", "banana"], lambda *a, **k: calls.append(a))
    assert rc == 2 and calls == []


def _engines(work, *dirs):
    """Real files, because the per-backend engine choice reads the work dir."""
    for d in dirs:
        exe = work / "strata" / d / "strata"
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text("")


def _lead_probe_main(extra, tmp_path, measure_lead):
    from install import _bench_lead as BL
    from install import services as SV
    if not (tmp_path / "strata").exists():
        _engines(tmp_path, "build-amd", "build-nvidia")
    pre = S.Prebuilt("/w/strata/build-amd/strata", {"amd": "/h/amd", "nvidia": "/h/nv"})
    from install import config as C
    mp = C.ModelPaths(exe="", cwd=str(tmp_path), pack="/p", native="/m/m.gguf", tokenizer="/t", model_name="m", log="/l", expert_profile="/w/strata/data/expert-profile.bin")
    report = SimpleNamespace(**vars(REPORT), toolkit_for=lambda v: next((t for t in REPORT.toolkits if t.vendor == v), None))   # detect's accessor
    with mock.patch.object(S, "detect_host", return_value=report), mock.patch.object(S, "format_report", return_value=""), \
            mock.patch.object(S, "find_prebuilt", return_value=pre), mock.patch.object(SV, "pick_python", return_value="/py"), \
            mock.patch("install._setup_model.list_packs", return_value=["x"]), mock.patch("install._setup_model.discover", return_value=mp), \
            mock.patch("install._setup_model.model_shards", return_value=[mp.native]), mock.patch("install._setup_model.resolve_profile", return_value=mp), \
            mock.patch.object(S, "_finish", return_value=0), \
            mock.patch.object(P, "read_model_facts", return_value=FACTS), mock.patch.object(BL, "measure_lead", measure_lead):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = S.main(["--yes", "--work-dir", str(tmp_path), *extra])
    return rc, out.getvalue()


def test_bench_lead_builds_a_launch_per_candidate_with_that_card_first_and_ranks_by_it(tmp_path):
    from install import _bench_lead as BL
    seen = {}

    def measure_lead(cands, plan, classes, runner):
        seen["classes"] = list(classes)
        res = {}
        for c in cands:
            pl = plan.placement(c)
            seen[(c.vendor, c.index)] = pl
            res[(c.vendor, c.index)] = BL.LeadResult({1: 24.0 if c.vendor == "nvidia" else 26.7}, {})
        return res
    rc, out = _lead_probe_main(["--bench-lead"], tmp_path, measure_lead)
    assert (rc, seen["classes"]) == (0, [1])
    assert all(t in out for t in ("2 cold engine start(s)", "card 1 ranked by bench",
                                  "main: RTX 3090 (measured lead job, 1 stream: 24.0 ms/window"))
    nv, amd = seen[("nvidia", 0)], seen[("amd", 0)]
    assert nv.cfg_json_path != amd.cfg_json_path
    assert [o.vendor for o in nv.others] == ["amd"]
    assert [o.vendor for o in amd.others] == ["nvidia"]


def test_bench_lead_not_asked_never_starts_an_engine_and_dry_run_says_so(tmp_path):
    calls = []
    _lead_probe_main([], tmp_path, lambda *a, **k: calls.append(a))
    assert calls == []
    rc, out = _lead_probe_main(["--bench-lead", "--dry-run"], tmp_path, lambda *a, **k: calls.append(a))
    assert calls == [] and "the lead-job probe was not run (dry-run)" in out


def test_shard_order_uses_index_not_prefix():
    from install import _setup_model as M
    names = ["zz-00001-of-00002.gguf", "aa-00002-of-00002.gguf"]
    assert M.shard_order(names) == names


def test_shard_order_drops_non_shards():
    from install import _setup_model as M
    assert M.shard_order(["notes.txt", "model.gguf", "tokenizer"]) == []

def _model_layout(tmp_path):
    """One installed pack with a four-shard model; files need only exist for inventory."""
    from install import _setup_model as M
    data = tmp_path / "data"
    (data / "packs" / "iq3_s" / "tokenizer").mkdir(parents=True)
    folder = data / "models" / "IQ3_S"
    folder.mkdir(parents=True)
    for i in range(1, 5):
        (folder / ("model-%05d-of-00004.gguf" % i)).write_text("")
    paths = M.discover(str(data), "iq3_s", "/b/strata", "/b", "/b/x.log")
    return folder, paths


def test_full_inventory_and_missing_middle_shard(tmp_path):
    from install import _setup_model as M
    folder, paths = _model_layout(tmp_path)
    assert len(M.model_shards(paths.native)) == 4
    (folder / "model-00002-of-00004.gguf").unlink()
    with pytest.raises(M.ModelSetupError, match="incomplete"):
        M.model_shards(paths.native)
    with pytest.raises(M.ModelSetupError, match="incomplete"):
        M.discover(str(tmp_path / "data"), "iq3_s", "/b/strata", "/b", "/b/x.log")


def test_profile_family_and_dimension_verification(tmp_path):
    import struct
    from dataclasses import replace
    from install import _setup_model as M
    _, paths = _model_layout(tmp_path)
    work = tmp_path / "work"
    profile_dir = work / "strata" / "data"
    profile_dir.mkdir(parents=True)
    for name, experts in (("expert-profile.bin", 512), ("expert-profile-coder.bin", 256)):
        (profile_dir / name).write_bytes(struct.pack("<4s5I", b"STRP", 1, 48, experts, 0, 0))
    facts = SimpleNamespace(layers=48, experts=512)
    with mock.patch("install._plan_model.read_model_facts", return_value=facts) as read:
        assert M.resolve_profile(paths, str(work)).expert_profile.endswith("expert-profile.bin")
        coder = replace(paths, model_name="coder-iq3_s")
        read.return_value = SimpleNamespace(layers=48, experts=256)
        assert M.resolve_profile(coder, str(work)).expert_profile.endswith("expert-profile-coder.bin")
        read.return_value = facts
        with pytest.raises(M.ModelSetupError, match="dimensions"):
            M.resolve_profile(coder, str(work))
    assert M.resolve_profile(coder, str(tmp_path / "not-built"), require_exists=False).expert_profile.endswith("expert-profile-coder.bin")

def test_discover_finds_the_strata_checkouts_expert_profile_and_config_refuses_without_one(tmp_path):
    """Finding 1 (run13): discover never set expert_profile, so the engine exited 2 at start.  The default layout has it at <work>/strata/data."""
    from install import _setup_model as M
    from install import config as C
    data = tmp_path / "data"
    (data / "packs" / "p" / "tokenizer").mkdir(parents=True)
    (data / "models" / "P").mkdir(parents=True)
    (data / "models" / "P" / "m-00001-of-00001.gguf").write_text("")
    work = tmp_path / "work"
    (work / "strata" / "data").mkdir(parents=True)
    assert M.discover(str(data), "p", "", str(work), "l").expert_profile is None
    (work / "strata" / "data" / "expert-profile.bin").write_text("")
    found = M.discover(str(data), "p", "", str(work), "l")
    assert found.expert_profile.endswith("expert-profile.bin")
    args = C._engine_args(SimpleNamespace(helpers=lambda: (), tiers=SimpleNamespace(ram_bytes=0)), found, 4096, C.ConfigOptions())
    assert args[args.index("--expert-profile") + 1] == found.expert_profile
    with pytest.raises(C.ConfigRefused, match="expert profile"):
        C._engine_args(SimpleNamespace(helpers=lambda: (), tiers=SimpleNamespace(ram_bytes=0)),
                       C.ModelPaths(**{**{f: getattr(found, f) for f in found.__slots__}, "expert_profile": None}), 4096, C.ConfigOptions())


def test_each_candidate_gets_the_engine_built_for_its_backend_or_is_unmeasured(tmp_path):
    """Finding 2 (run13): one HIP engine was started for the CUDA card (exit 127).  Never a mismatched binary."""
    from install import _bench_lead as BL
    _engines(tmp_path, "build-amd", "build-nvidia-main", "build-nvidia")
    assert S.engine_for(str(tmp_path), "amd").endswith("build-amd/strata".replace("/", __import__("os").sep))
    assert "build-nvidia" in S.engine_for(str(tmp_path), "nvidia") and "build-amd" not in S.engine_for(str(tmp_path), "nvidia")
    seen = {}

    def measure_lead(cands, plan, classes, runner):
        for c in cands:
            try:
                seen[c.vendor] = plan.placement(c).cfg_json_path
            except BL.PlacementError as exc:
                seen[c.vendor] = str(exc)
        return {(c.vendor, c.index): BL.LeadResult({}, {1: "x"}) for c in cands}
    _lead_probe_main(["--bench-lead"], tmp_path, measure_lead)
    import json
    assert "build-nvidia" in json.load(open(seen["nvidia"]))["exe"] and "build-amd" in json.load(open(seen["amd"]))["exe"]
    (tmp_path / "strata" / "build-nvidia" / "strata").unlink()
    (tmp_path / "strata" / "build-nvidia-main" / "strata").unlink()
    _lead_probe_main(["--bench-lead"], tmp_path, measure_lead)
    assert "no nvidia engine built" in seen["nvidia"] and "build-amd" in json.load(open(seen["amd"]))["exe"]
