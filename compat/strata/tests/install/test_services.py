# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for install/services.py: unit text for fixture plans, dry-run default, mocked-HTTP health."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from install import _setup_stages
from install import services as S
from install.config import SERVER_ENV
from install.plan import Build, ModelFacts, Plan, Role, TierSizes


@dataclass
class Cfg:                                  # stand-in for config's ConfigOutput (provisional contract)
    cfg_json_path: str = "/data/cfg/seabraid.json"
    # What config.make_config really hands over for two helpers: the full list and the fixed switches (services only renders them).
    server_env: dict = field(default_factory=lambda: {"HSA_SVM_FOR_PAGED_MEM": "0", "SEABRAID_STRATA_SHM": "/seabraid0,/seabraid1", **SERVER_ENV})
    helper_args: dict = field(default_factory=lambda: {0: ["--pack", "/data/pack", "--layers", "7", "--experts", "13"],
                                                       1: ["--pack", "/data/pack", "--device", "9"]})
    helper_env: dict = field(default_factory=lambda: {0: {"X": "1"}, 1: {}})
    port: int = 18080


def mkplan(n_help=2) -> Plan:
    roles = [Role("main", 0, "amd", "R9700", "gfx1201", 0, None, 1)]
    roles += [Role("helper", 1 + i, "nvidia", f"N{i}", "sm_86", 3 + i, i, 1) for i in range(n_help)]
    builds = (Build("amd", "rocm", ("strata",), ("gfx1201",), (0,), "/r", "7", "/r/llvm/bin/clang++", None, ("/r/lib", "/r/lib2")),
              Build("nvidia", "cuda", ("seabraid-helper",), ("sm_86",), (1, 2), "/c", "12", "/c/nvcc", "/h/gcc-13", ()))
    return Plan(ModelFacts(7, 13, 1, 1, 8), "all-vram", tuple(roles), builds, TierSizes({}, 0, 0, None))


FAKE_FS = {"/r/llvm/bin/clang++"}                  # fake ROCm root /r: classic layout, no rocm_sysdeps
render = lambda *a, **kw: S.render_units(*a, exists=FAKE_FS.__contains__, **kw)    # noqa: E731

STRATA = "/opt/bl/strata"
PY = "/opt/bl/strata/.venv/bin/python"
HELPERS = {"nvidia": "/opt/bl/strata/build-nvidia/seabraid/seabraid-helper"}


def units(cfg=None, plan=None, **kw):
    return render(plan or mkplan(**kw), cfg or Cfg(), STRATA, HELPERS, python=PY)


def test_ordering_server_requires_and_follows_all_helpers():
    u = units()
    srv = u[S.SERVER_UNIT]
    assert "Requires=seabraid-helper-0.service seabraid-helper-1.service" in srv
    assert "After=seabraid-helper-0.service seabraid-helper-1.service" in srv
    assert "After=" not in u["seabraid-helper-0.service"] and "Requires=" not in u["seabraid-helper-1.service"]
    assert set(u) == {"seabraid-helper-0.service", "seabraid-helper-1.service", S.SERVER_UNIT}


def test_helper_visibility_env_and_device_zero():
    u = units()
    h0, h1 = u["seabraid-helper-0.service"], u["seabraid-helper-1.service"]
    assert "Environment=SEABRAID_STRATA_SHM=/seabraid0" in h0 and "Environment=SEABRAID_STRATA_SHM=/seabraid1" in h1
    assert "Environment=CUDA_VISIBLE_DEVICES=1" in h0 and "Environment=CUDA_VISIBLE_DEVICES=2" in h1   # gpu_index 1, 2
    assert "Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID" in h0 and "HIP_VISIBLE_DEVICES" not in h0
    assert "--shm /seabraid0 --device 0" in h0 and "--device 3" not in h0           # plan ordinal 3 is NOT passed
    assert h1.count("--device") == 1 and "--device 0" in h1 and "--device 9" not in h1   # config's --device replaced
    assert "Environment=X=1" in h0


def test_nvidia_uuid_pins_helper_and_main_when_known_else_index():
    p = mkplan(1)
    uu = "GPU-11111111-2222-3333-4444-555555555555"
    roles = (replace(p.roles[0], vendor="nvidia", uuid="GPU-main"), replace(p.roles[1], uuid=uu))
    u = render(replace(p, roles=roles), Cfg(), STRATA, HELPERS, python=PY)
    h, srv = u["seabraid-helper-0.service"], u[S.SERVER_UNIT]
    assert f"Environment=CUDA_VISIBLE_DEVICES={uu}" in h and "CUDA_VISIBLE_DEVICES=1" not in h
    assert "Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID" in h and "--device 0" in h
    assert "Environment=CUDA_VISIBLE_DEVICES=GPU-main" in srv
    plain = render(p, Cfg(), STRATA, HELPERS, python=PY)["seabraid-helper-0.service"]    # no uuid known
    assert "Environment=CUDA_VISIBLE_DEVICES=1" in plain and "GPU-" not in plain


def test_amd_uuid_is_ignored_hip_keeps_index():
    p = mkplan(1)
    p = replace(p, roles=(replace(p.roles[0], uuid="GPU-x"), p.roles[1]))
    assert "Environment=HIP_VISIBLE_DEVICES=0" in render(p, Cfg(), STRATA, HELPERS, python=PY)[S.SERVER_UNIT]


def test_amd_helper_gets_hip_visible_devices():
    p = mkplan(1)
    p = Plan(p.model, p.mode, (p.roles[0], Role("helper", 4, "amd", "A", "gfx1100", 1, 0, 1)), p.builds, p.tiers)
    h = render(p, Cfg(), STRATA, {"amd": "/b/h"}, python=PY)["seabraid-helper-0.service"]
    assert "Environment=HIP_VISIBLE_DEVICES=4" in h and "CUDA_VISIBLE_DEVICES" not in h


def test_server_execstart_form():
    srv = units()[S.SERVER_UNIT]
    line = next(l for l in srv.splitlines() if l.startswith("ExecStart="))
    assert line == (f"ExecStart={PY} {STRATA}/serve/server.py --engine strata --config /data/cfg/seabraid.json "
                    "--host 127.0.0.1 --port 18080 --no-api-key --parallel 1")
    # A bind beyond loopback now needs an API key file (tests/install/test_exposure.py covers the key path itself).
    with pytest.raises(S.ServiceError, match="api-key-file"):
        render(mkplan(), Cfg(), STRATA, HELPERS, python=PY, bind="0.0.0.0")


def test_server_env_keys_and_main_visibility():
    srv = units()[S.SERVER_UNIT]
    for want in ("SEABRAID_STRATA_SHM=/seabraid0,/seabraid1", "SEABRAID_STRATA_EXCLUDE_HELPER_RAM=1", "STRATA_PREFILL_HELPER_SOURCE=1",
                 "STRATA_DISJOINT_ADAPT=1", "LD_LIBRARY_PATH=/r/lib:/r/lib2", "HIP_VISIBLE_DEVICES=0", "HSA_SVM_FOR_PAGED_MEM=0"):
        assert f"Environment={want}" in srv, want
    p = mkplan(1)
    nv_main = Plan(p.model, p.mode, (Role("main", 2, "nvidia", "N", "sm_86", 0, None, 1),), p.builds[1:], p.tiers)
    s2 = render(nv_main, Cfg(), STRATA, {}, python=PY)[S.SERVER_UNIT]
    assert "Environment=CUDA_VISIBLE_DEVICES=2" in s2 and "HIP_VISIBLE_DEVICES" not in s2


def test_readiness_execstartpre_lists_every_helper_shm():
    srv = units()[S.SERVER_UNIT]
    pre = next(l for l in srv.splitlines() if l.startswith("ExecStartPre="))
    # the names are DATA after the fixed script (positional args), not text inside it
    assert pre.endswith("seabraid-shm-wait 90 seabraid0 seabraid1") and "/dev/shm/$$n" in pre and "-lt $$w" in pre and "still missing after $${w}s" in pre
    assert "seabraid0" not in pre.split("seabraid-shm-wait")[0]
    assert srv.index("ExecStartPre=") < srv.index("ExecStart=/")
    assert "ExecStartPre" not in units(plan=mkplan(0))[S.SERVER_UNIT]
    assert "ExecStartPre" not in units()["seabraid-helper-0.service"]


def _fake_data(tmp_path, versions):
    for v in versions:
        d = tmp_path / "strata" / "terminal" / v / "python" / "bin"
        d.mkdir(parents=True)
        (d / "python3.12").write_text("")
        (d / "python3").write_text("")
        (d / "python3-config").write_text("")


def test_pick_python_highest_numeric_version_in_data_dir_wins(tmp_path):
    _fake_data(tmp_path, ["0.1.9", "0.1.56", "0.1.10"])          # string sort would pick 0.1.9
    out = []
    got = S.pick_python(STRATA, out.append, executable=lambda p: True, env={"XDG_DATA_HOME": str(tmp_path)})
    assert "0.1.56" in got and got.endswith("python3.12") and "bundled" in out[0]


def test_pick_python_derives_data_dir_from_home_and_skips_non_executable(tmp_path):
    _fake_data(tmp_path / ".local" / "share", ["0.1.56", "0.1.40"])
    hi = tmp_path / ".local" / "share" / "strata" / "terminal" / "0.1.56"
    got = S.pick_python(STRATA, lambda s: None, executable=lambda p: "0.1.56" not in p, env={"HOME": str(tmp_path)})
    assert "0.1.40" in got and str(hi) not in got                  # refused the non-executable higher one


def test_pick_python_falls_back_to_venv_then_system(tmp_path):
    out = []
    assert S.pick_python(STRATA, out.append, executable=lambda p: p == PY, env={"XDG_DATA_HOME": str(tmp_path)}) == PY
    assert PY in out[0]
    out.clear()
    got = S.pick_python(STRATA, out.append, executable=lambda p: False, env={})
    assert "python3" in got and "system python3" in out[0] and "HOME is set" in out[0]


def test_no_hardcoded_paths_in_unit_text():
    text = "\n".join(units().values())
    paths = set(re.findall(r"/[A-Za-z0-9_./-]+", text)) - {"/seabraid0", "/seabraid1", "/seabraid0,/seabraid1"}
    allowed = ("/opt/bl/", "/data/", "/r/", "/c/", "/h/", "/dev/shm", "/bin/sh", "/.nv")      # "/.nv" is the tail of the %h/.nv ReadWritePaths specifier, not a fixed path
    assert not [p for p in paths if not p.startswith(allowed)], paths
    assert not re.search(r"/home/|/opt/rocm|/usr/local", text)


def test_relative_paths_missing_helper_and_unready_plan_refused():
    with pytest.raises(S.ServiceError, match="absolute"):
        render(mkplan(), Cfg(), STRATA, HELPERS, python="python3")
    with pytest.raises(S.ServiceError, match="absolute"):
        render(mkplan(), Cfg(cfg_json_path="cfg.json"), STRATA, HELPERS, python=PY)
    with pytest.raises(S.ServiceError, match="no seabraid-helper binary"):
        render(mkplan(), Cfg(), STRATA, {}, python=PY)
    with pytest.raises(S.ServiceError, match="no helper args"):
        render(mkplan(), Cfg(helper_args={0: []}), STRATA, HELPERS, python=PY)
    p = mkplan()
    bad = Plan(p.model, "infeasible", p.roles, p.builds, p.tiers, refusals=("x",))
    with pytest.raises(S.ServiceError, match="not ready"):
        render(bad, Cfg(), STRATA, HELPERS, python=PY)


def test_no_helpers_means_no_shm_env_and_no_requires():
    srv = units(cfg=Cfg(server_env={}), plan=mkplan(0))[S.SERVER_UNIT]       # services adds none of it itself
    assert "SEABRAID_STRATA_SHM" not in srv and "Requires=" not in srv and "STRATA_DISJOINT_ADAPT" not in srv


def test_install_is_dry_run_by_default(tmp_path):
    calls, out = [], []
    acts = S.install_units(units(), str(tmp_path / "u"), run=calls.append, echo=out.append)
    assert not calls and not (tmp_path / "u").exists() and all(o.startswith("[dry-run]") for o in out)
    assert acts[-1].split()[-1] == S.SERVER_UNIT and "daemon-reload" in acts[-2]
    enable = [a for a in acts if "enable --now" in a][0].split()
    assert enable.index("seabraid-helper-0.service") < enable.index(S.SERVER_UNIT)


def test_install_apply_writes_then_reloads_then_enables(tmp_path):
    calls = []
    S.install_units(units(), str(tmp_path / "u"), apply=True, run=lambda c: calls.append(c) or 0, echo=lambda s: None)
    assert (tmp_path / "u" / S.SERVER_UNIT).read_text().startswith("[Unit]")
    assert calls[0][:3] == ["systemctl", "--user", "daemon-reload"] and calls[1][2:4] == ["enable", "--now"]
    with pytest.raises(S.ServiceError):
        S.install_units(units(), str(tmp_path / "u"), apply=True, run=lambda c: 1, echo=lambda s: None)


def test_no_start_writes_the_units_and_reloads_but_enables_and_starts_nothing(tmp_path):
    calls = []
    S.install_units(units(), str(tmp_path / "u"), apply=True, start=False, run=lambda c: calls.append(c) or 0, echo=lambda s: None)
    assert (tmp_path / "u" / S.SERVER_UNIT).read_text().startswith("[Unit]")      # the file is really there
    assert calls == [["systemctl", "--user", "daemon-reload"]]                     # no enable, no --now


# ---- health, HTTP mocked ----------------------------------------------------------------------------------------
class Http:
    def __init__(self, health_codes, chat):
        self.codes, self.chat, self.log = list(health_codes), chat, []

    def __call__(self, method, url, body, timeout):
        self.log.append((method, url, body))
        if url.endswith("/health"):
            code = self.codes.pop(0) if self.codes else 200
            if code == "boom":
                raise ConnectionRefusedError("refused")
            return code, {}
        return self.chat


GOOD = (200, {"choices": [{"message": {"content": "1, 2, 3"}}], "usage": {"completion_tokens": 100},
              "timings": {"predicted_per_second": 97.25}})


def test_health_polls_then_chats_and_reports_tok_s():
    t, out = [0.0], []
    h = Http(["boom", 503, 200], GOOD)
    res = S.check_health(18080, http=h, sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0], echo=out.append)
    assert res.ok and res.decode_tok_s == 97.25 and res.tokens == 100 and "97.2" in out[0]
    method, url, body = h.log[-1]
    assert (method, url) == ("POST", "http://127.0.0.1:18080/v1/chat/completions") and body["messages"][0]["content"] == S.TEST_PROMPT
    assert sum(1 for m, u, _ in h.log if u.endswith("/health")) == 3


def test_health_timings_variants_and_missing():
    assert S.parse_decode_tok_s({"timings": {"predicted_n": 50, "predicted_ms": 500}}) == 100.0
    assert S.parse_decode_tok_s({}) is None
    res = S.check_health(1, http=Http([200], (200, {"choices": [{"message": {"content": "x"}}], "usage": {"completion_tokens": 3}})), echo=lambda s: None)
    assert res.decode_tok_s is None and "NO timings" in res.detail


def test_unhealthy_server_fails_loudly():
    t = [0.0]
    with pytest.raises(S.ServiceError, match="not healthy"):
        S.check_health(1, wait_s=10, http=Http(["boom"] * 99, GOOD), sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
    with pytest.raises(S.ServiceError, match="unhealthy"):
        S.check_health(1, http=Http([200], (200, {"choices": [{"message": {"content": ""}}], "usage": {"completion_tokens": 0}})))
    with pytest.raises(S.ServiceError, match="status 500"):
        S.check_health(1, http=Http([200], (500, {})))


class _St:
    def __init__(self, size):
        self.st_size = size


def test_live_shm_size_must_match_the_plan():
    p = replace(mkplan(), shm_bytes_per_helper=27_676_672)
    seen, out = [], []
    def stat(path):
        seen.append(path)
        return _St(27_676_672)
    res = S.check_health(1, http=Http([200], GOOD), plan=p, stat=stat, echo=out.append)
    assert res.ok and seen == ["/dev/shm/seabraid0", "/dev/shm/seabraid1"] and any("matches the plan" in o for o in out)


def test_live_shm_mismatch_fails_loudly_with_both_numbers():
    p = replace(mkplan(), shm_bytes_per_helper=27_676_672)
    with pytest.raises(S.ServiceError) as e:
        S.check_health(1, http=Http([200], GOOD), plan=p, stat=lambda path: _St(26_038_272), echo=lambda s: None)
    msg = str(e.value)
    assert "26,038,272" in msg and "27,676,672" in msg and "drifted" in msg and "/dev/shm/seabraid0" in msg


def test_missing_shm_file_is_skipped_with_a_note():
    out = []
    def gone(path):
        raise FileNotFoundError(path)
    S.check_health(1, http=Http([200], GOOD), plan=replace(mkplan(), shm_bytes_per_helper=1), stat=gone, echo=out.append)
    assert sum("skipped" in o for o in out) == 2
    S.check_health(1, http=Http([200], GOOD), plan=mkplan(0), stat=gone, echo=out.append)       # no helpers: nothing to stat


def test_therock_runtime_ld_library_path_includes_rocm_sysdeps_and_opt_layout_does_not():
    """Same probe as the configure step: a TheRock engine needs <root>/lib/rocm_sysdeps/lib at start; /opt/rocm has none."""
    p = mkplan(1)
    tr = replace(p.builds[0], toolkit_root="/tr", compiler="/tr/lib/llvm/bin/clang++", lib_dirs=("/tr/lib",))
    fs = {"/tr/lib/llvm/bin/clang++", "/tr/lib/rocm_sysdeps", "/tr/lib/rocm_sysdeps/lib"}
    u = S.render_units(replace(p, builds=(tr, p.builds[1])), Cfg(), STRATA, HELPERS, python=PY, exists=fs.__contains__)
    assert "Environment=LD_LIBRARY_PATH=/tr/lib:/tr/lib/rocm_sysdeps/lib" in u[S.SERVER_UNIT]
    assert "rocm_sysdeps" not in units()[S.SERVER_UNIT]


def test_optional_helper_features_stay_off_with_only_a_primary(capsys):
    requested = SimpleNamespace(ladder=True, file_moves=True)
    effective = _setup_stages._available_flags(requested, mkplan(n_help=0))
    assert not effective.ladder and not effective.file_moves
    assert requested.ladder and requested.file_moves
    assert "Running the primary without ladder or file moves" in capsys.readouterr().out
    assert _setup_stages._available_flags(requested, mkplan(n_help=1)) is requested


def test_no_start_names_the_primary_service(capsys):
    args = SimpleNamespace(dry_run=False, no_start=True)
    assert _setup_stages._announce(args, Cfg(), None, "model", "units", {}, mkplan(0)) == 0
    assert f"systemctl --user start {S.SERVER_UNIT}" in capsys.readouterr().out


def test_setup_health_receives_the_planned_shmem_layout(monkeypatch):
    expected_plan = mkplan(2)
    cfg = Cfg()
    args = SimpleNamespace(dry_run=False, no_start=False, host="127.0.0.1")

    def check_health(port, *, host, model, plan=None):
        assert (port, host, model, plan) == (cfg.port, args.host, "model", expected_plan)
        return SimpleNamespace(decode_tok_s=None)

    monkeypatch.setattr(S, "check_health", check_health)
    assert _setup_stages._announce(args, cfg, None, "model", "units", {}, expected_plan) == 0
