# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for ov_proc.py (the witnesses) and ov_cache.py (page-cache eviction and residency) on a planted procfs, sysfs and cgroup tree."""
import os
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ovf_synth as S  # noqa: E402
import ov_cache  # noqa: E402
import ov_proc  # noqa: E402


@pytest.fixture
def tree(tmp_path):
    return S.Tree(tmp_path, S.load_cfg()["plans"]["O2"])


def test_disk_is_found_by_major_minor_with_its_whole_disk_parent_and_flag(tree):
    d = tree.host().disk_for_dev(259, 2)
    assert (d.name, d.parent, d.rotational) == (S.DISK, S.PARENT, False)
    assert tree.host().disk_for_dev(1, 1) is None
    spinning = S.Tree(pathlib.Path(tree.root) / "r", rotational=1).host().disk_for_dev(259, 2)
    assert spinning.rotational is True


def test_probe_reads_device_process_and_cgroup_counters(tree):
    probe = ov_proc.Probe(tree.host(), S.DISK, tree.pids)
    a = probe.snap()
    tree.bump(disk_bytes=1_000_000, proc_bytes=900_000)
    tree.set_events(S.CG, oom=0, hits=3, majflt=40)
    b = probe.snap()
    assert b.dev_bytes - a.dev_bytes == 1_000_000 // 512 * 512 and b.proc_bytes - a.proc_bytes == 900_000
    assert b.max_hits == 3 and b.majflt == 40 and b.peak == 9 << 30 and b.oom_kill == 0


def test_a_process_that_exits_never_makes_the_process_counter_fall(tree):
    probe = ov_proc.Probe(tree.host(), S.DISK, tree.pids)
    tree.bump(0, 5_000_000)
    first = probe.snap().proc_bytes
    for name in ("io", "cgroup", "comm"):
        os.remove(f"{tree.proc}/{S.ENGINE_PID}/{name}")  # the engine is gone
    assert probe.snap().proc_bytes == first


def test_an_unreadable_witness_is_none_never_zero(tmp_path):
    probe = ov_proc.Probe(ov_proc.Host(str(tmp_path / "p"), str(tmp_path / "s"), str(tmp_path / "c")), "nvme0n1p2", lambda: [1])
    s = probe.snap()
    assert (s.dev_bytes, s.proc_bytes, s.majflt, s.oom_kill, s.max_hits, s.peak) == (None,) * 6


def test_cgroup_report_reads_the_limit_back_from_the_process_own_cgroup(tree):
    rows = {r["comm"]: r for r in tree.host().cgroup_report(tree.pids())}
    assert rows["strata"]["memory_max"] == tree.plan.engine_cap_bytes and rows["strata"]["swap_max"] == 0
    assert rows["helper-cuda"]["memory_max"] == tree.plan.helper_cap_bytes
    tree.set_limit(S.CG, "max", 0)
    assert tree.host().cgroup_report([S.ENGINE_PID])[0]["memory_max"] == "max"


def test_cg_kill_refuses_anything_but_a_run_scope_of_the_user_manager(tree):
    h = tree.host()
    assert "refusing" in h.cg_kill("/user.slice/user-1000.slice/session-3.scope", None)
    assert "refusing" in h.cg_kill(S.CG, S.CG)  # the runner's own cgroup
    assert "refusing" in h.cg_kill("/system.slice/ssh.service", None)
    assert not os.path.exists(f"{tree.cgroot}{S.CG}/cgroup.kill")  # none of the refusals wrote anything
    assert h.cg_kill(S.CG, "/user.slice/user-1000.slice/session-3.scope") is None  # the allowed shape: a run-*.scope that is not the runner's own
    assert open(f"{tree.cgroot}{S.CG}/cgroup.kill", encoding="utf-8").read() == "1"
    assert "failed" in h.cg_kill("/user.slice/u/user@1000.service/app.slice/run-p9-i9.scope/../nope/run-q.scope", None)


def test_model_files_lists_every_shard_and_the_pack_files(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "pack").mkdir()
    for n in ("a-00001-of-00002.gguf", "a-00002-of-00002.gguf", "a-00001-of-00002.gguf.done"):
        (tmp_path / "m" / n).write_bytes(b"x")
    (tmp_path / "pack" / "dense.bin").write_bytes(b"x")
    files = ov_cache.model_files(str(tmp_path / "m" / "a-00001-of-00002.gguf"), str(tmp_path / "pack"))
    assert [os.path.basename(f) for f in files] == ["a-00001-of-00002.gguf", "a-00002-of-00002.gguf", "dense.bin"]


def fincore(res):
    return lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=res, stderr="")


def test_preflight_passes_when_eviction_leaves_little_in_the_cache(tmp_path):
    f = tmp_path / "w.gguf"
    f.write_bytes(b"x")
    seen = []
    pre = ov_cache.preflight([str(f)], 512.0, fadvise=lambda fd, o, n, adv: seen.append((o, n)), run=fincore("1000000\n2000000\n"))
    assert pre["ok"] and seen == [(0, 0)] and pre["resident_mb"] == pytest.approx(3.0)


def test_preflight_refuses_a_cache_that_is_still_warm_after_eviction(tmp_path):
    f = tmp_path / "w.gguf"
    f.write_bytes(b"x")
    pre = ov_cache.preflight([str(f)], 512.0, fadvise=lambda *a: None, run=fincore(f"{3_273_478_144}\n"))
    assert not pre["ok"] and pre["resident_mb"] > 512


def test_preflight_names_a_missing_fincore_and_a_failed_eviction_and_never_passes_them(tmp_path):
    def no_fincore(cmd, **kw):
        raise FileNotFoundError("fincore")

    def bad(fd, o, n, adv):
        raise OSError("EINVAL")

    f = tmp_path / "w.gguf"
    f.write_bytes(b"x")
    pre = ov_cache.preflight([str(f), str(tmp_path / "missing.gguf")], 512.0, fadvise=bad, run=no_fincore)
    assert not pre["ok"] and pre["resident_mb"] is None and len(pre["errors"]) == 3
