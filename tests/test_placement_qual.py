# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Dry run of compat/strata/scripts/placement-qual/run.py against a stub OpenAI-style server (S6 finish line).

The stub is a loopback http.server on an ephemeral port that answers every chat request with a canned completion and a
llama-server style ``timings`` block.  The dry run must print the planned step list, send the capped starts to the
stub without starting any engine, and write chain.tsv (the timings file), SUMMARY.txt, results.json and chain_report.html.
Bad configs must exit 2 before anything starts.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import subprocess
import sys
import threading

import pytest
from conftest import REPO

QUAL = os.path.join(REPO, "compat", "strata", "scripts", "placement-qual")
CANNED = {"choices": [{"message": {"content": "stub reply"}}], "usage": {"prompt_tokens": 12, "completion_tokens": 20},
          "timings": {"prompt_ms": 120.0, "prompt_per_second": 80.0, "predicted_n": 20, "predicted_per_second": 95.0}}
# the real seabraid round line (adapters/strata/placement_round_log.cpp:82-90) and the legacy ladder's (_retier_ladder.cpp:67-102 + generate.cpp:21044)
ROUNDS = "\n".join(
    f"placement round={i} epoch={i} planned=4 committed=2 rejected=0 restored=0 qmask=0x0 adopted=3 spares=0 defer=0 helper_share={min(0.1 * i, 0.6):.3f} "
    f"table_hash=ab{i % 2}000000000000{i % 10}0 plan_us=300 fence_us={40000 + i} stage_us=900 replace_us=1200 publish_us=50 follow_us=2000"
    for i in range(1, 13)) + "\n"
LEGACY = ("strata serve: retier round 1 (window 129, promote): proposed 4 moves, pairs 2, committed 2, failed 0, dropped 0; moves: L31/E501 card 2->card 1; "
          "ranker 1.4 ms, transition 16.2 ms; ownership epoch 5, STALE 0, held by both cards 0, card 2 open\n")


class Stub(http.server.BaseHTTPRequestHandler):
    seen: list[int] = []

    def do_POST(self):  # noqa: N802 - http.server API
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        Stub.seen.append(len(body))
        out = json.dumps(CANNED).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):  # keep pytest output quiet
        pass


class _NoDnsServer(http.server.ThreadingHTTPServer):
    """HTTPServer.server_bind calls socket.getfqdn(host): a reverse DNS lookup that can block about 30 s while the resolver is down
    (a once-seen 29.9 s run of this test); a loopback test server has no use for the name, so bind without it."""

    def server_bind(self):
        super(http.server.HTTPServer, self).server_bind()  # TCPServer's bind, skipping HTTPServer's getfqdn
        self.server_name, self.server_port = self.server_address[0], self.server_address[1]


@pytest.fixture()
def stub():
    srv = _NoDnsServer(("127.0.0.1", 0), Stub)
    Stub.seen = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def cfgdir(tmp_path):
    d = tmp_path / "cfg"
    shutil.copytree(os.path.join(QUAL, "prompts"), d / "prompts")
    cfg = json.load(open(os.path.join(QUAL, "arms.json"), encoding="utf-8"))
    cfg["vars"].update(gpu="0", rocm="/rocm", pack="/pack", native="/native.gguf")  # machine paths come from --var in a live run
    cfg["dry_log"] = "engine_canned.log"
    (d / "engine_canned.log").write_text(ROUNDS, encoding="utf-8")
    (d / "arms.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", os.path.join(QUAL, "run.py"), *args], capture_output=True, text=True, timeout=300)


def _check_plan(stdout: str) -> None:
    plan = [ln for ln in stdout.splitlines() if ln.startswith("PLAN ")]
    assert any(" W1 " in ln and " P2z " in ln for ln in plan), stdout
    assert plan[-1].startswith("PLAN total: ") and "engine starts" in plan[-1]
    assert sum(1 for ln in plan if " W1 " in ln) == 6 * 8  # full protocol n is planned even though the dry run sends 1 start


def _check_timings(out) -> None:
    rows = [ln.split("	") for ln in (out / "chain.tsv").read_text(encoding="utf-8").splitlines()]
    assert rows[0] == ["name", "start_epoch", "seconds", "rc", "status", "note"]
    steps = {r[0]: r for r in rows[1:]}
    assert "W1-P2-s0-start" in steps and "W1-P2-s0-decode-r3" in steps and "W5-P2-s0-fault-request" in steps
    assert all(float(r[2]) >= 0 and r[3] in ("0", "1") for r in rows[1:])
    _check_no_failed_steps(rows[1:])


def _check_no_failed_steps(rows: list) -> None:
    assert [r[0] for r in rows if r[4] == "FAIL" and not r[0].startswith("criterion-")] == []
    assert any(r[0].startswith("criterion-C") for r in rows)


def _check_report(out) -> None:
    html = (out / "chain_report.html").read_text(encoding="utf-8")
    assert "Findings, ranked" in html and "Ideas to chain next" in html and "W1-P2-s0-start" in html
    assert "fence_us" in (out / "SUMMARY.txt").read_text(encoding="utf-8")
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    w1 = [r for r in res if r["workload"] == "W1" and r["key"] == "P2"][0]
    assert w1["phases"]["decode"][1]["decode_tps"] == 95.0 and len(w1["rounds"]) == 12


def test_dry_run_plans_records_and_reports(stub, cfgdir, tmp_path):
    out = tmp_path / "out"
    p = run("--dry-run", "--server", stub, "--config", str(cfgdir / "arms.json"), "--out", str(out))
    assert p.returncode == 0, p.stdout + p.stderr
    _check_plan(p.stdout)
    _check_timings(out)
    assert Stub.seen, "the stub saw no request"
    _check_report(out)


def test_bad_config_exits_2_before_starting_anything(cfgdir, tmp_path):
    cfg = json.load(open(cfgdir / "arms.json", encoding="utf-8"))
    cfg["engine"]["argv"][0] = "/usr/bin/systemctl"
    (cfgdir / "bad.json").write_text(json.dumps(cfg), encoding="utf-8")
    p = run("--dry-run", "--config", str(cfgdir / "bad.json"), "--out", str(tmp_path / "o"))
    assert p.returncode == 2 and "not an engine process" in p.stderr
    assert not (tmp_path / "o").exists()
    p = run("--dry-run", "--arms", "NOPE", "--config", str(cfgdir / "arms.json"), "--out", str(tmp_path / "o2"))
    assert p.returncode == 2 and "unknown arm" in p.stderr


def test_unreachable_server_is_recorded_as_fail_not_crash(cfgdir, tmp_path):
    out = tmp_path / "out"
    p = run("--dry-run", "--server", "http://127.0.0.1:9", "--workloads", "W1", "--arms", "P0", "--config", str(cfgdir / "arms.json"), "--out", str(out))
    assert p.returncode == 1 and "FAIL W1-P0-s0-decode-r1" in p.stderr
    assert "FAIL" in (out / "chain.tsv").read_text(encoding="utf-8")


def test_round_line_parsers_read_the_real_formats():
    sys.path.insert(0, QUAL)
    import qual_stats as st

    rows = st.parse_round_lines(ROUNDS + LEGACY)
    assert len(rows) == 13 and rows[0]["qmask"] == "0x0" and rows[0]["table_hash"] == "ab100000000000010" and rows[0]["fence_us"] == 40001.0
    assert rows[-1]["legacy"] == 1.0 and rows[-1]["fence_us"] == 16200.0 and rows[-1]["qmask"] == "0x0"
    assert not st.quarantined(rows) and st.quarantined([{"qmask": "0x2"}]) and st.quarantined([{"qmask": "zz"}])
    assert st.quarantined(st.parse_round_lines(LEGACY.replace("card 2 open", "card 2 CLOSED")))
    assert st.driver_errors(ROUNDS + LEGACY + "strata serve: Seabraid placement: a round every 128 windows\n") == []
    assert len(st.driver_errors("seabraid: ERROR placement degraded: helper is gone (mask 0x2 epoch 4)\nstrata serve: placement: the private helper stages one slot size\n")) == 2
    assert st.commit_effect([2, 2, 0, 0, 1]) == 5 and st.commit_effect([2, 2, 0, 0, 1], 0.5) == 2 and st.commit_effect([0, 0]) is None


def test_n_override_and_rejudge_combine_invocations(stub, cfgdir, tmp_path):
    a, b, both = tmp_path / "a", tmp_path / "b", tmp_path / "both"
    cfg = str(cfgdir / "arms.json")
    p = run("--dry-run", "--server", stub, "--workloads", "W1", "--arms", "P0,P2", "--n", "W1=2", "--config", cfg, "--out", str(a))
    assert p.returncode == 0 and "4 engine starts" in p.stdout, p.stdout + p.stderr
    p = run("--dry-run", "--server", stub, "--workloads", "W2", "--arms", "P2", "--config", cfg, "--out", str(b))
    assert p.returncode == 0, p.stdout + p.stderr
    p = run("--rejudge", f"{a},{b}", "--config", cfg, "--out", str(both))
    assert p.returncode == 0, p.stdout + p.stderr
    tsv = (both / "chain.tsv").read_text(encoding="utf-8")
    assert "a/W1-P0-s0-start" in tsv and "b/W2-P2-s0-start" in tsv and "criterion-C1" in tsv
    assert len(json.loads((both / "results.json").read_text(encoding="utf-8"))) >= 3
    assert run("--n", "W9=2", "--config", cfg, "--out", str(tmp_path / "x")).returncode == 2
