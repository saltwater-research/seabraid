# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""tools/kl_gate.sh end to end with a fake server and fake helper (no GPU): OUT handling, missing inputs, empty rows.

The script is run from a COPY of tools/ so a relative OUT lands in the temp dir, never in the checkout.  The fake
server answers /health, and on the one POST writes the rows named by FAKE_ROWS to $STRATA_LOGPOS (or nothing)."""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from test_kl_gate import TOOLS, logits, write

# Windows: PATH's bash is a launcher stub, which this suite must not start; the Linux gate runs these.
pytestmark = pytest.mark.skipif(sys.platform == "win32" or not (shutil.which("bash") and shutil.which("curl")),
                                reason="needs a native bash and curl (Linux gate)")

FAKE_SERVER = '''
import argparse, http.server, os, shutil

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if os.environ.get("FAKE_ROWS"):
            shutil.copy(os.environ["FAKE_ROWS"], os.environ["STRATA_LOGPOS"])
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b"{}")

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int)
a, _ = ap.parse_known_args()
http.server.HTTPServer(("127.0.0.1", a.port), H).serve_forever()
'''


def posix(p) -> str:
    return pathlib.Path(p).as_posix()


@pytest.fixture
def rig(tmp_path):
    tools = tmp_path / "tools"
    shutil.copytree(TOOLS, tools)
    (tmp_path / "seabraid" / "serve").mkdir(parents=True)
    (tmp_path / "seabraid" / "serve" / "server.py").write_text(FAKE_SERVER)
    helper = tmp_path / "helper.sh"
    helper.write_bytes(b"#!/bin/sh\nexec sleep 60\n")
    helper.chmod(0o755)
    rows = tmp_path / "rows.tsv"
    write(rows, logits(n=60))
    for n in ("stock.json", "seabraid.json", "text.txt"):
        (tmp_path / n).write_text("x")
    port = 18000 + os.getpid() % 2000
    env = {**os.environ, "SEABRAID_STRATA_DIR": posix(tmp_path / "seabraid"), "STRATA_DATA": posix(tmp_path), "NATIVE_GGUF": "m.gguf",
           "PACK_NAME": "p", "HELPER_BIN": posix(helper), "LAYERS": "2", "EXPERTS": "4", "PORT": str(port), "SHM": "/kltest",
           "PY": posix(sys.executable), "POLL_SECS": "0.2", "HELPER_WAIT_SECS": "0.2",
           "STOCK_CFG": posix(tmp_path / "stock.json"), "SEABRAID_STRATA_CFG": posix(tmp_path / "seabraid.json"),
           "TEXT_FILE": posix(tmp_path / "text.txt"), "FAKE_ROWS": posix(rows)}

    def run(*args, **over):
        e = {**env, **over}
        e = {k: v for k, v in e.items() if v is not None}
        return subprocess.run(["bash", posix(tools / "kl_gate.sh"), *args], capture_output=True, text=True, env=e, timeout=120)
    return tmp_path, tools, run


def test_absolute_out_is_written_where_asked(rig):
    tmp, _, run = rig
    out = tmp / "abs-out"
    p = run(posix(out))
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert all((out / f"{n}.tsv").stat().st_size > 0 for n in ("stock1", "stock2", "seabraid")), p.stdout
    assert "seabraid vs stock" in p.stdout and "Traceback" not in p.stderr


def test_relative_out_is_relative_to_the_script_dir(rig):
    _, tools, run = rig
    p = run("rel-out")
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert (tools / "rel-out" / "seabraid.tsv").stat().st_size > 0


def test_missing_inputs_fail_fast_naming_each_one(rig):
    tmp, tools, run = rig
    p = run(posix(tmp / "o"), STOCK_CFG=None, SEABRAID_STRATA_CFG=None, TEXT_FILE=None)   # defaults point into tools/, where none ship
    assert p.returncode == 2, (p.stdout, p.stderr)
    for name in ("STOCK_CFG", "SEABRAID_STRATA_CFG", "TEXT_FILE"):
        assert f"error: {name} not found" in p.stderr, p.stderr
    assert "WITHOUT --expert-cache-device1" in p.stderr and "WITH --expert-cache-device1" in p.stderr
    assert not list((tmp / "o").glob("*")), "nothing may start before the inputs are known"
    assert str(tmp) not in p.stderr and posix(tmp) not in p.stderr, "messages name the input, not a path"


def test_one_missing_input_names_only_that_one(rig):
    tmp, _, run = rig
    p = run(posix(tmp / "o"), TEXT_FILE=posix(tmp / "gone.txt"))
    assert p.returncode == 2 and "TEXT_FILE not found" in p.stderr and "STOCK_CFG" not in p.stderr, p.stderr


def test_empty_rows_fail_clearly_not_with_a_traceback(rig):
    tmp, _, run = rig
    p = run(posix(tmp / "o"), FAKE_ROWS=None)          # the server logs nothing
    assert p.returncode != 0, (p.stdout, p.stderr)
    assert "arm stock1 logged no rows" in p.stderr and "Traceback" not in p.stdout + p.stderr
    assert not (tmp / "o" / "stock2.tsv").exists(), "the run must stop at the first empty arm"
