# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Release hardening of the add-on: loud errors, no full paths in logs, no proxies, safe unit text, honest prompts.

Each rule here has a class-level checker at the bottom (a scan that fails on the NEXT offender), not only a case test.
"""
from __future__ import annotations

import http.server
import logging
import re
import sys
import threading
from pathlib import Path

import pytest

from install import _http, _log, _probe, services as S
from install import seabraid_setup as CLI

from test_services import HELPERS, PY, STRATA, Cfg, mkplan, render

ADDON = Path(__file__).resolve().parents[2]


# ---- logging: scrubbing and configuration
def test_scrub_cuts_paths_to_basenames_and_one_line():
    assert _log.scrub("cannot open /srv/models/m.gguf: denied\nsecond line") == "cannot open m.gguf: denied"
    assert _log.scrub("D:\\data\\x.txt missing") == "x.txt missing"
    assert _log.scrub("") == ""
    assert len(_log.scrub("x" * 500)) == 160


def test_describe_names_the_error_but_not_the_file():
    try:
        open("/definitely/not/here/secret.key")
    except OSError as exc:
        d = _log.describe(exc)
    assert d.startswith("FileNotFoundError") and "secret" not in d and "definitely" not in d


def test_configure_is_idempotent():
    _log.configure(False)
    _log.configure(True)
    handlers = [h for h in logging.getLogger("install").handlers if getattr(h, "_seabraid", False)]
    assert len(handlers) == 1 and logging.getLogger("install").level == logging.DEBUG
    _log.configure(False)


# ---- probes say why they returned None
def test_missing_command_is_a_debug_note(caplog):
    with caplog.at_level(logging.DEBUG, logger="install"):
        assert _probe._run(["seabraid-no-such-tool-xyz", "--version"]) is None
    assert any(r.levelno == logging.DEBUG and "command not found" in r.getMessage() for r in caplog.records)


def test_failing_command_logs_exit_code_and_scrubbed_first_stderr_line(caplog):
    code = "import sys; sys.stderr.write('boom reading /secret/dir/file.txt\\nnext'); sys.exit(3)"
    with caplog.at_level(logging.DEBUG, logger="install"):
        assert _probe._run([sys.executable, "-c", code]) is None
    msg = next(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "exit 3" in msg and "file.txt" in msg and "/secret" not in msg and "next" not in msg


def test_unreadable_file_is_logged_without_its_directory(caplog, tmp_path):
    with caplog.at_level(logging.DEBUG, logger="install"):
        assert _probe._read(str(tmp_path)) is None                       # a directory: exists, cannot be read as text
        assert _probe._read(str(tmp_path / "absent")) is None
    text = " | ".join(r.getMessage() for r in caplog.records)
    assert "absent" in text and str(tmp_path.parent) not in text
    assert any(r.levelno == logging.WARNING for r in caplog.records) and any(r.levelno == logging.DEBUG for r in caplog.records)


# ---- no proxy for loopback
class _Echo(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"direct")

    def log_message(self, *a):
        pass


def _serve():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.mark.parametrize("module", ["install", "tools"])
def test_loopback_helpers_ignore_a_proxy_environment(monkeypatch, module):
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")          # nothing listens on the discard port
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    srv = _serve()
    url = "http://127.0.0.1:%d/" % srv.server_address[1]
    try:
        if module == "install":
            with _http.loopback_open(url, 5) as r:
                assert r.read() == b"direct"
        else:
            sys.path.insert(0, str(ADDON / "tools"))
            try:
                import _loopback
            finally:
                sys.path.remove(str(ADDON / "tools"))
            assert _loopback._OPENER.open(url, timeout=5).read() == b"direct"
    finally:
        srv.shutdown()


# ---- systemd unit text
def test_newline_or_control_character_in_any_unit_value_is_refused():
    for bad in ("a\nExecStartPre=/bin/evil", "a\rb", "a\x00b", "a\x1bb"):
        with pytest.raises(S.ServiceError, match="control character"):
            S._q(bad)
        with pytest.raises(S.ServiceError, match="control character"):
            S._unit(bad, [], ["/bin/true"], {}, 5)
        with pytest.raises(S.ServiceError, match="control character"):
            S._unit("d", [], ["/bin/true"], {"K": bad}, 5)
    with pytest.raises(S.ServiceError, match="not an environment variable name"):
        S._unit("d", [], ["/bin/true"], {"A\nB": "x"}, 5)
    with pytest.raises(S.ServiceError, match="control character"):
        render(mkplan(), Cfg(server_env={"SEABRAID_STRATA_SHM": "/seabraid0,/seabraid1", "K": "v\nExecStartPre=x"}), STRATA, HELPERS, python=PY)


def test_description_percent_is_escaped_so_a_gpu_name_cannot_become_a_specifier():
    assert "Description=a 100%% b" in S._unit("a 100% b", [], ["/bin/true"], {}, 5)


def test_shm_names_are_data_not_script():
    argv = S._shm_wait(["/seabraid0", "/seabraid1"])
    assert argv[:2] == ["/bin/sh", "-c"] and argv[3:] == ["seabraid-shm-wait", "90", "seabraid0", "seabraid1"]
    assert "seabraid0" not in argv[2] and "seabraid1" not in argv[2]
    for bad in ("/x; reboot", "/a b", "/../etc", "/$(id)", "/.."):
        with pytest.raises(S.ServiceError, match="shared-memory name"):
            S._shm_wait([bad])


def test_hardening_is_in_every_unit_and_never_hides_devices_or_shm():
    u = render(mkplan(), Cfg(), STRATA, HELPERS, python=PY)
    for name, text in u.items():
        for want in ("NoNewPrivileges=yes", "RestrictSUIDSGID=yes", "LockPersonality=yes"):
            assert want in text.splitlines(), (name, want)
        for never in ("PrivateDevices", "PrivateIPC", "PrivateTmp", "DevicePolicy", "MemoryDenyWriteExecute"):
            assert never not in text, (name, never)


# ---- prompts do not guess
def _answers(monkeypatch, *replies):
    it = iter(replies)

    def fake(prompt=""):
        r = next(it)
        if r is EOFError:
            raise EOFError
        return r
    monkeypatch.setattr("builtins.input", fake)


def test_eof_without_yes_aborts_instead_of_choosing_one(monkeypatch):
    _answers(monkeypatch, EOFError)
    with pytest.raises(CLI.PromptAborted, match="--yes"):
        CLI.ask_choice("which?", ["a", "b"], yes=False)


def test_out_of_range_reprompts_then_accepts(monkeypatch, capsys):
    _answers(monkeypatch, "7", "0", "x", "2")
    with pytest.raises(CLI.PromptAborted, match="3 tries"):
        CLI.ask_choice("which?", ["a", "b"], yes=False)
    _answers(monkeypatch, "7", "2")
    assert CLI.ask_choice("which?", ["a", "b"], yes=False) == "b"
    assert "number from 1 to 2" in capsys.readouterr().out


def test_empty_reply_takes_the_advertised_default_and_yes_is_unchanged(monkeypatch):
    _answers(monkeypatch, "")
    assert CLI.ask_choice("which?", ["a", "b"], yes=False) == "a"
    assert CLI.ask_choice("which?", ["a", "b"], yes=True) == "a"


def test_main_reports_an_aborted_model_question_with_status_130(monkeypatch, capsys):
    import types
    _answers(monkeypatch, EOFError)
    monkeypatch.setattr(CLI, "detect_host", lambda **kw: types.SimpleNamespace(blockers=[]))
    monkeypatch.setattr(CLI, "format_report", lambda r: "")
    import install._setup_model as M
    monkeypatch.setattr(M, "list_packs", lambda d: ["p1", "p2"])
    assert CLI.main(["--data-dir", "x"]) == 130
    assert "--yes" in capsys.readouterr().err


# ---- class checkers: the next offender fails here
def _py_files():
    return [p for d in ("install", "tools") for p in (ADDON / d).glob("*.py")]


def test_no_shell_true_anywhere_in_the_add_on():
    offenders = [p.name for p in _py_files() if re.search(r"shell\s*=\s*True", p.read_text(encoding="utf-8"))]
    assert not offenders, offenders


def test_urllib_urlopen_only_in_the_two_proxy_free_helpers_and_the_preflight_capture_tool():
    allowed = {"_http.py", "_loopback.py", "capture_expert_preflight.py"}      # the last builds its own ProxyHandler({}) opener
    offenders = [p.name for p in _py_files() if p.name not in allowed and re.search(r"urllib\.request\.urlopen\(|(?<![.\w])urlopen\(", p.read_text(encoding="utf-8"))]
    assert not offenders, offenders


def test_every_curl_in_the_shell_tools_bypasses_the_proxy():
    bad = []
    for sh in (ADDON / "tools").glob("*.sh"):
        for n, line in enumerate(sh.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"(^|[\s;&|(])curl\s", line) and not line.lstrip().startswith("#") and "--noproxy" not in line:
                bad.append(f"{sh.name}:{n}")
    assert not bad, bad
