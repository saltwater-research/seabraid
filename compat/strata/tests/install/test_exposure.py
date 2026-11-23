# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""install/_exposure.py: a server reachable beyond this PC needs a private API-key file; loopback is unchanged."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from types import SimpleNamespace

import pytest

from install import _exposure as X
from install import services as S

from test_services import HELPERS, PY, STRATA, Cfg, mkplan, render

KEY = "k" * 24
SH = shutil.which("sh")


def fake_stat(mode=0o600, uid=1000, kind=stat.S_IFREG):
    return lambda path: SimpleNamespace(st_mode=kind | mode, st_uid=uid)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.1.2.3", "localhost", "LOCALHOST", "::1", "[::1]"])
def test_loopback_hosts(host):
    assert X.is_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "203.0.113.5", "example.com", "127.0.0.1.evil.example", "localhost.evil.example"])
def test_everything_else_is_exposed(host):
    assert not X.is_loopback_host(host)


def _key(tmp_path, text=KEY + "\n", mode=0o600):
    p = tmp_path / "api.key"
    p.write_text(text)
    os.chmod(p, mode)
    return str(p)


def test_loopback_needs_no_key_and_exposed_host_is_refused_without_one():
    X.require_key_for_host("127.0.0.1", None)
    with pytest.raises(X.ExposureError, match="--api-key-file"):
        X.require_key_for_host("0.0.0.0", None)


def test_exposed_host_with_a_private_key_file_is_accepted(tmp_path):
    X.require_key_for_host("0.0.0.0", _key(tmp_path))
    assert X.read_key_file(_key(tmp_path)) == KEY


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o660, 0o644])
def test_group_or_world_readable_key_file_is_refused_with_the_fix(tmp_path, mode, monkeypatch):
    monkeypatch.setattr(os, "fstat", fake_stat(mode))        # fstat on the OPEN descriptor is what the check reads
    with pytest.raises(X.ExposureError, match="chmod 600"):
        X.read_key_file(_key(tmp_path), posix=True)


def test_private_file_passes_and_another_owners_file_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(os, "fstat", fake_stat(0o600, uid=1000))
    assert X.read_key_file(_key(tmp_path), posix=True) == KEY
    monkeypatch.setattr(os, "fstat", fake_stat(0o600, uid=0))
    with pytest.raises(X.ExposureError, match="not owned by you"):
        X.read_key_file(_key(tmp_path), posix=True)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW is POSIX")
def test_symlinked_key_file_is_refused(tmp_path):
    real = _key(tmp_path)
    link = tmp_path / "link.key"
    try:
        os.symlink(real, link)
    except OSError:
        pytest.skip("cannot create symlinks here")
    with pytest.raises(X.ExposureError, match="symbolic link"):
        X.read_key_file(str(link))


def test_key_is_judged_on_the_open_descriptor_not_a_second_stat(tmp_path, monkeypatch):
    """Class checker for the TOCTOU: os.stat must never be consulted for the key file."""
    monkeypatch.setattr(os, "stat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("path stat used")))
    assert X.read_key_file(_key(tmp_path)) == KEY


def test_cleartext_warning_only_beyond_loopback():
    assert X.cleartext_warning("127.0.0.1") is None
    assert "TLS reverse proxy" in X.cleartext_warning("0.0.0.0")


@pytest.mark.parametrize("text, why", [("short\n", "shorter than"), ("\n", "shorter than"), ("a" * 20 + " " + "b" * 20, "one key on one line"),
                                       ("a" * 20 + "\nb" * 20, "one key on one line")])
def test_weak_or_malformed_keys_are_refused(tmp_path, text, why):
    with pytest.raises(X.ExposureError, match=why):
        X.read_key_file(_key(tmp_path, text))


def test_missing_file_and_directory_are_refused(tmp_path):
    with pytest.raises(X.ExposureError, match="cannot read"):
        X.read_key_file(str(tmp_path / "nope"))
    with pytest.raises(X.ExposureError, match="not a regular file|cannot read"):    # Windows refuses to open a directory outright
        X.read_key_file(str(tmp_path))


def test_the_message_never_carries_the_key_or_the_directory(tmp_path):
    with pytest.raises(X.ExposureError) as e:
        X.read_key_file(_key(tmp_path, "tooShort"))
    assert "tooShort" not in str(e.value) and str(tmp_path) not in str(e.value)


# ---- the units


def test_loopback_unit_is_unchanged_and_exposed_bind_without_key_renders_nothing():
    srv = render(mkplan(), Cfg(), STRATA, HELPERS, python=PY)[S.SERVER_UNIT]
    assert "--no-api-key" in srv and "STRATA_API_KEY" not in srv
    with pytest.raises(S.ServiceError, match="api-key-file"):
        render(mkplan(), Cfg(), STRATA, HELPERS, python=PY, bind="0.0.0.0")


def test_exposed_server_unit_reads_the_key_at_start_and_never_contains_it(tmp_path):
    kf = "/srv/seabraid/api.key"       # the unit only NAMES the file; render_units validates it through the seam below
    srv = render(mkplan(), Cfg(), STRATA, HELPERS, python=PY, bind="0.0.0.0", api_key_file=kf, key_check=lambda host, f: None)[S.SERVER_UNIT]
    line = next(l for l in srv.splitlines() if l.startswith("ExecStart="))
    assert KEY not in srv and "--no-api-key" not in srv and "--host 0.0.0.0" in line
    assert line.startswith("ExecStart=/bin/sh -c ") and 'STRATA_API_KEY=$$(cat -- \\"$$1\\")' in line
    # the real command rides as arguments of the wrapper, after the key path
    assert line.index(kf) < line.index("server.py")


def _run_wrapper(argv):
    assert SH, "a POSIX sh is needed to run the wrapper"
    return subprocess.run([SH, *argv[1:]], capture_output=True, text=True)


def test_wrapper_exports_the_key_and_runs_the_command_with_it(tmp_path):
    kf = _key(tmp_path)
    argv = X.key_wrapper_argv([SH, "-c", 'printf %s "$STRATA_API_KEY"'], kf)
    out = _run_wrapper(argv).stdout
    assert out == KEY


def test_wrapper_fails_closed_when_the_key_file_is_gone(tmp_path):
    argv = X.key_wrapper_argv([SH, "-c", "echo started"], str(tmp_path / "gone.key"))
    r = _run_wrapper(argv)
    assert r.returncode != 0 and "started" not in r.stdout


def test_check_health_sends_the_key_only_when_given():
    seen = []

    def http(method, url, body, timeout, api_key=None):
        seen.append(api_key)
        return 200, {"choices": [{"message": {"content": "1,2"}}], "usage": {"completion_tokens": 2}}

    S.check_health(1, http=http, echo=lambda s: None, api_key=KEY)
    assert seen == [KEY, KEY]
    seen.clear()
    S.check_health(1, http=lambda m, u, b, t: http(m, u, b, t), echo=lambda s: None)
    assert seen == [None, None]


def test_setup_refuses_an_exposed_host_before_probing_anything(capsys):
    from install import seabraid_setup as CLI
    rc = CLI.main(["--detect-only", "--host", "0.0.0.0"])
    assert rc == 2 and "--api-key-file" in capsys.readouterr().err
