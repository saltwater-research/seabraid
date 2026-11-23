# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Security-review fixes: root refusal, Python-version guard, unit sandbox table, cleartext-key warning."""
from __future__ import annotations

import logging

import pytest

from install import seabraid_setup as CLI
from install import services as S

from test_services import HELPERS, PY, STRATA, Cfg, mkplan, render


# ---- root
def test_root_is_refused_without_allow_root(capsys):
    assert CLI.refuse_root(False, geteuid=lambda: 0) == 2
    assert "--allow-root" in capsys.readouterr().err


def test_allow_root_proceeds_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert CLI.refuse_root(True, geteuid=lambda: 0) is None
    assert any("--allow-root" in r.getMessage() for r in caplog.records)


def test_normal_user_and_windows_are_not_refused():
    assert CLI.refuse_root(False, geteuid=lambda: 1000) is None
    assert CLI.refuse_root(False, geteuid=None) is None          # Windows: os.geteuid does not exist


def test_main_refuses_root_before_detecting_anything(monkeypatch):
    monkeypatch.setattr(CLI, "refuse_root", lambda allow, **k: 2 if not allow else None)
    monkeypatch.setattr(CLI, "detect_host", lambda **k: pytest.fail("detect ran as root"))
    assert CLI.main(["--detect-only"]) == 2


# ---- python version
def test_old_python_is_refused_with_a_clear_message(capsys):
    assert CLI.check_python((3, 8, 10)) == 2
    err = capsys.readouterr().err
    assert "3.10" in err and "3.8.10" in err


@pytest.mark.parametrize("v", [(3, 10, 0), (3, 12, 3), (4, 0, 0)])
def test_supported_python_passes(v):
    assert CLI.check_python(v) is None


# ---- unit sandbox
def _units():
    return render(mkplan(), Cfg(), STRATA, HELPERS, python=PY)


def test_every_table_row_lands_in_exactly_the_units_it_names():
    units = _units()
    for directive, roles, why in S.SANDBOX:
        assert why.strip(), directive
        for name, text in units.items():
            role = "server" if name == S.SERVER_UNIT else "helper"
            assert (directive in text.splitlines()) == (role in roles), (name, directive)


def test_helpers_are_the_strictest_and_keep_unix_and_netlink():
    helper = _units()[S.helper_unit_name(0)]
    for want in ("RestrictAddressFamilies=AF_UNIX AF_NETLINK", "ProtectSystem=strict", "ProtectHome=read-only", "NoNewPrivileges=yes",
                 "SystemCallArchitectures=native", "LockPersonality=yes", "RestrictRealtime=yes", "ProtectKernelTunables=yes",
                 "ProtectControlGroups=yes", "UMask=0077"):
        assert want in helper.splitlines(), want


def test_server_keeps_the_network_and_the_filesystem():
    server = _units()[S.SERVER_UNIT]
    for never in ("RestrictAddressFamilies", "ProtectSystem", "ProtectHome"):
        assert never not in server                              # it serves HTTP and writes logs/cache
    for want in ("NoNewPrivileges=yes", "RestrictRealtime=yes", "UMask=0077"):
        assert want in server.splitlines()


def test_no_unit_hides_devices_or_shm():
    for text in _units().values():
        for never in ("PrivateDevices", "PrivateIPC", "PrivateTmp", "DevicePolicy", "MemoryDenyWriteExecute"):
            assert never not in text


# ---- cleartext key warning
def test_exposed_render_warns_that_the_key_travels_in_clear(caplog):
    with caplog.at_level(logging.WARNING):
        render(mkplan(), Cfg(), STRATA, HELPERS, python=PY, bind="0.0.0.0", api_key_file="/k/api.key", key_check=lambda h, f: None)
    assert any("TLS reverse proxy" in r.getMessage() for r in caplog.records)


def test_loopback_render_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        render(mkplan(), Cfg(), STRATA, HELPERS, python=PY)
    assert not any("TLS" in r.getMessage() for r in caplog.records)


def test_summary_prints_the_sentence_only_when_exposed(capsys):
    CLI._print_cleartext("127.0.0.1")
    assert capsys.readouterr().out == ""
    CLI._print_cleartext("0.0.0.0")
    assert "TLS reverse proxy" in capsys.readouterr().out
