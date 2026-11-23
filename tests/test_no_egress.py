# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Sabotage proof for checks/check_no_egress.py: every known evasion is planted in a throwaway tree and must FAIL.

The planted snippets are written with a ``~`` break character that ``d()`` removes at run time, so this file does not
itself contain the shapes the checker scans the whole tree for.  Each evasion was found by a reviewer who got zero
hits from the old checker; the table below is the regression record (listed in checks/SABOTAGE.md).
Also here: the link-level gate (undefined socket symbols of a built libseabraid.so, Linux only).
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

import pytest
from conftest import REPO, run_tool

import check_no_egress as egress  # noqa: E402  (conftest put checks/ on sys.path)


def d(text: str) -> str:
    return text.replace("~", "")


def plant(root, rel: str, data) -> None:
    path = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data if isinstance(data, bytes) else data.encode("utf-8"))


def scan(root):
    return egress.scan(str(root))[0]


CLEAN = {
    "src/lib.rs": "pub fn f() -> u32 { 1 }\n",
    "adapters/ggml/a.cpp": "// a ~socket( in a comment is not a call\nint f() { return 1; }\n",
    "bindings/python/seabraid/__init__.py": "# import so~cket in a comment\nX = 1\n",
    "compat/strata/tools/ok.py": 'URL = "http:~//127.0.0.1:8080/health"\nV6 = "http:~//[::1]:80/x"\nL = "http:~//localhost:9/"\n',
    "README.md": "See https:~//example.org/paper and https:~//github.com/x/y for background.\n",
}


@pytest.fixture
def clean(tmp_path, monkeypatch):
    monkeypatch.setattr(egress, "ALLOW", [])  # the real rows belong to the real tree; a planted tree must stand alone
    for rel, text in CLEAN.items():
        plant(tmp_path, rel, d(text))
    return tmp_path


def test_clean_tree_passes(clean):
    assert scan(clean) == []


def test_cli_exit_codes():
    code, out = run_tool(REPO, os.path.join("checks", "check_no_egress.py"))
    assert code == 0 and out.rstrip().endswith("check_no_egress: ok"), out[-1500:]


def test_cli_fails_on_a_planted_leak(clean, tmp_path):
    plant(clean, "src/leak.c", d("#include <sys/so~cket.h>\n"))
    code, out = run_tool(REPO, os.path.join("checks", "check_no_egress.py"), "--root", str(clean))
    assert code == 1 and "src/leak.c:1: egress: " in out, out


RT = "src/x.c"
PY = "bindings/python/seabraid/x.py"
# (case id, path, planted text).  Text uses ~ as the break character removed by d().
EVASIONS = [
    ("include-socket-h", RT, "#include <sys/so~cket.h>\n"),
    ("include-curl", RT, "#include <cu~rl/curl.h>\nint x;\n"),
    ("include-netdb", RT, "#include <net~db.h>\n"),
    ("include-arpa", RT, "#  include <arpa/i~net.h>\n"),
    ("include-netinet", RT, "#include <neti~net/in.h>\n"),
    ("rust-cfg-use", "src/x.rs", "#[cfg(unix)] use std::n~et::TcpS~tream;\n"),
    ("rust-tokio-net", "src/x.rs", "use tokio::n~et::TcpS~tream;\n"),
    ("rust-reqwest", "src/x.rs", "use req~west::blocking::get;\n"),
    ("c-star-line", RT, "void f(int *out) {\n*out = soc~ket(2, 1, 0);\n}\n"),
    ("c-global-socket", RT, "int s = ::soc~ket(2, 1, 0);\n"),
    ("c-global-connect", RT, "int r = ::conn~ect(s, a, n);\n"),
    ("c-sendto", RT, "ssize_t n = send~to(s, b, 1, 0, a, l);\n"),
    ("c-sendmsg", RT, "ssize_t n = send~msg(s, &m, 0);\n"),
    ("c-recvfrom", RT, "ssize_t n = recv~from(s, b, 1, 0, a, l);\n"),
    ("c-getaddrinfo", RT, "int r = getaddr~info(h, p, 0, &res);\n"),
    ("c-gethostbyname", RT, "void* h = gethost~byname(n);\n"),
    ("c-system-curl", RT, 'int r = system("cu~rl -s example.org");\n'),
    ("c-popen-wget", RT, 'FILE* f = popen("wg~et -O- x", "r");\n'),
    ("c-execv-ssh", RT, 'char* a[] = {"ss~h", "host", 0};\n'),
    ("c-dlopen-curl-lib", RT, 'void* h = dlopen("libcu~rl.so.4", RTLD_NOW);\n'),
    ("c-url-ftp", RT, 'const char* u = "ftp:~//example.org/f";\n'),
    ("c-url-ws", RT, 'const char* u = "ws:~//example.org/f";\n'),
    ("c-url-wss", RT, 'const char* u = "wss:~//example.org/f";\n'),
    ("rust-command-curl", "src/x.rs", 'let _o = Command::new("cu~rl").arg("x");\n'),
    ("rust-unixstream", "src/x.rs", "let s: UnixSt~ream;\n"),
    ("rust-udpsocket", "src/x.rs", "let s = UdpSo~cket::bind(a);\n"),
    ("rust-tcplistener", "src/x.rs", "let s = TcpLis~tener::bind(a);\n"),
    ("py-dunder-import", PY, '__imp~ort__("so~cket")\n'),
    ("py-import-module", PY, 'importlib.import_module("so~cket")\n'),
    ("py-from-http-import-client", PY, "from ht~tp import cli~ent\n"),
    ("py-urllib3", PY, "import urll~ib3\n"),
    ("py-httpx", PY, "import htt~px\n"),
    ("py-aiohttp", PY, "import aioh~ttp\n"),
    ("py-requests", PY, "import req~uests\n"),
    ("py-websocket", PY, "import web~socket\n"),
    ("py-after-semicolon", PY, "x = 1; import so~cket\n"),
    ("py-import-list", PY, "import os, so~cket\n"),
    ("py-subprocess-curl", PY, 'subprocess.run(["cu~rl", "-s", "x"])\n'),
    ("py-smtplib", PY, "from smtp~lib import SMTP\n"),
    ("localhost-in-trailing-comment", "compat/strata/tools/x.py", 'import json\nrequests.post("https:~//example.org/collect")  # local~host\n'),
    ("localhost-lookalike-host", "compat/strata/tools/x.py", 'U = "http:~//local~host.example.org/x"\n'),
    ("loopback-lookalike-ip", "compat/strata/tools/x.py", 'U = "http:~//127.0.0.1.exam~ple.org/x"\n'),
    ("compat-api-without-url", "compat/strata/tools/x.py", "import so~cket\n"),
    ("runtime-loopback-url", "src/x.rs", 'const U: &str = "http:~//127.0.0.1:1/";\n'),
    ("unscanned-build-rs", "build.rs", "use std::n~et::Tcp~Stream;\n"),
    ("unscanned-root-cmake", "CMakeLists.txt", "file(DOWN~LOAD http:~//example.org/x y)\n"),
    ("unscanned-cmake-dir", "cmake/x.cmake", "execute_process(COMMAND cu~rl x)\n"),
    ("unscanned-tools", "tools/x.py", "import so~cket\n"),
    ("unscanned-native-tests", "native-tests/x.sh", "wg~et http:~//example.org/x\n"),
    ("unscanned-scripts", "scripts/x.py", "import htt~px\n"),
    ("unscanned-examples-sh", "examples/x.sh", "if ! cu~rl -sf x; then exit 1; fi\n"),
    ("extensionless-sh", "scripts/run", "#!/bin/sh\nnc~at example.org 80\n"),
    ("extensionless-python", "scripts/serve", "#!/usr/bin/env python3\nimport so~cket\n"),
    ("patch-added-include", "adapters/x.patch", "--- a/f.c\n+++ b/f.c\n@@ -1,1 +1,2 @@\n x\n+#include <sys/so~cket.h>\n"),
    ("patch-added-python", "adapters/p/x.patch", "--- a/f.py\n+++ b/f.py\n@@ -1 +1,2 @@\n x\n+import req~uests\n"),
    ("binary-with-nul", "docs/blob.dat", "MZ\x00\x00 import so~cket"),
]


@pytest.mark.parametrize("case,rel,text", EVASIONS, ids=[e[0] for e in EVASIONS])
def test_evasion_is_detected(clean, case, rel, text):
    plant(clean, rel, d(text).encode("latin1"))
    bad = scan(clean)
    assert any(h.startswith(f"{rel}:") for h in bad), (case, bad)


def test_patch_context_and_removed_lines_are_not_scanned(clean):
    plant(clean, "adapters/p/x.patch", d("--- a/f.c\n+++ b/f.c\n@@ -1,2 +1,1 @@\n #include <sys/so~cket.h>\n-#include <net~db.h>\n+int ok;\n"))
    assert [h for h in scan(clean) if "x.patch" in h] == []


def test_localhost_word_on_the_same_line_does_not_allow_a_remote_url(clean):
    plant(clean, "compat/strata/tools/y.py", d('U = "http:~//127.0.0.1:1/"; V = "https:~//exam~ple.org/c"\n'))
    bad = scan(clean)
    assert any(d("https:~//example.org/c") in h for h in bad) and not any("127.0.0.1" in h for h in bad), bad


def test_unreadable_tracked_file_is_a_hit(clean, monkeypatch):
    real = egress.common.read_text

    def boom(root, rel):
        if rel == "src/lib.rs":
            raise PermissionError("denied")
        return real(root, rel)

    monkeypatch.setattr(egress.common, "read_text", boom)
    assert any(h.startswith("src/lib.rs:1: egress-unreadable") for h in scan(clean))


def test_real_tree_is_clean():
    assert egress.scan(REPO)[0] == []


# --- link-level gate ---------------------------------------------------------------------------------------------------
FORBIDDEN = {"socket", "connect", "bind", "listen", "accept", "getaddrinfo", "gethostbyname", "sendto", "recvfrom"}


def undefined_symbols(nm_output: str) -> set[str]:
    """Symbol names from ``nm -D --undefined-only`` output (version suffix after '@' dropped)."""
    return {line.split()[-1].split("@")[0] for line in nm_output.splitlines() if line.split()}


def test_link_gate_parser_flags_a_planted_import():
    planted = "                 U malloc@GLIBC_2.2.5\n                 U connect@GLIBC_2.2.5\n"
    assert undefined_symbols(planted) & FORBIDDEN == {"connect"}
    assert undefined_symbols("                 U malloc@GLIBC_2.2.5\n                 w __gmon_start__\n") & FORBIDDEN == set()


def test_link_gate_decision_run_skip_or_fail(tmp_path):
    lib = tmp_path / "libseabraid.so"
    lib.write_bytes(b"x")
    assert link_gate({"SEABRAID_LIBRARY": str(lib)}, None) == ("run", str(lib))
    assert link_gate({}, "found.so")[0] == "run" and link_gate({}, None)[0] == "skip", "local runs keep the skip"
    # CI sets the switch: a library that is not there is a failure, never a skip (a skip reads as a pass)
    assert link_gate({"SEABRAID_REQUIRE_EGRESS_LINK_TEST": "1"}, None)[0] == "fail"
    assert link_gate({"SEABRAID_REQUIRE_EGRESS_LINK_TEST": "1"}, "found.so")[0] == "run"
    # a path that was asked for and is absent is a configuration error whether or not the switch is set
    assert link_gate({"SEABRAID_LIBRARY": str(tmp_path / "nope.so")}, "found.so")[0] == "fail"


def _found_lib() -> str | None:
    for prof in ("release", "debug"):
        found = glob.glob(os.path.join(REPO, "target", prof, "libseabraid.so"))
        if found:
            return found[0]
    return None


def link_gate(env, found: str | None) -> tuple[str, str]:
    """-> ("run", library) | ("skip", why) | ("fail", why).  OWNER of the rule for when the link-level gate may skip.

    WHY one env name, SEABRAID_LIBRARY: it is the name every other consumer of "the built library" reads (the Python package,
    CTest); the old short alias was a second spelling of the same fact.  WHY a switch and not "always fail": a developer
    without a Rust build still runs the suite; CI sets SEABRAID_REQUIRE_EGRESS_LINK_TEST=1 so a missing library cannot read as
    a pass (the original gate never ran in CI: pytest ran before cargo build, and the missing library skipped).
    """
    named = env.get("SEABRAID_LIBRARY")
    if named:
        return ("run", named) if os.path.isfile(named) else ("fail", f"SEABRAID_LIBRARY names no file: {named}")
    if found:
        return "run", found
    if env.get("SEABRAID_REQUIRE_EGRESS_LINK_TEST") == "1":
        return "fail", "SEABRAID_REQUIRE_EGRESS_LINK_TEST=1 but no built libseabraid.so (run `cargo build --release` first)"
    return "skip", "no built libseabraid.so (target/release, target/debug or SEABRAID_LIBRARY); run `cargo build --release` first"


def test_built_library_imports_no_socket_symbols():
    if not sys.platform.startswith("linux"):
        pytest.skip(f"link-level egress gate needs Linux `nm -D`; run it on the Linux box (platform: {sys.platform})")
    verdict, detail = link_gate(os.environ, _found_lib())
    if verdict == "fail":
        pytest.fail(detail)
    if verdict == "skip":
        pytest.skip(detail)
    out = subprocess.run(["nm", "-D", "--undefined-only", detail], capture_output=True, text=True, check=True).stdout
    assert undefined_symbols(out) & FORBIDDEN == set(), f"{detail} imports network symbols"
