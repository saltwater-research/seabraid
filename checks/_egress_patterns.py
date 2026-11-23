# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""What counts as network use, per language family: the pattern tables behind check_no_egress.py.

OWNER: this module is the only definition of "a network API"; the checker owns scanning, allow rows and reporting.
CONTRACT: ``FAMILY_PATTERNS[family]`` is a list of compiled regexes applied to one comment-stripped line; ``URL``
and ``loopback_host`` serve the URL rule.  Pure data and one pure function; imports nothing from the checkers.
FAILURE: a pattern that is too broad costs a reviewed allow row (loud, visible); one that is too narrow is the
escape this file exists to prevent, so when unsure a pattern is broad.  CHECKER: tests/test_no_egress.py plants one
evasion per row of these tables.
"""

from __future__ import annotations

import re

_PY_MODS = (
    r"socket(?:server)?|http\.client|http\.server|urllib(?!\.parse)|urllib3|requests|httpx|aiohttp|websockets?|ftplib|"
    r"smtplib|poplib|imaplib|telnetlib|xmlrpc|pycurl|paramiko|grpc"
)
# WHY a separate tool list: the same names are matched quoted (argv lists, system strings) and bare (shell words).
_TOOLS = r"curl|wget|nc|ncat|netcat|socat|ssh|scp|sftp|rsync|telnet"
_C_CALLS = r"socket|connect|bind|listen|accept4?|sendto|sendmsg|sendmmsg|recvfrom|recvmsg|recvmmsg|getaddrinfo|gethostbyname2?|gethostbyaddr|getnameinfo"
_INCLUDES = (
    r"sys/socket\.h|sys/un\.h|netdb\.h|arpa/inet\.h|netinet/[\w.]+|curl/[\w.]+|winsock2?\.h|ws2tcpip\.h|ifaddrs\.h|"
    r"openssl/ssl\.h|asio[./]\w*|boost/asio\S*|httplib\.h|zmq\.h"
)


def _c(*rx: str) -> list[re.Pattern]:
    return [re.compile(r) for r in rx]


_COMMON = _c(
    # a tool name opening a quoted string: argv element, Command::new("curl"), system("curl ...") alike
    rf"[\"'](?:{_TOOLS})(?:\.exe)?(?=[\"'\s])",
    r"\b(?:Tcp|Udp|Unix)(?:Stream|Listener|Socket|Datagram)\b",
    r"\blib(?:curl|ssl|nghttp2)\b[\w.\-]*",
    r"\b(?:dlopen|LoadLibrary\w*)\s*\([^)]*(?:curl|ssl|winhttp|wininet|ws2_32)",
)
_C_FAMILY = _COMMON + _c(
    rf"(?<![\w.>:])(?:::)?(?:{_C_CALLS})\s*\(",
    rf"#\s*include\s*[<\"](?:{_INCLUDES})",
    r"\bcurl_\w+|\bWSAStartup\b|\bWinHttp\w+|\bInternetOpen\w*",
)
_RUST = _COMMON + _c(
    r"\b(?:std|tokio|async_std|smol|mio)::net\b",
    r"\b(?:reqwest|ureq|hyper|surf|isahc|attohttpc|socket2|curl)\b",
)
_PY = _COMMON + _c(
    rf"(?:^|[;:\"'])\s*(?:import\s+(?:[\w.]+(?:\s+as\s+\w+)?\s*,\s*)*(?:{_PY_MODS})\b|from\s+(?:{_PY_MODS})\b|from\s+urllib\s+import\b|from\s+http\s+import\s+[\w, ]*\b(?:client|server)\b)",
    rf"\b(?:__import__|import_module)\s*\(\s*[\"'](?:{_PY_MODS})",
    r"\basyncio\.(?:open_connection|start_server|create_server)\b",
)
# WHY any word position, not only command position: `if ! curl`, `then wget`, `timeout 5 nc`, `xargs curl` all run it.
_SH = _COMMON + _c(rf"(?<![\w.-])(?:{_TOOLS})(?![\w-])", r"/dev/(?:tcp|udp)/")
_CMAKE = _SH + _c(
    r"\bfile\s*\(\s*(?:DOWNLOAD|UPLOAD)\b",
    r"\bFetchContent_\w+|\bExternalProject_Add\b|\bGIT_REPOSITORY\b|\bfind_package\s*\(\s*CURL\b",
)

FAMILY_PATTERNS = {"c": _C_FAMILY, "rust": _RUST, "py": _PY, "sh": _SH, "cmake": _CMAKE, "hash": _SH, "raw": [], "doc": []}

# scheme://authority; the authority group is judged by loopback_host, never by words elsewhere on the line.
URL = re.compile(
    r"\b(?:https?|ftps?|wss?|sftp|ssh|git|tcp|udp|grpcs?|mqtts?|amqps?|rediss?|smtps?)://(?P<auth>\[[0-9a-fA-F:.]+\]|[^\s/?#\"'`)>\]\\]*)[^\s\"'`)>\]\\]*",
    re.I,
)
_LOOPBACK = re.compile(r"localhost|127(?:\.\d{1,3}){3}|\[?::1\]?", re.I)


def loopback_host(authority: str) -> bool:
    """True when the URL authority (user@host:port) names exactly a loopback host (127/8, localhost, ::1)."""
    host = authority.rsplit("@", 1)[-1]
    host = host if host.startswith("[") else host.split(":", 1)[0]
    if host.startswith("[") and "]" in host:
        host = host[: host.index("]") + 1]
    return bool(_LOOPBACK.fullmatch(host))
