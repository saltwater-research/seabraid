# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""HTTP to the engine server the installer started, never through a proxy.

OWNER     : the installer's only HTTP client (health check, lead-probe requests).  tools/_loopback.py is the same
            rule for the standalone scripts, which ship without this package.
CONTRACT  : ``loopback_open(url_or_request, timeout)`` -> a urllib response (context manager), raising what urllib raises.
WHY       : urllib honours http_proxy/HTTPS_PROXY.  On a machine that sets one, a request to 127.0.0.1 would leave the host
            for the proxy, carrying the prompt and the API key; an opener with an empty ProxyHandler never does.
CHECKER   : tests/install/test_hardening.py (a proxy environment variable is not used; urlopen appears only in
            the proxy-free helpers).
"""

from __future__ import annotations

import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def loopback_open(url_or_request, timeout: float):
    return _OPENER.open(url_or_request, timeout=timeout)
