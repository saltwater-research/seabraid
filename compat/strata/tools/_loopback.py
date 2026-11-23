# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""HTTP to the local engine server without ever touching a proxy.

OWNER   : the tools' single way to POST to a seabraid server.  (The installer has its own copy, install/_http.py,
          because these scripts ship and run without the installer package.)
CONTRACT: ``post_json(url, body, timeout) -> dict``; raises whatever urllib raises (HTTPError, URLError, ValueError).
WHY     : urllib honours http_proxy/HTTPS_PROXY; on a machine that has one set, a request to 127.0.0.1 would leave the
          host (prompt text included) for the proxy.  An opener with an empty ProxyHandler never does.
"""
import json
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post_json(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as resp:
        return json.load(resp)
