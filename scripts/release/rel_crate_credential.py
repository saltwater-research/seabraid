# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Authorize Cargo's publish request only for the sealed crate.

This provider speaks Cargo's version-one stdio protocol. It opens no network
connection and never stores a token. The uploader supplies the expected
crate identity, checksum and witness location in an isolated environment.
"""

import json
import os
from pathlib import Path
import re
import sys


REGISTRY_INDEXES = {
    "sparse+https://index.crates.io/",
    "https://github.com/rust-lang/crates.io-index",
}
MAX_MESSAGE = 65536


def proven_crate(request, environment):
    """Check Cargo's payload identity before making its credential available."""
    if (request.get("v"), request.get("kind"), request.get("operation")) != (1, "get", "publish"):
        raise ValueError("unsupported Cargo credential request")
    registry = request["registry"]
    if registry.get("name") not in (None, "crates-io") or registry.get("index-url") not in REGISTRY_INDEXES:
        raise ValueError("unexpected Cargo registry")
    expected = {
        "name": environment["RELEASE_CRATE_NAME"],
        "vers": environment["RELEASE_CRATE_VERSION"],
        "cksum": environment["RELEASE_CRATE_SHA256"],
    }
    if not re.fullmatch(r"[0-9a-f]{64}", expected["cksum"]):
        raise ValueError("invalid sealed checksum")
    if {name: request.get(name) for name in expected} != expected:
        raise ValueError("Cargo request does not match the sealed crate")
    return expected


def authorize(request, environment):
    expected = proven_crate(request, environment)
    token = environment["RELEASE_CRATE_TOKEN"]
    if not token or not token.isascii() or any(not 33 <= ord(char) <= 126 for char in token):
        raise ValueError("invalid publication credential")
    # A fresh witness lets the caller prove Cargo used this provider. Failed
    # authentication leaves no token in the file or in the publisher log.
    path = Path(environment["RELEASE_CRATE_WITNESS"])
    with path.open("x", encoding="utf-8", newline="\n") as witness:
        witness.write(json.dumps(expected) + "\n")
    return {"kind": "get", "token": token, "cache": "never", "operation_independent": False}


def request_line(stdin):
    raw = stdin.readline(MAX_MESSAGE + 1)
    if len(raw) > MAX_MESSAGE or not raw.endswith("\n"):
        raise ValueError("incomplete Cargo credential message")
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("registry"), dict):
        raise ValueError("malformed Cargo credential message")
    return value


def emit(stdout, value):
    print(json.dumps(value, separators=(",", ":")), file=stdout, flush=True)


def serve(stdin, stdout, environment):
    emit(stdout, {"v": [1]})
    try:
        response = authorize(request_line(stdin), environment)
    except (ValueError, TypeError, KeyError, OSError):
        # Cargo displays error responses. Keep request data and the credential
        # out of that channel even when reading or writing fails.
        emit(stdout, {"Err": {"kind": "other", "message": "sealed crate authorization refused"}})
        return 1
    emit(stdout, {"Ok": response})
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["--cargo-plugin"]:
        raise SystemExit("requires Cargo credential provider invocation")
    raise SystemExit(serve(sys.stdin, sys.stdout, os.environ))
