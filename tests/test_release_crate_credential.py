# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Exercise the Cargo credential handshake with a synthetic token."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

RELEASE = Path(__file__).resolve().parents[1] / "scripts" / "release"
sys.path.insert(0, str(RELEASE))
from rel_crate_credential import REGISTRY_INDEXES  # noqa: E402

IDENTITY = {"name": "test-crate", "vers": "1.0.0", "cksum": "a" * 64}
SENTINEL = "synthetic-test-credential"


@pytest.fixture
def auth(tmp_path):
    witness = tmp_path / "witness.json"
    env = {
        **os.environ,
        "RELEASE_CRATE_NAME": IDENTITY["name"],
        "RELEASE_CRATE_VERSION": IDENTITY["vers"],
        "RELEASE_CRATE_SHA256": IDENTITY["cksum"],
        "RELEASE_CRATE_TOKEN": SENTINEL,
        "RELEASE_CRATE_WITNESS": str(witness),
    }
    request = {"v": 1, "kind": "get", "operation": "publish", **IDENTITY}
    request["registry"] = {"index-url": next(iter(REGISTRY_INDEXES)), "name": "crates-io"}
    return witness, env, request


def invoke(env, request):
    result = subprocess.run(
        [sys.executable, "-B", str(RELEASE / "rel_crate_credential.py"), "--cargo-plugin"],
        input=json.dumps(request) + "\n", env=env, text=True, capture_output=True, check=False, timeout=10,
    )
    hello, reply = map(json.loads, result.stdout.splitlines())
    assert hello == {"v": [1]}
    assert not result.stderr
    return result.returncode, reply


@pytest.mark.parametrize("index", sorted(REGISTRY_INDEXES))
def test_matching_payload_gets_an_operation_scoped_credential(auth, index):
    witness, env, request = auth
    request["registry"]["index-url"] = index
    code, reply = invoke(env, request)
    assert code == 0
    assert reply == {"Ok": {"kind": "get", "token": SENTINEL, "cache": "never", "operation_independent": False}}
    assert json.loads(witness.read_text(encoding="utf-8")) == IDENTITY
    assert SENTINEL not in witness.read_text(encoding="utf-8")


@pytest.mark.parametrize("change", [
    {"name": "other-crate"}, {"vers": "1.0.1"}, {"cksum": "b" * 64}, {"operation": "read"},
    {"registry": {"index-url": "http://localhost/", "name": "crates-io"}},
])
def test_wrong_payload_or_registry_never_gets_the_token(auth, change):
    witness, env, request = auth
    code, reply = invoke(env, {**request, **change})
    assert code == 1 and "Err" in reply
    assert SENTINEL not in json.dumps(reply) and not witness.exists()


def test_stale_witness_is_not_overwritten(auth):
    witness, env, request = auth
    witness.write_text("old witness", encoding="utf-8")
    code, reply = invoke(env, request)
    assert code == 1 and "Err" in reply
    assert SENTINEL not in json.dumps(reply)
    assert witness.read_text(encoding="utf-8") == "old witness"
