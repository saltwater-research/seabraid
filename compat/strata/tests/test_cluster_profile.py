# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for tools/cluster_profile.py: --help works, and a tiny STRP v1 profile is rewritten with the helper layers
ranked last (hot-first order kept inside each set), the ranking completed to every (layer, expert), and the rank table
consistent.  CPU only, stdlib + pytest."""
import pathlib
import struct
import subprocess
import sys

TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "cluster_profile.py"
L, E = 3, 2


def run(*args):
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True)


def test_help_exits_zero():
    p = run("--help")
    assert p.returncode == 0 and "HELPER_LAYERS" in p.stdout


def test_helper_layers_rank_last_and_ranking_is_completed(tmp_path):
    hot = [(2, 1), (0, 1), (1, 0)]   # hottest first; the other three pairs are unranked and get appended
    src, dst = tmp_path / "in.strp", tmp_path / "out.strp"
    table = [-1] * (L * E)
    for r, (l, e) in enumerate(hot):
        table[l * E + e] = r
    src.write_bytes(b"STRP" + struct.pack("<5I", 1, L, E, len(hot), len(hot))
                    + b"".join(struct.pack("<2H", *p) for p in hot) + struct.pack(f"<{L*E}i", *table))
    p = run(str(src), str(dst), "2")
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "4 main-GPU pairs (layers not in [2]), 2 helper pairs; --expert-cache 4"
    raw = dst.read_bytes()
    ver, l, e, want, n = struct.unpack_from("<5I", raw, 4)
    assert (ver, l, e, want, n) == (1, L, E, 6, 6)
    ranked = [struct.unpack_from("<2H", raw, 24 + 4 * i) for i in range(n)]
    assert ranked == [(0, 1), (1, 0), (0, 0), (1, 1), (2, 1), (2, 0)]
    out_table = struct.unpack_from(f"<{L*E}i", raw, 24 + 4 * n)
    assert all(out_table[a * E + b] == r for r, (a, b) in enumerate(ranked))
