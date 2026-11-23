#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""cluster_profile.py IN OUT HELPER_LAYERS

Rewrites a Strata expert profile (STRP v1) so that every expert of the layers NOT in HELPER_LAYERS ranks first
(hot-first order kept inside that set), followed by the helper layers' experts. Layer and expert counts are read
from the STRP header, not assumed. Setting --expert-cache to (experts per layer) x (number of non-helper layers)
makes the main GPU hold whole layers, so the seabraid helper gets exactly HELPER_LAYERS.
Arguments:
  IN              source STRP v1 profile
  OUT             rewritten profile to write
  HELPER_LAYERS   layers that go to the helper: comma list and/or ranges, e.g. 37-47 or 0,5,37-47
No environment variables. Prints the main/helper pair counts and the --expert-cache value to use.
"""
import argparse
import struct


def parse_layers(spec):
    helper = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        helper.update(range(int(a), int(b or a) + 1))
    return helper


def read_profile(path):
    """-> (layers, experts, ranked (layer, expert) pairs completed to every pair) from a STRP v1 file."""
    raw = open(path, "rb").read()
    assert raw[:4] == b"STRP"
    ver, L, E, want, n = struct.unpack_from("<5I", raw, 4)
    assert ver == 1
    pairs = [struct.unpack_from("<2H", raw, 24 + 4 * i) for i in range(n)]
    seen = set(pairs)
    pairs += [(l, e) for l in range(L) for e in range(E) if (l, e) not in seen]   # complete the ranking
    return L, E, pairs


def write_profile(path, L, E, ranked):
    """Write `ranked` as a STRP v1 file: header, the ranked pairs, then the rank table indexed layer*E+expert."""
    table = [-1] * (L * E)
    for r, (l, e) in enumerate(ranked):
        table[l * E + e] = r
    with open(path, "wb") as f:
        f.write(b"STRP" + struct.pack("<5I", 1, L, E, len(ranked), len(ranked)))
        for l, e in ranked:
            f.write(struct.pack("<2H", l, e))
        f.write(struct.pack(f"<{L*E}i", *table))


def main():
    ap = argparse.ArgumentParser(description="Reorder a Strata STRP v1 expert profile so helper layers rank last.")
    ap.add_argument("src", metavar="IN", help="source STRP v1 profile")
    ap.add_argument("dst", metavar="OUT", help="output profile")
    ap.add_argument("helper", metavar="HELPER_LAYERS", help="helper layers, e.g. 37-47 or 0,5,37-47")
    a = ap.parse_args()
    helper = parse_layers(a.helper)

    L, E, pairs = read_profile(a.src)
    main_pairs = [p for p in pairs if p[0] not in helper]
    side = [p for p in pairs if p[0] in helper]
    write_profile(a.dst, L, E, main_pairs + side)
    print(f"{len(main_pairs)} main-GPU pairs (layers not in {sorted(helper)}), {len(side)} helper pairs; --expert-cache {len(main_pairs)}")


if __name__ == "__main__":
    main()
