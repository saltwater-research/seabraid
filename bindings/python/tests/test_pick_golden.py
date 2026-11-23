# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The reference provider against src/picks/golden.txt, the pick rule's table shared with Rust and the ggml adapter.

OWNER   : src/picks.rs owns the rule and the table; this file only reads it. CHECKER for the rule in reference.py.
CONTRACT: every `case` row is submitted to ReferenceSession and must be accepted (`ok`) or refused INVALID (any other
          verdict). The reference provider names no kind or pick, so those parts of the verdict are the Rust test's.
FAILURE : fail-fast; a missing or empty table FAILS (it must never read as "no cases").
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from seabraid import SeabraidError
from seabraid._model import Shape
from seabraid.constants import INVALID
from seabraid.reference import ReferenceSession

TABLE = Path(__file__).resolve().parents[3] / "src" / "picks" / "golden.txt"


def _weight(token: str) -> float:
    return -0.0 if token == "-0" else float(token)


def _rows() -> list[tuple[str, int, bool, int, list[int], list[float], str]]:
    rows = []
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("case "):
            continue
        name, shape, ids_text, overrides, verdict = line.split(" | ")
        experts, partial, top_k = (int(part) for part in shape.split())
        ids = [int(part) for part in ids_text.split()]
        weights = [0.5] * len(ids)
        for entry in overrides.split():
            if entry != "-":
                pick, value = entry.split("=")
                weights[int(pick)] = _weight(value)
        rows.append((name.removeprefix("case "), experts, partial == 1, top_k, ids, weights, verdict))
    return rows


class GoldenPickTests(unittest.TestCase):
    def test_every_row_is_accepted_or_refused_as_the_table_says(self) -> None:
        rows = _rows()
        self.assertGreaterEqual(len(rows), 30, f"{TABLE} lost its rows")
        for name, experts, partial, top_k, ids, weights, verdict in rows:
            with self.subTest(row=name):
                tokens = len(ids) // top_k
                self.assertEqual(len(ids), tokens * top_k)
                session = ReferenceSession(partial_routes=partial)
                session.prepare(Shape(hidden=1, experts=experts, max_tokens=tokens, top_k=top_k))
                if verdict == "ok":
                    ticket = session.submit(tokens, [1.0] * tokens, ids, weights, 0)
                    self.assertTrue(all(math.isfinite(value) for value in session.collect(ticket, 0, tokens)))
                else:
                    with self.assertRaises(SeabraidError) as caught:
                        session.submit(tokens, [1.0] * tokens, ids, weights, 0)
                    self.assertEqual(caught.exception.status, INVALID)


if __name__ == "__main__":
    unittest.main()
