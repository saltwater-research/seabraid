# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Quickstart: open a Seabraid runtime, weave one batch, and order two cards.

It asks for the reference transform with Runtime(reference=True), which is the same call on both paths. With
SEABRAID_LIBRARY unset it runs the pure-Python reference provider, an API demonstration and test transform, not model
math. With SEABRAID_LIBRARY set to the built native library, the same code runs the engine's own reference transform
natively. (A bare Runtime() with the variable set is refused: a native library needs reference=True or a provider=.)
"""

from typing import NamedTuple

import seabraid

print(f"seabraid {seabraid.version()}")

with seabraid.Runtime(reference=True) as runtime:
    prepared = runtime.prepare(hidden=2, experts=3, max_tokens=2, top_k=2)
    # One token, activations [1, 2], routed to experts 0 and 2 with weights 0.25 and 0.75.
    result = prepared.weave([1.0, 2.0], [0, 2], [0.25, 0.75])
    print(f"provider={runtime.provider} result={result}")  # both providers give [2.5, 5.0]
    # An engine that moves experts tags each batch with its placement epoch: a stale tag is refused.
    prepared.weave([1.0, 2.0], [0, 2], [0.25, 0.75], placement_epoch=2)
    try:
        prepared.weave([1.0, 2.0], [0, 2], [0.25, 0.75], placement_epoch=1)
    except seabraid.SeabraidError as error:
        print(f"stale epoch refused: status {error.status}")


class Card(NamedTuple):
    """The fields seabraid.ranking reads from a card."""

    key: tuple
    name: str
    arch: str
    free: int
    total: int
    pcie: int


GIB = 1 << 30
cards = [
    Card(("cuda", 0), "RTX 3060", "sm_86", 11 * GIB, 12 * GIB, 16),
    Card(("rocm", 0), "RX 7900 XTX", "gfx1100", 23 * GIB, 24 * GIB, 16),
]
# Measured lead-job ms per decode window (lower wins card 1) and decode expert rows/s (orders the rest).
ordered, basis = seabraid.ranking.rank(
    cards, {c.key: ms for c, ms in zip(cards, (41.0, 28.0))}, {c.key: r for c, r in zip(cards, (900.0, 2400.0))},
    fits_main=lambda card: True,
)
print("card order:", " > ".join(card.name for card in ordered), f"(basis: {basis})")
