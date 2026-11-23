# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Hottest-first spillover placement, bound to the native seabraid_placement_fill (native only).

OWNER   : the Python call into seabraid_placement_fill and the shape validation in front of it. The fill ALGORITHM is
          native (src/placement); this module decides nothing about who owns which expert.
CONTRACT: fill(layers, experts, slots, heat, library_path=None) -> list[int]. `slots` holds one capacity per card
          (lead card first, then helpers in rung order), so cards = len(slots); `heat` and the result are layer-major
          (index layer * experts + expert). A result value in 0..cards-1 is a card rung; value `cards` is the
          fallback rung (host/model source). Imports: stdlib + siblings.
SIZE    : under 120 lines.
FAILURE : bad shapes (including more than 2^26 layer x expert cells, as native) raise ValueError before any native call; a missing native library or a library without
          seabraid_placement_fill raises RuntimeError (there is no Python fallback: a second copy of the algorithm
          would be a second owner); a nonzero ABI status raises SeabraidError with the native failure text.
CHECKER : tests/test_placement.py (mocked library: marshalling, validation, absent symbol, status).
FINISH  : those tests pass.
"""

from __future__ import annotations

import ctypes as ct
import math
from collections.abc import Sequence
from pathlib import Path

from . import _abi
from ._model import _UINT32_MAX, checked_integer
from .constants import ABI_VERSION
from .errors import SeabraidError

_UINT64_MAX = (1 << 64) - 1
# Mirrors SEABRAID_PLACEMENT_MAX_CARDS (checked by tests/test_header_parity.py): the native quarantine mask is one
# u64 bit per rung, so more than 63 cards is a native SEABRAID_INVALID; reject it here as a ValueError instead.
_MAX_CARDS = 63
# Mirrors SEABRAID_PLACEMENT_MAX_CELLS (also checked by tests/test_header_parity.py): native refuses a request with more
# (layer, expert) cells than this, so Python refuses it first, before it builds the 8-byte-per-cell heat and owner arrays.
_MAX_CELLS = 1 << 26


def _checked_request(
    layers: int, experts: int, slots: Sequence[int], heat: Sequence[float],
) -> tuple[int, int, list[int], list[float]]:
    layers = checked_integer(layers, 1, _UINT32_MAX, "layers")
    experts = checked_integer(experts, 1, _UINT32_MAX, "experts")
    if layers * experts > _MAX_CELLS:
        raise ValueError(f"layers * experts must be at most {_MAX_CELLS} cells, got {layers * experts}")
    if not 1 <= len(slots) <= _MAX_CARDS:
        raise ValueError(f"slots must hold one capacity for each of 1 to {_MAX_CARDS} cards")
    capacities = [checked_integer(s, 0, _UINT64_MAX, "slot capacity") for s in slots]
    if len(heat) != layers * experts:
        raise ValueError("heat must hold one value per (layer, expert), layer-major")
    heats = [float(h) for h in heat]
    if not all(math.isfinite(h) for h in heats):
        raise ValueError("heat values must be finite")
    return layers, experts, capacities, heats


def fill(
    layers: int, experts: int, slots: Sequence[int], heat: Sequence[float],
    library_path: str | Path | None = None,
) -> list[int]:
    """Assign every (layer, expert) to a card rung, hottest first; see the module contract."""
    layers, experts, capacities, heats = _checked_request(layers, experts, slots, heat)
    library = _abi.resolve_library(library_path)
    if library is None:
        raise RuntimeError(
            "placement.fill needs a native Seabraid library: pass library_path or set SEABRAID_LIBRARY"
        )
    native = getattr(library, "seabraid_placement_fill", None)
    if native is None:
        raise RuntimeError("This Seabraid library does not expose placement fill")
    cards = len(capacities)
    packed_slots = (ct.c_uint64 * cards)(*capacities)
    packed_heat = (ct.c_double * len(heats))(*heats)
    request = _abi._PlacementFillV1(
        ABI_VERSION, layers, experts, cards, packed_slots, cards, packed_heat, len(heats),
    )
    owners = (ct.c_uint8 * len(heats))()
    status = native(ct.byref(request), owners, len(owners))
    if status != 0:
        raise SeabraidError("seabraid_placement_fill", status, _abi.last_error(library))
    return list(owners)
