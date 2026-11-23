# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The revocable view a provider callable receives when numpy is not installed.

OWNER   : the one object type through which a callable touches engine-owned memory without numpy, and the rule that it
          can be neither sliced into a surviving window nor re-exported as a buffer.
CONTRACT: ProviderView(memoryview) -> memoryview-like: len(), view[i], view[i] = x, view[a:b] = values (copies INTO the
          memory), iteration, .tolist(), .tobytes(), bytes(view), .format/.itemsize/.nbytes/.readonly. view[a:b] (a slice
          READ) raises TypeError, there is no .cast()/.toreadonly()/buffer protocol, so no derived window can exist.
          .revoke() releases the one real memoryview: every later use of the view raises ValueError.
          Imports: stdlib only.
SIZE    : under 100 lines.
FAILURE : slice read -> TypeError (loud at the point of the mistake, inside the call); use after revoke -> ValueError
          (loud, never a read of freed memory); writing a read-only view -> TypeError (memoryview's own refusal).
CHECKER : tests/test_provider.py (stash slices and the whole view, with and without numpy).
FINISH  : that test passes.
WHY not memoryview itself: view[0:4] and view.cast() return NEW memoryviews that share the underlying buffer and
          survive release() of the parent, so a callable could keep reading engine memory after the engine reused or
          freed it (R7 MED 2). A class that never hands out another memoryview closes that hole by construction.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterator, Sequence

_NO_SLICE = ("a provider view cannot be sliced (a slice would outlive the call): index it, iterate it, call .tolist() "
             "or .tobytes() for a copy, or assign view[a:b] = values to write")


class ProviderView:
    """A memoryview-like handle on engine memory that is only valid during one provider call."""

    __slots__ = ("_view",)

    def __init__(self, view: memoryview) -> None:
        self._view = view

    def revoke(self) -> None:
        """Make every later use fail with ValueError; never raises (nothing exports this view)."""
        self._view.release()

    def __len__(self) -> int:
        return len(self._view)

    def __getitem__(self, index: int) -> float | int:
        if isinstance(index, slice):
            raise TypeError(_NO_SLICE)
        return self._view[index]

    def __setitem__(self, index: int | slice, value: float | int | Sequence) -> None:
        if isinstance(index, slice):
            # The values are copied into the memory here, so nothing derived from the view is created or kept.
            value = array(self._view.format, value)
        self._view[index] = value

    def __iter__(self) -> Iterator[float | int]:
        return iter(self._view)

    def tolist(self) -> list:
        return self._view.tolist()

    def tobytes(self) -> bytes:
        return self._view.tobytes()

    def __bytes__(self) -> bytes:
        return self._view.tobytes()

    @property
    def format(self) -> str:
        return self._view.format

    @property
    def itemsize(self) -> int:
        return self._view.itemsize

    @property
    def nbytes(self) -> int:
        return self._view.nbytes

    @property
    def readonly(self) -> bool:
        return self._view.readonly

    def __repr__(self) -> str:
        return "<ProviderView: valid only during the provider call>"
