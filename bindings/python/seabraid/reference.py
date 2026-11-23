# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python reference provider: an API demonstration and test transform, NOT model math.

OWNER   : the Python mirror of src/batch.rs (Config.validate, lengths, Batch.load, reference) and of the one-ticket
          session rules in src/session.rs. Used when no native library is configured, and as the parity oracle in tests.
CONTRACT: ReferenceSession(partial_routes=False, provider=None, compact_rows=False) with the session interface shared
          with _native.NativeSession: provider, capabilities(), status(), prepare(shape), submit(tokens, activations,
          expert_ids, weights, origins, operation_id=0, *, placement_epoch=0, activation_format=F32) -> ticket,
          collect(...) -> list[float], close(). Output = sum_k weight_k * activation * (expert_id_k + 1), every
          operation rounded to float32.
          With a PythonProvider the callable replaces that transform (called in submit, on the caller's thread); a
          failing callable surfaces at collect as SeabraidError(PROVIDER), like the native worker's failure.
          Batch v2 / provider v3 semantics mirrored from src/session.rs and src/batch.rs: a nonzero placement_epoch below the highest
          accepted one is BUSY_OR_STATE, 0 after any nonzero epoch is INVALID, and a rejected batch never moves the
          high-water mark; F16/BF16 activations are INVALID unless the provider declared them (the built-in transform
          declares none) and otherwise reach the callable as raw bytes; COMPACT_ROWS ships only the rows with a
          real pick (src/rows.rs, mirrored in _rows.py) and zero-fills the rest, so output is bit-identical to an
          uncompacted run; status() carries rows_submitted / rows_shipped.
          Imports: stdlib + siblings; no network, no files, no environment.
SIZE    : under 320 lines.
ORDER   : two defects in one batch are reported in the engine's order (operation, state, origins and known format, counts,
          epoch, declared format, picks); the ABI promises no order, so tests/test_parity_fuzz.py pins it here and an
          engine reorder is mirrored, not tolerated. Sums start at -0.0 like the native fold (src/batch.rs); failed tickets count as
          provider_failures, not collected.
FAILURE : the same SeabraidError(operation, status) the native library raises for the same input (INVALID, BUSY_OR_STATE);
          compact_rows without partial_routes is SeabraidError(INVALID) at construction, as at native open;
          this provider cannot time out (work finishes inside submit), so collect never returns TIMEOUT.
CHECKER : tests/test_reference.py (known value, every validation branch) and its native parity test (SEABRAID_LIBRARY).
FINISH  : the known value hidden=2 experts=3 max_tokens=2 top_k=2 [1,2] ids [0,2] w [0.25,0.75] -> [2.5, 5.0] holds
          and the parity test passes against the built library.
"""

from __future__ import annotations

import ctypes as ct
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

from . import _rows
from ._model import ELEMENT_BYTES, BatchInfo, Capabilities, SessionStatus, Shape
from ._provider import PythonProvider, provider_flags, run_in_process
from .constants import (
    ALL_SOURCE_MODALITIES, BUSY_OR_STATE, FORMAT_PARTIAL_F32, FORMAT_ROUTED_F32, INVALID,
    OPERATION_ROUTED_MOE, PROVIDER, SKIP_EXPERT_ID, SourceModality, ABI_VERSION,
)
from .errors import SeabraidError, submit_operation

# Both wire elements are 32 bits; src/batch.rs also bounds the byte extent by Rust's isize slice limit (64-bit).
_ELEMENT_LIMIT = ((1 << 63) - 1) // 4
# Distinct across sessions in this process, like the native ticket counter.
_TICKETS = itertools.count(1)


def _f32(value: float) -> float:
    """Round to float32 the way the native binding's c_float packing and Rust f32 arithmetic do."""
    return ct.c_float(value).value


def _element_count(tokens: int, width: int, operation: str) -> int:
    elements = tokens * width
    if elements == 0 or elements > _ELEMENT_LIMIT:
        raise SeabraidError(operation, INVALID)
    return elements


def _lengths(shape: Shape, tokens: int, operation: str) -> tuple[int, int]:
    """Return (rows, picks) element counts for a batch of `tokens`; mirrors Config::lengths."""
    if tokens == 0 or tokens > shape.max_tokens:
        raise SeabraidError(operation, INVALID)
    return (_element_count(tokens, shape.hidden, operation),
            _element_count(tokens, shape.top_k, operation))


def _validate_shape(shape: Shape) -> None:
    """Mirror Config::validate (the ABI version is fixed by this package)."""
    if shape.experts == 0 or shape.top_k > shape.experts:
        raise SeabraidError("seabraid_prepare", INVALID)
    _lengths(shape, shape.max_tokens, "seabraid_prepare")


def _pick_invalid(expert: int, weight: float, experts: int, partial: bool) -> bool:
    if expert == SKIP_EXPERT_ID and partial:
        return weight != 0.0  # a skipped pick must carry no weight (NaN also fails)
    return expert < 0 or expert >= experts or not math.isfinite(weight)


def _check_counts(shape: Shape, tokens: int, counts: tuple[int, int, int], operation: str) -> None:
    """Mirror Lengths::check: the token range and the three element counts."""
    rows, picks = _lengths(shape, tokens, operation)
    if counts != (rows, picks, picks):
        raise SeabraidError(operation, INVALID)


def _check_picks(
    shape: Shape, tokens: int, ids: Sequence[int], weights: Sequence[float], partial: bool, operation: str,
) -> None:
    """Mirror src/picks.rs check: range, skip rule, finite weight, no repeated id in a row (golden.txt pins it)."""
    if any(_pick_invalid(i, w, shape.experts, partial) for i, w in zip(ids, weights)):
        raise SeabraidError(operation, INVALID)
    if any(_repeats_an_expert(ids[row * shape.top_k:(row + 1) * shape.top_k]) for row in range(tokens)):
        raise SeabraidError(operation, INVALID)


def _repeats_an_expert(row: Sequence[int]) -> bool:
    """Mirror pick_error in src/picks.rs: a real expert id may appear once per row; skip markers may repeat."""
    real = [expert for expert in row if expert != SKIP_EXPERT_ID]
    return len(real) != len(set(real))


def _row_value(activation: float, ids: Sequence[int], weights: Sequence[float]) -> float:
    """Sum one output element over a token's picks in order, rounding like f32 arithmetic.

    WHY -0.0 and not 0.0: the native transform (src/batch.rs `reference`) folds explicitly from -0.0 on every
    toolchain (`Iterator::sum` only started at -0.0 in rustc 1.83, and the MSRV 1.79 started at +0.0), so a row whose
    every product is -0.0 (or that has no real pick) is -0.0, not +0.0.
    """
    total = -0.0
    for expert, weight in zip(ids, weights):
        scale = _f32(_f32(expert) + 1.0)
        total = _f32(total + _f32(weight * _f32(activation * scale)))
    return total


def _transform(
    shape: Shape, tokens: int, activations: Sequence[float], ids: Sequence[int], weights: Sequence[float],
) -> list[float]:
    """Mirror batch::reference. Illustrative transform only: it is not a mixture-of-experts forward pass."""
    output: list[float] = []
    for token in range(tokens):
        picks = slice(token * shape.top_k, (token + 1) * shape.top_k)
        row = activations[token * shape.hidden:(token + 1) * shape.hidden]
        output.extend(_row_value(a, ids[picks], weights[picks]) for a in row)
    return output


@dataclass
class _Pending:
    ticket: int
    output: list[float] | None  # None: the provider callable failed; collect reports it
    count: int


def _accepted_formats(provider: PythonProvider | None) -> frozenset:
    """F32 always; F16/BF16 only when the provider declared them (the built-in transform declares none)."""
    return frozenset((FORMAT_ROUTED_F32,)) | (provider.accepts if provider else frozenset())


class ReferenceSession:
    """One in-process session with the native session's state rules and no worker thread."""

    provider = "reference"

    def __init__(
        self, *, partial_routes: bool = False, provider: PythonProvider | None = None, compact_rows: bool = False,
    ) -> None:
        provider_flags(partial_routes, compact_rows)  # the open-time rule, raised before anything exists
        self._partial = partial_routes
        self._compact = compact_rows
        self._provider = provider
        self._shape: Shape | None = None
        self._pending: _Pending | None = None
        self._submitted = self._collected = self._failures = 0
        self._rows_submitted = self._rows_shipped = 0
        self._epoch_high = 0
        self._last_origins = SourceModality(0)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            ABI_VERSION, OPERATION_ROUTED_MOE, FORMAT_ROUTED_F32, FORMAT_PARTIAL_F32,
            ALL_SOURCE_MODALITIES,
        )

    def status(self) -> SessionStatus:
        return SessionStatus(
            self._submitted, self._collected, self._failures, 0, self._last_origins, self._pending is not None,
            self._rows_submitted, self._rows_shipped,
        )

    def prepare(self, shape: Shape) -> None:
        if self._shape is not None:
            raise SeabraidError("seabraid_prepare", BUSY_OR_STATE)
        _validate_shape(shape)
        self._shape = shape

    def _check_epoch(self, epoch: int, operation: str) -> None:
        """Mirror Session::check_epoch: 0 is unversioned until the first nonzero epoch, then every batch needs one."""
        if epoch == 0 and self._epoch_high != 0:
            raise SeabraidError(operation, INVALID)
        if epoch != 0 and epoch < self._epoch_high:
            raise SeabraidError(operation, BUSY_OR_STATE)

    def _provide(
        self, shape: Shape, operation_id: int, tokens: int, acts: object, ids: list[int], wts: list[float],
        info: BatchInfo,
    ) -> list[float] | None:
        """Run the provider (or the built-in transform) on exactly the rows it is given."""
        if self._provider is None:
            return _transform(shape, tokens, acts, ids, wts)
        return run_in_process(self._provider, operation_id, shape, tokens, acts, ids, wts, info)

    def _shipped(
        self, shape: Shape, operation_id: int, tokens: int, acts: Sequence | bytes, ids: list[int],
        wts: list[float], info: BatchInfo,
    ) -> tuple[list[float] | None, int]:
        """(output, rows handed to the provider); with COMPACT_ROWS only kept rows go, the rest are zeros."""
        if not self._compact:
            return self._provide(shape, operation_id, tokens, acts, ids, wts, info), tokens
        rows = _rows.kept_rows(ids, shape.top_k)
        if not rows:
            return [0.0] * (tokens * shape.hidden), 0  # the provider is not called, as in the native worker
        raw = isinstance(acts, bytes)
        # F16/BF16 arrive as raw bytes, so a row is hidden * element-size bytes wide.
        width = shape.hidden * (ELEMENT_BYTES[info.activation_format] if raw else 1)
        row_acts = _rows.gather(acts, rows, width)
        compact = self._provide(
            shape, operation_id, len(rows), bytes(row_acts) if raw else row_acts, _rows.gather(ids, rows, shape.top_k),
            _rows.gather(wts, rows, shape.top_k), info,
        )
        if compact is None:
            return None, len(rows)
        return _rows.scatter(compact, rows, shape.hidden, tokens), len(rows)

    def _admit(
        self, tokens: int, counts: tuple[int, int, int], origins: int, operation_id: int, epoch: int, fmt: int,
        operation: str,
    ) -> Shape:
        """Every check that needs no copy, in the engine's order today: operation, state, origins and known format,
        counts, epoch, declared format. (Picks follow in submit.) The ABI promises no order between two defects, so a
        deliberate engine reorder is mirrored here; the parity fuzz flags it."""
        shape = self._shape
        if operation_id >= (self._provider.operations if self._provider else 1):
            raise SeabraidError(operation, INVALID)
        if self._pending is not None or shape is None:
            raise SeabraidError(operation, BUSY_OR_STATE)
        if origins & ~int(ALL_SOURCE_MODALITIES) or fmt not in ELEMENT_BYTES:
            raise SeabraidError(operation, INVALID)
        _check_counts(shape, tokens, counts, operation)
        self._check_epoch(epoch, operation)
        if fmt not in _accepted_formats(self._provider):
            raise SeabraidError(operation, INVALID)
        return shape

    def submit_name(self, origins: int, operation_id: int = 0) -> str:
        """The operation name this session reports for a submission (the per-shape names, as in a v1 library)."""
        return submit_operation(origins, operation_id)

    def submit(
        self, tokens: int, activations: Sequence[float], expert_ids: Sequence[int],
        weights: Sequence[float], origins: int, operation_id: int = 0, *, placement_epoch: int = 0,
        activation_format: int = FORMAT_ROUTED_F32, ticket_slot: ct.c_uint64 | None = None,
    ) -> int:
        """`activations` holds floats for F32, or a ctypes uint16 array (raw 16-bit elements) for F16/BF16.

        `ticket_slot` mirrors NativeSession.submit: the ticket is written into it as well as returned.
        """
        try:
            operation = self.submit_name(origins, operation_id)
            shape = self._admit(
                tokens, (len(activations), len(expert_ids), len(weights)), origins, operation_id, placement_epoch,
                activation_format, operation,
            )
            # Round inputs to float32 first: the native binding stores them as c_float, and inf must fail the finite check.
            acts = [_f32(a) for a in activations] if activation_format == FORMAT_ROUTED_F32 else bytes(activations)
            wts = [_f32(w) for w in weights]
            ids = list(expert_ids)
            _check_picks(shape, tokens, ids, wts, self._partial, operation)
            info = BatchInfo(activation_format, placement_epoch)
            # WHY the provider runs before anything is committed: a KeyboardInterrupt inside it propagates (see
            # PythonProvider.invoke), and an interrupted submit must leave the session exactly as it was.
            output, shipped = self._shipped(shape, operation_id, tokens, acts, ids, wts, info)
            ticket = next(_TICKETS)
            self._epoch_high = max(self._epoch_high, placement_epoch)  # only after every check passed
            self._pending = _Pending(ticket, output, len(activations))
            self._submitted += 1
            self._rows_submitted += tokens
            self._rows_shipped += shipped
            self._last_origins = SourceModality(origins)
            if ticket_slot is not None:
                ticket_slot.value = ticket
            return ticket
        finally:
            # WHY: a refusal's traceback keeps this frame; the packed arrays may alias the caller's buffers.
            activations = expert_ids = weights = None

    def collect(self, ticket: int, timeout_ms: int, count: int, out: ct.Array | None = None) -> list[float] | None:
        pending = self._pending
        if pending is None:
            raise SeabraidError("seabraid_collect", BUSY_OR_STATE)
        if pending.ticket != ticket or count != pending.count:
            raise SeabraidError("seabraid_collect", INVALID)
        self._pending = None
        if pending.output is None:
            self._failures += 1  # a failed ticket is not a collected one, as in the engine's counters
            raise SeabraidError("seabraid_collect", PROVIDER)
        self._collected += 1
        if out is None:
            return pending.output
        out[:] = pending.output
        return None


    def close(self, timeout_ms: int | None = None) -> None:
        # Nothing runs in the background here, so there is nothing to wait for or abandon.
        self._pending = None
