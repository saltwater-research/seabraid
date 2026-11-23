# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Public Runtime and PreparedMoE: pick a provider, validate caller input, track one pending batch.

OWNER   : the Python session API (Runtime, PreparedMoE) and the choice between the native library and the reference
          provider. Routing and expert math belong to the native runtime (or, for demos, reference.py).
CONTRACT: Runtime(library_path=None, *, provider=None, operations=1, partial_routes=False, reference=False,
          compact_rows=False, accepts=()) -> prepare(...) -> PreparedMoE.weave(..., operation=0, out=None,
          placement_epoch=0, activation_format=F32)/collect(out=None); Runtime.reference(...)
          is Runtime(..., reference=True); Runtime.capabilities(), .status(), .provider, .close(timeout_ms=None).
          A path (argument or env SEABRAID_LIBRARY) selects the native library, which then REQUIRES provider= or
          reference=True (the engine refuses a null provider so the reference transform is never used by accident);
          with no path, the pure-Python reference provider runs and ONE warnings.warn plus one logger "seabraid"
          warning say so (reference=True is the explicit request and stays silent).
          provider= is a Python callable hosted as the engine's provider (see _provider.py) on either path; it gets
          ProviderView objects without numpy and numpy arrays with it, valid only during the call.
          Batch v2 / provider v3 options, all defaulted off so existing callers see no change: weave(placement_epoch=N) tags the batch
          (stale or untagged-after-tagged epochs are refused, see reference.py); accepts=(FORMAT_ROUTED_F16 and/or
          FORMAT_ROUTED_BF16) declares formats the provider takes, after which weave(activation_format=...) takes the
          raw bytes of the 16-bit elements (no conversion here; numpy not required); compact_rows=True ships the
          provider only rows with a real pick and needs partial_routes. The reference transform (reference=True)
          accepts partial_routes and compact_rows (it is the parity oracle) but no formats.
          Inputs are float32/int32 buffers (fast, see _buffers.py) or plain sequences (slow); results are a list, or
          the caller's `out` buffer. One RLock per Runtime serialises prepare/weave/collect/status/close, so a close()
          from another thread waits for the call in flight (bounded by that call's timeout_ms) instead of freeing the
          session under it. Imports: stdlib + siblings.
SIZE    : under 400 lines.
FAILURE : native or reference failures raise SeabraidError(operation, status[, detail]); bad Python arguments raise
          ValueError/TypeError before any provider call; use after close, double prepare and any call from inside this runtime's own provider callable raise RuntimeError
          (the last at once: waiting would outlast the caller's timeout); Ctrl-C in weave after the engine took the batch keeps it collectable, Ctrl-C in collect forgets the batch when the engine
          already finished it (the result is lost), else leaves it collectable;
          a count refused by the Python precheck is SeabraidError(INVALID, detail) named after the call the session
          would have made, with one WARNING record, like the engine's own refusal;
          operations/partial_routes/accepts without provider= (or reference=True for partial_routes and compact_rows) raise
          ValueError; compact_rows without partial_routes raises SeabraidError(INVALID) before any C call; a native library without provider= or
          reference=True raises ValueError; close(timeout_ms) raises SeabraidError(TIMEOUT) when the provider is stuck; an unloadable explicit library raises OSError (there is never a silent fallback to the reference).
CHECKER : tests/test_mock_runtime.py, tests/test_reference.py, tests/test_native_runtime.py.
FINISH  : all pass; the reference/native parity test passes with SEABRAID_LIBRARY set.
"""

from __future__ import annotations

import ctypes as ct
import logging
import os
import threading
import warnings
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import NoReturn, TypeVar

from . import _abi
from ._buffers import FLOAT32, INT32, Operand, RawHalfOperand, output_target
from ._model import _UINT32_MAX, _UINT64_MAX, ELEMENT_BYTES, Capabilities, SessionStatus, Shape, checked_integer
from ._native import NativeSession
from ._provider import PythonProvider, calling_provider, provider_flags
from .constants import BUSY_OR_STATE, FORMAT_ROUTED_F32, INVALID, TIMEOUT, SourceModality
from .errors import SeabraidError
from .reference import ReferenceSession

__all__ = ["Capabilities", "SeabraidError", "PreparedMoE", "Runtime", "SessionStatus", "SourceModality"]

_LOG = logging.getLogger("seabraid")
# The caller's `out` buffer: weave/collect hand back that same object, so the annotation says "whatever you passed".
_Out = TypeVar("_Out")
_DEFAULT_TIMEOUT_MS = 30_000
_PROVIDER_NOTICE = (
    "Seabraid has no native library, so your provider callable runs inside the pure-Python reference session on the "
    "caller's thread. Set SEABRAID_LIBRARY or pass library_path to host it on the native runtime's worker thread."
)
_REFERENCE_NOTICE = (
    "Seabraid is using the pure-Python reference provider, an API demonstration and test transform, not model math. "
    "Set SEABRAID_LIBRARY or pass library_path to use the native runtime."
)

# Every Runtime, so a forked child can replace a lock that a vanished parent thread may have held (see _renew_locks).
_RUNTIMES: "weakref.WeakSet[Runtime]" = weakref.WeakSet()


def _renew_locks() -> None:
    """Child side of os.fork(): give every Runtime a fresh, unlocked lock.

    WHY: fork copies only the calling thread, so a lock another thread held at that instant stays locked forever in
    the child and every call would hang before reaching the "belongs to the parent process" RuntimeError. Nothing in
    the child can legitimately own the old lock, so replacing it is safe.
    """
    for runtime in list(_RUNTIMES):
        runtime._lock = threading.RLock()


if hasattr(os, "register_at_fork"):  # absent on Windows, where there is no fork
    os.register_at_fork(after_in_child=_renew_locks)


def _select_session(
    library_path: str | Path | None, provider: PythonProvider | None, partial_routes: bool, reference: bool,
    compact_rows: bool,
) -> NativeSession | ReferenceSession:
    """Native when a path is given or SEABRAID_LIBRARY is set; otherwise the reference provider, announced loudly."""
    path = _abi.configured_path(library_path)
    if path is not None:
        return NativeSession(path, provider, reference=reference, partial_routes=partial_routes,
                             compact_rows=compact_rows)
    session = ReferenceSession(partial_routes=partial_routes, provider=provider, compact_rows=compact_rows)
    if not reference:
        # A silent fallback would let a deployment believe it runs the native engine, so say so twice (warning + log).
        notice = _REFERENCE_NOTICE if provider is None else _PROVIDER_NOTICE
        warnings.warn(notice, RuntimeWarning, stacklevel=3)
        _LOG.warning(notice)
    return session  # reference=True was asked for, so no notice


def _make_provider(
    function: object, operations: int, partial_routes: bool, compact_rows: bool, accepts: Sequence[int],
) -> PythonProvider | None:
    """Wrap `function` for hosting; the extra arguments only mean something with a function to host."""
    if function is not None:
        return PythonProvider(function, operations, partial_routes, compact_rows, accepts)
    if operations != 1 or partial_routes or compact_rows or accepts:
        raise ValueError("operations, partial_routes, compact_rows and accepts require provider=")
    return None


class Runtime:
    """Own a single session; close explicitly or use a with block.

    Calls from several threads are serialised by one lock, so close() cannot free the session under a weave/collect.
    A provider callable must still not call back into its Runtime (on the native path it runs on another thread).
    """

    def __init__(
        self, library_path: str | Path | None = None, *, provider: Callable | None = None,
        operations: int = 1, partial_routes: bool = False, reference: bool = False, compact_rows: bool = False,
        accepts: Sequence[int] = (),
    ) -> None:
        """`provider(operation_id, config, activations, expert_ids, weights, output)` serves the experts itself.

        It receives zero-copy views valid only during the call, runs on the native runtime's worker thread (the
        caller's thread on the reference path), must fill `output`, and must not call back into this Runtime.
        `reference=True` asks for the built-in reference transform (an API demonstration, not model math) and
        excludes provider=; a native library needs one of the two. `compact_rows=True` (needs partial_routes) gives the
        provider only the rows that have a real pick; `accepts` lists the extra activation formats (F16/BF16 IDs from
        seabraid.constants) a provider takes, which then also receives a trailing BatchInfo argument.
        """
        if reference and provider is not None:
            raise ValueError("provider and reference are mutually exclusive")
        if reference and accepts:
            raise ValueError("the reference transform takes only F32, so reference excludes accepts")
        if reference:
            provider_flags(partial_routes, compact_rows)  # same pre-C refusal as for a hosted provider
            hosted = None
        else:
            hosted = _make_provider(provider, operations, partial_routes, compact_rows, accepts)
        if reference and operations != 1:
            raise ValueError("operations requires provider=")
        self._session = _select_session(library_path, hosted, partial_routes, reference, compact_rows)
        self._hosted = hosted
        self._open = True
        self._shape: Shape | None = None
        # WHY one RLock (not a busy flag): the engine documents "callers must serialize calls", so serialising here turns
        # a close()-vs-collect() use-after-free into a wait; re-entrant because weave() calls collect() itself.
        self._lock = threading.RLock()
        _RUNTIMES.add(self)

    @classmethod
    def reference(cls, library_path: str | Path | None = None) -> Runtime:
        """A runtime on the built-in reference transform: an API demonstration and test transform, not model math."""
        return cls(library_path, reference=True)

    @property
    def provider(self) -> str:
        """Which provider serves this runtime: "native" or "reference"."""
        return self._session.provider

    def _serialised(self) -> threading.RLock:
        """The call lock, refused up front when this runtime's own provider callable is the caller.

        WHY a RuntimeError and not the lock: the callable runs while a weave/collect holds the lock (on the native path
        on another thread), so waiting would block for that call's whole timeout and then act on a batch in flight.
        """
        if self._hosted is not None and calling_provider() is self._hosted:
            raise RuntimeError("a provider callable must not call back into its own Runtime")
        return self._lock

    def _require_open(self) -> None:
        if not self._open:
            raise RuntimeError("Seabraid runtime is closed")

    def capabilities(self) -> Capabilities:
        """Describe the loaded runtime's routed-activation interface."""
        with self._serialised():
            self._require_open()
            return self._session.capabilities()

    def status(self) -> SessionStatus:
        """Report counts observed by the caller and the current ticket state."""
        with self._serialised():
            self._require_open()
            return self._session.status()

    def prepare(self, *, hidden: int, experts: int, max_tokens: int, top_k: int) -> PreparedMoE:
        """Prepare this session once; v1 binds its shape to the provider session."""
        with self._serialised():
            self._require_open()
            if self._shape is not None:
                raise RuntimeError("Seabraid runtime is already prepared")
            shape = Shape(
                checked_integer(hidden, 1, _UINT32_MAX, "hidden"),
                checked_integer(experts, 1, _UINT32_MAX, "experts"),
                checked_integer(max_tokens, 1, _UINT32_MAX, "max_tokens"),
                checked_integer(top_k, 1, _UINT32_MAX, "top_k"),
            )
            self._session.prepare(shape)
            self._shape = shape
            return PreparedMoE(self, shape)

    def close(self, timeout_ms: int | None = None) -> None:
        """Release the session, including any pending batch, exactly once.

        Without timeout_ms this waits for a running provider callback however long it takes. With timeout_ms a
        stuck provider is abandoned: SeabraidError(TIMEOUT) is raised, the worker thread is leaked (logged at
        ERROR by the engine) and the provider callable is kept alive for the process lifetime. The runtime is
        closed either way. A library without seabraid_close_timeout raises RuntimeError and leaves the runtime
        open, so close() without a timeout still works.
        """
        if timeout_ms is not None:
            timeout_ms = checked_integer(timeout_ms, 0, _UINT32_MAX, "timeout_ms")
        with self._serialised():  # waits for a weave/collect in flight on another thread
            if self._open:
                # The provider close consumes the handle even when it reports failure, so never retry it ...
                self._open = False
                try:
                    self._session.close(timeout_ms)
                except SeabraidError as error:
                    # ... except BUSY_OR_STATE: the engine refused (a busy handle, or a close from its own provider
                    # thread) and kept the handle, so the runtime stays open and a later close() can try again.
                    if error.status == BUSY_OR_STATE:
                        self._open = True
                    raise
                except RuntimeError:
                    # A session refusing before it touched the handle (a library without seabraid_close_timeout):
                    # nothing was closed, so the runtime stays open and close() without a timeout still works.
                    self._open = True
                    raise

    def __enter__(self) -> Runtime:
        self._require_open()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True)
class _Pending:
    ticket: int
    output_count: int


def _refuse(operation: str, detail: str) -> NoReturn:
    """Log one WARNING like the engine's own refusal and raise INVALID with the same text as the detail."""
    _LOG.warning("%s: %s", operation, detail)
    raise SeabraidError(operation, INVALID, detail)


class PreparedMoE:
    """Submit packed rows using the fixed shape set by Runtime.prepare()."""

    def __init__(self, runtime: Runtime, shape: Shape) -> None:
        self._runtime = runtime
        self._shape = shape
        self._pending: _Pending | None = None

    def _check_submit_shape(self, tokens: int, id_count: int, weight_count: int, operation: str) -> None:
        """Mirror the engine's count check before touching caller data or allocating provider storage.

        WHY detail and a log record here: the engine's refusal carries its reason and logs once (every error is
        logged); a Python refusal that did neither would be the only silent one. `operation` is the call the session
        would have made (session.submit_name), so the message matches the engine's own.
        """
        shape = self._shape
        picks = tokens * shape.top_k
        if tokens > shape.max_tokens:
            detail = f"tokens {tokens} outside 1..={shape.max_tokens}"
        elif id_count != picks:
            detail = f"expert_id_count {id_count} != tokens {tokens} * top_k {shape.top_k}"
        elif weight_count != picks:
            detail = f"weight_count {weight_count} != tokens {tokens} * top_k {shape.top_k}"
        else:
            return
        _refuse(operation, detail)

    def weave(
        self, activations: Sequence[float], expert_ids: Sequence[int], weights: Sequence[float],
        *, timeout_ms: int = _DEFAULT_TIMEOUT_MS, source_modalities: SourceModality = SourceModality(0),
        operation: int = 0, out: _Out | None = None, placement_epoch: int = 0,
        activation_format: int = FORMAT_ROUTED_F32,
    ) -> list[float] | _Out:
        """Weave flat, row-major buffers on prepared `operation`; retry a timed-out batch with collect().

        Inputs may be C-contiguous float32/int32 buffers (array.array, memoryview, numpy; zero-copy, fast) or plain
        sequences (converted element by element, slow). Returns a list, or `out` itself when a writable float32
        buffer of exactly tokens * hidden elements is given.

        `placement_epoch` (default 0, unversioned) tags the batch: once a nonzero epoch was accepted, a lower one is
        refused (status 2) and 0 is refused (status 1). `activation_format` is FORMAT_ROUTED_F32 (default) or, for a
        provider that declared it, FORMAT_ROUTED_F16 / FORMAT_ROUTED_BF16, in which case `activations` is any
        C-contiguous buffer holding the raw 16-bit elements (bytes, array('H'), a numpy float16 array): they are
        passed through, never converted. Output is always float32.
        """
        with self._runtime._serialised():
            self._runtime._require_open()
            if self._pending is not None:
                raise RuntimeError("A Seabraid batch is pending; call collect() before weaving again")
            # WHY try/finally: the operands (and the packed arrays inside submit) hold buffer exports on the caller's
            # arrays; a refusal's traceback keeps these frames, so without clearing them the caller could not
            # append to or resize its array until the exception was freed (R7 LOW 10).
            rows = ids = wts = None
            try:
                # Validate before submit so bad caller input never strands a ticket.
                timeout = checked_integer(timeout_ms, 0, _UINT32_MAX, "timeout_ms")
                fmt = checked_integer(activation_format, 0, _UINT32_MAX, "activation_format")
                epoch = checked_integer(placement_epoch, 0, _UINT64_MAX, "placement_epoch")
                if fmt not in ELEMENT_BYTES:
                    _refuse(self._runtime._session.submit_name(0, 0), f"unknown activation format {fmt}")
                rows = Operand(activations, FLOAT32, "activations") if fmt == FORMAT_ROUTED_F32 \
                    else RawHalfOperand(activations, "activations")
                ids, wts = Operand(expert_ids, INT32, "expert_ids"), Operand(weights, FLOAT32, "weights")
                tokens, remainder = divmod(rows.count, self._shape.hidden)
                if remainder:
                    raise ValueError("activations must contain whole hidden-sized rows")
                tokens = checked_integer(tokens, 1, _UINT32_MAX, "tokens")
                origins = checked_integer(source_modalities, 0, _UINT32_MAX, "source_modalities")
                operation_id = checked_integer(operation, 0, _UINT32_MAX, "operation")
                session = self._runtime._session
                self._check_submit_shape(tokens, ids.count, wts.count, session.submit_name(origins, operation_id))
                if out is not None:
                    output_target(out, rows.count)  # reject a bad `out` now, before a ticket exists
                # WHY a caller-owned slot: the session writes the ticket into it as the native call returns, so an
                # interrupt that lands anywhere before `_pending` is set still leaves the ticket recoverable here.
                slot = ct.c_uint64()
                try:
                    ticket = self._runtime._session.submit(
                        tokens, rows.pack(), ids.pack(), wts.pack(), origins, operation_id, placement_epoch=epoch,
                        activation_format=fmt, ticket_slot=slot,
                    )
                    self._pending = _Pending(ticket, rows.count)
                except KeyboardInterrupt:
                    if self._pending is None and slot.value:  # the engine accepted the batch: keep it collectable
                        self._pending = _Pending(slot.value, rows.count)
                    raise
                return self.collect(timeout_ms=timeout, out=out)
            finally:
                rows = ids = wts = None

    def _settle_after_interrupt(self) -> None:
        """After Ctrl-C in collect, forget the batch if the engine already completed it.

        WHY: a signal is handled only once the native wait returns, so the ticket is usually consumed (result lost)
        while `_pending` still names it, and the next weave would be refused as "pending". The engine's own
        `outstanding` flag says which case this is; if it cannot be read, the batch stays pending for collect().
        """
        try:
            if not self._runtime._session.status().outstanding:
                self._pending = None
        except Exception:  # cannot tell: leave it to collect(), which reports the truth
            _LOG.warning("seabraid could not read the session status after an interrupt; the batch stays pending")

    def collect(self, *, timeout_ms: int = _DEFAULT_TIMEOUT_MS, out: _Out | None = None) -> list[float] | _Out:
        """Collect a pending batch; a timeout leaves it pending for another attempt.

        Pass the same `out` again when retrying a weave(out=...) that timed out.
        """
        with self._runtime._serialised():
            self._runtime._require_open()
            pending = self._pending
            if pending is None:
                raise RuntimeError("No Seabraid batch is pending")
            timeout = checked_integer(timeout_ms, 0, _UINT32_MAX, "timeout_ms")
            target = None if out is None else output_target(out, pending.output_count)
            try:
                output = self._runtime._session.collect(pending.ticket, timeout, pending.output_count, target)
            except SeabraidError as error:
                if error.status != TIMEOUT:
                    self._pending = None
                raise
            except KeyboardInterrupt:
                self._settle_after_interrupt()
                raise
            self._pending = None
            return out if target is not None else output
