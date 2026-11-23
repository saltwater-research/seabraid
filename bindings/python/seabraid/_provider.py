# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Host a Python callable as the Seabraid provider (seabraid_provider_v2/v3 .execute).

OWNER   : the translation between the C provider callback and a Python callable, shared by the native session
          (ctypes trampoline) and the reference session (direct call), so both behave identically.
CONTRACT: PythonProvider(function, operations, partial_routes, compact_rows=False, accepts=()). function(operation_id,
          config, activations, expert_ids, weights, output) -> None, where config is a Shape and the other four are
          zero-copy views valid ONLY during the call: ProviderView objects (read-only float32/int32/float32 and one
          writable float32; see _views.py: indexable and iterable, no slice reads, no memoryview re-export) or numpy
          arrays over the same memory when numpy is importable. The callable must fill every output element. With COMPACT_ROWS the views hold only the kept rows (fewer tokens than were submitted).
          A provider that declares accepts=(FORMAT_ROUTED_F16 and/or FORMAT_ROUTED_BF16) gets ONE extra trailing
          argument, a BatchInfo(activation_format, placement_epoch), and `activations` is a read-only view of RAW BYTES
          (2 bytes per element, ProviderView format "B"; numpy uint8) whenever that batch is F16/BF16: Python never
          converts them, so numpy is not required. F32 batches keep the float32 view.
          .struct() is the seabraid_provider_v2 for seabraid_open_multi; .struct_v3(config) is the
          seabraid_provider_v3 for seabraid_session_open; .invoke(...) is the guarded call.
          provider_flags() is the one place the flag combination rules are checked (before any C call).
          Imports: stdlib + siblings; numpy only if already installed.
SIZE    : under 340 lines.
FAILURE : nothing raised by the callable ever crosses the C boundary: it is caught, ONE record is logged on logger
          "seabraid" with the exception type and operation id (never tensor data), and SEABRAID_PROVIDER is returned.
          A callable that keeps a numpy array, or any slice of one, past the call (the buffer is still exported or the
          array still alive) also gets SEABRAID_PROVIDER and one ERROR: numpy memory cannot be revoked, so that case is
          detected and failed, not made unusable. Without numpy nothing aliasing engine memory can be kept at all.
          A bad operation id or null pointer returns SEABRAID_INVALID. provider_flags raises SeabraidError(INVALID) for
          COMPACT_ROWS without PARTIAL_ROUTES (the same rule the engine enforces at open) and ValueError for a format
          that is not F16/BF16.
CHECKER : tests/test_provider.py (trampoline driven through the real CFUNCTYPE, exceptions, zero-copy, numpy).
FINISH  : that test passes; with SEABRAID_LIBRARY set the same callable runs on the native worker thread.
REENTRY : calling_provider() names the provider whose callable runs on this thread; Runtime uses it to refuse (RuntimeError) a
          call made from inside its own provider. invoke(reraise=True) is the reference path's: KeyboardInterrupt and
          SystemExit propagate there instead of becoming a provider failure (the native worker thread never sets it).
THREADING: on the native path the callable runs on the runtime's worker thread; ctypes takes the GIL for it. It must
          not call back into its own Runtime, and the Runtime must be closed before interpreter exit.
"""

from __future__ import annotations

import ctypes as ct
import logging
import threading
import weakref
from array import array
from collections.abc import Callable, Sequence
from functools import cache

from . import _abi
from ._model import _UINT32_MAX, ELEMENT_BYTES, BatchInfo, Shape, checked_integer
from .constants import (
    FORMAT_ROUTED_BF16, FORMAT_ROUTED_F16, FORMAT_ROUTED_F32, INVALID, MULTI_PROVIDER_VERSION, OK, PROVIDER,
    PROVIDER_ACCEPTS_BF16, PROVIDER_ACCEPTS_F16, PROVIDER_COMPACT_ROWS, PROVIDER_PARTIAL_ROUTES,
    PROVIDER_V3_VERSION,
)
from ._views import ProviderView
from .errors import SeabraidError

_LOG = logging.getLogger("seabraid")
_PYBUF_READ, _PYBUF_WRITE = 0x100, 0x200
_SHAPE_CACHE_MAX = 8
# The provider whose callable is running on this thread (None outside one); see calling_provider().
_RUNNING = threading.local()
_KEPT_MESSAGE = ("seabraid provider callable kept a reference to an input or output array past the call; do not store "
                 "or return the arrays it receives (copy them instead)")


@cache
def _from_memory():
    """PyMemoryView_FromMemory: a memoryview over runtime-owned memory, with no copy and no owner."""
    function = ct.pythonapi.PyMemoryView_FromMemory
    function.restype = ct.py_object
    function.argtypes = [ct.c_void_p, ct.c_ssize_t, ct.c_int]
    return function


def _numpy_or_none():
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _typed_view(pointer: object, count: int, code: str, writable: bool, size: int = 4) -> memoryview:
    address = ct.cast(pointer, ct.c_void_p).value
    if not count or not address:
        return memoryview(b"").cast("B").cast(code)
    raw = _from_memory()(address, count * size, _PYBUF_WRITE if writable else _PYBUF_READ)
    return raw.cast(code)


_ACCEPT_FLAGS = {FORMAT_ROUTED_F16: PROVIDER_ACCEPTS_F16, FORMAT_ROUTED_BF16: PROVIDER_ACCEPTS_BF16}


def provider_flags(partial_routes: bool, compact_rows: bool, accepts: Sequence[int] = ()) -> int:
    """The provider flag bits for a declaration, refusing a combination the engine would refuse at open.

    WHY here and before C: COMPACT_ROWS renumbers rows, which only means something when rows can be skipped, so a
    caller who forgot partial_routes gets the error at construction, not as an opaque INVALID from a later open.
    F32 in `accepts` is allowed and adds nothing (it is always accepted).
    """
    if compact_rows and not partial_routes:
        raise SeabraidError("seabraid_session_open", INVALID, "compact_rows requires partial_routes")
    flags = (PROVIDER_PARTIAL_ROUTES if partial_routes else 0) | (PROVIDER_COMPACT_ROWS if compact_rows else 0)
    for fmt in accepts:
        if fmt not in ELEMENT_BYTES:
            raise ValueError(f"unknown activation format {fmt!r}")
        flags |= _ACCEPT_FLAGS.get(fmt, 0)
    return flags


def calling_provider() -> PythonProvider | None:
    """The provider whose callable is running on this thread right now, or None.

    WHY: Runtime refuses a call made from inside its own provider. The callable runs while that Runtime's lock is held
    by a weave/collect (on the native path by another thread), so re-entering would wait out the caller's whole
    timeout and then act on a session mid-batch; refusing at once is the only answer that cannot hang.
    """
    return getattr(_RUNNING, "provider", None)


def _failure(error: BaseException, operation: int, reraise: bool) -> int:
    """Log a callable's exception once (type only) and return PROVIDER, or re-raise a non-Exception when `reraise`."""
    if reraise and not isinstance(error, Exception):
        raise error
    _LOG.error("seabraid provider callable raised %s in operation %d", type(error).__name__, operation)
    return PROVIDER


def _counts_consistent(cfg: object, data: object, output_count: int) -> bool:
    """True when every count the engine passed equals what the batch's own token count implies.

    WHY checked before any memory is mapped: the views are sized from these counts and the callable may write the
    output, so a count that disagrees with `tokens` (COMPACT_ROWS makes `tokens` smaller than the submitted batch,
    never the counts out of step with it) would map too much memory. Counts are elements, not bytes.
    """
    tokens = data.tokens
    picks = tokens * cfg.top_k
    return (tokens <= cfg.max_tokens and data.activation_count == tokens * cfg.hidden
            and data.expert_id_count == picks and data.weight_count == picks
            and output_count == tokens * cfg.hidden)


def _release(views: Sequence[memoryview]) -> bool:
    """Invalidate our views so a stashed memoryview fails loudly; False when the callable still holds an export."""
    clean = True
    for view in views:
        try:
            view.release()
        except BufferError:
            clean = False
    if not clean:
        # WHY an error and not a debug note: a kept numpy array would alias buffers the runtime reuses on the next submit.
        _LOG.error(_KEPT_MESSAGE)
    return clean


def _revoke(arguments: Sequence[object]) -> None:
    """Revoke every ProviderView among `arguments`. WHY a function: a loop variable here would outlive the loop in
    invoke() and hold the last numpy array, which the kept-reference check then reads as a leak."""
    for argument in arguments:
        if isinstance(argument, ProviderView):
            argument.revoke()


class PythonProvider:
    """A Python callable plus the C callback object that lets the native runtime call it."""

    def __init__(
        self, function: Callable, operations: int = 1, partial_routes: bool = False, compact_rows: bool = False,
        accepts: Sequence[int] = (),
    ) -> None:
        if not callable(function):
            raise TypeError("provider must be callable")
        self._function = function
        self.operations = checked_integer(operations, 1, _UINT32_MAX, "operations")
        self.partial_routes = bool(partial_routes)
        self.compact_rows = bool(compact_rows)
        self.flags = provider_flags(self.partial_routes, self.compact_rows, accepts)
        # WHY a flag and not "always pass info": a provider that declared nothing keeps the six-argument call.
        self.extended = bool(self.flags & (PROVIDER_ACCEPTS_F16 | PROVIDER_ACCEPTS_BF16))
        self.accepts = frozenset(fmt for fmt in accepts if fmt != FORMAT_ROUTED_F32)
        # Imported here, once, so the worker thread never pays for it and the package never requires it.
        self._numpy = _numpy_or_none()
        # WHY built once: `invoke` runs per MoE layer per token on the worker thread; the tuples never change.
        np = self._numpy
        self._dtypes = None if np is None else (np.float32, np.int32, np.float32, np.float32)
        self._raw_dtypes = None if np is None else (np.uint8, np.int32, np.float32, np.float32)
        # WHY a bounded cache: Shape is frozen and a session uses one geometry, so reuse avoids a dataclass build
        # (with validation) per call; the cap stops a pathological caller growing it without limit.
        self._shapes: dict[tuple, Shape] = {}
        # Both objects must live as long as the session: the engine keeps raw pointers to them.
        self._callback = _abi.ProviderExecute(self._execute)
        self._callback_v3 = _abi.ProviderExecuteV3(self._execute_v3)
        self._struct = _abi._ProviderV2(
            MULTI_PROVIDER_VERSION, self.operations, self.flags & PROVIDER_PARTIAL_ROUTES, None, self._callback,
        )

    def struct(self) -> _abi._ProviderV2:
        return self._struct

    def struct_v3(self, config: _abi._ConfigV1) -> _abi._ProviderV3:
        """The v3 descriptor; `config` is copied by the engine at open, so it only has to live until then."""
        return _abi._ProviderV3(
            PROVIDER_V3_VERSION, ct.sizeof(_abi._ProviderV3), self.operations, self.flags, ct.pointer(config), None,
            self._callback_v3,
        )

    def _wrap(self, views: Sequence[memoryview], raw: bool) -> list:
        if self._numpy is None:
            # WHY a wrapper and not the memoryview: view[0:4] and view.cast() are NEW memoryviews that survive the
            # parent's release(), so a stashed slice would read engine memory after the call (R7 MED 2).
            return [ProviderView(view) for view in views]
        return [self._numpy.frombuffer(v, dtype=d) for v, d in zip(views, self._raw_dtypes if raw else self._dtypes)]

    def _shape(self, cfg: object) -> Shape:
        key = (cfg.hidden, cfg.experts, cfg.max_tokens, cfg.top_k)
        shape = self._shapes.get(key)
        if shape is None:
            shape = Shape(*key)
            if len(self._shapes) < _SHAPE_CACHE_MAX:
                self._shapes[key] = shape
        return shape

    def _call(self, *arguments: object) -> None:
        outer = getattr(_RUNNING, "provider", None)
        _RUNNING.provider = self
        try:
            self._function(*arguments)
        finally:
            _RUNNING.provider = outer

    def invoke(
        self, operation: int, shape: Shape, views: Sequence[memoryview], info: BatchInfo | None = None,
        *, reraise: bool = False,
    ) -> int:
        """Call the function with (activations, expert_ids, weights, output) views; return a status, never raise.

        An extending provider also gets `info`; its activations view is raw bytes when the batch is F16/BF16.
        `reraise=True` (the reference path, where the callable runs on the caller's own thread) lets KeyboardInterrupt,
        SystemExit and the like propagate: nothing is unwinding into C there, and Ctrl-C must not turn into a
        provider failure. The native worker thread never sets it.
        """
        arguments = None
        watch: list[weakref.ref] = []
        try:
            raw = info is not None and info.activation_format != FORMAT_ROUTED_F32
            arguments = self._wrap(views, raw)
            if self._numpy is not None:
                # WHY weakrefs: numpy copies the pointer and does not pin the memoryview, so view.release() cannot tell
                # that a stashed array still aliases runtime-owned memory; a live array after the call can.
                watch = [weakref.ref(argument) for argument in arguments]
            self._call(operation, shape, *arguments, *((info,) if self.extended else ()))
        except BaseException as error:  # nothing may unwind into C; BaseException so SystemExit cannot either
            return _failure(error, operation, reraise)
        finally:
            _revoke(arguments or ())  # the call is over: every kept reference now raises instead of reading
            arguments = None  # drop our numpy references before checking for the callable's
        if any(ref() is not None for ref in watch):
            _LOG.error(_KEPT_MESSAGE)
            return PROVIDER
        return OK

    def _execute(self, _context: object, operation: int, config: object, batch: object,
                 output: object, count: int) -> int:
        """The v1/v2 C trampoline: build zero-copy views over the engine's buffers, call, release."""
        if operation >= self.operations or not config or not batch or not output:
            return INVALID
        data = batch.contents
        return self._serve(operation, config, output, count, data, FORMAT_ROUTED_F32, 0)

    def _execute_v3(self, _context: object, config: object, batch: object, output: object, count: int) -> int:
        """The v3 C trampoline: the operation, format and epoch travel in the batch."""
        if not config or not batch or not output:
            return INVALID
        data = batch.contents
        # WHY struct_size first: the header says fields past batch->struct_size do not exist, but this ctypes mirror
        # would read all of them; struct_size sits at offset 4, inside every version, so it is read before any other
        # field and a shorter struct is refused instead of read past its end.
        if data.struct_size < ct.sizeof(_abi._BatchV2):
            _LOG.error("seabraid provider got a batch of %d bytes, shorter than the %d this binding reads",
                       data.struct_size, ct.sizeof(_abi._BatchV2))
            return INVALID
        if data.operation_id >= self.operations:
            return INVALID
        return self._serve(data.operation_id, config, output, count, data, data.activation_format,
                           data.placement_epoch)

    def _serve(self, operation: int, config: object, output: object, count: int, data: object, fmt: int,
               epoch: int) -> int:
        views: list[memoryview] = []
        try:
            cfg = config.contents
            size = ELEMENT_BYTES.get(fmt)
            if size is None or not _counts_consistent(cfg, data, count):
                return INVALID
            views = [
                _typed_view(data.activations, data.activation_count, "f" if size == 4 else "B", False, size),
                _typed_view(data.expert_ids, data.expert_id_count, "i", False),
                _typed_view(data.weights, data.weight_count, "f", False),
                _typed_view(output, count, "f", True),
            ]
            status = self.invoke(operation, self._shape(cfg), views, BatchInfo(fmt, epoch))
        except BaseException as error:
            _LOG.error("seabraid provider trampoline failed with %s in operation %d", type(error).__name__, operation)
            _release(views)
            return PROVIDER
        # A failed release (stashed array) is a provider error even though the callable itself returned normally.
        return status if _release(views) else PROVIDER


def run_in_process(provider: PythonProvider, operation: int, shape: Shape, tokens: int, activations: Sequence[float] | bytes,
                   expert_ids: Sequence[int], weights: Sequence[float], info: BatchInfo | None = None,
                   ) -> list[float] | None:
    """Reference-path call: the same views over private copies; the output list, or None when the callable failed.

    `activations` is float32 values, or the raw bytes of an F16/BF16 batch (then `info` names the format).
    """
    raw = isinstance(activations, bytes)
    first = memoryview(activations).toreadonly() if raw else memoryview(array("f", activations)).toreadonly()
    output = array("f", bytes(4 * tokens * shape.hidden))
    views = [first, memoryview(array("i", expert_ids)).toreadonly(), memoryview(array("f", weights)).toreadonly(),
             memoryview(output)]
    try:
        status = provider.invoke(operation, shape, views, info, reraise=True)
    finally:
        released = _release(views)
    return list(output) if status == OK and released else None
