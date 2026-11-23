# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Native provider: one session on the compiled Seabraid library, called through ctypes.

OWNER   : the translation of one session's calls into seabraid_* ABI calls and statuses into SeabraidError.
CONTRACT: NativeSession(path, provider=None, *, reference=False, partial_routes=False, compact_rows=False) with the
          session interface shared with reference.ReferenceSession: provider, capabilities(), status(), prepare(shape),
          submit(tokens, activations, expert_ids, weights, origins, operation_id=0, *, placement_epoch=0,
          activation_format=F32) -> ticket (arrays are ctypes arrays), collect(ticket, timeout_ms, count, out=None),
          close(timeout_ms=None).
          OPEN: a library that exports seabraid_session_open opens at prepare(shape) with a seabraid_provider_v3
          (the config rides in the descriptor; a PythonProvider is bridged through ProviderExecuteV3, reference=True
          sets the REFERENCE flag); an older library opens at construction through seabraid_open_multi (a
          PythonProvider) or seabraid_open_reference and prepares with seabraid_prepare. reference=True is REQUIRED
          without a provider (the engine refuses a null provider). partial_routes/compact_rows only configure the
          reference transform; a provider carries its own flags. v2-only features on an older library raise
          RuntimeError instead of being dropped. Imports: stdlib + siblings.
          A session dropped without close() is closed by a weakref finalizer BEFORE its provider thunk is released.
          After os.fork() the child DISOWNS every inherited handle (os.register_at_fork): it never calls seabraid_close
          on it (the worker thread exists only in the parent, so a join would hang), and a later call on that session
          raises RuntimeError; the provider thunk stays referenced so nothing is freed under the parent's copy.
SIZE    : under 320 lines.
FAILURE : every nonzero status raises SeabraidError carrying the native last-error text when the library has it;
          absent optional functions raise RuntimeError (older library) instead of failing obscurely.
CHECKER : tests/test_mock_runtime.py (injected statuses), tests/test_native_runtime.py (linked library).
FINISH  : both pass.
"""

from __future__ import annotations

import ctypes as ct
import logging
import os
import weakref
from pathlib import Path

from . import _abi
from .constants import (
    ABI_VERSION, BATCH_V2_VERSION, BUSY_OR_STATE, FORMAT_ROUTED_F32, PROVIDER_PARTIAL_ROUTES,
    PROVIDER_REFERENCE, PROVIDER_V3_VERSION, TIMEOUT, SourceModality,
)
from .errors import SeabraidError, submit_operation
from ._model import Capabilities, SessionStatus, Shape
from ._provider import PythonProvider, provider_flags

_LOG = logging.getLogger("seabraid")
# Providers whose worker thread may still be running after a timed-out close. WHY forever: the worker can call
# the ctypes thunk at any moment, and freeing it then would jump into freed memory.
_LEAKED: list[object] = []
# Every _Owned that may hold a live handle, so a forked child can find and disown them. Weak: a closed session drops out.
_LIVE: "weakref.WeakSet[_Owned]" = weakref.WeakSet()


class _Owned:
    """What must outlive the engine session: the library, the handle and the provider (its callback thunk)."""

    __slots__ = ("library", "session", "provider", "disowned", "__weakref__")

    def __init__(self, library: ct.CDLL, provider: PythonProvider | None) -> None:
        self.library = library
        self.session = ct.c_void_p()
        self.provider = provider
        self.disowned = False  # True in a forked child: the handle belongs to the parent process
        _LIVE.add(self)


def _disown_after_fork() -> None:
    """Child side of os.fork(): forget every inherited native handle without closing it.

    WHY: seabraid_close joins the engine's worker thread, which does not exist in the child (fork copies only the
    calling thread), so closing there can hang or free state the parent still uses. The provider stays referenced,
    never freed, because the parent's copy of the thunk memory is not ours to release either.
    """
    for owned in list(_LIVE):
        owned.session = ct.c_void_p()
        owned.disowned = True  # even before prepare(): a v3 child must not open a session inside the parent's record
        if owned.provider is not None:
            _LEAKED.append(owned.provider)
            owned.provider = None


if hasattr(os, "register_at_fork"):  # absent on Windows, where there is no fork
    os.register_at_fork(after_in_child=_disown_after_fork)


def _close_owned(owned: _Owned, timeout_ms: int | None) -> int | None:
    """Close the handle (None when already closed), then and only then release the provider reference.

    BUSY_OR_STATE means the engine REFUSED and kept the handle (busy, or called from its own worker thread): the
    handle and the provider are put back so nothing leaks and a later close can retry. Every other status, INVALID
    ("not live") included, means the handle is gone, so it is never closed again.
    The handle is out of `owned` during the engine call: only the Runtime lock keeps other threads out of that window.
    """
    session, owned.session = owned.session, ct.c_void_p()
    if not session:
        return None
    if timeout_ms is None:
        status = owned.library.seabraid_close(session)
    else:
        status = owned.library.seabraid_close_timeout(session, timeout_ms)
    if status == BUSY_OR_STATE:
        owned.session = session
        return status
    if status == TIMEOUT:
        _LEAKED.append(owned.provider)  # the stuck worker may still call it
    owned.provider = None
    return status


def _finalize(owned: _Owned) -> None:
    """Runs when a NativeSession is collected (or at exit) without close(): close before the thunk can be freed."""
    status = _close_owned(owned, None)
    if status == BUSY_OR_STATE:
        # Nobody will retry a collected session, and its worker may still call the thunk: keep everything alive.
        _LEAKED.append(owned)
    if status:
        _LOG.error("seabraid session closed by the garbage collector reported status %d", status)


class NativeSession:
    """Own one native session handle; the Runtime serialises calls and guards use-after-close."""

    provider = "native"

    def __init__(
        self, library_path: str | Path, provider: PythonProvider | None = None, *, reference: bool = False,
        partial_routes: bool = False, compact_rows: bool = False,
    ) -> None:
        self._library = _abi.load_library(library_path)
        # Checked after the load so an unloadable library is still reported as such, never as a usage error.
        if provider is None and not reference:
            raise ValueError("a native session needs a provider, or reference=True for the built-in reference transform")
        if provider is not None and reference:
            raise ValueError("provider and reference are mutually exclusive")
        self._flags = provider.flags if provider is not None else provider_flags(partial_routes, compact_rows)
        # The engine keeps raw pointers into the provider, so the _Owned record keeps it alive until the close.
        self._owned = _Owned(self._library, provider)
        # The finalizer holds _Owned (not self), so it can close the session after self is gone.
        self._finalizer = weakref.finalize(self, _finalize, self._owned)
        try:
            self._open_early(provider)
        except BaseException:
            self._finalizer.detach()
            raise

    def _has_v3(self) -> bool:
        """True when the library opens sessions with seabraid_session_open (provider v3).

        WHY this is the one open/submit-family choice: v3 open carries the config and every v2-only feature
        (formats, compaction, reference flags), so a library that has it is driven only through it, and one that
        lacks it keeps the exact v1 calls it always had.
        """
        return getattr(self._library, "seabraid_session_open", None) is not None

    def _open_early(self, provider: PythonProvider | None) -> None:
        """Open now on a v1 library; on a v3 library opening waits for prepare(shape), which supplies the config."""
        if self._has_v3():
            return
        # open_multi's descriptor carries PARTIAL_ROUTES itself; every later flag needs the v3 open.
        if self._flags & ~PROVIDER_PARTIAL_ROUTES or (provider is None and self._flags):
            raise RuntimeError("This Seabraid library does not expose seabraid_session_open (needed for compact_rows, "
                               "activation formats and a reference session with partial_routes)")
        out = ct.byref(self._owned.session)
        if provider is None:
            open_reference = self._optional("seabraid_open_reference", "seabraid_open_reference (deprecated v1 open)")
            self._check("seabraid_open_reference", open_reference(out))
            return
        open_multi = self._optional("seabraid_open_multi", "multi-operation providers")
        self._check("seabraid_open_multi", open_multi(ct.byref(provider.struct()), out))

    @property
    def _session(self) -> ct.c_void_p:
        if self._owned.disowned:
            raise RuntimeError("this Seabraid session belongs to the parent process (the process forked); "
                               "open a new Runtime in the child")
        return self._owned.session

    @property
    def _provider(self) -> PythonProvider | None:
        return self._owned.provider

    @property
    def library(self) -> ct.CDLL:
        return self._library

    def _check(self, operation: str, status: int) -> None:
        if status == 0:
            return
        # A timeout is an expected, retryable outcome with no failure text to fetch.
        detail = None if status == TIMEOUT else _abi.last_error(self._library)
        raise SeabraidError(operation, status, detail)

    def _optional(self, name: str, what: str):
        function = getattr(self._library, name, None)
        if function is None:
            raise RuntimeError(f"This Seabraid library does not expose {what}")
        return function

    def capabilities(self) -> Capabilities:
        query = self._optional("seabraid_query_capabilities", "capability discovery")
        info = _abi._CapabilitiesV1(ABI_VERSION, ct.sizeof(_abi._CapabilitiesV1))
        self._check("seabraid_query_capabilities", query(ct.byref(info)))
        return Capabilities(
            info.abi_version, info.operation, info.input_format, info.output_format,
            SourceModality(info.source_modalities),
        )

    def status(self) -> SessionStatus:
        if not self._session:  # a v3 session does not exist until prepare(): nothing has been submitted
            return SessionStatus(0, 0, 0, 0, SourceModality(0), False)
        snapshot = self._optional("seabraid_session_status", "session status")
        info = _abi._SessionStatusV1(ABI_VERSION, ct.sizeof(_abi._SessionStatusV1))
        self._check("seabraid_session_status", snapshot(self._session, ct.byref(info)))
        return SessionStatus(
            info.submitted, info.collected, info.provider_failures, info.wait_timeouts,
            SourceModality(info.last_source_modalities), bool(info.outstanding),
            info.rows_submitted, info.rows_shipped,
        )

    def prepare(self, shape: Shape) -> None:
        """Prepare once: seabraid_prepare on a v1 session, or the v3 open (which carries the config) otherwise."""
        config = _abi._ConfigV1(
            ABI_VERSION, shape.hidden, shape.experts, shape.max_tokens, shape.top_k,
        )
        if not self._has_v3():
            prepare = self._optional("seabraid_prepare", "seabraid_prepare (deprecated v1 prepare)")
            self._check("seabraid_prepare", prepare(self._session, ct.byref(config)))
            return
        if self._session:
            raise SeabraidError("seabraid_session_open", BUSY_OR_STATE, "session is already prepared")
        provider = self._provider
        if provider is not None:
            descriptor = provider.struct_v3(config)
        else:  # REFERENCE: execute and context stay NULL
            descriptor = _abi._ProviderV3(
                PROVIDER_V3_VERSION, ct.sizeof(_abi._ProviderV3), 1, self._flags | PROVIDER_REFERENCE, ct.pointer(config),
                None, _abi.ProviderExecuteV3(),
            )
        out = ct.byref(self._owned.session)
        self._check("seabraid_session_open", self._library.seabraid_session_open(ct.byref(descriptor), out))

    def submit_name(self, origins: int, operation_id: int = 0) -> str:
        """The ABI call submit() makes, so a Python refusal names the call the engine would have refused.

        WHY here: the library decides (seabraid_session_submit when it has it, else the per-shape submit calls);
        Runtime asks instead of guessing, so the name cannot drift from what submit() does.
        """
        if getattr(self._library, "seabraid_session_submit", None) is not None:
            return "seabraid_session_submit"
        return submit_operation(origins, operation_id)

    def submit(
        self, tokens: int, activations: ct.Array, expert_ids: ct.Array, weights: ct.Array, origins: int,
        operation_id: int = 0, *, placement_epoch: int = 0, activation_format: int = FORMAT_ROUTED_F32,
        ticket_slot: ct.c_uint64 | None = None,
    ) -> int:
        """Submit packed ctypes arrays (see _buffers); the engine copies them before this returns.

        `ticket_slot`, when given, is the c_uint64 the engine writes the ticket into (it is returned as well), so a
        caller can recover the ticket even if an interrupt hits before it receives the return value.

        `activations` is a float array for F32 or a uint16 array of raw elements for F16/BF16. Uses
        seabraid_session_submit (one call for every operation and origin) when the library has it, else the
        older per-shape submit calls, which carry neither an epoch nor a non-F32 format.
        """
        batch = None
        try:
            ticket = ticket_slot if ticket_slot is not None else ct.c_uint64()
            session_submit = getattr(self._library, "seabraid_session_submit", None)
            if session_submit is not None:
                batch = _abi._BatchV2(
                    BATCH_V2_VERSION, ct.sizeof(_abi._BatchV2), operation_id, origins, placement_epoch, tokens,
                    activation_format,
                    # WHY addressof and not ct.cast: cast stores the array inside its own _objects dict, a reference
                    # cycle that keeps the caller's buffer export-locked until the cyclic GC runs (R7 LOW 10). The
                    # array outlives this call and the engine copies the bytes before it returns.
                    ct.addressof(activations), len(activations), expert_ids, len(expert_ids), weights,
                    len(weights),
                )
                self._check("seabraid_session_submit", session_submit(self._session, ct.byref(batch), ct.byref(ticket)))
                return ticket.value
            if placement_epoch or activation_format != FORMAT_ROUTED_F32:
                raise RuntimeError("This Seabraid library does not expose seabraid_session_submit (needed for "
                                   "placement_epoch and activation formats)")
            operation = self.submit_name(origins, operation_id)
            submit = self._optional(operation, f"{operation} (deprecated v1 submit)")
            batch = _abi._BatchV1(
                ABI_VERSION, tokens, activations, len(activations), expert_ids, len(expert_ids), weights, len(weights),
            )
            if operation_id:
                status = submit(self._session, operation_id, ct.byref(batch), origins, ct.byref(ticket))
            elif origins:
                status = submit(self._session, ct.byref(batch), origins, ct.byref(ticket))
            else:
                status = submit(self._session, ct.byref(batch), ct.byref(ticket))
            self._check(operation, status)
            return ticket.value
        finally:
            # WHY: a refusal's traceback keeps this frame, and the packed arrays hold exports on the caller's buffers.
            activations = expert_ids = weights = batch = None

    def collect(self, ticket: int, timeout_ms: int, count: int, out: ct.Array | None = None) -> list[float] | None:
        """Wait for the ticket; fill `out` in place when given (returning None), else return a new list."""
        output = (ct.c_float * count)() if out is None else out
        status = self._library.seabraid_collect(self._session, ticket, timeout_ms, output, count)
        self._check("seabraid_collect", status)
        return list(output) if out is None else None

    def close(self, timeout_ms: int | None = None) -> None:
        """Close the engine session; with timeout_ms, give up on a stuck provider (TIMEOUT, worker leaked loudly)."""
        if timeout_ms is not None:
            self._optional("seabraid_close_timeout", "a bounded close")
        # Native close consumes a valid handle even when joining the worker reports failure, and the
        # finalizer must not close it a second time.
        self._finalizer.detach()
        status = _close_owned(self._owned, timeout_ms)
        if status == BUSY_OR_STATE:  # refused, handle kept: the finalizer is the backstop again until a retry works
            self._finalizer = weakref.finalize(self, _finalize, self._owned)
        if status is not None:
            self._check("seabraid_close" if timeout_ms is None else "seabraid_close_timeout", status)
