# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Mirror the native C ABI for ctypes and bind its exported functions.

OWNER   : the ctypes declaration of every exported seabraid_* function and struct (layouts follow include/seabraid/seabraid.h
          and src/layout/abi_layout.txt, which tests/test_layout.py checks field by field).
CONTRACT: load_library(path) -> CDLL with argtypes/restype set; a library whose seabraid_abi_epoch() differs from
          constants.ABI_EPOCH (or lacks it) is refused with OSError before any other symbol is bound;
          optional symbols stay absent on older libraries;
          last_error(library) -> str | None; active_library() -> CDLL | None. Imports: stdlib only.
SIZE    : under 300 lines (one ctypes declaration per exported function and struct of the header, so it grows with the ABI).
FAILURE : an unloadable library (missing, a directory, not a shared library, an invalid path) raises OSError without the
          path in the message (the OS reason is kept, the path text replaced); a missing REQUIRED symbol raises
          OSError; a failing seabraid_last_error is logged on logger "seabraid" and yields None.
CHECKER : tests/test_mock_runtime.py (binding and degrade paths), tests/test_native_runtime.py (layouts, linked).
FINISH  : both pass; the linked run also passes against the built native library.
"""

from __future__ import annotations

import ctypes as ct
import logging
import os
from pathlib import Path

from .constants import ABI_EPOCH

_LOG = logging.getLogger("seabraid")


class _CapabilitiesV1(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("struct_size", ct.c_uint32),
        ("operation", ct.c_uint32),
        ("input_format", ct.c_uint32),
        ("output_format", ct.c_uint32),
        ("source_modalities", ct.c_uint32),
    ]


class _SessionStatusV1(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("struct_size", ct.c_uint32),
        ("submitted", ct.c_uint64),
        ("collected", ct.c_uint64),
        ("provider_failures", ct.c_uint64),
        ("wait_timeouts", ct.c_uint64),
        ("last_source_modalities", ct.c_uint32),
        ("outstanding", ct.c_uint32),
        ("rows_submitted", ct.c_uint64),
        ("rows_shipped", ct.c_uint64),
    ]


class _ConfigV1(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("hidden", ct.c_uint32),
        ("experts", ct.c_uint32),
        ("max_tokens", ct.c_uint32),
        ("top_k", ct.c_uint32),
    ]


class _BatchV1(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("tokens", ct.c_uint32),
        ("activations", ct.POINTER(ct.c_float)),
        ("activation_count", ct.c_size_t),
        ("expert_ids", ct.POINTER(ct.c_int32)),
        ("expert_id_count", ct.c_size_t),
        ("weights", ct.POINTER(ct.c_float)),
        ("weight_count", ct.c_size_t),
    ]


class _PlacementFillV1(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("layers", ct.c_uint32),
        ("experts", ct.c_uint32),
        ("cards", ct.c_uint8),
        ("slots", ct.POINTER(ct.c_uint64)),
        ("slot_count", ct.c_size_t),
        ("heat", ct.POINTER(ct.c_double)),
        ("heat_count", ct.c_size_t),
    ]


class _PlacementPolicyV2(ct.Structure):
    """seabraid_placement_policy_v2 (seabraid_placement_open_v2): caller-filled and sized; set abi_version = 2 and
    struct_size = sizeof. Declared for layout parity; the binding does not open placements."""

    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("struct_size", ct.c_uint32),
        ("max_moves_per_round", ct.c_uint16),
        ("ladder_boundaries", ct.c_uint8),
        ("fallback_boundary", ct.c_uint8),
        ("max_hops_per_round", ct.c_uint8),
        ("reserved", ct.c_uint8 * 3),
        ("cooldown_rounds", ct.c_uint64),
        ("promote_margin", ct.c_double),
        ("max_helper_layer_share", ct.c_double),
    ]


# int32_t (*stage)(void *context, uint8_t rung, uint32_t layer, uint32_t expert, uint8_t *output, size_t output_bytes,
#                  size_t *bytes_written)
PlacementStage = ct.CFUNCTYPE(
    ct.c_int32, ct.c_void_p, ct.c_uint8, ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_uint8), ct.c_size_t,
    ct.POINTER(ct.c_size_t),
)
# int32_t (*replace)(void *context, uint8_t rung, uint32_t layer, uint32_t old_expert, uint32_t new_expert,
#                    const uint8_t *input, size_t input_bytes)
PlacementReplace = ct.CFUNCTYPE(
    ct.c_int32, ct.c_void_p, ct.c_uint8, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_uint8), ct.c_size_t,
)
# int32_t (*replace_from_fallback)(void *context, uint8_t rung, uint32_t layer, uint32_t old_expert,
#                                  uint32_t new_expert)
PlacementReplaceFromFallback = ct.CFUNCTYPE(
    ct.c_int32, ct.c_void_p, ct.c_uint8, ct.c_uint32, ct.c_uint32, ct.c_uint32,
)


class _PlacementProviderV2(ct.Structure):
    """seabraid_placement_provider_v2 (seabraid_placement_apply_v2): caller-filled and sized; set abi_version = 2 and
    struct_size = sizeof. Declared for layout parity; the binding does not apply placements."""

    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("struct_size", ct.c_uint32),
        ("context", ct.c_void_p),
        ("layer_slot_bytes", ct.POINTER(ct.c_size_t)),
        ("layer_count", ct.c_size_t),
        ("stage", PlacementStage),
        ("replace", PlacementReplace),
        ("from_fallback_rungs", ct.c_uint64),
        ("replace_from_fallback", PlacementReplaceFromFallback),
    ]


class _ProviderV2(ct.Structure):
    """seabraid_provider_v2; `execute` is a ProviderExecute (declared below the log callback type)."""


# int32_t (*execute)(void *context, uint32_t operation_id, const seabraid_config_v1 *, const seabraid_batch_v1 *,
#                    float *output, size_t output_count)
ProviderExecute = ct.CFUNCTYPE(
    ct.c_int32, ct.c_void_p, ct.c_uint32, ct.POINTER(_ConfigV1), ct.POINTER(_BatchV1),
    ct.POINTER(ct.c_float), ct.c_size_t,
)
_ProviderV2._fields_ = [
    ("abi_version", ct.c_uint32),
    ("operation_count", ct.c_uint32),
    ("flags", ct.c_uint32),
    ("context", ct.c_void_p),
    ("execute", ProviderExecute),
]

class _BatchV2(ct.Structure):
    """seabraid_batch_v2, submitted with seabraid_session_submit. `activations` is raw: its element type is
    `activation_format`."""

    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("struct_size", ct.c_uint32),
        ("operation_id", ct.c_uint32),
        ("source_modalities", ct.c_uint32),
        ("placement_epoch", ct.c_uint64),
        ("tokens", ct.c_uint32),
        ("activation_format", ct.c_uint32),
        ("activations", ct.c_void_p),
        ("activation_count", ct.c_size_t),
        ("expert_ids", ct.POINTER(ct.c_int32)),
        ("expert_id_count", ct.c_size_t),
        ("weights", ct.POINTER(ct.c_float)),
        ("weight_count", ct.c_size_t),
    ]


class _ProviderV3(ct.Structure):
    """seabraid_provider_v3, opened with seabraid_session_open; `execute` is a ProviderExecuteV3."""


# int32_t (*execute)(void *context, const seabraid_config_v1 *, const seabraid_batch_v2 *, float *output,
#                    size_t output_count)
ProviderExecuteV3 = ct.CFUNCTYPE(
    ct.c_int32, ct.c_void_p, ct.POINTER(_ConfigV1), ct.POINTER(_BatchV2),
    ct.POINTER(ct.c_float), ct.c_size_t,
)
_ProviderV3._fields_ = [
    ("abi_version", ct.c_uint32),
    ("struct_size", ct.c_uint32),
    ("operation_count", ct.c_uint32),
    ("flags", ct.c_uint32),
    ("config", ct.POINTER(_ConfigV1)),
    ("context", ct.c_void_p),
    ("execute", ProviderExecuteV3),
]

# void (*)(void *context, uint32_t level, int32_t status, const char *message)
LogCallback = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_uint32, ct.c_int32, ct.c_char_p)

# name -> (argtypes, restype). REQUIRED symbols are the non-deprecated calls every library has (collect and close);
# OPTIONAL ones are newer, or are the v1 entry points the header deprecates (removed at 1.0), and may be absent: a call
# that needs one says so (NativeSession._optional) instead of the whole library being refused at load.
_REQUIRED = {
    "seabraid_collect": (
        [ct.c_void_p, ct.c_uint64, ct.c_uint32, ct.POINTER(ct.c_float), ct.c_size_t], ct.c_int32),
    "seabraid_close": ([ct.c_void_p], ct.c_int32),
}
_OPTIONAL = {
    "seabraid_open": ([ct.c_void_p, ct.POINTER(ct.c_void_p)], ct.c_int32),
    "seabraid_open_reference": ([ct.POINTER(ct.c_void_p)], ct.c_int32),
    "seabraid_prepare": ([ct.c_void_p, ct.POINTER(_ConfigV1)], ct.c_int32),
    "seabraid_submit": ([ct.c_void_p, ct.POINTER(_BatchV1), ct.POINTER(ct.c_uint64)], ct.c_int32),
    "seabraid_close_timeout": ([ct.c_void_p, ct.c_uint32], ct.c_int32),
    "seabraid_query_capabilities": ([ct.POINTER(_CapabilitiesV1)], ct.c_int32),
    "seabraid_session_status": ([ct.c_void_p, ct.POINTER(_SessionStatusV1)], ct.c_int32),
    "seabraid_submit_routed": (
        [ct.c_void_p, ct.POINTER(_BatchV1), ct.c_uint32, ct.POINTER(ct.c_uint64)], ct.c_int32),
    "seabraid_open_multi": ([ct.POINTER(_ProviderV2), ct.POINTER(ct.c_void_p)], ct.c_int32),
    "seabraid_submit_operation": (
        [ct.c_void_p, ct.c_uint32, ct.POINTER(_BatchV1), ct.c_uint32, ct.POINTER(ct.c_uint64)], ct.c_int32),
    "seabraid_session_open": ([ct.POINTER(_ProviderV3), ct.POINTER(ct.c_void_p)], ct.c_int32),
    "seabraid_session_submit": (
        [ct.c_void_p, ct.POINTER(_BatchV2), ct.POINTER(ct.c_uint64)], ct.c_int32),
    "seabraid_set_log_sink": ([LogCallback, ct.c_void_p, ct.c_uint32], ct.c_int32),
    "seabraid_log": ([ct.c_uint32, ct.c_int32, ct.c_char_p], ct.c_int32),
    "seabraid_last_error": ([ct.c_char_p, ct.c_size_t, ct.POINTER(ct.c_size_t)], ct.c_int32),
    "seabraid_version": ([], ct.c_char_p),
    "seabraid_placement_fill": (
        [ct.POINTER(_PlacementFillV1), ct.POINTER(ct.c_uint8), ct.c_size_t], ct.c_int32),
}

# The most recently loaded library, so version() and enable_logging() can find it without a Runtime.
_active: ct.CDLL | None = None


def active_library() -> ct.CDLL | None:
    """Return the native library loaded most recently in this process, or None."""
    return _active


def _bind(library: ct.CDLL, table: dict, required: bool) -> None:
    for name, (argtypes, restype) in table.items():
        function = getattr(library, name, None)
        if function is None:
            if required:
                raise OSError(f"Seabraid native library lacks a required function ({name})")
            continue
        function.argtypes = argtypes
        function.restype = restype


def _check_epoch(library: ct.CDLL) -> None:
    """Refuse a library whose struct layouts differ from the ones this module declares.

    WHY first and fatal: every struct above is a hand-copied layout, so binding another epoch would misread
    memory silently instead of failing. A library without the symbol predates the epoch, hence is another layout.
    """
    function = getattr(library, "seabraid_abi_epoch", None)
    if function is None:
        raise OSError(
            f"Seabraid native library has no seabraid_abi_epoch(): it predates layout epoch {ABI_EPOCH}; rebuild it"
        )
    function.argtypes = []
    function.restype = ct.c_uint32
    found = function()
    if found != ABI_EPOCH:
        raise OSError(
            f"Seabraid native library has ABI layout epoch {found}, this binding needs {ABI_EPOCH}; "
            "install matching versions of the library and the package"
        )


def load_library(path: str | Path) -> ct.CDLL:
    """Resolve the caller's path to avoid loading a library from a search path."""
    global _active
    given, resolved = str(path), ""
    try:
        resolved = str(Path(path).resolve(strict=True))
        if os.path.isdir(resolved):
            # WHY here: dlopen on a directory says only "OSError" on Linux and "FileNotFoundError" on Windows.
            raise OSError("the path is a directory")
        library = ct.CDLL(resolved)
    except ValueError:  # an embedded NUL: Path raises ValueError on Linux where Windows raises OSError
        raise OSError("Seabraid native library could not be loaded: the path is not valid") from None
    except OSError as error:
        # The OS message may embed the path and messages here must not, so keep only the reason (path text removed).
        reason = error.strerror or str(error) or type(error).__name__
        for text in filter(None, (resolved, given)):
            reason = reason.replace(text, "<library>")
        raise OSError(f"Seabraid native library could not be loaded: {reason}") from None
    _check_epoch(library)
    _bind(library, _REQUIRED, True)
    _bind(library, _OPTIONAL, False)
    _active = library
    return library


def configured_path(library_path: str | Path | None) -> str | Path | None:
    """The one rule for which library path applies: the explicit argument, else env SEABRAID_LIBRARY, else None.

    WHY here: Runtime (which then loads it) and the module-level calls (placement, logging) must agree on it, so it
    is written once; an empty SEABRAID_LIBRARY counts as unset.
    """
    return library_path if library_path is not None else (os.environ.get("SEABRAID_LIBRARY") or None)


def resolve_library(library_path: str | Path | None) -> ct.CDLL | None:
    """Pick the library for a module-level call: explicit path, then env SEABRAID_LIBRARY, then the active one."""
    path = configured_path(library_path)
    return load_library(path) if path is not None else _active


def last_error(library: ct.CDLL) -> str | None:
    """Fetch this thread's last native failure text; None when the library cannot say."""
    function = getattr(library, "seabraid_last_error", None)
    if function is None:
        return None
    capacity, needed = 256, ct.c_size_t()
    for _ in range(2):  # the second pass only runs when the first buffer was too small
        buffer = ct.create_string_buffer(capacity)
        status = function(buffer, capacity, ct.byref(needed))
        if status != 0:
            _LOG.warning("seabraid_last_error failed (status %d); native failure text unavailable", status)
            return None
        if needed.value < capacity:
            break
        capacity = min(needed.value + 1, 1 << 16)
    return buffer.value.decode("utf-8", "replace") or None
