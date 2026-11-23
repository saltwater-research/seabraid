# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""One error type for every native or reference failure, carrying the ABI status.

OWNER   : the mapping from ABI status to words, and the names of the ABI operations used in messages.
CONTRACT: SeabraidError(operation, status, detail=None) exposes .operation, .status, .detail; submit_operation(origins).
SIZE    : under 60 lines.
FAILURE : a status outside the table still raises, labelled "unknown status"; nothing is ever swallowed.
CHECKER : tests/test_header_parity.py (status table equals the header) and tests/test_mock_runtime.py (messages).
FINISH  : both pass.
Messages hold only the operation, the status words and native text; never paths, hostnames, env values or tensors.
"""

from __future__ import annotations

from . import constants as c

_STATUS_NAMES = {
    c.INVALID: "invalid argument",
    c.BUSY_OR_STATE: "busy or invalid state",
    c.TIMEOUT: "timeout",
    c.PROVIDER: "provider failure",
    c.INTERNAL: "internal failure",
}


class SeabraidError(RuntimeError):
    """Expose the native status so callers can distinguish timeout from failure."""

    def __init__(self, operation: str, status: int, detail: str | None = None) -> None:
        self.operation = operation
        self.status = status
        self.detail = detail
        text = f"{operation}: {_STATUS_NAMES.get(status, 'unknown status')} (status {status})"
        # The native text explains WHY; the status alone only says which class of failure.
        super().__init__(f"{text}: {detail}" if detail else text)

    def __reduce__(self) -> tuple:
        # WHY: Exception.args holds only the message, so default unpickling would call __init__ with one argument;
        # multiprocessing and ProcessPoolExecutor pickle every exception they carry back to the parent.
        return (type(self), (self.operation, self.status, self.detail))


def submit_operation(origins: int, operation_id: int = 0) -> str:
    """Name the ABI call used for a submission; operation IDs above zero and tagged work need their own entry points."""
    if operation_id:
        return "seabraid_submit_operation"
    return "seabraid_submit_routed" if origins else "seabraid_submit"
