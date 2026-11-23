# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Seeded parity fuzz: the pure-Python reference session and the native library must agree on every outcome.

OWNER   : the executable statement "reference.py mirrors the runtime": per weave step, output bits, the `out` buffer's
          bits, what a hosted provider was shown, the error class and ABI status, and the session counters.
CONTRACT: run_case(case, build_runtime) -> list of events, one per construction/prepare/step/close; the native test
          runs every seed on both sessions and compares the lists. Seeds are stdlib `random` (see _parity_cases.py), so a
          failure prints `seed=N` and `python -m unittest ...test_parity_fuzz.NativeFuzz` reruns it with
          SEABRAID_PARITY_SEEDS=1 SEABRAID_PARITY_FIRST=N. Without SEABRAID_LIBRARY only the reference-side
          invariants run (determinism, compaction is invisible to a zero-writing provider, no unexpected exception).
SIZE    : under 260 lines.
FAILURE : a mismatch fails with the seed, the step, and both events.
CHECKER : this file is the checker; _parity_cases.py is its generator.
FINISH  : 0 mismatches over the default seed range on the Linux test machine (SEABRAID_LIBRARY set), in about ten seconds.
"""

from __future__ import annotations

import os
import struct
import unittest
from array import array

from seabraid import SeabraidError

from ._parity_cases import Case, build, build_out, make_case
from ._v2_support import LIBRARY, reference_runtime

SEEDS = int(os.environ.get("SEABRAID_PARITY_SEEDS", "2000"))
REFERENCE_SEEDS = min(SEEDS, 300)  # the reference-only checks run on every desktop, so they stay quick
FIRST = int(os.environ.get("SEABRAID_PARITY_FIRST", "0"))
FORMAT_F16 = 3


def _decode_half(raw: bytes, fmt: int) -> float:
    if fmt == FORMAT_F16:
        return struct.unpack("<e", raw)[0]
    return struct.unpack("<f", b"\0\0" + raw)[0]  # bf16 is the high half of an f32


def make_provider(log: list, state: dict):
    """A deterministic provider: out = sum over real picks of weight * activation * (expert + 1); records what it saw."""

    def serve(operation: int, config: object, acts: object, ids: object, weights: object, out: object,
              *info: object) -> None:
        batch = info[0] if info else None
        log.append((operation, (config.hidden, config.experts, config.max_tokens, config.top_k),
                    acts.tobytes(), ids.tolist(), weights.tobytes(), batch))
        if state["mode"] == "raise":
            raise ValueError("injected")
        raw = batch is not None and batch.activation_format != 1
        hidden, top_k, mode = config.hidden, config.top_k, state["mode"]
        data = acts.tobytes() if raw else acts  # a provider view is not a buffer: copy the raw bytes once
        for token in range(len(out) // hidden):
            for col in range(hidden):
                index = token * hidden + col
                value = _decode_half(bytes(data[2 * index:2 * index + 2]), batch.activation_format) if raw \
                    else float(data[index])
                total = 0.0
                for pick in range(token * top_k, (token + 1) * top_k):
                    if ids[pick] >= 0:
                        total += float(weights[pick]) * value * (int(ids[pick]) + 1)
                if mode == "half" and index % 2:
                    continue  # leaves the zero the runtime pre-filled
                out[index] = array("f", [float("nan") if mode == "nan" else total])[0]  # f32-rounded: no numpy cast warning

    return serve


def _error(error: BaseException) -> tuple:
    return ("err", type(error).__name__, getattr(error, "status", None))


def _out_bits(out: object) -> bytes | None:
    if out is None or isinstance(out, list):
        return None if out is None else repr(out).encode()
    return bytes(memoryview(out))


def _status(runtime: object) -> tuple:
    try:
        s = runtime.status()
    except Exception as error:
        return _error(error)
    return (s.submitted, s.collected, s.provider_failures, s.wait_timeouts, int(s.last_source_modalities),
            s.outstanding, s.rows_submitted, s.rows_shipped)


def _runtime_kwargs(case: Case, provider: object) -> dict:
    kwargs = {"partial_routes": case.partial, "compact_rows": case.compact}
    if case.builtin:
        return {**kwargs, "reference": True}
    return {**kwargs, "provider": provider, "operations": case.operations, "accepts": case.accepts}


def _weave(prepared: object, step: object, out: object) -> tuple:
    """One weave step -> ("ok", output bits) or the error class and status."""
    try:
        result = prepared.weave(
            build(step.activations), build(step.expert_ids, True), build(step.weights),
            placement_epoch=step.epoch, activation_format=step.activation_format, operation=step.operation,
            source_modalities=step.origins, out=out,
        )
    except Exception as error:  # every refusal is an outcome to compare, whatever its class
        return _error(error)
    return ("ok", None if out is not None else array("f", result).tobytes())


def run_case(case: Case, build_runtime: object) -> list:
    log: list = []
    state = {"mode": "ok"}
    events: list = []
    try:
        runtime = build_runtime(_runtime_kwargs(case, make_provider(log, state)))
    except Exception as error:
        return [("construct", _error(error))]
    prepared = None
    for shape in ([case.bad_prepare] if case.bad_prepare else []) + [case.shape]:
        try:
            prepared = runtime.prepare(**shape)
            events.append(("prepare", "ok", _status(runtime)))
        except Exception as error:
            events.append(("prepare", _error(error), _status(runtime)))
    for number, step in enumerate(case.steps):
        state["mode"] = step.provider_mode
        mark, out = len(log), build_out(step.out)
        events.append((number, _weave(prepared, step, out), _out_bits(out), tuple(log[mark:]), _status(runtime)))
    runtime.close()
    runtime.close()  # closing twice is a no-op
    events.append(("closed", _status(runtime)))
    return events


class ReferenceInvariants(unittest.TestCase):
    """The parts that need no native library."""

    def test_a_seed_is_reproducible(self) -> None:
        for seed in range(FIRST, FIRST + 40):
            # repr, not ==: a NaN in the data never equals itself.
            self.assertEqual(repr(make_case(seed)), repr(make_case(seed)), f"seed={seed}")

    def test_the_reference_side_never_raises_an_unexpected_exception(self) -> None:
        for seed in range(FIRST, FIRST + REFERENCE_SEEDS):
            events = run_case(make_case(seed), reference_runtime_for)
            self.assertTrue(events, f"seed={seed}")
            for event in events:
                flat = repr(event)
                self.assertNotIn("TypeError", flat, f"seed={seed}: {flat}")
                self.assertNotIn("OverflowError", flat, f"seed={seed}: {flat}")
                self.assertNotIn("Internal", flat, f"seed={seed}: {flat}")

    def test_the_generator_reaches_every_outcome_class(self) -> None:
        seen: set = set()
        for seed in range(FIRST, FIRST + REFERENCE_SEEDS):
            for event in run_case(make_case(seed), reference_runtime_for):
                if isinstance(event[0], int):
                    seen.add(event[1][0] if event[1][0] == "ok" else event[1][1:])
        for expected in ("ok", ("SeabraidError", 1), ("SeabraidError", 2), ("SeabraidError", 4), ("ValueError", None)):
            self.assertIn(expected, seen)


def reference_runtime_for(kwargs: dict) -> object:
    return reference_runtime(**kwargs)


@unittest.skipUnless(LIBRARY, "set SEABRAID_LIBRARY for native parity")
class NativeFuzz(unittest.TestCase):
    def test_reference_and_native_agree_on_every_seed(self) -> None:
        from seabraid import Runtime

        def native(kwargs: dict) -> object:
            return Runtime(LIBRARY, **kwargs)

        for seed in range(FIRST, FIRST + SEEDS):
            case = make_case(seed)
            expected, actual = run_case(case, reference_runtime_for), run_case(case, native)
            self.assertEqual(len(expected), len(actual), f"seed={seed}")
            for left, right in zip(expected, actual):
                self.assertEqual(left, right, f"seed={seed} (rerun: SEABRAID_PARITY_FIRST={seed} "
                                              f"SEABRAID_PARITY_SEEDS=1)\nreference={left}\nnative   ={right}")


if __name__ == "__main__":
    unittest.main()
