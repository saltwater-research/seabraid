# Python binding

Package `seabraid` (sources in `bindings/python/seabraid/`, built from the repository-root `pyproject.toml`).

| File | Owns | Excludes |
|---|---|---|
| `seabraid/__init__.py` | Public names and `__version__` | Behavior |
| `seabraid/runtime.py` | `Runtime`/`PreparedMoE`, provider choice, input validation, the one pending batch | Routing and expert math |
| `seabraid/_buffers.py` | Buffer/sequence inputs to ctypes arrays, `out=` targets, buffer validation | Shape rules, the ABI |
| `seabraid/_provider.py` | `PythonProvider`: a Python callable behind `seabraid_provider_v2.execute` and `seabraid_provider_v3.execute`; `provider_flags` (the flag-combination rules, checked before any C call); the guarded call shared by both sessions | Provider choice, expert math |
| `seabraid/_native.py` | One native session: ctypes calls, statuses to `SeabraidError`; the one choice between `seabraid_session_open` (v3, opens at `prepare`) and the v1 open calls | Provider choice |
| `seabraid/reference.py` | Pure-Python reference provider mirroring `src/batch.rs` and the v2 session rules (placement epoch, format refusal, COMPACT_ROWS, row counters); an API demonstration, not model math | Native ABI, real expert math |
| `seabraid/_rows.py` | Python mirror of `src/rows.rs`: which rows COMPACT_ROWS keeps, gather, scatter (reference session only) | The row rule itself (Rust owns it) |
| `seabraid/_bundled.py` | `bundled_library()`: path of the native library a platform wheel ships in `seabraid/_lib/`, else None | Choosing native mode (runtime.py stays opt-in) |
| `seabraid/_abi.py` | ctypes struct layouts, symbol binding, native last-error text, the active library | Session behavior |
| `seabraid/constants.py` | ABI numbers (status, modality, log level, ABI version) | Anything not in `seabraid.h` |
| `seabraid/errors.py` | `SeabraidError` and the status-to-words table | Retry policy |
| `seabraid/_model.py` | `Capabilities`, `SessionStatus` (with `rows_submitted` / `rows_shipped`), `BatchInfo`, `Shape`, activation element sizes, the integer-bounds guard | I/O |
| `seabraid/diagnostics.py` | `version()`, `enable_logging()`, `disable_logging()` | Native log policy |
| `seabraid/placement.py` | `placement.fill`, the binding for `seabraid_placement_fill` (native only) | The fill algorithm |
| `seabraid/ranking.py` | Card-ordering rules R1-R3 (rule text in its docstring, checked by `tests/test_ranking.py`) and the per-card memory-bandwidth table; the compat add-on installer imports it | Hardware detection |
| `tests/` | Mock, reference, batch v2 / provider v3 (`test_abi_v2.py`, `test_compaction.py`), robustness (pickle, trampoline guards, lock, fork), quickstart, header-parity, the pick-rule table (`test_pick_golden.py`, reads `src/picks/golden.txt`), diagnostics, placement, API and linked native tests (`test_native_edges.py`: placement and logging) | Compiler setup |

Python calls the C ABI through the standard library's `ctypes`: the v2 entry points (`seabraid_session_open`,
`seabraid_session_submit`) when the library exports them, else the v1 calls. Imports never
load a native library. `Runtime(library_path=None)` uses the native library when a
path is passed or `SEABRAID_LIBRARY` is set, and a native library then REQUIRES
`provider=fn` or `reference=True` (a bare `Runtime()` raises `ValueError`: the engine
never picks a transform for you). Otherwise it runs the pure-Python reference
provider and says so with one `RuntimeWarning` and one `logging` warning on logger
`seabraid`, unless `reference=True` asked for it (then it stays silent); it never
falls back silently from an explicit path.
Inputs are flat. A C-contiguous float32/int32 buffer (`array.array`, `memoryview`,
numpy, bytes-like) is used without per-element Python work (aliased when writable and
4-byte aligned, copied once when read-only or misaligned; a misaligned `out` is a `ValueError`); a plain sequence still works but converts element by
element, so use buffers for real batches. numpy is never imported by the package
itself. `weave(..., out=buffer)` fills a writable float32 buffer of exactly
`tokens * hidden` elements and returns it; without `out` a list is returned.
Python checks counts against the prepared shape before packing. The native runtime verifies
routing and buffer counts and owns expert computation.
One lock per `Runtime` serialises its calls, so `close()` from another thread waits for a
`weave`/`collect` in flight instead of freeing the session under it. After `os.fork()` the
child disowns every inherited native handle (it never closes it; using it raises
`RuntimeError`): open a new `Runtime` in the child.

Hosting a provider: `Runtime(provider=fn, operations=N, partial_routes=False)` serves
experts from a Python callable `fn(operation_id, config, activations, expert_ids,
weights, output)`. `config` is a `Shape`; the four buffers are zero-copy views valid
only during the call and `output` is writable and must be filled. Without numpy they are
`ProviderView` objects: index them, iterate them, `.tolist()` / `.tobytes()` for a copy,
assign `output[a:b] = values`; a slice READ (`view[a:b]`) raises `TypeError`, there is no
`.cast()` and no buffer protocol (no `memoryview(view)`), so nothing aliasing engine memory
can be kept, and every kept view raises `ValueError` after the call. With numpy importable
they are numpy arrays; numpy memory cannot be revoked, so keeping an array (or a slice of
one) past the call is detected and fails the call with status 4 and one ERROR record. `weave(..., operation=i)`
selects a prepared operation. On the native path `fn` runs on the runtime's worker
thread (ctypes takes the GIL) and must not call back into its Runtime; close the
Runtime before the interpreter exits. An exception in `fn` is caught, logged once on
logger `seabraid` (type only, no data) and reported as `SeabraidError` status 4. On
the reference path the same callable runs on the caller's thread at `weave`, and
failure surfaces at collect with the same status.

Batch v2 / provider v3 options, all off by default (existing callers see no change). They need a library that exports
`seabraid_session_open` / `seabraid_session_submit`; an older one raises `RuntimeError` rather than dropping them.

- `weave(..., placement_epoch=N)`: tags the batch. After the first nonzero epoch a lower one is refused
  (`SeabraidError` status 2) and `0` is refused (status 1); a refused batch never moves the high-water mark.
- `Runtime(provider=fn, accepts=(FORMAT_ROUTED_F16, FORMAT_ROUTED_BF16))` declares the 16-bit formats `fn` takes
  (ids in `seabraid.constants`; F32 is always accepted). Then `weave(raw, ..., activation_format=FORMAT_ROUTED_F16)`
  takes any C-contiguous buffer holding the raw 16-bit elements (`bytes`, `array('H')`, a numpy float16 array) and
  passes the bytes through; nothing is converted in Python and numpy is not required. `fn` gets one extra trailing
  argument, `BatchInfo(activation_format, placement_epoch)`, and `activations` is a read-only view of raw bytes
  (`ProviderView` format `"B"`, or numpy `uint8`) for F16/BF16 batches. A format the provider did not declare is
  refused (status 1), on the reference path as natively. Output stays float32.
- `Runtime(provider=fn, partial_routes=True, compact_rows=True)`: `fn` is called only with the rows that have a real
  (non-skip) pick, and the result is zero-filled in the others, bit-identically to an uncompacted run for a provider
  that writes zeros for skipped rows. `compact_rows` without `partial_routes` raises `SeabraidError` (status 1) at
  construction, before any C call. `Runtime(reference=True, partial_routes=True, compact_rows=True)` does the same
  with the built-in transform; `reference=True` takes no formats.
- `runtime.status()` also reports `rows_submitted` and `rows_shipped`, so the share of rows compaction saved is
  `1 - rows_shipped / rows_submitted`.

With a v3 library the session opens at `prepare()` (the config travels in the open call), so `status()` before
`prepare()` reports zeros.

```python
import seabraid

with seabraid.Runtime(reference=True) as runtime:   # pure Python, or the engine's reference transform if SEABRAID_LIBRARY is set
    prepared = runtime.prepare(hidden=2, experts=3, max_tokens=2, top_k=2)
    result = prepared.weave([1.0, 2.0], [0, 2], [0.25, 0.75])   # [2.5, 5.0]
```

Each activation row has `hidden` values; its expert ID and weight rows have
`top_k` values each. A `SeabraidError` exposes `operation`, `status` and, when the
native library supplies it, `detail` (its last-error text). A provider callable that calls back into its own Runtime gets `RuntimeError` at once (it
runs while that Runtime's call holds the lock). Ctrl-C inside a reference-path provider propagates and leaves the
session unchanged; Ctrl-C during a native `collect` loses that batch's result and the next `weave` works. A timeout (status `3`)
leaves the current batch pending; call `prepared.collect()` to wait again, or
close the runtime to release the session. `examples/quickstart.py` runs end to end, with
and without `SEABRAID_LIBRARY` (`tests/test_quickstart.py` runs both).

Diagnostics: `seabraid.version()`, `seabraid.enable_logging(min_level="WARNING")` (native
records become `logging` records on logger `seabraid`, status in `record.seabraid_status`)
and `seabraid.disable_logging()`. `seabraid.placement.fill(layers, experts, slots, heat)`
returns one card rung per `(layer, expert)`, layer-major; `slots` is one capacity per
card. Both need a native library that exports the functions and say so clearly when
it does not. Placement is bound only as `seabraid_placement_fill`; the placement controller
(`seabraid_placement_open`, `_plan`, `_apply`, `_snapshot`, `_update`, `_quarantine`, `_close`) has no Python
binding, so use the C ABI for it.

`tests/test_parity_fuzz.py` (seeded, stdlib `random`) runs thousands of small valid and invalid batches on the reference and
native sessions and compares output bits, `out` bits, what a provider saw, error class and status, and the counters; a failure
prints its seed (`SEABRAID_PARITY_FIRST=N SEABRAID_PARITY_SEEDS=1` reruns it). `tests/test_binding_edges.py` pins bad
library paths, close/use-after-close, re-entry, Ctrl-C and `out` aliasing.

From the repository root, run `python -B -m unittest discover -s bindings/python -v`.
Set `SEABRAID_LIBRARY` to the built shared library path to also run the native
integration and reference-parity tests (epochs, formats, compaction, counters). `tests/test_header_parity.py` parses
`include/seabraid/seabraid.h` and fails when a Python constant drifts from it.
The Python complexity gate checks the binding and fixtures for branch depth.
