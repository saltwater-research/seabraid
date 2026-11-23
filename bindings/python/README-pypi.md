# Seabraid

An engine-neutral runtime that lets GPUs from different vendors serve one mixture-of-experts layer together. This
package is the Python binding.

**Linux x86-64 is the supported release platform.** The native wheel targets glibc 2.34 or newer.
Windows users can try the experimental core runtime from source. A native Windows DLL and end-to-end
inference still need verification. The ABI can change before 1.0.

## Two ways to run

- **Pure Python (reference provider).** Requires Python 3.11 or newer and runs on the CPU. It is an API demonstration and
  test transform, not model math, and it says so with a warning unless you ask for it explicitly.
- **Native.** The Linux x86-64 wheel ships the native library; `seabraid.bundled_library()` returns its path (or `None` on a
  pure install). Pass that path as `library_path`, or set the `SEABRAID_LIBRARY` environment variable. A native library
  never picks a transform for you: pass `reference=True` for the engine's reference transform, or host your own with
  `provider=fn`. The Strata installer is part of the full Seabraid source release.

## Example

```python
import seabraid

with seabraid.Runtime(reference=True) as runtime:
    prepared = runtime.prepare(hidden=2, experts=3, max_tokens=2, top_k=2)
    # One token: activations [1, 2], routed to experts 0 and 2 with weights 0.25 and 0.75.
    print(prepared.weave([1.0, 2.0], [0, 2], [0.25, 0.75]))   # [2.5, 5.0]
```

The same code runs natively when `SEABRAID_LIBRARY` points at the built library.

## Good to know

- Inputs are flat, row-major float32/int32 buffers (`array.array`, `memoryview`, numpy, bytes-like) used without
  per-element Python work; plain lists work but are slow.
- `Runtime` serialises its calls with one lock, so closing it from another thread waits for a call in flight.
- Placement: only `seabraid.placement.fill` (native `seabraid_placement_fill`) is bound. The placement controller
  (`seabraid_placement_open`, `_plan`, `_apply`, ...) is C ABI only.
- After `os.fork()` the child cannot use an inherited native session; open a new `Runtime` there.
- Errors are `seabraid.SeabraidError` with `.operation`, `.status` and `.detail`; it pickles, so it crosses process pools.

## License

The Python package and native runtime are licensed under Apache-2.0.
Copyright 2026 Saltwater Research LLC.
