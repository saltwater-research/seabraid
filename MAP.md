# Module ownership

One owner per concept. Dependencies point one way:
runtime <- adapters <- compat. The core never imports an adapter, an engine or
a binding, and nothing imports `compat`. One build input crosses the other way,
and it is a tree you supply, not a path in this repository: `adapters/strata`
builds against a Strata source tree that has had `compat/strata`'s patch series
(and `adapters/strata/strata-v7-seabraid.patch`) applied for the engine's own headers; the helper wire is not part of
that tree, it is `adapters/strata/wire`, which compat and the patched engine include from here. Nothing in `src/`,
`include/`, `cmake/`, `examples/`, `native-tests/` or `tools/` builds from or
depends on `compat` (one comment in `src/placement/tests/parity.rs` names the
generator that wrote its golden file). `checks/check_compat_edges.py` enforces this direction (it runs in CI).

| Module | Owns | Depends on |
|---|---|---|
| `src/lib.rs` | The crate root: module list, the lint policy (`deny(unsafe_code)`; the only `allow` is inside `src/ffi.rs`) | Every module below |
| `src/batch.rs` (+ `src/batch/`, tests) | Shape checks, owned input storage, reference math | Rust standard library |
| `src/picks.rs` (+ `src/picks/`: `src/picks/golden.txt`, `src/picks/tests/golden.rs`, `src/picks/tests/layering.rs`) | The pick rule (one table): expert range, skip marker, weight, no repeated expert in a row; `golden.txt` is its recorded outcomes, also read by `bindings/python/tests/test_pick_golden.py` | `batch.rs` (the skip marker), `diag` |
| `src/rows.rs` | Which rows of a batch a provider sees (compact rows): derive, gather, scatter | `batch.rs`, `diag` |
| `src/submission.rs` | The shape of one submission between the FFI layer and the session | Nothing |
| `src/session.rs` (+ `src/session/`) | Worker, tickets, buffer reuse, caller-observed status | Batch module |
| `src/diag.rs` (+ `src/diag/`: `src/diag/sink.rs`, tests) | Error construction, log levels, the process-wide sink and last-error text | Rust standard library |
| `src/gate.rs` | The handles' misuse detector: liveness cookie, busy flag, poison bit | Rust standard library |
| `src/layout.rs` (+ `src/layout/`: `src/layout/abi_layout.txt`, the frozen `abi_layout.epochN.txt` snapshots) | Compile-time proof that every `repr(C)` struct matches `src/layout/abi_layout.txt` | The layout table |
| `src/header_parity.rs` | Test-only: every `#define` and enum value in the C headers equals its Rust constant | `include/seabraid/`, the Rust constants |
| `src/capabilities.rs` | Operation, input and output formats, accepted origins | Rust standard library |
| `src/placement.rs`, `src/placement/` | Expert ownership, fill, live swap policy and transactions | Rust standard library and adapter-provided copy operations |
| `src/ffi.rs`, `src/ffi/` (including `src/ffi/placement/`) | C layout, callback versions, diagnostics entry points, pointer contracts | Runtime |
| `.github/` | CI: `workflows/ci.yml`, the release gates (its policy is tested in `tests/test_ci_workflow.py` and `tests/test_ci_coverage.py`) | `checks/`, `tests/`, the build files |
| `include/seabraid/` | The public C headers (`seabraid.h`, `placement.h`); Apache-2.0 OR MIT | Nothing |
| `cmake/`, `CMakeLists.txt` | Offline Cargo build, the installed SDK, `Seabraid::runtime` | Installed Cargo and optional test toolchains |
| `bindings/python/seabraid` | Python package: runtime, reference provider, ranking, placement helpers, logging | C ABI and the Python standard library |
| `adapters/strata` | Multi-layer weighted Qwen execution, private helper lifetime, placement bridge | Seabraid ABI and patched Strata headers |
| `adapters/ggml` | Routed SwiGLU over resident weights; opt-in llama.cpp hook patch | Seabraid ABI and the engine's ggml backend |
| `examples/` | Minimal provider (C) and a quickstart (Python) | Public ABI only |
| `compat/strata` | The Strata add-on: patch series, helper, installer, tools, hardware measurements | Pinned Strata and vendor toolkits; `adapters/strata` (`strata-ext/seabraid.cmake` puts `adapters/strata/wire`, the helper wire and option header, on the include path); `bindings/python` (`install/__init__.py` imports the ranking module) |
| `checks/` | Repository gates, one file each: `check_abi_append` (ABI append-only), `check_compat_edges` (the one-way compat edge), `check_complexity`, `check_error_paths`, `check_helper_tier`, `check_links`, `check_map` (this file against the tree), `check_names`, `check_no_egress`, `check_no_local_refs`, `check_no_placeholders` (no unfilled fill-in marker), `check_pid_probes` (no bare-pid probe or signal, one `pidfd_open` owner, no bare channel wait), `check_pins` (tool pins, MSRV, CMake floor), `check_size`, `check_size_budgets`, `check_spdx`, `check_version`; `run_all.py` runs them as CI does | Python standard library |
| `scripts/` | Developer scripts: `add_spdx.py`, `rename.py`, `build_wheel.py` (the platform wheel), `wheel_smoke.py`, `journey/` (the first-time-user replay), `release/` (the release pipeline: `publish.py` runs preflight, a clean one-commit export, build and, only with `--publish` and a typed confirmation per step, the uploads; its guide is `docs/RELEASING.md`), `ggml_fixtures/` (the ggml adapter fixtures against a ggml source) | Python standard library; `checks/` (the shared file list and SPDX rule); `setuptools` and `wheel` (the wheel build); the installed `seabraid` wheel (`wheel_smoke.py`); cargo and cmake at run time |
| `tests/` | Root hygiene tests | `checks/` |
| `native-tests/` | C and C++ ABI, lifecycle and placement fixtures | Seabraid and the C++ standard library |
| `tools/sim/` | `seabraid_sim` load simulator (`-DSEABRAID_BUILD_TOOLS=ON`) and `matrix.sh` sweep | Public ABI only |
| `docs/` | Runtime documentation: [ABI](docs/ABI.md), [adapter guide](docs/ADAPTER-GUIDE.md), [architecture](docs/ARCHITECTURE.md), [placement](docs/PLACEMENT-ABI.md), [technical note](docs/TECHNICAL-NOTE.md), [roadmap](docs/ROADMAP.md) | None |
| `compat/strata/docs` | The add-on's design log, protocol, Strata integration notes, N-card design, research | None |

`allow(unsafe_code)` appears only in: `src/ffi.rs` (the one inner attribute, which lets the whole ffi tree hold the
`extern "C"` functions and pointer contracts); the crate root carries only `deny(unsafe_code)`. Everything else is `deny(unsafe_code)`;
`checks/check_map.py` fails on an attribute anywhere else and on a listed path that no longer holds one.

Source files stay at or under
500 lines (split at 400: `checks/size_ratchet.json` lists the files still past it) and functions at cyclomatic complexity 10 or
below in every code directory. The legal files at the root ([LICENSE](LICENSE), [LICENSE-MIT](LICENSE-MIT),
[NOTICE](NOTICE)) own the licensing of each part.
