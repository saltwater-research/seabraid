# Changelog

All notable changes are recorded here. The project follows semantic
versioning once it reaches 1.0; until then any alpha may change the ABI.

## 0.5.0-alpha.8 - 2026-11-22

- Linux x86-64 is the supported platform for this release of the mixed-vendor expert runtime. The native Python wheel targets glibc 2.34 or newer.
- Windows core runtime source builds are experimental. The README gives the build command and lists the unverified Windows paths. The Strata installer targets Linux.
- Strata users can enable `--file-moves` to let helpers read experts from the model pack. The installer builds or verifies its checksum manifest on Linux.
- The release checklist accepts Linux build-host verification. Windows and macOS CI jobs remain available for optional portability checks.

## 0.5.0-alpha.7 - 2026-11-22

- Use the complete pinned Strata patch series in setup. Find the built engine,
  helper and sums targets and resolve the model's profile after checkout.
- Size the plan for the requested context and preserve the selected GPU in the
  serving process. Continue on a usable primary when no helper can run.
- Keep `--file-moves` opt-in. Use the selected native shard and expert geometry
  to build and verify its checksums.
- Validate retier prerequisites and reject non-finite named helper decode
  replies before merging their results.
- Record preparation checks in the release receipt. Refuse publication when an
  earlier stage skipped a required check.
- Normalize Python source archive metadata for reproducible builds with the same
  compression toolchain.

## 0.5.0-alpha.6 - 2026-11-22

- Run the Strata patch step from its checkout so Bash expands the patch series correctly.
- Correct the Linux wheel quickstart and run its clean install with an explicit setuptools dependency.
- This Linux alpha carries the opt-in Strata `--file-moves` path for moving experts by file reference.

## 0.5.0-alpha.3 - 2026-11-21

### CI

- Pass the pinned Strata checkout to the compatibility tests so they build and run the xxHash-backed file-move fixtures.

### Added

- The old add-on name is retired (owner decision, 2026-11-21): the add-on is the Strata integration under `compat/strata`, and the public release presents
  Strata and Seabraid only.  One scripted step, `scripts/retire_name.py`, did the whole rename: directories (`compat/strata`, `adapters/strata/wire/strata/seabraid`),
  the C++ namespace `seabraid`, binaries and units `seabraid-helper`, `seabraid-sums`, `seabraid-server`, `seabraid-helper-N`, every engine and
  CMake variable `SEABRAID_STRATA_*`, the log prefix `seabraid:`, and the shared-memory defaults `/seabraid0` and `/seabraid1`.  The Strata patch series and the v7
  mbox were regenerated, not edited (git am of the old series, the same script on the Strata checkout, git format-patch).  **Wire v11**: the channel magic text
  is `SEABRD1` and the layout is unchanged, so an older engine and helper refuse each other: rebuild both from the same series.  `helper-begin-finish` is the one
  deviation from the name map (`seabraid_helper_begin` and `seabraid_helper_finish`, because the private Qwen tier already owns `seabraid_begin` and `seabraid_finish`).
  `check_names.py --forbid` and `rel_guard.PROVISIONAL` refuse the retired name in every exported file and as `--name`; see docs/RELEASING.md for the redeploy names.
- Docs: record helper loss on GPUs (runs HL and HL2, 2026-11-20): a killed or idle-killed helper closes only its card and the engine keeps serving; the graceful-stop restart failure of HL is fixed by patch 0020. A restart while a live helper is still in its shutdown state stays CPU-tested only.
- Docs: record live run L2 (2026-11-20): one expert through all eight transitions and file-reference moves on GPUs, with the limits, in the research log, status, README, how-it-works, technical note and paper.
- Add-on installer, opt-in (track FR-D of the file-reference expert moves; the engine, helper and wire tracks are separate):
  `--file-moves` (needs `--ladder`) builds or verifies each pack's checksum manifest with `seabraid-sums`, then adds
  `--retier-file-moves 1` to the engine and `--file-io auto` to every helper; it refuses when the manifest cannot be built.
  Off by default. `placement-qual` gained the opt-in `file-moves` persona (`--persona file-moves`: arms F0/F1, workloads
  W8 to W10 = FR-A1 to FR-A3) and the F1 verdict (file counters present, zero bad sums), whose log-line format is
  provisional until the engine track defines it. The options catalog accepts a switch that is not in source yet only
  when its row says `pending: track FR-<letter>`.

- Misuse detection on session handles: a liveness cookie (cleared on close) makes a double close or a
  garbage handle return `SEABRAID_INVALID`, and a busy flag makes a call that overlaps another call on
  the same session (another thread, or the provider callback calling into its own session) return
  `SEABRAID_BUSY_OR_STATE` and do nothing. Best effort; a freed or unreadable handle is still undefined
  behavior (documented in `seabraid.h` and `docs/ABI.md`).
- `SEABRAID_DEPRECATED` marks the seven v1 session entry points (removed at 1.0). It expands to nothing
  unless the consumer defines `SEABRAID_WARN_DEPRECATED`; `native-tests/check_deprecated.cmake` checks
  both modes in strict C89.
- `seabraid.h` refuses to compile when `sizeof(void *)` is not 8 (C89-safe negative-array assert).
- `docs/ABI.md` opens with a "Which number means what" table (ABI version, epoch, per-struct versions,
  struct_size, SONAME).
- Batch v2 / provider v3 entry points, integrated end to end. `seabraid_session_open` opens a session from a
  `seabraid_provider_v3` (config moved into open; flags `PARTIAL_ROUTES`, `COMPACT_ROWS`,
  `ACCEPTS_F16`, `ACCEPTS_BF16`, `REFERENCE`) and `seabraid_session_submit` submits a
  `seabraid_batch_v2` (activation format F32, F16 or BF16, `placement_epoch`). F16 and BF16
  bytes reach a provider that declared them unconverted. A stale `placement_epoch`
  returns `SEABRAID_BUSY_OR_STATE`, and 0 after a nonzero epoch returns `SEABRAID_INVALID`.
  `COMPACT_ROWS` hides fully skipped rows from the provider and writes results back to their
  original rows. `seabraid_session_status_v1` grew from 48 to 64 bytes (`rows_submitted`,
  `rows_shipped`); a v1-sized caller is unaffected. The v1 and v2 entry points are unchanged.
  See [docs/ABI.md](docs/ABI.md#batch-v2-and-provider-v3-entry-points).
- `strata-v7-seabraid.patch`: `--retier on --retier-promote seabraid` (opt-in) runs live
  placement through Seabraid's placement driver: one fenced round per
  `--retier-every` windows at a quiet boundary, one published executor table that
  both helper tiers' `holds()` read, and the RAM copy following each committed pair.
  It works with the private Qwen helper (the only live publisher it accepts: the
  v7 refusal of `--retier` is narrowed to this pair) and with the named channel
  next to `legacy`. `--retier-promote legacy` is the old ownership-map ladder;
  `on` still parses and means `legacy`. `seabraid` is refused at startup (exit 2) with
  `--retier zero-op`, `--batch` above 1 or a second helper tier. First timed on a live
  engine on 2026-11-18 (matched arms, two to four starts per cell): it did not pass on
  mean decode, and the host page-migration stalls need a cause; see
  [docs/RESEARCH-LOG.md](docs/RESEARCH-LOG.md#2026-11-18--first-matched-arm-run-of-the-live-placement-publisher-s8).

### Changed

- Source layout and checks, no behavior change (the C ABI layouts and every exported symbol are unchanged): the pick rule
  now has one table, `src/picks/golden.txt`, that the Rust core, the ggml adapter (`test_routes.c`) and the Python
  reference provider are each held to, with `NARROW_ROW` and the ggml `MAX_PICKS` recorded in it; `SKIP_EXPERT_ID` moved
  from `batch.rs` to `picks.rs`, which ends a module cycle; `placement.h` gained `SEABRAID_PLACEMENT_MAX_SLOT_BYTES`
  (4294967295, previously only prose) with a Rust and a Python parity test; the ABI layout table moved from
  `native-tests/` to `src/layout/` (the core build no longer reads a test directory); every `unsafe` block outside tests
  has a specific `// SAFETY:` comment and `clippy::undocumented_unsafe_blocks` is denied; all `unsafe` code is now under
  `src/ffi/` (the C log sink is called from `ffi/logging.rs`; `diag` holds a safe closure); `seabraid.ranking` carries its
  own rule text and checker in the wheel.
- Python, breaking: without numpy, a provider callable now receives a `ProviderView` instead of a `memoryview`. It supports
  `len`, indexing, iteration, `.tolist()`, `.tobytes()` and slice assignment (`out[a:b] = values`), but a slice read, a
  `.cast()` or `memoryview(v)` raises `TypeError`, and every use after the callback returns raises `ValueError`. A plain
  memoryview's slices outlived its release, so a provider that kept `out[0:4]` could read engine memory after it was
  reallocated (R7). With numpy, a kept array or slice is still detected and fails the call (`SEABRAID_PROVIDER`, one ERROR).
- Python: `seabraid.version()` returns the PEP 440 form (`0.5.0a2`) before and after a native library is loaded; the
  deprecated v1 symbols are optional, so a library that exports only the v3 entry points loads.


- Placement handles carry the same liveness cookie and busy flag as sessions: a double close or a foreign pointer
  returns `SEABRAID_INVALID` and an overlapping call `SEABRAID_BUSY_OR_STATE` where detectable (the residual
  undefined behavior is unchanged and documented in `docs/ABI.md`).
- Internal, no ABI change: placement's C entry points moved from `src/placement/ffi*` to `src/ffi/placement*`;
  `ABI_VERSION` and `CAPABILITIES` moved from `batch.rs`/`capabilities.rs` to `src/ffi.rs`; `Config::validate` no longer
  checks `abi_version` (the ffi checks it before the struct is read); the `placement` module is crate-private; the
  handle gate moved to `src/gate.rs` and the lease is shared by both handle types.
- Documentation: `README.md` no longer says an engine needs no rewrite, and lists which engines are
  integrated and which are planned; "Strata integration" is defined at its first use; the quickstart text names
  `SEABRAID_PROVIDER_REFERENCE` instead of the deprecated `seabraid_open_reference`. `docs/ROADMAP.md`
  marks the finished "Before an ABI freeze" items and has a working ABI anchor. Links to the placement
  qualification scripts point at `compat/strata/scripts/placement-qual/`. `MAP.md` names
  `checks/check_compat_edges.py` as the checker of the one-way compat edge.
- ggml adapter: the direct `seabraid_ggml_execute` path now refuses a non-finite (NaN or Inf) weight on a
  real pick with `SEABRAID_INVALID` instead of returning a poisoned row, and reports an out-of-range expert
  id and a repeated one as separate reasons (the repeat check was mislabeled "out-of-range or repeated").
  The session path is unchanged (the runtime already refuses both). New edge fixtures cover route faults,
  token bounds, cache lifecycle, the transfer counter under threads and `append_swiglu` refusal reporting.
- Strata adapter: `QwenDecodeTier` keeps a layer local, with one log record, when the host detaches its
  placement view after the session accepted a nonzero epoch (the session refuses epoch 0 as `INVALID`,
  which used to fail the engine's layer). New edge fixture covers full-window and over-window batches, row
  compaction with one and no kept row, a NaN router weight and a helper killed mid-compacted batch.
- ggml adapter CMake: a normal (non-fixture) build links `Seabraid::runtime` (or stops with a one-line configure
  error); before, a program linking only `seabraid_ggml`, which is the llama.cpp hook's link line, failed with
  undefined `seabraid_log` and `seabraid_abi_epoch`. MAP/README now point at `seabraid_session_open` and
  `seabraid_session_submit` and say the ggml adapter has no provider v3 yet.
- Strata adapter tests: `private_qwen_test.cpp` split into the lifecycle/admission file and
  `private_qwen_runtime_test.cpp` (no logic change); the scripted helper can answer any row count.
- Documentation, from a recorded first-time-user run: the README Quickstart is a venv block (with the offline
  `--no-build-isolation` variant), has a Requirements section with the toolchain floors, the native quickstart
  command with its expected output (including the intentional stale-epoch WARN), the wheel install in a fresh
  venv, the plain `gcc` line for the C example and the load simulator build and run. The pure quickstart is
  documented as silent with `reference=True`. `CONTRIBUTING.md` lists the CI-pinned test tools, what a skip
  means and every checker CI runs (including `check_links` and `check_error_paths --strict-runtime`).
  `adapters/ggml/MAP.md` says where to get ggml and how to build it; `compat/strata/README.md` says
  `--plan-only` exits non-zero on a blocker.
- Docs and header comments only (no declaration or behavior change), from an adapter author's walkthrough:
  `docs/ADAPTER-GUIDE.md` gains "Versioned submits and stale epochs" (pass the snapshot epoch each batch; on
  `SEABRAID_BUSY_OR_STATE` stale, re-snapshot, re-route, resubmit), "Reading counters", a log-level table
  (refusals and stale epochs WARN, provider failures ERROR; `min_level` ERROR hides refusals), `COMPACT_ROWS` in the
  provider-flags prose, and a pointer to `examples/adapter_walkthrough.c`. `docs/ABI.md` now leads with
  `seabraid_session_open`, `seabraid_session_submit`, batch v2 and provider v3 and moves the deprecated v1/v2
  calls to an appendix, and `seabraid.h` and `docs/ABI.md` state: partially skipped rows ship unchanged under
  `COMPACT_ROWS` and the provider gets no kept-row map; a failed ticket is consumed, leaves `output` untouched and
  `collected` unchanged, counts in `provider_failures` and keeps the epoch mark, with the callback's own code only
  in the log text and `seabraid_last_error`; `seabraid_last_error` persists until the next failure on that thread;
  the library may or may not pre-zero `output`; the callback always runs on the session's one worker thread;
  `seabraid_session_status` from another thread during a call returns `SEABRAID_BUSY_OR_STATE`; the log sink runs on
  whichever thread logs. The epoch and SONAME are named as the compatibility contract, with a CMake
  `find_package(Seabraid 0.5 CONFIG REQUIRED)` and C `seabraid_abi_epoch()` check; version examples no longer name a
  stale release.
- README: the "about 20 lines" claim for a provider now counts the callback in `examples/minimal_provider.c` (21
  lines, 95 with its driver); the CMake build, test and install commands drop `--config Release` (single-config
  generators ignore it, and CMake always runs `cargo build --release`), and the standalone `cargo build` is said to be
  needed only for the Rust and Python path.
- Documentation and header comments only, no behavior change: `PLACEMENT-ABI.md` gains the policy table (range, unit and
  default of every field; `max_moves_per_round` counts expert writes, one swap is 2), the heat rule (first window as
  given, then `0.75 * previous + 0.25 * window`, kept across apply and epochs), the outcome and epoch arithmetic, a fault
  to outcome table (including that a `FAILED` lower write closes that card without rewriting it), and the reopen
  guidance (a reopened handle restarts at epoch 1; reopen the sessions or stamp batches with an engine-owned
  `generation << 32 | epoch`). `slots` is documented as cells across all layers. The stale-epoch guard is described as
  a per-session monotonic mark that is not linked to the placement handle, with a provider-side ownership check
  recommended. `ADAPTER-GUIDE.md` gets an F16/BF16 activations section; `INPUT-CAPABILITIES.md` and `ABI.md` no longer say
  the input is FP32 only and state that `seabraid_query_capabilities` reports the built-in F32 path.
- The installed SDK follows GNUInstallDirs (`CMAKE_INSTALL_LIBDIR`, default `lib`; headers, docs and the Windows DLL
  likewise) and is split into `Runtime` (library, licenses) and `Development` (headers, `libseabraid.so` link, import
  library, package config) install components; a plain `cmake --install` installs both. The library is installed
  executable (0755), and the `libseabraid.so` development symlink is listed in `install_manifest.txt`.
- Release builds are stripped of non-exported symbols (`[profile.release] strip = "symbols"`; 813 kB to 628 kB, same
  27 exports; use `CARGO_PROFILE_RELEASE_STRIP=none` for a symbolised backtrace). `panic = "unwind"` is now stated
  and explained next to it, because the FFI guard depends on it.
- The crate builds as a `cdylib` only (the unused `rlib` is gone); `cargo test` is unchanged.
- The imported `Seabraid::runtime` target in `SeabraidConfig.cmake` is `GLOBAL`, like the one in the build tree.
- Windows builds pick the import library by ABI: `seabraid.dll.lib` (MSVC) or `libseabraid.dll.a` (windows-gnu);
  untested here, no Windows host. Only Linux records a SONAME in CMake (FreeBSD and other ELF systems get a plain
  `libseabraid.so`, matching `build.rs`).
- `docs/ABI.md` and `build.rs` state the macOS limit: dyld refuses an older library but still loads a newer one, so
  the start-up `seabraid_abi_epoch()` check is the only 0.x enforcement there; the CMake package version drops the
  pre-release suffix and skips its 32-bit check.

- Documentation: the README's verification table now cites dated entries in
  [docs/RESEARCH-LOG.md](docs/RESEARCH-LOG.md) and no longer prints exact test counts or
  figures that no record in the tree supports; the technical note, the changelog and the
  README agree on what was run (the KL gate on v0.2.15 passed, the llama.cpp hook is
  bit-identical to stock on four backends, the placement publisher had one first timed run).
- NOTICE reproduces the MIT notices of Strata and llama.cpp/ggml for the patch series.
- The library fills a smaller caller's `struct_size` honestly: every library-filled struct gets
  `min(caller struct_size, current size)` bytes and that count written back as `struct_size` (one owner,
  `write_out`), where a smaller caller used to be told the library's own size. The ggml adapter's
  `seabraid_ggml_get_transfer_stats` now refuses a `struct_size` below its v1 size and clamps the same way.
- An expert id may not repeat within one row: refused with `SEABRAID_INVALID` in the runtime
  (`pick_error`). Adapters keep their own copies for now.
- `seabraid_provider_v2` is frozen to `PARTIAL_ROUTES`: `COMPACT_ROWS` on a v2 descriptor is refused
  with `SEABRAID_INVALID` (row compaction is a provider v3 feature).
- Docs and header comments: one written epoch rule ("the epoch changes only for a non-append change;
  appends ride on `struct_size`"), the corrected library-filled rule, and `seabraid_session_submit`
  promises only that a refused batch changes nothing (the fixed validation order is dropped from the
  contract).

- The compatibility series now targets Strata v0.2.15 (`c77572806`, was v0.2.13.1
  `628f74955`). Patches 0001-0007 and 0010-0016 were rebased with `git am -3`;
  patch 0016 lost its `src/dsv41` hunk, and `strata-v7-seabraid.patch` applies after it.
  Strata's own build flags are followed with one deliberate difference: the installer
  passes `-DSTRATA_MMQ=ON` on CUDA and HIP, because Strata 0.2.14 turned MMQ off by
  default and patch 0013 (D4) hooks llama.cpp's MMQ prompt path (Strata's own prompt
  kernels are the default since 0.2.14).

### Fixed

- Strata patch 0020: an engine restarted after a graceful stop no longer exits when its helper is dead. A graceful stop leaves the channel in
  `kShutdown`, and `seabraid_open` refused on that state ("not ready for a plan (state 5)", exit 1) before it looked at the helper's identity,
  so the stale-channel rule of 0019 never ran. The identity check now comes first: a dead or recycled helper opens as a closed card with one
  WARN whatever the state word says, and a live helper still in `kShutdown` is waited for (up to 30 s). Strata tests L11 and L12.
- Session and placement handles no longer break Rust's aliasing rules (R6a H2, found by Miri): the cookie, busy flag
  and poison bit now sit in a header in front of the session or placement instead of inside it, so a call's exclusive
  borrow never covers the bytes another thread's refused call reads, and close frees the memory last, after the
  session's own close returned. The C ABI, every exported symbol and every status are unchanged. Miri now runs the FFI
  tests under both Stacked Borrows and Tree Borrows in CI; the tests Miri cannot run (guard pages, `/proc`, a megabyte
  error) are skipped under it only, and three test-side faults it found are fixed (a wide-struct pointer that covered
  only the known prefix, a misalignment that depended on where the stack array landed, and a close of a freed handle).
- Python (R7): a misaligned float32 buffer is copied instead of aliased (native refused it while the reference accepted
  it); a misaligned `out=` is refused. The count precheck names `seabraid_session_submit`, carries the engine's detail text
  and logs once. `close(timeout_ms)` against a library without `seabraid_close_timeout` leaves the runtime open. A refused
  native weave no longer keeps the caller's `array.array` export-locked. `ranking.rank` validates its inputs (and knows
  the RTX 3060 Ti at 448 GB/s); Python `placement.fill` refuses more than 2^26 cells like native.

- The package config an absolute-`LIBDIR` install writes is now mode 0644 under any umask (it was 0600 under `umask 077`, so
  no other user could `find_package` the SDK), is removed before it is written (a symlink left there was written
  through, and a read-only copy kept its mode), prints its "Installing:" line only when `CMAKE_INSTALL_MESSAGE` is not
  `NEVER`, and names `/include` for `--prefix /` (the empty prefix was resolved to the working directory). A `;` or a
  newline in the install prefix is refused by the first install rule of both components, in every layout, before any file
  is copied. The same install-time replacement covers the Windows runtime path under an absolute `LIBDIR` (written
  without a Windows host: not run there). `check_layouts.sh` gained a `umask 077` step, a `--prefix /` step under
  `DESTDIR`, symlink, read-only and `CMAKE_INSTALL_MESSAGE` steps, and a `;` refusal check per component; the test pins
  every `ok:` line of the reinstall block. `CC` holding a path with a space no longer fails the compiler probe.
- `seabraid_sim`'s work estimate counts the picks: `picks * (22 + 12 * devices)` ns per layer call (the accepted shape
  `--hidden 1 --tokens 262144 --experts 4096 --top-k 64 --devices 1 --layers 1000 --steps 15 --warmup 1` took about an
  hour against the ten-minute cap, and the old estimate was 5.4 times under at 65536 tokens and top-k 64), a fixed call
  cost of 6000 + 2000 per device, and placement as 100 ns per cell once plus 55 per cell per round (it was 40 per cell
  per round and no one-time cost, under the measured time for one to five rounds). Re-measured on the reference box
  (Release, 49 shapes): the estimate is at or above every measured time, the least margin 1.14. The simulator now
  compiles with `-O2` when no `CMAKE_BUILD_TYPE` is given (a plain configure had no `-O` and ran up to 3.6 times over the
  estimate), and as `-std=c11` (CMake's `c_std_11` feature gave `-std=c2x` on gcc 15). `matrix.sh` no longer
  pathname-expands `--extra` words.
- Release gate tests: `code_only` treats `;` and `{` (not `${#x}`) as comment predecessors, follows a double-quoted
  string across lines, handles `$'...'` and CMake bracket arguments, and its docstring states the exact limits left;
  `test_release_hygiene` takes its file list from `checks/_common.tracked` (it had a second walk that scanned an in-tree
  `logs/` directory); the sabotage and rename tree copy fails loudly on a top-level entry missing from
  `SHIPPED_TOP_LEVEL`; `SEABRAID_SKIP_SDK_TESTS` semantics are tested (only `1` skips); the `TIMEOUT=0` test xfails on a
  CMake newer than the keyword test's ceiling instead of failing.
- `seabraid_close_timeout` with a zero (or very short) bound on a session with no outstanding ticket no longer
  returns `SEABRAID_TIMEOUT` and logs a false "worker thread leaked" ERROR: the exit wait now applies only while
  a provider callback can be running.
- A batch row with `top_k` above 64 picks is checked for a repeated expert id in O(top_k log top_k) instead of
  O(top_k^2) (a single 64,000-pick row took 12.7 s unoptimized); the refusals and their text are unchanged.
- A placement heat table whose every value is finite but whose per-layer sum overflows `f64` is still refused
  with `SEABRAID_INVALID`, now with a message that says so (`heat of one layer sums past f64::MAX`) instead of
  "heat must be finite"; `placement.h` states the per-layer-sum rule.
- `seabraid_placement_fill` now applies that same per-layer-sum rule (one shared check with the planner): a heat
  table whose layer sum overflows `f64` returns `SEABRAID_INVALID` with the same message instead of `SEABRAID_OK`
  and an owner table. The Python `placement.fill` gets the refusal through the native status.

- `seabraid_close` and `seabraid_close_timeout` called from the session's own provider callback (the worker thread)
  are refused with `SEABRAID_BUSY_OR_STATE` and a logged reason instead of waiting for the thread they run on.
- The reference transform's all-zero result is `-0.0` on every toolchain (an explicit fold from `-0.0`; `Iterator::sum`
  starts at `+0.0` before rustc 1.83, and the MSRV is 1.79).
- Every exported call (not only close) made from the session's own provider callback is refused with `SEABRAID_BUSY_OR_STATE`
  and a logged reason, whether or not another call is inside the session: a callback calling `seabraid_collect` on its own
  session used to wait out the caller's timeout for a completion only that thread could send.
- A batch row wider than 64 picks that holds several defects names the same pick and text as a narrow row: the lowest-index
  defect wins whatever its kind (the sort path used to report a bad id or weight anywhere in the batch before a repeat).
- A panic inside a call poisons its handle before the busy flag is released (the poison flag moved into the gate as an
  atomic), so a racing caller can no longer enter the half-updated handle unpoisoned; the lease no longer keeps a `&Gate`
  that aliases the `&mut` it hands out (it holds a raw pointer to the gate).
- ggml adapter: a pick with several defects is refused with the runtime's reason (range, then non-finite weight, then
  repeat; the adapter used to name the repeat before the weight), and a skipped pick's zero weight is a bit test, so a
  NaN skip weight is refused even when a host builds the adapter with `-ffast-math` (a float compare may fold there).
- ggml adapter: a row wider than 64 picks is checked for a repeated expert id by sorting a copy (O(top_k log top_k),
  two 64,000-pick rows in milliseconds instead of about 0.9 s); narrow rows keep the scan, refusal text unchanged.

- The Strata decode tier no longer treats every `SEABRAID_BUSY_OR_STATE` carrying a nonzero epoch as a stale
  placement epoch: only an epoch below the highest one its session accepted falls back locally; a busy,
  closing, not-prepared or pending-ticket refusal is now a normal submit failure instead of a silent local fallback.
- `adapters/ggml/sdk_check/run.sh` configures and links `adapters/ggml` standalone (tests off) against an installed
  SDK, so the `find_package(Seabraid)` path the llama.cpp hook uses is exercised.
- Python: Ctrl-C landing between the native submit returning and the ticket being recorded no longer strands the
  batch; `weave` keeps it collectable (`collect()`), where before every later call said "no batch is pending" or
  BUSY_OR_STATE until `close()`.
- Python: a `Runtime` whose lock another thread held at `os.fork()` no longer hangs in the child; the lock is renewed
  in the child, so the "belongs to the parent process" error is reached.
- Python: a `close()` the engine refuses with `BUSY_OR_STATE` (a busy handle, or a close from the provider thread) keeps
  the handle, the provider and the finalizer instead of leaking them, and the runtime stays open for a retry.
- `scripts/build_wheel.py`: `CARGO_BUILD_RUSTFLAGS` and `CARGO_TARGET_<TRIPLE>_RUSTFLAGS` are merged into
  `CARGO_ENCODED_RUSTFLAGS` (they were silently dropped), and the leak scan also looks for the checkout path with
  `/`, `\` and `\\` separators and for the home directory.
- The refusal for an expert id outside the provider's range no longer says "skips need a partial-routes provider" to a
  provider that already declares `PARTIAL_ROUTES`: that hint is given only for the skip id (-1) sent to a provider
  without the flag; any other bad id is told the valid range and that -1 is the skip marker. Status and refusal order
  are unchanged.
- With `CARGO_BUILD_TARGET` set, `cmake --build` used to exit 0 and leave only a dangling `libseabraid.so.0.5` symlink
  (the library was under `cargo/<triple>/release`), which `cmake --install` then shipped. The build now clears the
  variable for its cargo call and fails loudly if any output is missing (for example with `build.target` in a cargo
  config file).
- CTest ignored the exit code of tests that set `PASS_REGULAR_EXPRESSION` (`echo ok; exit 3` passed):
  `seabraid_minimal_provider` and `seabraid_adapter_walkthrough` now run through `native-tests/check_output.cmake`.
- `checks/check_version.py` now compares the `seabraid` entry of `Cargo.lock`: bumping `VERSION` and `Cargo.toml`
  without the lock file passed the check and then failed every `--locked` build.
- Editing `SEABRAID_ABI_EPOCH` in the header (or `native-tests/abi_layout.txt`) now re-runs configure and the cargo
  step; `SeabraidConfig.cmake` kept the old epoch. The Strata adapter's tests no longer depend on a `BUILD_TESTING`
  variable nothing defines.
- The installed `libseabraid.so.MAJOR.MINOR` was mode 0644; it is 0755.
- `seabraid_sim` judges a run by a relative error (scaled by `max(1, |reference|)`), not an absolute one: correct runs
  with 1024 or more experts used to fail with exit 2 (`max_scaled_err` joins `max_abs_err` in the JSON line). A run that
  verified no layer now fails (exit 2) unless `--fail-rate` is 1; `--warmup` plus `--steps` is summed in 64 bits and
  both are capped at 1000000 (`--warmup 4294967295 --steps 1` used to wrap, run nothing and print `ok:true`);
  `--hidden` is capped at 1048576 and `--devices` at 1024; size products are overflow-checked and an allocation failure
  prints an "out of memory" line (exit 1) instead of exiting silently; numeric flags must start with a digit (a leading
  space or `+` was accepted, so `--seed " -1"` meant 2^64-1).
- `tools/sim/matrix.sh`: a value flag with no value (`--bin` last) exits 64 instead of looping forever; grid values must
  be integers; a sweep that runs no cell exits 64 instead of 0.
- `examples/consumer`: the copy instruction names both `.c` files (copying one failed at configure), the programs carry
  `INSTALL_RPATH` `$ORIGIN/../lib` so an installed copy starts (it exited 127 without), and its tests use
  `check_output.cmake` so a nonzero exit fails them (ctest's `PASS_REGULAR_EXPRESSION` ignores the exit code).
- `scripts/build_wheel.py` reads the library from `target/<CARGO_BUILD_TARGET>/release` when that variable is set,
  refuses a wheel that lacks `seabraid/_lib/<name>` (or holds it without an execute bit on POSIX), and takes a per-checkout
  lock so two builds in one checkout cannot delete each other's library (the second exits at once).
- `-DCMAKE_INSTALL_LIBDIR=lib64` (or `lib/x86_64-linux-gnu`) given untyped on the command line installed the library
  under the current directory instead of the prefix: the cache entry is now a STRING. The development symlink step
  also handles an absolute LIBDIR, creates its directory (a `--component Development` install into a fresh prefix
  failed), and its install_manifest entry no longer carries DESTDIR (CMake's own entries do not).
  `native-tests/installed/check_layouts.sh` checks default, lib64, multiarch, absolute-with-DESTDIR, an absolute LIBDIR
  consumed through `find_package` (see the next entries), Runtime-only and Development-only installs.
- With an absolute `CMAKE_INSTALL_LIBDIR` and a `cmake --install --prefix` that differs from the configure-time
  `CMAKE_INSTALL_PREFIX`, the installed package config resolved its include directory against the configure prefix:
  `find_package` passed and the consumer then failed with "seabraid/seabraid.h: No such file". The config now names the
  install-time prefix's include directory (an install step writes the final config from the build tree's pristine copy, so
  a second `cmake --install`, a different `--prefix` or a `touch -r` of the installed file work; the prefix is made
  absolute and escaped, and a `;` in it is refused; a relative LIBDIR is unchanged and still relocatable).
  `check_layouts.sh` builds and runs a consumer against that layout and reinstalls it five ways.
- Configure refuses a `CMAKE_INSTALL_LIBDIR` or `CMAKE_INSTALL_INCLUDEDIR` containing `"`, `$`, a backslash, `;`, a newline or a leading `~`:
  the value is pasted into install-time script text, where those characters changed what was installed (`~lib` counted
  as absolute and installed a directory named `~lib` under the current directory).
- `native-tests/installed/check_sdk.sh` removes its scratch directory only after moving the SDK back succeeded; when the
  move fails (a read-only parent), the directory holding the only copy is kept and its path is printed, and the script
  exits 1 even when every check passed.
- `native-tests/check_output.cmake` requires `TIMEOUT` to be a whole number of seconds and no longer builds the
  `execute_process` text from it (a value with a newline followed by `file(WRITE ...)` ran that command; `abc` was an
  unrelated parse error); `-DTIMEOUT=0` is documented as "time out at once", not "no limit". It declares
  `cmake_minimum_required(VERSION 3.22)`; the keyword list is checked against the installed CMake's synopsis by a test
  (a failure up to CMake 4.2, a warning and xfail on a newer CMake, so a runner image update does not fail the gate).
- `native-tests/check_output.cmake` passes empty arguments, `;`, brackets and a command named `false`/`off`/`no` through
  unchanged (it used a CMake list, which dropped or misread them); an argument equal to an execute_process keyword is
  refused with a message. A command that overruns TIMEOUT fails with "timed out" even when EXPECT_RC names the timeout.
  On a TIMEOUT the script returns promptly and leaves a grandchild running, orphaned; on a normal exit a grandchild that
  keeps the output pipe open delays the return until it closes it (both documented in the script and pinned by a test).
- `checks/check_version.py`: a missing `Cargo.lock` beside `Cargo.toml` fails (builds use `--locked`) instead of being skipped.
- `seabraid_sim` fails a NaN or infinite output (the running maximum skipped NaN, so a NaN kernel printed `ok:true`).
  The pass line is now the float32 rounding bound derived from the computation, `(picks + devices + 3) * FLT_EPSILON *
  sum |term|` per element, instead of a flat `1e-4` relative: an off-by-one expert multiplier used to pass from 262144
  experts (`max_scaled_err` is in units of that bound; 1 passes). That bound grows with top-k and devices, so a wrong
  multiplier hides under it on large shapes; those shapes are now refused instead of passed (see the next entry).
  New caps: `--experts` and `--tokens` 1048576, `--top-k` 4096, `--layers` 65536. `--fail-rate` refuses hex floats
  such as `0x0.8`. The kernels `nan`, `inf` and `off_by_one` are wrong on purpose, for testing the verifier.
- `tools/sim/check_cli.sh` detects a sanitizer build with `nm` (or `otool -L` on macOS) instead of `ldd` alone and runs the
  out-of-memory case only where `ulimit -v` demonstrably holds. `tools/sim/matrix.sh` requires `--bin` to be an executable
  regular file, `--steps` and `--layers` (new) to be a single integer, and every grid value to be within the simulator's caps.
- `scripts/build_wheel.py` takes the per-checkout lock before its cargo build, so a second builder is refused at once and
  not after building; `main()` is covered end to end with cargo and setuptools faked.
- `seabraid_placement_plan` refuses a window whose per-layer raw sum overflows on every call, not only when the blended heat
  does (`plan([1,1])` then `plan([1.7e308, 1.7e308])` returned a plan while `seabraid_placement_fill` refused the same heat);
  `docs/PLACEMENT-ABI.md` now states exactly that.
- `seabraid_sim` refuses a shape whose result it could not verify: the rounding bound grows with top-k and devices, and the
  deliberately wrong `off_by_one` kernel passed at 1048576 experts with top-k 32, at 8192 experts with top-k 4096 and at
  262144 experts with top-k 64 on 8 devices. The joint limit `(experts + 1) * (top-k + devices + 3) * sqrt(top-k)` is
  derived in `sim_cli.c` (the limit is 2^22, see the entry below; it was first set to 2^23) and checked by an experts x
  top-k x devices sweep in `check_cli.sh` (off_by_one fails and a real run passes on every accepted shape). A combined
  memory limit (4 GiB estimated: tokens x hidden x devices, tokens x top-k x devices, layers x steps) refuses a product of
  in-cap flags with a message instead of an out-of-memory. `--validate-only` checks the flags and exits 0; `--print-caps`
  prints every cap.
- `tools/sim/matrix.sh` reads the caps from `seabraid_sim --print-caps` (it kept a second copy that a test could not tell was
  wrong), refuses an unknown `--kernel`, an unverifiable or over-memory cell, `--steps 0`, a padded value (`" 2"`) and a
  zero-padded value (`00000000008`, once reported as "above the cap") before any cell runs, each with the simulator's own
  words. The kernel list has one owner, the table in `sim_providers.c`; the CLI validates `--kernel` and prints `--help`
  from it.
- `seabraid_sim`'s combined memory estimate undercounted the measured peak RSS by 1.3 to 2.2x (4 bytes per cell per device
  against 12 measured; the per-pick copies, the placement buffers and the process baseline were left out), so a shape it
  called "under 4 GiB" could take twice that. It now follows measured `/usr/bin/time -v` peaks (the constants and the
  shapes behind them are in `tools/sim/sim_budget.c`): 8 MiB + 16 KiB per device + 20 + 15 per device per cell (+ 4 per
  cell with `--compact-rows`) + 8 + 18 per device per pick + 17 per latency sample + 48 per placement cell. The run line
  prints it as `estimated_bytes`, and `check_cli.sh` requires it to be at least the peak RSS on ten shapes, two of them
  `--compact-rows` (110 to 128 percent of it on the large ones) wherever GNU `/usr/bin/time` exists; it is skipped on a
  sanitizer build, where RSS is inflated. The margin over 31 measured shapes is at least 9 percent (the earlier "at least
  10 percent" was not true: the first constants were 3 percent over on large 1-device runs and under on `--compact-rows`). The out-of-memory case
  in `check_cli.sh` uses `--tokens 64` (the larger shape is now refused as over 4 GiB, which is the estimate being right).
- `seabraid_sim`'s joint verifiable-shape limit is 2^22, not 2^23: at 2^23 there was no room at top-k 1, where one checked
  element (`--hidden 1 --tokens 1 --layers 1 --steps 1`) let `off_by_one` through in 7 of 1000 seeds (1048575 experts on
  4 devices), 6 (699049 on 8), 4 (419429 on 16) and 2 (123360 on 64). At 2^22 the same eighteen edge shapes (top-k 1, 2 and
  8 on 1 to 64 devices) were detected 1000 of 1000 times; that is a measurement, not a guarantee, and the comments no
  longer say "guaranteed" or "exactly". The largest accepted expert count is now 838859 (the `--experts` cap, 1048576, is
  past the limit for every top-k). `check_cli.sh` runs the sweep on a one-element run as well and a hundred seeds on each
  of eight edge shapes.
- `seabraid_sim` bounds the running time of a configuration: `--placement-rounds` is capped at 100000 (4000000000 never
  finished), and a combined work estimate (`(warmup + steps) * layers * (cells * (8 + 2 * top-k * devices) + 5000 + 2000 *
  devices)` nanoseconds, plus 40 per cell and 20000 per round of placement, at most 6e11, about ten minutes) refuses what
  would never finish, including `--warmup 1000000 --layers 65536`, in `--validate-only` and so in `matrix.sh` before any
  cell runs. `--print-caps` prints it as `work`. The comment that said the per-flag caps bound the work is corrected.
- `tools/sim/matrix.sh` refuses `--validate-only`, `--help`, `-h` and `--print-caps` inside `--extra` (exit 64: the
  simulator prints no run line, so every cell passed unrun: "16 runs, 0 failed", exit 0) and counts a cell that exits 0
  without printing an `"ok":` run line as failed.
- `seabraid_sim`: the default kernel is the first row of the kernel table, not a second copy of its name; the `self_test`
  column of that table is used (`--help` prefixes those kernels with "SELF-TEST, wrong on purpose:") instead of being
  dead; the `libm` link is skipped where `find_library(m)` finds nothing (macOS, musl) instead of failing the configure
  with `SEABRAID_SIM_LIBM-NOTFOUND`.

- `checks/_common.py` `tracked()`: in a tree without `.git` the walk scans only the top-level entries the repository ships
  (`SHIPPED_TOP_LEVEL`, compared with `git ls-files` by a test); an SDK installed into the tree or a `logs/` directory no
  longer makes `checks/run_all.py` and `pytest tests` fail. README and CONTRIBUTING say to install outside the source tree.
- Tests: `SEABRAID_SKIP_SDK_TESTS` skips only when it is `1`; `test_ci_workflow` requires Python 3.11 in the matrix;
  `tests/test_complexity.py` skips instead of aborting collection when lizard is missing.
- Whole-runtime audit R6a, each with a test that fails against the previous code:
  - `seabraid_close_timeout(0)` (or any short bound) after a ticket whose callback had returned but that nobody
    collected no longer answers `SEABRAID_TIMEOUT` with a false "worker thread leaked" ERROR (2000 of 2000 in the
    reproducer): a callback is "still running" only while the ticket's completion is not yet queued. A callback that
    really is running still times out, and the uncollected failure is still logged.
  - `seabraid_placement_plan` is no longer O(challengers x incumbents) when no swap can pass the helper-share ceiling
    (3+ cards, share 0, or a helper over its cap): one layer of 80,000 experts took 26 s, now it is a sort and one
    binary search per challenger. The search is exact (the ceiling test is monotone in the incumbents' heat order);
    a differential test runs 20,000 random rounds against the old scan and requires identical plans.
  - After a provider callback panics the session's worker stays stopped (documented in `seabraid.h` and `docs/ABI.md`,
    which used to say the session was free for the next submit). A submit that raced the worker's exit was accepted,
    raising the placement epoch mark and the counters, and failed only at collect; it is now refused up front with
    `SEABRAID_PROVIDER` and changes nothing.
  - A provider failure during `seabraid_placement_apply` is one log record, not two, and says what happened (a stage
    that ACKs but writes fewer bytes than the slot is no longer "did not acknowledge"; an unknown code names the code).
    PROVIDER and INTERNAL records are ERROR as `seabraid.h` says; a clean refusal and the quarantine notice stay WARN
    (`seabraid.h` now states the placement rule).
  - A corrupt `layer_slot_bytes` can no longer OOM-kill the host: entries are capped at 2^32 - 1 bytes (`SEABRAID_INVALID`
    before any I/O, ticket kept, stated in `placement.h`) and the stage buffer is zero-allocated without touching its
    pages (a 1 GiB slot used to commit 1 GiB before the callback ran).
  - Library-filled structs carry zero padding, never stack garbage: the placement structs have explicit pad fields (same
    layout), and a build-time check fails if a library-filled struct grows implicit padding.
  - A `struct_size` that ends inside a field (50 for the 64-byte status) is `SEABRAID_INVALID` for every library-filled
    struct, as it already was for the caller-filled ones; it must be a multiple of the struct's alignment. The outcome
    array's reset of element 0's `struct_size` is documented next to the stride rule.
  - A session handle given to a placement call, or the reverse, is refused as `SEABRAID_INVALID` without reading past the
    first 8 bytes of either allocation: the liveness gate is the first field of both handles and each kind has its own cookie.
  - `seabraid_submit_operation` reads the version word through a raw pointer before forming `&BatchView`, and takes the
    handle lease before validating `source_modalities`, as `seabraid_session_submit` does, so a poisoned session answers
    `SEABRAID_INTERNAL` on both entries.

- `seabraid_last_error` no longer records its own argument errors, so a bad call cannot replace the
  message being asked for.
- `strata-v7-seabraid.patch`: the private Qwen helper's experts now stay in the
  engine's RAM copy, like the named-channel helper's (`SEABRAID_STRATA_EXCLUDE_HELPER_RAM=1`
  still leaves them out). Before, the rule checked only for the named-channel helper,
  so prompt reads of the private helper's experts went to the drive. A RAM budget
  that cannot hold both now fills the experts no GPU holds first (decode reads them)
  and the helper's after them, so keeping the helper's experts does not push
  decode's experts to the drive.
- `strata-v7-seabraid.patch`: `--retier-promote seabraid` now starts on the private helper with mixed per-layer expert quants (the helper rung takes one slot size per layer; before, the engine exited 2 with "the private helper stages one slot size"), and the per-round placement record (INFO) now reaches stderr once the driver is open (the default Seabraid threshold, WARN, hid it).

### Removed

- Patches 0008 and 0009 (DeepSeek-V4.1 helper link and `SEABRAID_STRATA_LINK_CHECK`): Strata
  removed its DeepSeek-V4.1 engine in 0.2.14, so there is nothing to apply them to.
  Numbers are not reused; the series is 0001-0007 and 0010-0016.
- The DeepSeek-V4.1 add-on code that only served those patches: `seabraid-helper-dsv41`,
  its executor, plan check, parity and end-to-end tests, the dsv41 KL gate
  (`tools/kl_dsv41.py`, `tools/kl_gate_dsv41.sh`, `tools/_kl_paired.py`) and the
  installer's dsv41 channel-geometry parameter. The DeepSeek measurements stay in the
  docs as history, labelled with the Strata version they ran on (v0.2.13.1).

### Tests

- Public examples: `examples/adapter_walkthrough.c` (CTest `seabraid_adapter_walkthrough`, also built by
  `examples/consumer`), a commented provider_v3 reference adapter covering `PARTIAL_ROUTES` and `COMPACT_ROWS`, the ABI
  epoch check, a log sink, a stale-epoch refusal and resubmit, a refused malformed batch, a device failure and the status
  counters; and `examples/consumer/CMakeLists.txt`, the minimal `find_package(Seabraid)` consumer built against an
  installed SDK (CI's `sdk` job builds and runs it).
- Verification added: `scripts/journey/` replays the README Quickstart and CONTRIBUTING "Build and test" commands in a
  fresh tree (`tests/test_journey_steps.py` fails when a step's command leaves its document; CI's `journey` job runs it on
  a `git archive` of the commit under Python 3.11, 3.12 and 3.14); `native-tests/installed/check_sdk.sh` checks the
  installed library, its exports, the install manifest and `find_package(Seabraid <major.minor>)` after the prefix moves;
  `checks/check_links.py` fails CI on broken relative links and `#anchors` in tracked Markdown; `checks/run_all.py` runs
  every checker with CI's arguments; the Miri job covers the FFI and diagnostics test modules, and a test fails if its
  filter misses one.
- Platform gap: `adapter_walkthrough` (C11 `<stdatomic.h>`) is not built or run on MSVC, in the root build or
  `examples/consumer` (unverified on Windows). At the time, CONTRIBUTING required a green manual Windows/macOS CI run before a tag.
- The per-test regression detail of this release (the poison-test race rates, the wheel-builder mutants, the
  simulator and SDK-layout checks and similar) is recorded in the project's internal notes, not in these release notes.

## 0.5.0-alpha.1 - 2026-11-15

First public alpha. Use at your own risk; the ABI can change before 1.0.

### Added

- Rust core: sessions with one worker per session, single-use tickets,
  buffer reuse, timeouts that preserve the ticket, and status counters.
- C ABI (`include/seabraid/seabraid.h`): provider callbacks v1 (one operation) and v2
  (a dense table of operations sharing one worker), partial routes
  (`SEABRAID_SKIP_EXPERT_ID`) so several devices split one layer, capability
  discovery with activation provenance, and session status.
- Live placement (`include/seabraid/placement.h`): Rust ownership map ported from
  the one our patch 0012 adds to Strata, heat-based promotion with cooldown, margin and move budget,
  acknowledged swaps at a drained engine fence, restore and quarantine, a
  verified fallback rung, and executor snapshots. Limits are in the header:
  `SEABRAID_PLACEMENT_MAX_CARDS` (63) and `SEABRAID_PLACEMENT_MAX_CELLS` (2^26).
- Spillover fill (`seabraid_placement_fill`): hottest-first initial placement
  across cards.
- Diagnostics: a log sink (`seabraid_set_log_sink`, `seabraid_log`), per-thread last
  error (`seabraid_last_error`), and `seabraid_version`.
- CMake SDK: `find_package(Seabraid)` and `Seabraid::runtime`, relocatable.
- Python package: `seabraid.Runtime` with a pure-Python reference provider that
  runs without a GPU, native library loading, `seabraid.ranking` (card ranking is untested on hardware),
  `seabraid.placement`, and logging helpers.
- Strata adapter (`adapters/strata`): sealed descriptor handoff to a private
  Qwen v7 helper, pidfd watchdog, helper retirement with local fallback, and
  the fenced placement bridge.
- ggml adapter (`adapters/ggml`): routed SwiGLU on any ggml backend with a
  cached graph, verified on a Q4_0 fixture on CPU, CUDA and Vulkan, plus an
  opt-in llama.cpp scheduler hook patch.
- Strata integration (`compat/strata`): the original Strata add-on, with its
  patch series 0001-0016, helper, installer, tools and hardware measurements.
- Documentation: ABI reference, adapter guide, technical note, roadmap.

### Changed

- Every ABI struct starts with `abi_version`; structs the caller allocates for the
  library to fill carry `struct_size` second, and the library writes at most
  `min(struct_size, its own size)`. `seabraid_query_capabilities(out)` and
  `seabraid_session_status(session, out)` take the version from the struct, not an
  extra argument. Batch initialisers begin with `SEABRAID_ABI_VERSION`.
- `seabraid_abi_epoch()` / `SEABRAID_ABI_EPOCH` (1): the Python binding refuses a
  library with another epoch, the CMake package exports it, and the Strata and ggml
  adapters check it at start-up.
- Linux SONAME is `libseabraid.so.0.5` (DT_NEEDED checked by CTest).
- One layout table, `native-tests/abi_layout.txt`, drives the C static asserts, the
  Rust const asserts and the Python ctypes checks.
- A NULL provider is invalid; `seabraid_open_reference()` selects the reference
  transform explicitly.
- A panic caught at the FFI boundary poisons the handle (close still frees it).
  Placement provider panics keep quarantining the rung.
- Added `seabraid_close_timeout()`; Python `close(timeout_ms)` and a finalizer that
  closes the session before releasing the provider thunk.
- `seabraid_placement_open` nulls `*out` on failure, quarantine reports the real
  error, and placement copies fail as logged errors instead of aborting.
- Public headers are C89-comment clean (tested with `-pedantic-errors`).
- The ggml adapter's llama.cpp hook patch targets release v1.0.0.

### Security

- Egress checker rebuilt as a tripwire over every tracked file (comments stripped per
  language, patch additions scanned, loopback-only URL hosts); a link-level test
  asserts `libseabraid.so` imports no socket symbols. It is a tripwire, not a proof.
- Strata helper spawn and kill hardened: spawn uses `pidfd_spawn` or a verified
  `pidfd_open`, is refused while `SIGCHLD` is ignored, and refuses an equal-descriptor
  handoff; only `ECHILD` means the helper is gone, other `waitid` errors fall back to
  `waitpid`/poll; a setuid/setgid helper or one in a world-writable non-sticky
  directory is refused; `SEABRAID_STRATA_*` is an explicit list and `STRATA_*` debug switches
  are dropped unless the launcher asks. Compiler flags: `-Wall -Wextra -Wformat=2`,
  stack protector, `_FORTIFY_SOURCE=3`, libstdc++ assertions, PIE and full RELRO.
- Installer refuses root without `--allow-root`, generated systemd units carry a
  sandbox (no network families, `ProtectSystem=strict`, `NoNewPrivileges`), and the
  GGUF header reader bounds every length by the bytes left in the file.
- ggml hook patch retargeted to llama.cpp v1.0.0 after a supply-chain review of the
  build-touched paths.
- Patch 0016: every engine-side read of helper-written shared-memory text goes through
  `seabraid::bounded_text` (at most cap-1 bytes, never past the field, control bytes
  stripped), so a helper that omits the NUL can no longer make the engine over-read or
  inject terminal escapes; the private attach returns a structured result instead of
  a string the adapter had to match. The adapter's helper entry sets `no_new_privs`.

### Verified

- On Linux, 2026-11-16: Rust tests 53 of 53; CTest 13 of 13 when configured with
  `-DSEABRAID_BUILD_NATIVE_TESTS=ON -DSEABRAID_BUILD_STRATA_ADAPTER=ON -DSTRATA_SOURCE=<strata checkout>`
  (8 without the Strata adapter: provider smoke, placement smoke, diagnostics smoke, minimal provider,
  SONAME, run from another directory, Rust core, Python linked; the Strata adapter adds static, private,
  helper-process, runtime and placement fixtures). An installed-SDK consumer and an ASan/UBSan build also passed.
- Live cross-vendor run, 2026-11-16: Strata on an AMD R9700 (HIP) with the Seabraid
  helper tier on an RTX 3090 (CUDA), Qwen3.8-Flash-Next IQ3_S. Greedy 2000-token decode
  at 89.5 to 95.0 tok/s over six runs; a SIGKILLed helper fails the in-flight request
  in 0.33 to 0.44 s and the next request succeeds after 1.16 to 1.21 s, with the
  engine still serving.
- ggml Q4_0 fixture on CUDA (RTX 3090), HIP (R9700) and Vulkan (both cards).

### Fixed

- Strata adapter: a killed helper retired the lease only after its last thread
  exited (0.55 to 0.63 s for a 22 GB CUDA helper), and teardown then waited out a
  300 ms SIGTERM grace and the kernel's cleanup before the engine could report the
  error. The provider now also watches `/proc/<pid>/stat` and retires the lease
  when the main thread exits; teardown of such a helper sends SIGKILL at once and
  reaps it in the background after 20 ms.

### Not yet run

- llama.cpp loaded-model parity.
- Windows handle handoff.

### Removed

- The `SEABRAID_STRATA_PTRACER` debug switch (patch 0015).
