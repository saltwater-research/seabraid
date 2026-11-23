# Strata weighted decode adapter

| File | Owns | Must not contain |
|---|---|---|
| `static_qwen.hpp` / `static_qwen.cpp` | Exclusive, fixed-layer translation from Seabraid's C provider interface to a prepared Strata channel | Expert kernels, router, placement, loading, engine-thread state, fallback |
| `static_qwen_test.cpp` | v7 request/reply and rejection fixture, using a scripted peer on the real channel layout | Claims of GPU or model parity |
| `private_qwen.hpp` / `private_qwen.cpp` | Dedicated Qwen process, descriptor handoff, one plan and reaping | Wire execution, placement policy, rediscovery or retry in the old mapping |
| `session_v2_test.cpp` | `seabraid_session_open`/`_submit` through the provider: stale epoch refused, COMPACT_ROWS output equals full output | GPU math |
| `qwen_stale_epoch_test.cpp` | Decode tier keeps a stale-epoch layer local and nothing else (busy/not-prepared refusals are failures); `compact_rows` launch option opens | Helper protocol parity |
| `qwen_edge_test.cpp` | Decode-tier and runtime edge cases against the scripted helper: view replaced by "none" after a nonzero epoch, windows of `kWindow` and `kWindow + 1` rows, row compaction with one, two and no kept rows, a NaN router weight, helper death mid-compacted batch | Helper protocol parity, GPU math |
| `static_qwen_peers.hpp` | Test-only in-process helper peer and geometry constants | Non-test use |
| `private_qwen_test.cpp` | Exec-based startup, timeout, reaping and replacement fixture using the helper's option parser, plus helper-path, environment and option admission (no Rust runtime) | GPU behavior |
| `private_qwen_runtime_test.cpp` | The same exec-based peer behind the Rust runtime: `QwenRuntime` routing and reaping, the weighted decode tier's claim and retirement, the private helper rung's slot table | Helper lifecycle and admission cases |
| `placement.hpp` / `placement.cpp` | Fenced Rust apply and executor export through acknowledged Strata `RungIO` calls | Scheduling, model identity, admission, a second ownership map |
| `placement_driver.hpp` / `placement_driver.cpp` / `placement_driver_open.cpp` | The cadence and the one fenced round: adopt, plan, apply, verify, publish; open and the first publish | Liveness or loss handling, a second ownership map |
| `placement_qual.hpp` / `placement_qual.cpp` | `SEABRAID_PLACEMENT_QUAL`: the qualification heat-bias file, parsed at open, added to the planner's heat only | Routing, anything outside qualification |
| `placement_qual_test.cpp` | Walk-file grammar, every refusal, the bias rule | Driver behaviour |
| `placement_liveness.cpp` | Liveness on every tick (`helper_alive` per open helper rung, F11), `degrade` (one rung closed, planning continues), `die` | Planning, adoption |
| `placement_round.hpp` / `placement_round_log.cpp` | Round counters, the pure table comparisons and the round line | ABI calls, state |
| `placement_trace.hpp` / `placement_trace.cpp` | `SEABRAID_TRACE` / `SEABRAID_TRACE_EVERY`: the JSON-lines move ledger and latency trace (record schema, on/off switch, window statistics, call-to-move attribution); fail-open-loud | Serving or placement decisions, engine includes |
| `placement_driver_trace.cpp` | Which driver events become trace records (round, helper up/closed/served_after_loss, per-move times, fallback counter deltas) | Record formatting, the round itself |
| `placement_trace_test.cpp` / `placement_driver_trace_test.cpp` | Record formats, fail-open, window math; every outcome kind's line and record through a scripted driver | GPU or helper parity |
| `placement_liveness_test.cpp` | Three helpers, two successive losses: survivors keep planning and keep being checked | GPU or helper protocol parity |
| `file_moves.hpp` / `file_moves.cpp` | File-reference moves, engine side: the pack manifest (`PackSums`: load once, freshness, main's own read check, the cells that never move) and one helper's decision (`FileMoves`: every refusal, the helper's answers, the bad-sum switch-off, the counters and the `helper file io:` line); the engine facts arrive through `FileBackend` | The manifest format (the wire's `expert_sums.hpp`), the FileRef rules (`file_ref.hpp`), where an expert lives in the files (Strata's `file_extents`), the helper's read |
| `file_moves_test.cpp` / `placement_file_moves_test.cpp` | Every refusal and answer on a fake backend, the manifest load's WARN-off, the counter line; then the real runtime through the driver: by-reference commit, REFUSED takes the byte path in the same swap, FAILED rolls back, a bad sum clears the helper's mask bit | GPU or helper parity |
| `engine_flags.hpp` / `engine_flags.cpp` / `engine_flags_test.cpp` | The adapter's own engine flags (`--seabraid-compact-rows 0\|1`, default off): spelling, values, the error | The `--retier-*` flags (compat's `retier_loop`) |
| `wire/strata/seabraid/file_stamp.hpp` | The freshness stamp of a pack's files (size and mtime, table order): the one formula for the manifest writer and the engine's check at start | Hashing, the manifest bytes |
| `placement_test.cpp` | Scripted same-layer slots behind the real `RungIO` interface and an engine publication fence | GPU or helper protocol parity |
| `diag.hpp` / `diag.cpp` | The adapter's only log sink (atomic function pointer, stderr default) and status names | Anything that calls `seabraid_log`, or log text with paths, environment values or tensor data |
| `diag_seabraid.hpp` / `diag_seabraid.cpp` | The one forwarder into `seabraid_log`, linked only by runtime-linked targets | Log call sites |
| `diag_capture.hpp` | Test-only capturing sink and `require` assertions | Non-test use |
| `spawn_policy.hpp` / `spawn_policy.cpp` | Helper-path trust rules (checks on an `O_PATH` descriptor of the vetted inode) and the child environment allow-list | Process creation, reaping |
| `helper_process.hpp` / `helper_process.cpp` | The one place the helper is spawned (exec of the vetted descriptor, descriptor closing, pidfd), signalled and reaped | Wire access, path vetting, plan management |
| `helper_exit.hpp` / `helper_exit.cpp` | The non-consuming exit probe (pidfd and `/proc/<pid>/stat`) with its log text (`signal N` / `status N`) | Syscall wrappers, reaping, spawning |
| `helper_kernel.hpp` / `helper_kernel.cpp` | The pidfd syscall wrappers (`pidfd_open`, `waitid(P_PIDFD)`, `pidfd_send_signal`), the launch gate (`pidfd_supervision_fault`, `pidfd_launch_gate`: no pidfd signalling, no launch), `names_unreaped_child`, `retry_eintr`, and the `KernelCalls` test seam | Exit parsing, reaping, spawning, any `kill(2)` (F1: never signal or reap by number) |
| `wire/strata/seabraid/channel_layout.hpp` | The ONE owner of the helper wire's bytes: constants (v8: `kTierSwapBlob`, `TierWireOp::blob_slot`), `Header`, regions, `Layout::of`, `ChannelLimits` (pinned by compat's channel golden files) | Operations, process control |
| `wire/strata/seabraid/channel.hpp` | The wire's operations: create/connect/map and the doorbells (includes `channel_residency.hpp` at its end, so the include spelling is unchanged) | Engine or helper policy |
| `wire/strata/seabraid/channel_residency.hpp` | The tier-residency publish (helper) and consistency check (main): free functions over `TierWireOp`, split from `channel.hpp` for the size limit | Engine or helper policy |
| `wire/strata/seabraid/finite_rows.hpp` | Checks an owned binary32 reply for NaN and infinity under fast-math, for both named and private decode | Copying shared memory or choosing a recovery policy |
| `wire/strata/seabraid/peer_watch.hpp` | Which process a helper is and when the engine learns it is gone: `announce_identity` (the helper writes `helper_pid` and `helper_start` before `kHelperUp`) and `PeerWatch` (pidfd opened first, starttime compared, one thread polling the pidfd and the stat descriptor every 100 ms, one atomic; wire v10) | Closing a card, any signal to a process, the CPU fallback |
| `wire/strata/seabraid/proc_stat.hpp` | The grammar of one `/proc/<pid>/stat` line: state, starttime (field 22), exit status (field 52) and the wait-status decode; shared with `helper_exit.cpp` | File or syscall access |
| `wire/strata/seabraid/pidfd.hpp` | The raw `pidfd_open` syscall (`sys_pidfd_open`, `SEABRAID_STRATA_HAVE_PIDFD`), named nowhere else (`check_pid_probes`) | Waiting, signalling |
| `wire/strata/seabraid/channel_names.hpp` | The SEABRAID_STRATA_SHM channel list grammar and cap: `parse_channel_list` (at most three; the engine's `channel_name` picks entry card = rung - 1) | Channel mapping, engine policy, the lookup itself |
| `channel_names_test.cpp` | Wire v8 host fixture: list parse and refusal cases, `blob_slot` rule, blob-swap residency replay | GPU or engine behaviour |
| `peer_watch_test.cpp` | Host fixture W1-W4 over real forked children: the stat grammar, every `PeerWatch::start` refusal, detection timing and the destructor's join, and the publication order (10,000 iterations plus a control) | GPU, the engine, closing a card |
| `wire_v10_test.cpp` | Wire v10 host fixture (W5): the layout golden `wire_v10_golden.txt` and the refusal of a v9 peer naming both versions | FileRef rules (`wire_v9_test.cpp`) |
| `wire/strata/seabraid/expert_spec.hpp` | The v6 engine spec region (`LayerSpec`, `RoleSpec`, path table) | Pack parsing |
| `cmake/seabraid_harden.cmake` | The one definition of the hardened compile and link flags, used here and by compat's `seabraid-helper` | Target lists |
| `wire/helper_options.hpp` | The real helper's CLI admission (named or private transport), private-descriptor close, controller watch, and `forbid_new_privileges` (`PR_SET_NO_NEW_PRIVS`, the helper's first act; failure is logged and the helper continues) | Channel math, GPU work |
| `unique_fd.hpp` | The adapter's only descriptor RAII wrapper | Anything else |
| `helper_process_test.cpp` | Inherited-descriptor, swapped-file and pidfd-teardown fixture over a scripted child | Channel or GPU behavior |
| `test_support.hpp` | Test-only `check` and the scratch-executable guard (removed on failed checks too) | Non-test use |
| `README.md` | Host integration and verification instructions | Default wiring |
| `CMakeLists.txt` | Optional Linux static library and CPU fixture | Downloads, a Strata build, Rust or GPU build steps |
| `qwen_decode.hpp` / `qwen_decode.cpp` | Strata's serial weighted ticket, disjoint pick mask, reusable buffers, and retired helper state | Request rollback, model routing, GPU math, tier moves |
| `strata-v7-seabraid.patch` | Optional v7 engine wiring as a two-commit mbox: (1) the private tier, checkpoint validation and the serial verifier boundary; (2) the expert pyramid ladder (k helper rungs from the SEABRAID_STRATA_SHM list, `FallbackRung`, `replace_resident_from`, blob swaps, RAM_ADAPT guards, D4/legacy refusals with k > 1) | New routing policy, changing the default build, upstream Windows support |


The adapter is the leaf that knows both Seabraid's ABI and Strata's v7 channel, and it
owns that channel: `wire/` is the one definition of the helper wire, which compat's
helper and (through patch 0001's include path) the patched Strata engine include.
The order is runtime <- adapters <- compat <- patched Strata; nothing in this
directory needs a compat-patched tree for the wire. Seabraid never imports it. The host owns admission and model identity. For the external
channel entry point, it also owns helper bootstrap and shutdown.
`StaticQwenProvider::bind` takes a connected channel for one loaded
helper generation and fixes the layer and deadline. `config()` gives the matching
Seabraid config; `provider()` lends the callback and context to exactly one session,
with the adapter alive until `seabraid_close` returns. Each source targets 200 lines
(hard cap 500; split wire publication from binding before 400). Guard functions
target complexity <= 10 and nesting <= 4; separate shape, residency and wire
state checks. Before growing `bind`, isolate the handoff drain check; before
growing `extents_match`, separate counts from pointer preconditions; before
growing `run`, isolate slot admission from the publish/wait/copy sequence.

The boundary fails closed without publishing unsupported batches or returning
partial rows. Any wire failure retires the adapter: no retry can overwrite a
request still used by the helper. The fixture checks complete weighted replies,
shape and residency refusals, stale/bad replies and deadline quarantine; compile
with warnings as errors. Run the bundled complexity checker and use Lizard for
C++, which the bundled checker does not parse. Remove an output or retirement
guard to prove the tests catch it. Finish means a callback that speaks the real v7
contract and a fixture that distinguishes host protocol evidence from GPU
evidence. Full GPU parity is a separate gate.

The route follows Strata `src/core/remote_experts.cpp` (`seabraid_helper_begin` and
`seabraid_helper_finish`) and `strata-ext/src/helper.cpp` (weighted decode reply). It uses
the pinned `Channel` rather than copying offsets or atomics; the helper remains
the only owner of weighted GPU math. No `RemoteExperts` object crosses the Seabraid
worker boundary: tier fences can depend on Strata's engine thread. The v7 wire
does not arbitrate producers, so a live handoff is unsafe unless the controller
has stopped all other publishers. If Seabraid adds a layer-aware or partial-output
ABI, revisit the fixed-layer and all-resident checks together with the fixture.

The optional build takes the patched source (a Strata tree with the `compat/strata` series and the v7 patch applied) and its generated geometry include
as inputs because building Strata would select devices and toolkits outside the
adapter's contract. Keep the build description under 80 lines; reject missing
inputs and non-Linux targets. The finish check is CMake configure and library
compilation with explicit inputs, with CTest as the native fixture entry point.

`PrivateQwenWorker::start` returns one exclusive provider after the dedicated
helper loads its plan. The controller depends only on Linux process APIs, the
pinned `Channel` and the existing adapter. The real helper's
`wire/helper_options.hpp` owns named/private CLI admission
and private descriptor close, so both transports enter the same service loop.
The controller reports startup, load and geometry failures and reaps all
children it starts. It creates a fresh memfd on every start because a late
reply into an old slot cannot affect another generation. The exec fixture
checks the transport selection, plan/state exchange, timeout quarantine and
process cleanup. The controller, helper options and fixture each target 200
lines, with a hard cap of 500. Separate child spawning/reaping from plan
management before 400 lines, and check new functions with Lizard at CCN 10.

`apply_placement_at_fence` borrows the caller's placement handle, Rust ticket,
stable RungIO bindings and validated layer slot sizes. The engine passes staging
space and retains admission ownership until it publishes the returned executors.
Geometry is checked before provider I/O because discovery after a physical swap
would leave dispatch with an obsolete table. Each callback translates the native
acknowledgement explicitly. An exception quarantines the affected card because
completion is uncertain. The caller keeps the engine fenced on an ABI error.
Rust owns swap order, restoration and quarantine; the adapter keeps no map.
The bridge, header and fixture each target 175 lines, hard cap 500. Split slot
translation from orchestration at 400. Check CCN <= 10 with Lizard and exercise
ordered physical transfers, restoration, quarantine and executor publication
through the linked `strata_placement` fixture.

Every failure branch logs exactly once through `diag`, at the site that knows
the cause: `bind` refusals, extent and residency refusals, each retirement
reason, startup stages, teardown escalation, placement provider failures and
the decode tier's fall-back transitions. Counts, ids and status names only.
`checked_rung` bounds the runtime-chosen rung before indexing the card table
and fails closed. `spawn_policy.cpp` refuses a relative, unresolvable,
non-regular, foreign-owned, group/world-writable or non-executable helper and
spawns the resolved file with an allow-listed environment. Checkers: the three
CTest fixtures (log records captured and counted; a helper that exits if it
sees a non-allow-listed variable) and Lizard at CCN 10.

Helper liveness and spawn hardening. `PrivateQwenWorker` lends the provider the
helper's pidfd (`watch_helper`); the provider's reply wait probes it once per 256
predicate calls (a `waitid(P_PIDFD, WNOWAIT)` syscall, never a reap), so a killed
helper retires the lease in milliseconds with the single record
`retired at layer N: helper exited (signal S | status S)`. A reply that landed
before the death still wins; the next layer finds the corpse. `collect` copies the
reply rows and rejects any NaN or Inf (bit test on the exponent, immune to
`-ffast-math`), zeroing the output and retiring with a logged reason.
`HelperProcess::spawn` owns three hardening decisions, each with a one-sentence
WHY at its site: it closes every inherited descriptor but stdio and the handoff
(closefrom on glibc 2.34+, enumeration fallback), execs the vetted inode through
`/proc/self/fd/N` instead of re-resolving the path, and signals and reaps through
a pidfd opened right after the spawn instead of a bare pid. Checkers: the
`strata_helper_process` and `strata_private_qwen` CTest cases; each fails when
its one line is reverted. Functions stay at CCN <= 10 (Lizard) and every file under
500 lines.
