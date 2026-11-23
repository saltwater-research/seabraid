# Strata Qwen v7 provider

This C++ adapter sends a Seabraid batch to the existing `seabraid-helper` weighted
decode path. The helper's `RemoteExperts` and `RemoteExpertOpt` execute and
reduce on its GPU; the adapter only packs the request and accepts the reply.

## Dedicated Linux worker

For an engine caller, `QwenRuntime::start` opens the Rust
session in one `seabraid_session_open` call (a v3 provider carries the geometry), binds it to the private worker, and arranges for the session to
drain before the helper is stopped and reaped. Link `seabraid_strata_runtime`
from the root CMake build with `SEABRAID_BUILD_STRATA_ADAPTER=ON` and the
`STRATA_SOURCE` settings below. Submit the layer index with the routed
batch, then collect that ticket before merging output into the engine's
activation rows. Keep the engine's fallback policy at that merge.

Build the real `seabraid-helper` against the pinned Strata tree with the Seabraid
patches and its GPU backend. Match its geometry header and `STRATA_VERIFY_MAX_T`
in the adapter build. The dedicated controller supports Strata Qwen geometries
with up to ten picks per token, which is the capacity of the helper's
`RemoteExperts` buffers. The host selects the helper executable, device, expert
pack, optional native GGUF shard, and ranked placement plan. Match those
artifacts to the host model because v7 carries shape and engine tags and has no
model fingerprint.

1. Include `private_qwen.hpp`. Fill `PrivateQwenLaunch` with the executable and
   model paths and `PrivateQwenPlan` with ranked `{layer, expert}` entries, the
   slot limit, and Strata's optional auto-sizing. Call
   `PrivateQwenWorker::start(launch, plan, layer, decode_timeout_ms, error)`.
2. The controller creates a fresh memfd, starts the Qwen helper with one
   explicit `--shm-fd` handoff, and attaches to its size-sealed mapping. It
   checks the direct child's identity and geometry, publishes the plan after
   `kHelperUp` and binds at `kLoaded`. Use close-on-exec for other host
   descriptors. Leave the child waitable and let the controller reap it.
   The helper watches the controller process through a pidfd before opening
   the GPU. Controller exit ends the helper even without destructors. The
   private launcher needs Linux 5.4 or newer (pidfd_open, waitid(P_PIDFD) and pidfd_send_signal; container seccomp profiles must allow all three).
   The launcher refuses any helper that is not an absolute path to a regular
   executable owned by the current user or root and not group- or
   world-writable (a default `umask 002` build leaves binaries group-writable;
   `chmod g-w` the installed helper). It resolves symlinks, opens the result
   `O_PATH|O_NOFOLLOW`, runs the owner, mode and executability checks on that
   descriptor, and executes that same inode through `/proc/self/fd/N`, so a file
   swapped in after the check is not what runs (this needs `/proc`; without it
   the launch fails closed). The child inherits stdio and the one `--shm-fd`
   descriptor only: everything else is closed in the child with
   `posix_spawn_file_actions_addclosefrom_np` (glibc 2.34 or later) or, on
   older glibc, by closing what `/proc/self/fd` lists at spawn time. The
   fallback cannot close a descriptor that another thread opens between that
   listing and the spawn, so keep close-on-exec on every other host descriptor.
   The launcher holds a pidfd for the child from the spawn and signals and reaps it only
   through that pidfd. With glibc 2.39 or newer the pidfd comes from `pidfd_spawn` itself; on
   older C libraries the launcher spawns, then opens the pidfd and checks it names its own
   unreaped child, so a host thread that reaps arbitrary children (`waitpid(-1)`) could in
   principle let a reused process ID pass that check (known limit; use glibc 2.39+). Where the kernel or a seccomp profile refuses `pidfd_open`,
   `waitid(P_PIDFD)` or `pidfd_send_signal`, the launch is refused before any process
   exists; there is no `kill(pid)` or `waitpid` fallback, because a pid can be recycled by
   the time a fallback fires. The helper inherits only exact environment names listed in
   `spawn_policy.cpp`. These include trusted loader paths (`PATH`, `LD_LIBRARY_PATH`),
   locale and cache directories, CUDA and HIP device selection, and the supported kernel
   and channel settings. With `debug_env`, the launcher also passes the named diagnostics
   and card benchmark controls. Unlisted names and the server's `STRATA_API_KEY` are
   excluded in both modes. The launcher reserves a reaper before creating the helper.
   After the 300 ms SIGTERM grace and 20 ms SIGKILL wait, that worker can finish collection
   even under descriptor or thread exhaustion. If the kernel denies a signal or wait, the
   worker retains the pidfd and retries. Keep the adapter loaded while retirement is pending.
   The vetting also refuses a helper with a setuid or setgid bit and one in a
   world-writable directory that lacks the sticky bit (a swap there would not need
   write access to the file). It still does not check the other parent
   directories; install the helper where only you can write. A spawn is refused
   while `SIGCHLD` is ignored or `SA_NOCLDWAIT` is set, because the kernel would
   then reap the child and its pid could be reused, and a descriptor handoff whose
   source and target descriptors are equal is refused. On a kernel older than 5.4,
   or under a seccomp profile that removes `pidfd_open`, `waitid(P_PIDFD)` or
   `pidfd_send_signal`, the launch is refused before any process exists, with one
   error naming the refused call ("this kernel or seccomp profile refuses pidfd
   signalling; the private helper needs it"). There is no `waitpid`/`kill(pid)`
   fallback: a pid can be recycled by the time a fallback fires.
3. Open the session with `worker->adapter().session_provider(config, compact_rows)` and
   `seabraid_session_open` (the supported path; the bundled `QwenRuntime` does exactly this),
   and submit batch v2 with `seabraid_session_submit`. The older `provider()` with
   `seabraid_open` and `seabraid_prepare` (one v1 lease serving the layer selected at startup)
   is deprecated and removed at 1.0. The runtime submits batch v2 with `placement_epoch = ExecutorView::epoch()`:
   a stale epoch is refused with `SEABRAID_BUSY_OR_STATE`, and `QwenDecodeTier::begin` keeps that
   layer local (one log record, tier not retired) only when the tier's own record of the highest epoch its
   session accepted proves the batch is older; any other `BUSY_OR_STATE` (busy, closing, not prepared, a
   pending ticket) is an ordinary submit failure. A batch that would carry epoch 0 after the
   session accepted a nonzero one (the host detached the view) is kept local the same way, with
   its own once-logged record, instead of failing the layer. `PrivateQwenLaunch::compact_rows` (default off)
   adds `SEABRAID_PROVIDER_COMPACT_ROWS`, so the helper receives only rows with a real pick; the engine
   switch is `--seabraid-compact-rows 0|1` (`engine_flags.hpp`; the v7 patch passes it to the launch).
   Keep the worker alive through `seabraid_close`. (The bundled `QwenRuntime` uses
   `seabraid_close_timeout` with the decode deadline plus 5 s, and leaks the worker
   if that times out, because the detached session worker may still call the
   provider.) Then destroy it to stop and
   reap the helper. Close sends SIGTERM and gives the helper 300 ms to exit
   before sending SIGKILL and waiting for reaping. A helper whose main thread
   has already exited gets SIGKILL at once instead; if it is still exiting after
   20 ms (a killed GPU helper can spend most of a second freeing device memory),
   a background thread finishes the reap so the caller does not wait. Startup or
   plan failure also runs this cleanup before returning the error.
   While the helper runs, the provider also watches it between spin and nap
   iterations, through its pidfd and through `/proc/<pid>/stat` (opened at spawn,
   while the pid is still an unreaped child). The pidfd fires only when the last
   thread exits; the stat view shows the main thread's exit at once. A helper
   that is SIGKILLed or OOM-killed mid-request retires the lease within
   milliseconds with one record such as `helper exited (signal 9), process still
   exiting`, instead of every layer waiting the full decode deadline.
   A helper reply carrying NaN or Inf in any output row retires the lease too;
   the rows are zeroed and never reach the merge.
4. After a wire fault, finish `seabraid_close`, destroy the worker, and call
   `start` again to load the plan in a fresh process and mapping. A late reply
   can only write the old mapping. The host decides when to start a replacement.

The helper still accepts its existing named `--shm` mode and defaults to
`/seabraid0` when neither transport is given. For an already-prepared Qwen v7
channel, hosts can also call
`StaticQwenProvider::bind(std::move(channel), layer, timeout_ms, error)`.
The host must stop and drain every other publisher before handing over a
prepared channel, and owns that helper's teardown and replacement.

Every assigned pick must be resident on that helper. The multi-operation
provider accepts `SEABRAID_SKIP_EXPERT_ID` with zero weight for picks assigned
to another device. A wholly skipped batch contributes zero and leaves the
channel idle.
Missing picks are rejected before publication, because returning the helper's
subset as the full result would silently lose contributions. The host still
selects the picks, keeps local and helper contributions disjoint, and combines
the results. The batch adapter leaves D4 prefill, tier transitions and promotions to
the existing Strata path. Do not run those publishers on the leased channel.

The adapter's helper deadline is independent of `seabraid_collect`'s ticket timeout.
If the callback cannot prove a reply is current and complete, it returns a
provider error without copying output and retires the lease. The controller
must retire or replace that helper generation, never reset the slot in place or
reattach to it after an adapter fault: a delayed helper may still complete that
request even though the Seabraid callback returned. Helper work only touches its own
shared mapping, so the callback never lends Rust's borrowed buffers to the GPU.

With Linux C++17 and a Strata checkout at the pin, run from the Seabraid repository
root. The helper wire is this adapter's own (`wire/`), so the decode, private-helper
and hardening fixtures need only upstream Strata headers; the runtime-linked
placement targets also need the engine side of the compat series (patch 0012's
`ownership_map.hpp`). The standalone build has no Rust or GPU build
dependency; the embedding host links both `seabraid_strata_static_qwen` and Seabraid.

```sh
cmake -S adapters/strata -B "$TEST_DIR" -DSTRATA_SOURCE="$STRATA_SOURCE"
cmake --build "$TEST_DIR"
ctest --test-dir "$TEST_DIR" --output-on-failure
```

When targeting a real helper, pass `STRATA_WINDOW_MAX_T` with the helper's
configured decode window. `auto` in Strata's CMake can resolve differently by
backend; use the numeric value it selects. If the helper has a generated
geometry header, pass `STRATA_GEOMETRY_INCLUDE` with the include root that
contains `strata/core/geometry_selected.hpp`. Or build the fixture directly:

```sh
c++ -std=c++17 -O2 -pthread -Wall -Wextra -Werror \
  -Iinclude -Iadapters/strata/wire -isystem "$STRATA_SOURCE/include" \
  adapters/strata/static_qwen.cpp adapters/strata/helper_exit.cpp adapters/strata/diag.cpp \
  adapters/strata/static_qwen_test.cpp \
  -o "$TEST_DIR/static_qwen_test" -lrt
"$TEST_DIR/static_qwen_test"
```

The fixed-layer fixture uses a scripted peer on a real private v7 mapping. It
exercises wire shape and copying, lease ownership, residency, bad replies and
late responses. The `strata_private_qwen` CTest case execs a separate scripted
peer through the real helper option parser. It checks startup and load exit,
geometry and plan rejection, descriptor close, escalation and reaping after a
timeout, a helper killed mid-request (provider failure in under a second, one
record), and a weighted reply from a fresh replacement. The
`strata_helper_process` CTest case launches a scripted copy of itself through
`HelperProcess` and proves that a non-close-on-exec descriptor is not inherited
(both closing strategies), that a file swapped after vetting is not what runs,
and that teardown signals through the pidfd even if the recorded pid names
another process. Non-finite reply rows are covered by `strata_static_qwen`.

Diagnostics: every failure branch writes one record through `diag.hpp`
(`seabraid-strata: <LEVEL>: <message>` on stderr by default). A host that links the
runtime gets the records in `seabraid_log` automatically: `QwenRuntime::start` and
`apply_placement_at_fence` install the forwarding sink unless the host already
chose one with `diag::set_sink`. Records carry codes, counts and ids only; the
helper's own failure text is returned to the caller in `error` but is not
logged, because it can name files. Retirement records name the cause (reply
timeout, epoch change, malformed reply, sequence fault); a fall-back to local
work is logged once per transition, not per call. The fixtures capture the
sink and fail when a record is missing, duplicated or leaks the helper text.
Helper options (`helper_options.hpp`) parse numbers with a whole-string,
range-checked parser: `--layers` 1..1024, `--experts` 32..512 (Strata's
geometry rule), `--device` 0..999, `--prefill-tokens` 0..INT32_MAX/16. The
helper prints the rejection (`seabraid-helper: --layers requires an integer
from 1 to 1024`) and exits 2.

To prove the guards bite, remove the `out_rows` comparison and
check for `reject wrong row count from helper`; restore it, then skip the
`retired_` check and look for `a bad reply retires the generation lease`.
To check the controller fixture, omit the child descriptor duplication and
look for a startup failure, then restore it. Remove SIGKILL escalation and
check that CTest times out against the peer stopped during a request. Restore
the guards and rerun. Before adopting the adapter in Strata's default route,
compare this route against
the current weighted decode path with the same prepared Qwen plan, held picks,
activations, weights and GPU build, then compare at the engine's layer join.
The Linux subprocess fixtures run in CTest (`strata_private_qwen`,
`strata_helper_process`, `strata_runtime_qwen`) and passed on Linux on 2026-11-16.
A layer-level Qwen GPU parity comparison against the weighted decode path has
not been run.

## Opt-in Strata source integration

Apply [strata-v7-seabraid.patch](strata-v7-seabraid.patch) after the whole
compatibility series, patches 0001–0006 and 0010–0020 (0007 is now the adapter-owned `wire/`; 0008 and 0009 do not exist any more), on the Strata commit pinned in `compat/strata/patches/strata/STRATA_BASE` (its one owner; v0.2.15): first
`git am compat/strata/patches/strata/*.patch`, then `git am` this patch. It is a
three-commit mbox: commit 1 is the private decode tier and the one-helper placement
driver; commit 2 ("the expert pyramid ladder") adds k helper rungs, the fallback rung
and blob swaps, and needs 0017; commit 3 (file-reference expert moves) needs 0018. That
order was verified in the 2026-11-15 live test on the previous pin (Strata v0.2.13.1, `628f74955`; the series applied in sequence and
the engine built and served) and again on v0.2.15 (see below). The 2026-11-19 and 2026-11-20 live runs built from the
whole mbox: L1 and L2 with the series through 0018, HL with 0019 and HL2 with 0020
([research log](../../docs/RESEARCH-LOG.md)). Applying it before 0017 or on a different pin is
unverified. It is separate from the compatibility installer
so default installations retain the named-channel path. Configure the patched
Linux GPU engine with `-DSTRATA_SEABRAID_SOURCE_DIR=/path/to/seabraid`. If building the
helper in another build tree, use the same Strata geometry and verifier
window limits. The Linux host needs pidfd support.

Use serial `--serve --remote-expert-opt` and one explicit
`--expert-cache-device1` slot count. Set `SEABRAID_QWEN_HELPER` to the patched helper
executable. The helper uses the engine's resolved expert pack and optional
native GGUF shard. Existing `SEABRAID_QWEN_PACK` and `SEABRAID_QWEN_NATIVE` settings
are optional. If set, they must point to the same filesystem objects as the
engine's paths. Symlink aliases work. A separate copy or a missing override
path prevents startup. The helper uses its backend's visible ordinal zero.

The private helper's experts stay in the engine's page-locked RAM copy, as the
named-channel helper's do, unless `SEABRAID_STRATA_EXCLUDE_HELPER_RAM=1`. The helper
cannot hand its blobs back to the prompt path, so without the RAM copy every
prompt read of those experts goes to the drive (2026-11-17, 32k prompt on an
R9700 main card: about 58 GB read and 907-1,002 tok/s, against 1,774-2,043
tok/s on the named channel, which keeps the copy). When the RAM budget cannot
hold everything, it takes the experts no GPU holds first, because decode reads
them, and the helper's experts after them, each group hottest first.
The first engine slice requires a compiled routing width that matches the
pinned Strata weighted reducer.
Keep the named helper, hot share, split stages, batch serving, and the
legacy re-tier modes disabled for this opt-in tier (live placement is
`--retier-promote seabraid`, below). Its prefill runs locally.

Choosing the `--expert-cache-device1` slot count. The installer's plan does not
compute one for this path, so derive it from the helper card:

- Per-expert bytes = the expert pack's bytes / (layers x experts). The installer
  plan prints both inputs on its `Model:` line (`L layers x E experts, experts X GiB`).
- Slots = floor((free VRAM on the helper card - headroom) / per-expert bytes).
  Read the free VRAM from the installer's hardware report or `nvidia-smi` with
  nothing else on that card, and keep a few GiB of headroom for the CUDA
  context and the helper's own buffers. Start low and raise it; the slot count
  is an explicit number: this tier takes one fixed count.
- The live test used 11000 slots for IQ3_S on a 24 GiB RTX 3090 helper. That is one
  measured value, not a rule; the failure mode of an oversized count was not probed.

GPU visibility. The helper is spawned with the allow-listed environment above,
so it sees only what the ENGINE's environment already carries. Set
`CUDA_VISIBLE_DEVICES` (NVIDIA helper) or `HIP_VISIBLE_DEVICES` (AMD helper) to
the helper card in the engine's own environment, for example its systemd unit.
The helper then uses that backend's visible ordinal zero. When the engine and the
helper use the same backend, the variable also hides the other cards from the
engine, so give the engine the card list it needs and check which ordinal
comes first.

Strata starts the helper after ranking the slots and excludes primary or
already claimed experts. It passes the same router weights its own reducer
uses and lets local CPU/GPU work continue during the helper request. A
helper completion failure in an explicitly discardable serial window drains
the verifier, ends the request with an error, and clears its live-cache
marker before the next request. The retired tier keeps its runtime and
mapping inaccessible to future decode work and releases the helper process
at the request boundary. The next request routes through local tiers.
Engine faults outside that serial recovery contract keep their ordinary
failure behavior.

Measured on 2026-11-15 on real hardware (R9700 main, RTX 3090 helper, IQ3_S): a
SIGKILL of the helper during a long generation errored the in-flight request,
retired the tier, and the next request succeeded on the local tiers. The
kill-to-next-success time was 34.6 s (one run, before the fail-fast liveness change
described above). Re-measured on 2026-11-16 with that change (same hardware and
model): the in-flight request ended with `SEABRAID_PROVIDER` 0.93 to 0.96 s after
the kill and the next request succeeded 1.51 s and 1.54 s after it. A trace of
that second showed the helper's main thread a zombie within 0.1 ms, but its
pidfd readable only after 0.55 to 0.63 s while the kernel freed 22 GB of device
memory, and teardown then blocked the error on a SIGTERM grace and that reap.
With the stat view and the background reap described above (three kills): the
lease retired 1 to 2 ms after the kill, the client received the error after
0.33 to 0.44 s and the next request succeeded after 1.16 to 1.21 s, with no
zombie left under the engine. Of the remaining time, about 0.3 s passes between
the engine printing `ERR` and the client receiving it, in Strata's server.
Each kill wrote four records (helper exit with signal, collect status, tier
retired, local fallback) through the forwarded sink, which prints them with the
runtime's `seabraid:` prefix. The engine kept running throughout. The Strata integration
named-channel path did not at that date (a lost helper froze the engine for 30 s and
ended it); patches 0019 and 0020 changed that, and the difference is in
`compat/strata/README.md`, known limits.

The CPU-only checkpoint fixture runs on Linux with `STRATA_BUILD_TESTS=ON`
using `ctest -R '^seabraid_checkpoint_test$' --output-on-failure` from the Strata
build directory. It covers both checkpoint paths and accepted aliases.

The complete series applies on the pin. (Historical, previous pin v0.2.13.1: on 2026-11-16 it applied without
rejects, the HIP engine (R9700) and CUDA helper (RTX 3090) built from it, and
the engine served greedy 2000-token requests at 89.5 to 95.0 tok/s decode
(six runs) and recovered from two helper kills as described above.)

On 2026-11-17 the same was repeated on Strata v0.2.15 (the `STRATA_BASE` pin of that day) from the repository's own patch files (plain `git am`, then this
patch), built with the installer's flags including `-DSTRATA_MMQ=ON`. R9700 HIP engine with the RTX 3090 CUDA helper: 94.8 tok/s mean
(sd 3.0, six 2,000-token greedy runs, 91.1 to 98.4; 94.75 on the previous pin). Roles reversed, RTX 3090 CUDA engine with the R9700
HIP helper (16,000 helper slots): 106.7 tok/s mean (sd 2.0, 104.6 to 110.2). Killing the helper with SIGKILL in the middle of a request
(three times per arm, a fresh engine each time) failed the request after 0.33 to 0.46 s (HIP engine) and 0.51 to 0.73 s (CUDA engine); a
fresh request was served 1.7 to 1.8 s and 2.5 to 2.7 s after the kill; no zombie was left. One machine, no repeat across days; the raw
output and the step timings are kept with the run records, not in this repository.

## Placement at the decode fence

`seabraid_strata_placement` links the Rust placement ABI and translates synchronous
slot transfers through the engine's `strata::core::RungIO` interface. The
engine owns the handle, verifies the initial slot layout, and submits all-rung
routing observations to the Rust planner. It calls `apply_placement_at_fence`
on the engine thread with a quiet window, all affected queues drained, and
admission closed. The staged executor table must have one cell per expert and
layer. Publish it and its epoch together before reopening admission, even when
a provider outcome is a rejection, rollback or quarantine. An ABI error keeps
admission closed until the engine can verify and publish a table or stop.

The legacy `--retier-promote` call site is in Strata
`src/program/generate.cpp` next to the `!ajob && pending.empty()` check.
Use that boundary to replace the legacy round scheduler and ownership map:

- Use the quiet window between decode submissions. Drain the lead GPU,
  adaptive copies, helper decode and D4 or blob readers before both stages.
  `RemoteExperts::seabraid_retier` drains its own pending decode and fences
  one wire operation. The outer engine fence has to cover both stages, both
  writes, possible restoration and the final route publication.
- Coordinate maintenance through the private worker's exclusive channel owner.
  The fixed-layer batch provider checks the channel epoch captured at bind,
  while Strata's `seabraid_retier` advances that epoch after acknowledgement.
  The existing decode lease cannot accept an external maintenance publisher.
- `PrimaryRung::replace` checks the lead slot by readback and uploads the
  residency table before acknowledging. `RemoteRung::replace` reloads
  from its own pack, so the two pack identities and per-layer bytes must
  match. The legacy wire lacks an in-place restore for a helper slot.
  A torn helper slot belongs in quarantine. For a lead quarantine, stop the
  engine until every lead route can honor the fallback table.
- `RemoteRung::stage` still reports a failed `copy_cached_blobs` as a
  refusal (patch 0012, `ownership_rungs.cpp`), and that read can time out after publishing its
  shared blob request. Since patch 0019 a missed deadline or a lost helper closes
  the card (codes 4 and 3) and `stage` refuses at once on a closed card, so a late
  reply cannot be reused. Treat inconsistent helper residency, an invalid
  acknowledgement and a torn helper slot as uncertain: they close the card or
  quarantine the slot.
- Switch decode, prefill, blob lending and adaptation together to the same
  published executor table or inhibit those paths during qualification.
  `RemoteExperts::prefill_usable` checks the card's closed state since patch 0019.
  The old engine derives routing from `host_res` and helper `held[]` and
  can hold physical duplicates. Require disjoint resident cards before opening
  the Seabraid handle. After a completed external adaptive fill, refresh ownership
  while fenced before planning. Out-of-band loss is a quarantine at the same
  fence. A replacement needs verified bytes and a new generation.

### Running the placement driver (`--retier-promote seabraid`)

Opt-in, never the default. `strata-v7-seabraid.patch` wires `PlacementDriver` into the
`generate.cpp` window-boundary hook next to the legacy ladder, so the two are
selected by one value and cannot both run. Which values each tier accepts:

| Tier | `--retier on` (two-set) | `--retier on --retier-promote legacy` (alias `on`) | `--retier on --retier-promote seabraid` |
|---|---|---|---|
| Named channel (`SEABRAID_STRATA_SHM`) | yes | yes | yes |
| Private Qwen helper (`SEABRAID_QWEN_HELPER`) | refused, exit 2 | refused, exit 2 | yes |

- `seabraid` reuses `--retier-every`, `--retier-moves`, `--retier-margin`,
  `--retier-cooldown` and `--retier-ceiling` and, like `legacy`, needs `--adapt-every`
  and `--adapt-swaps` above 0 for the heat counts. It is refused at startup (exit 2)
  with `--retier zero-op` (no moves would be sent), `--batch` above 1, an in-process
  remote GPU tier beside the helper processes, a layer split or `--peer-device`.
- The expert pyramid (patch commit 2): `SEABRAID_STRATA_SHM=/seabraid0,/seabraid1[,/seabraid2]`
  lists one channel per named helper process; entry r is helper tier r + 1 and
  Seabraid rung r + 1 for the session (at most three: Strata holds three remote tiers;
  each listed tier needs its `--expert-cache-device<r+1>`). `seabraid` drives all of
  them; `--retier on` and `legacy` refuse more than one helper. `--retier-fallback 1`
  plans swaps between the last helper and the fallback rung (RAM copy + the pack on
  NVMe; with `SEABRAID_STRATA_EXCLUDE_HELPER_RAM=1` an expert leaving for a card gives its RAM
  place to the one arriving), `--retier-hops N` lets one expert climb N boundaries in a
  round. Every helper write carries the engine's bytes (wire v8 `kTierSwapBlob`) and is
  read back by the helper before it acks; each committed hop logs
  `placement move L/E r->r src=card|fallback bytes=N verified=1` (`src=` is the rung the expert left: `fallback` when the lower rung of the hop is the fallback, else `card`; the diag line never says `file`, because tools parse it, and it stays so for a by-reference hop). Liveness is checked on
  every tick: losing one helper closes only its rung and planning continues among the
  rest (shown on GPUs on 2026-11-20, runs HL and HL2: the card closed 22 to 93 ms after a kill and the engine kept
  serving; [entry](../../docs/RESEARCH-LOG.md#2026-11-20--helper-loss-on-gpus-hl-hl2)). `STRATA_PREFILL_REMOTE_COMPUTE` (D4) with more than one helper process exits 2.
  The private helper (`SEABRAID_QWEN_HELPER`) stays one helper (k private helpers are
  phase 2). `SEABRAID_PLACEMENT_QUAL=<walk file>` is for qualification only: it biases
  the planner's heat (never routing) and every round says `qual=1`.
- File moves (patch commit 3, `--retier-file-moves 1`, default 0, needs the pyramid and a named helper): a swap whose
  upper write is a helper's may ask the helper to load the promoted expert from the model file itself (wire v9
  `kTierSwapFile`) instead of receiving its bytes, through Seabraid's `provider_v2` (`replace_from_fallback`; the rung
  mask is every helper rung that still reports `reads_files()`, recomputed before each apply). REFUSED means "use
  bytes" and the library runs the byte path in the same swap, so it is never logged as a failure. The pack's
  `expert-sums.v1` (built once by `seabraid-sums`) is loaded at the first quiet boundary: missing, malformed or stale
  turns file moves off with one WARN. A helper whose read disagrees with the manifest (`kTierBadSum`) is switched off
  for the session with one WARN and takes bytes only; a cell whose own read by the engine disagrees keeps serving,
  logs an ERROR and never moves. The cumulative counters are one INFO line after each worked round:
  `helper file io: direct_bytes=N buffered_bytes=N reads=N bad_sum=N`; a round that asked also writes
  `placement round=N fr=A fr_ok=B fr_bytes=C`; with `SEABRAID_TRACE` a by-reference hop's move record has
  `"src":"file"` (the trace record only; a by-reference hop's diag line still says `src=card|fallback`). The rule is `file_moves.cpp` (host-tested); `EngineFileBackend` in the patch only answers facts.
  The engine side (patch commit 3) was shown on GPUs in
  [live run L2, 2026-11-20](../../docs/RESEARCH-LOG.md#2026-11-20--live-run-l2-one-expert-through-every-level): 124 hops by reference
  (48 from the fallback to helper 2, 76 from helper 2 to helper 1) and 190 by bytes, 256,307,200 direct bytes in 372 reads,
  `bad_sum=0`; by-reference writes were 2.6 to 3.0 times slower than byte writes, because the helper reads the drive itself, and the main
  card's stage time was 12 to 15% lower. Decode-path KL with file moves live passed. L1 (2026-11-19) had not exercised them. One
  machine, load-contaminated timings; the catalog rows are in
  [OPTIONS.md](../../docs/OPTIONS.md#re-tiering-and-residency).
- Trace (opt-in): `SEABRAID_TRACE=<path>` writes one JSON line per move, round, window and helper up or
  down event (`placement_trace.hpp`); `SEABRAID_TRACE_EVERY` sets the latency window. `verified=1` appears
  only when the helper's read-back matched. A trace that cannot be written logs one warning and turns itself
  off. Read it with `compat/strata/tools/trace_report.py TRACE.jsonl [--log ENGINE_LOG] [--out DIR]`, which
  prints each moved expert's journey and the latency table (helper submit to collect, time blocked in collect,
  round, stage and write).
  No trace file from a live run exists yet, so no sample report is shown. The [live run L1, 2026-11-19](../../docs/RESEARCH-LOG.md#2026-11-19--live-run-l1-the-pyramid-on-real-cards) used lab-only log lines, not
  the shipped trace.
- The installer's `--ladder` passes `--retier on --retier-promote seabraid --retier-fallback --retier-hops
  --retier-moves --adapt-every --adapt-swaps` for the plan's helper count (`compat/strata/install/_ladder.py`);
  `--file-moves` adds the file-reference flags and needs `--ladder`. The installer builds or verifies
  the checksum manifest for that native shard and expert geometry before writing services.
  Both options default to off. See the [installer guide](../../compat/strata/README.md) for setup.
- The driver opens at the first quiet boundary (no adaptive job, no copy in flight),
  because the residency tables and `res_upload` are final only there. Until it
  publishes its first table no helper cell is claimed, so the first request's prompt
  and first window read the helper's experts from the RAM copy.
- The helper's `holds()` is `published table AND physical residency AND alive`.
  `RemoteExperts::physically_holds` is the ungated physical test the driver verifies
  the table against, and `prefill_usable` also requires the rung to be open.
- The private helper's slot size is per layer (mixed per-layer quants make expert blobs differ in size across layers), and opening the driver makes the per-round INFO record (`seabraid: INFO: placement round=...`) reach stderr, with WARN and ERROR kept.
- After a committed pair the engine moves the RAM copy with `replace_resident` so
  prompt reads of the helper's experts stay off the drive.
- The Strata-side `PlacementHost` and the `holds()` gate have no host test (they sit in
  a 20k-line `main`). The live invariants and the A/A arm of `compat/strata/scripts/placement-qual`
  are their checker.

The linked `strata_placement` CTest fixture uses scripted cards implementing
the pinned Strata `RungIO` interface. It checks byte order, both write
acknowledgements, restoration, malformed staging, and an uncertain helper
which has an old index but torn bytes. Run it from the root build with
`SEABRAID_BUILD_STRATA_ADAPTER=ON`, `BUILD_TESTING=ON` and the same
`STRATA_SOURCE`. To prove the publication check, remove the final executor
snapshot and verify the committed swap fails the fixture. Restore the snapshot
and rerun before live verification.
