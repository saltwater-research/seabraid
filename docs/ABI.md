# C ABI reference

Headers: [`include/seabraid/seabraid.h`](../include/seabraid/seabraid.h) (sessions, providers,
capabilities, diagnostics) and
[`include/seabraid/placement.h`](../include/seabraid/placement.h) (placement). Both are
available under Apache-2.0 OR MIT. The ABI is alpha and can change before 1.0.
For a walkthrough, read the [adapter guide](ADAPTER-GUIDE.md) first.
The ABI targets 64-bit platforms only: `seabraid.h` refuses to compile when `sizeof(void *)` is not 8.

## Which number means what

| Name | Today | What it tells you | Changes when |
|---|---|---|---|
| `SEABRAID_ABI_VERSION` | 1 | The layout generation of the original structs (config, v1 batch, v1 provider, capabilities, status, placement). | Never for an existing struct; a new layout gets a new struct and its own number. |
| `SEABRAID_ABI_EPOCH` / `seabraid_abi_epoch()` | 1 | Whether a consumer and the library agree on every struct layout. A mismatch means refuse to load. | Only for a non-append change. Appends ride on `struct_size` and leave it alone. |
| Per-struct `abi_version` | 1, 2 or 3 | Which layout that one struct has: `seabraid_provider_v1` 1, `seabraid_provider_v2` 2, `seabraid_batch_v2` 2, `seabraid_provider_v3` 3, every other struct 1. | When a struct is replaced by a new struct. |
| `struct_size` | per struct | How many bytes of an appendable struct the caller has (caller-filled) or the library wrote (library-filled). | Grows when fields are appended. |
| SONAME `libseabraid.so.0.5` | 0.5 | Which library file a consumer was linked against. | Every minor bump while the major version is 0. |

"batch v2" and "provider v3" are names of the newest structs and of the two entry points that use them
(`seabraid_session_open`, `seabraid_session_submit`). They are not a library-wide ABI version.

General rules:

- Every function returns `int32_t` status unless noted. `SEABRAID_OK` is zero.
- Calls on one session or one placement handle must be serialized by the
  caller. Different handles are independent. The library detects the common
  violations and answers with a status instead of corrupting state: a call that
  overlaps another call on the same session from another thread returns `SEABRAID_BUSY_OR_STATE`
  and does nothing, and a closed or foreign session or placement handle returns `SEABRAID_INVALID`.
  Any call on a session from its own provider callback (the worker thread), including
  `seabraid_close` and `seabraid_close_timeout`, returns `SEABRAID_BUSY_OR_STATE` with a logged
  reason and changes nothing, whether or not the caller is inside a call: it would wait for the
  thread it runs on. Call from another thread. The check compares thread ids, so it only sees the
  worker thread itself: a provider that hands work to a thread of its own and calls into the session
  from there is not detected, and a close from that thread waits for the worker (misuse; do not).
  This is best effort. It reads the memory behind the handle, so a double close
  or a garbage pointer is still undefined behavior whenever that block was freed
  and reused or is unreadable, and the detection never replaces serializing. "A closed handle
  returns `SEABRAID_INVALID`" holds only while the freed block still reads as retired: once the
  allocator reuses it, a stale pointer can look live (or be unreadable), and the call is then
  undefined behavior, not a status.
  Residual, stated plainly: the busy flag and liveness cookie are atomics in a header beside the state a
  running call holds by `&mut`; no reference to a whole handle is formed. Miri reported no undefined
  behavior in the FFI tests under both Stacked and Tree Borrows (2026-11-19, on a test machine; the CI job
  runs both), but it sees only what the tests exercise: the detector is a diagnostic for caller bugs, not a
  proof of memory safety.
- Pointers must be valid and aligned for the duration of the call. Seabraid copies
  what it keeps; it never retains a caller pointer after a call returns,
  except the provider `context`, which must stay valid until `seabraid_close`.
- Callbacks must return, must not throw across the ABI, and must not call back
  into their own session. They run on one dedicated worker thread per session; see [Threads](#threads).

## Versioning

Every struct that crosses the ABI starts with `uint32_t abi_version`, that struct's layout
generation (not a library-wide number). Existing values never change meaning; a new layout
gets a new struct and a new value. Today `seabraid_provider_v1` carries 1,
`seabraid_provider_v2` carries `SEABRAID_MULTI_PROVIDER_VERSION` (2),
`seabraid_batch_v2` carries `SEABRAID_BATCH_V2_VERSION` (2) and `seabraid_provider_v3`
carries `SEABRAID_PROVIDER_V3_VERSION` (3); every other struct carries
`SEABRAID_ABI_VERSION` (1).

A struct the **caller allocates for the library to fill** (`seabraid_capabilities_v1`,
`seabraid_session_status_v1`, and the placement ticket, outcome, applied and snapshot
structs) also has `uint32_t struct_size` second. The caller sets both fields before the
call. The library returns `SEABRAID_INVALID` (logged) when `abi_version` is unknown or
`struct_size` is below the v1 size. It writes `min(caller struct_size, its own current size)`
bytes and sets the `struct_size` field of the caller's struct to that count, so an older
caller is never overrun and never told about bytes it does not have, and a newer
caller's unknown tail is left untouched. A newer caller reads `struct_size` back to
learn which tail fields were filled. `struct_size` must also be a multiple of the struct's alignment
(8 for every such struct except `seabraid_capabilities_v1`, which is 4), as the rule for the structs the
caller fills already required: a size that would end inside a field (50 for the 64-byte status) is
`SEABRAID_INVALID` and writes nothing, it is never rounded. Padding bytes inside what the library writes
are zero. In an array of such structs
(`seabraid_placement_apply` outcomes) the element stride is `struct_size`, read from
element 0, which the caller initializes. Like any library-filled struct, each element's `struct_size` is
overwritten with the bytes the library wrote, `min(stride, 64)`: a caller whose struct is wider (72, say) finds
64 there after the call and must set element 0 back to its stride before it reuses the array, or the next call
strides by 64. The bytes past 64 in each element are never touched.
Structs the caller fills for the library to read (`seabraid_batch_v1`, config, layout,
policy, providers v1 and v2) carry `abi_version` only. The two structs of the batch v2 / provider v3 entry points that the
caller fills (`seabraid_batch_v2`, `seabraid_provider_v3`) are the exception: they also
carry `struct_size` second, so a later library can append fields. See
[Batch v2 and provider v3 entry points](#batch-v2-and-provider-v3-entry-points). `seabraid_placement_issue_v1` is embedded in
the outcome and has no header of its own.

### Layout epoch

`uint32_t seabraid_abi_epoch(void)` returns the layout epoch the library was built with;
the header declares the same number as `SEABRAID_ABI_EPOCH`. The rule, written once:
**the epoch changes only for a non-append change; appends ride on `struct_size`.**
(`seabraid_session_status_v1` grew from 48 to 64 bytes with the epoch still 1.) A consumer must compare the two at start-up and refuse on a
mismatch: the Python binding does this on load (an `OSError`), and C/C++ embedders should
do the same. The CMake package exports `Seabraid_ABI_EPOCH` for build-time checks. Every
size and offset is recorded in [`src/layout/abi_layout.txt`](../src/layout/abi_layout.txt);
the C compiler, rustc and the Python tests all check against that one table.

The epoch and the SONAME below are the compatibility contract; the version string is only a label.
A consumer checks both, at build time and at start-up:

```cmake
find_package(Seabraid 0.5 CONFIG REQUIRED)     # same minor series (SameMinorVersion); sets Seabraid_ABI_EPOCH
```

```c
if (seabraid_abi_epoch() != SEABRAID_ABI_EPOCH) return 1;   /* header and library disagree: refuse */
```

### Library file names

While the major version is 0 any minor bump may break the ABI, so the Linux SONAME is
`libseabraid.so.MAJOR.MINOR` (today `libseabraid.so.0.5`): a consumer built against 0.5
will not load 0.6. The installed file has that name, with `libseabraid.so` as the
development symlink. macOS keeps `libseabraid.dylib` and sets a compatibility version of
`MAJOR.MINOR.0`; dyld only refuses a library whose compatibility version is lower than the one the
client recorded, so a client built against 0.5 still loads a 0.6 library, which Linux refuses.
That is a weaker guarantee than the SONAME gives, so on macOS the start-up `seabraid_abi_epoch()` check
is the only enforcement (a versioned macOS file name would match Linux, but it is not built because it
cannot be tested without a Mac). Windows keeps `seabraid.dll`, which carries no ABI version, so check
`seabraid_abi_epoch()` there.

Two version-number limits of the CMake package: CMake's version has no pre-release part, so
`0.5.0-alpha.8` is `0.5.0` to `find_package` (use `seabraid_version()` to tell builds apart); and the
project declares no languages, so the package version file skips its 32-bit/64-bit check, which the
header's 64-bit-pointer typedef does instead. Only Linux builds record a SONAME; other ELF systems get a plain
`libseabraid.so`. The installed library is not stripped by `cmake --install --strip`; release builds are
already stripped of non-exported symbols by Cargo (`[profile.release] strip`).

## Status codes

| Code | Name | Meaning |
|---|---|---|
| 0 | `SEABRAID_OK` | Success. |
| 1 | `SEABRAID_INVALID` | Bad argument, shape or geometry. Nothing happened. |
| 2 | `SEABRAID_BUSY_OR_STATE` | Wrong state: a ticket is outstanding, no ticket is outstanding when one is collected, or a placement epoch is stale. A session ticket that is not the outstanding one returns `SEABRAID_INVALID`; a placement ticket that is not the pending plan (stale, unknown, already applied or discarded) returns `SEABRAID_BUSY_OR_STATE`, because a plan ticket is single-use and its plan can be superseded or voided by an epoch change. |
| 3 | `SEABRAID_TIMEOUT` | The wait timed out. The ticket is preserved; wait again. |
| 4 | `SEABRAID_PROVIDER` | The provider callback returned nonzero or panicked, or its worker thread stopped. |
| 5 | `SEABRAID_INTERNAL` | An internal failure, including a panic caught at the boundary and the library's own bookkeeping around a callback failing (for example no memory for the output scratch, which also counts in `provider_failures`). |

## Sessions

```c
typedef struct seabraid_session seabraid_session;

int32_t seabraid_session_open(const seabraid_provider_v3 *provider, seabraid_session **out);
int32_t seabraid_session_submit(seabraid_session *session, const seabraid_batch_v2 *batch,
                            uint64_t *out_ticket);
int32_t seabraid_collect(seabraid_session *session, uint64_t ticket, uint32_t timeout_ms,
                     float *output, size_t output_count);
int32_t seabraid_close(seabraid_session *session);
int32_t seabraid_close_timeout(seabraid_session *session, uint32_t timeout_ms);
```

A session owns one dedicated worker thread and one in-flight batch.
`seabraid_session_open` validates the provider descriptor, copies it and its config
(`seabraid_config_v1`: `hidden`, `experts`, `max_tokens`, `top_k`), and prepares the
session in one call. `seabraid_session_submit` validates the shape and copies the inputs into
Rust-owned storage, so the caller can reuse its arrays at once, and returns a
ticket. `seabraid_collect` waits up to `timeout_ms` (zero polls once), writes
`tokens * hidden` floats, and consumes the ticket. A timeout leaves the ticket
and its storage leased. `seabraid_close` drains the worker and consumes the session
even on error; it waits without limit for a running provider callback.
`seabraid_close_timeout` waits at most `timeout_ms`: if the worker is still inside a
callback it is detached (leaked, with an ERROR record naming the stuck ticket) and the
call returns `SEABRAID_TIMEOUT`. A worker that is not inside a callback is never "stuck": with no
ticket outstanding, or one whose callback has returned but was never collected, the close succeeds
whatever the bound (zero included), and a failure nobody collected is logged. The session is consumed
either way, and after a timeout the provider context and callback code must stay valid for the process lifetime.

A null provider returns `SEABRAID_INVALID`: silently running a demonstration transform would
pass for model math. `SEABRAID_PROVIDER_REFERENCE` opens the built-in reference transform on
purpose. On every failed open `*out` is NULL. The calls from before `seabraid_session_open`
(`seabraid_open*`, `seabraid_prepare`, `seabraid_submit*`) still work and are described in the
[deprecated appendix](#appendix-deprecated-v1-and-v2-entry-points); they are removed at 1.0.

A handle of the wrong kind (a session passed to a placement call, or the reverse) is refused as
`SEABRAID_INVALID` like a closed one: both handle types begin with a liveness word that differs per kind, and
it is the only memory the check reads before the handle is accepted.

A panic caught at the boundary poisons the session (and likewise a placement handle):
every later call except close returns `SEABRAID_INTERNAL` ("handle poisoned by earlier
panic") and does nothing. Close still works and frees the handle. A C++ exception that
escapes a provider callback is undefined behavior and aborts the process (Rust 1.81 and
later), so providers must catch everything and return a status.

### Threads

- **The worker thread.** Every session starts one worker thread when it opens, and every callback of that
  session runs on that one thread, for every call. A thread-bound device context (a CUDA or HIP context
  made current on a thread, thread-local handles) therefore stays valid across calls if it is set up from
  inside the callback. The thread is not the caller's, so nothing the caller made current on its own thread
  is current there.
- **Calls from other threads.** Calls on one session are serialized by the caller. While a call is inside
  the session (a blocking `seabraid_collect` is the long one), any other call on it, `seabraid_session_status`
  included, returns `SEABRAID_BUSY_OR_STATE` and does nothing. A metrics thread therefore cannot poll
  `seabraid_session_status` during a collect; have the thread that owns the session read the counters
  between calls (see [Reading counters](ADAPTER-GUIDE.md#reading-counters)). The same refusal applies to a
  call made from the provider callback.
- **The log sink.** It runs on whichever thread logs: the caller's thread for the library's own records,
  the worker thread for a `seabraid_log` made inside a callback.

## Batch v2 and provider v3 entry points

`seabraid_session_open` and `seabraid_session_submit` are the entry points for formats
other than F32, for versioned placement and for row compaction. The deprecated v1 and v2
open/submit calls keep working unchanged until 1.0; a v1 submit is a v2 submit with F32 data and
placement epoch 0. Both new calls are additive: no status code and no existing struct
layout changed except `seabraid_session_status_v1` (see below).

```c
typedef struct seabraid_provider_v3 {
    uint32_t abi_version;                 /* SEABRAID_PROVIDER_V3_VERSION */
    uint32_t struct_size;                 /* sizeof, at least 40 */
    uint32_t operation_count;             /* at least 1 */
    uint32_t flags;                       /* PARTIAL_ROUTES | COMPACT_ROWS | ACCEPTS_F16 | ACCEPTS_BF16 | REFERENCE */
    const seabraid_config_v1 *config;     /* copied at open; no seabraid_prepare */
    void *context;
    int32_t (*execute)(void *context, const seabraid_config_v1 *config,
                       const seabraid_batch_v2 *batch, float *output, size_t output_count);
} seabraid_provider_v3;

typedef struct seabraid_batch_v2 {
    uint32_t abi_version;                 /* SEABRAID_BATCH_V2_VERSION */
    uint32_t struct_size;                 /* sizeof, at least 80 */
    uint32_t operation_id;
    uint32_t source_modalities;
    uint64_t placement_epoch;             /* 0 = unversioned */
    uint32_t tokens;
    uint32_t activation_format;           /* SEABRAID_FORMAT_ROUTED_F32 / _F16 / _BF16 */
    const void *activations;  size_t activation_count;   /* elements, tokens * hidden */
    const int32_t *expert_ids; size_t expert_id_count;   /* tokens * top_k */
    const float *weights;      size_t weight_count;      /* tokens * top_k */
} seabraid_batch_v2;
```

- **Sized rule.** Both structs are zero-initialized, then `abi_version` and `struct_size` are
  set. `struct_size` must be at least the base size (40 and 80), at most 1024 and a
  multiple of 8; every byte past what the library knows must be zero, so a newer caller
  that sets a field this library lacks is refused rather than misread.
- **Open.** `seabraid_session_open` validates the descriptor, copies it and its config, and
  prepares the session. `*out` is NULL on every failure. `REFERENCE` selects the built-in
  reference transform and requires null `execute` and `context` and no `ACCEPTS_*` flag;
  without it `execute` must be non-null. `COMPACT_ROWS` requires `PARTIAL_ROUTES`.
- **Formats.** F32 is always accepted. `ACCEPTS_F16` and `ACCEPTS_BF16` let a provider receive
  16-bit elements; the library passes their bits through unchanged (NaN payloads included) and never
  converts. `activation_count` is in elements, and the caller's array must be 2-byte aligned (an odd address is
  refused with `SEABRAID_INVALID`). The provider sees `activation_format` and `activations` aligned to 4 bytes, with
  any padding after the last element. A format the provider did not declare, or an unknown format number, is refused at
  submit with `SEABRAID_INVALID` and changes nothing. The output is always F32. Worked guidance:
  [F16 and BF16 activations](ADAPTER-GUIDE.md#f16-and-bf16-activations).
- **Placement epoch.** `placement_epoch` 0 means unversioned and is accepted until the
  session has accepted a nonzero epoch. After that, 0 returns `SEABRAID_INVALID` and an epoch
  below the highest accepted one returns `SEABRAID_BUSY_OR_STATE`. The mark is per session and only an
  accepted submit raises it (one accepted huge epoch raises it for good), and the provider sees the epoch of the
  batch it is given. The session is not linked to any placement handle, so this is a monotonic guard on the number,
  not a check of the routing table: the engine's fence and drain keep routing correct, and a provider that owns only
  some experts should check its own ownership of each pick. A reopened placement handle restarts at epoch 1; see
  [Reopening](PLACEMENT-ABI.md#reopening-and-the-epoch-mark).
  An engine with live placement passes the epoch of the snapshot it routed with on every batch; see
  [Versioned submits and stale epochs](ADAPTER-GUIDE.md#versioned-submits-and-stale-epochs) for what to do
  on a stale refusal.
- **Row compaction.** With `COMPACT_ROWS`, a row whose picks are all `SEABRAID_SKIP_EXPERT_ID`
  is not shown to the provider: `batch->tokens`, `activation_count`, `expert_id_count` and
  `weight_count` describe only the kept rows, in their original order, and `output_count` is
  `tokens * hidden` of the kept rows. The library writes each result row back to its original
  position, and fully skipped rows are exactly zero. If every row is skipped the provider is
  not called. `seabraid_collect` still returns `tokens * hidden` floats for the submitted tokens.
  Only a row whose picks are **all** skipped is dropped. A partially skipped row is shipped unchanged, with its
  skipped picks still carrying `SEABRAID_SKIP_EXPERT_ID` and weight zero. The provider receives no map from the
  kept rows to the original rows (the library scatters the results back itself), so it must not need one; it
  numbers its rows 0 to `tokens - 1`.
- **Provider view.** The callback receives the library's own `seabraid_batch_v2`: `struct_size`
  is the size this library knows, `source_modalities` and `placement_epoch` are the submitted
  values, and `operation_id` selects the operation. The `config` and `batch` it is handed are copies owned by
  the library, not the caller's memory.
- **Submit.** The one promise is that **a refused batch changes nothing**: not the buffers,
  the epoch mark or any counter, and `*out_ticket` stays untouched. Every check runs before
  the first copy. The order in which several simultaneous problems are reported is not part
  of the contract and may change. The list of refusals is in `seabraid.h`.
- **One expert once per row.** Within one row the expert ids must be distinct; a repeat is
  refused with `SEABRAID_INVALID` ("pick N: expert id E repeats an earlier pick of the same
  row"). `SEABRAID_SKIP_EXPERT_ID` may appear several times in a row. The runtime owns this
  rule, so a provider can rely on it.

### Callback rules

- `config` and `batch` are library-owned copies that expire when the callback returns; so does `output`.
  Finish all device work that touches them before returning, on every path, including errors.
- Fill every element of `output` (`output_count` floats) and return 0, or return nonzero to report a failure. The
  library may or may not have zeroed `output` before the call: never read it and never rely on its prior contents.
- One callback runs at a time per session, on the worker thread (see [Threads](#threads)). It must return, must not
  throw across the ABI and must not call back into its own session.

### Provider failure

A callback that returns nonzero fails that ticket; the caller sees the failure at `seabraid_collect`:

- `seabraid_collect` returns `SEABRAID_PROVIDER` (4), whatever nonzero value the callback returned.
- The ticket is consumed, so the session accepts the next submit. `output` is untouched, `collected` is not
  incremented and `provider_failures` is.
- The placement epoch mark that the submit raised stays raised: a failed ticket does not lower it.
- The callback's own return code is not a status. It appears only in the log text ("provider callback returned
  status N", at ERROR) and in `seabraid_last_error` on the thread that called collect.
- Failures of the library's own bookkeeping around the callback (no memory for the output scratch, say) return
  `SEABRAID_INTERNAL` rather than `SEABRAID_PROVIDER`, and count in `provider_failures` all the same.
- A callback that **panics** (a Rust provider only) fails the ticket the same way, and the session's worker thread
  stops for good: the provider's state is unknown, and a replacement thread would break the promise that every
  callback runs on one thread. Every later submit returns `SEABRAID_PROVIDER` ("provider worker stopped") and
  changes nothing: no counter, no placement epoch mark. Close the session and open another.

The engine decides what to do next (compute the picks locally, retire the helper, resubmit).

### Partial routes

With `SEABRAID_PROVIDER_PARTIAL_ROUTES`, an `expert_ids` entry may be
`SEABRAID_SKIP_EXPERT_ID` (-1) with weight zero. That pick belongs to another
provider. A row whose picks are all skipped contributes zero. This is how
several devices split one layer: the engine sends each device the same rows
with the picks it does not own masked, and sums the partial outputs.

## Capabilities and status

```c
int32_t seabraid_query_capabilities(seabraid_capabilities_v1 *out);
int32_t seabraid_session_status(seabraid_session *session, seabraid_session_status_v1 *out);
```

`seabraid_capabilities_v1` reports the operation (`SEABRAID_OPERATION_ROUTED_MOE`),
input and output formats (`SEABRAID_FORMAT_ROUTED_F32`, `SEABRAID_FORMAT_PARTIAL_F32`:
the built-in F32 path; a provider's own `ACCEPTS_F16` / `ACCEPTS_BF16` support cannot be advertised here, see
[Formats](#batch-v2-and-provider-v3-entry-points) above)
and accepted source modalities (text, image, audio, video, other). Modalities
only tag already-encoded, already-routed activations; Seabraid does not decode
media. `seabraid_session_status_v1` carries `submitted`, `collected`,
`provider_failures`, `wait_timeouts`, `outstanding` and the last source tag, and from
this version `rows_submitted` (the sum of `tokens` over accepted submits) and
`rows_shipped` (rows handed to the provider; fewer only when `COMPACT_ROWS` dropped
some). The struct grew from 48 to 64 bytes; a caller built against the 48-byte layout
passes `struct_size` 48 and receives the first 48 bytes unchanged. It is a caller-side
snapshot: it is a call like any other, so from another thread while a call is inside the session it returns
`SEABRAID_BUSY_OR_STATE` (see [Threads](#threads)). The counters change as follows: `submitted` and
`rows_submitted` at an accepted submit; `collected` at a successful collect; `provider_failures` at a collect
whose callback failed (or whose worker died, or whose bookkeeping around the callback failed); `wait_timeouts` at each collect that times out. The adapter guide
shows how to read them ([Reading counters](ADAPTER-GUIDE.md#reading-counters)).

## Diagnostics

```c
enum seabraid_log_level { SEABRAID_LOG_DEBUG = 0, SEABRAID_LOG_INFO = 1, SEABRAID_LOG_WARN = 2,
                      SEABRAID_LOG_ERROR = 3, SEABRAID_LOG_OFF = 4 };
typedef void (*seabraid_log_fn)(void *context, uint32_t level, int32_t status, const char *message);

int32_t seabraid_set_log_sink(seabraid_log_fn sink, void *context, uint32_t min_level);
int32_t seabraid_log(uint32_t level, int32_t status, const char *message);
int32_t seabraid_last_error(char *buffer, size_t capacity, size_t *needed);
const char *seabraid_version(void);
```

- `seabraid_set_log_sink` installs a process-wide sink. A null sink restores the
  default, which writes to stderr at WARN and above. `min_level` filters.
  `SEABRAID_LOG_OFF` silences everything. The sink can be called from any thread: the calling thread for the
  library's own records, the session's worker thread for a `seabraid_log` made inside a callback.
  Which event logs at which level is tabulated in the [adapter guide](ADAPTER-GUIDE.md#5-diagnostics).
  Seabraid releases its own lock before calling the sink, so a sink may call `seabraid_log`.
- The sink does not see Rust panic messages. A panic is caught at the boundary and reported through
  the sink as `SEABRAID_INTERNAL`, but before that, Rust's default panic hook prints
  `thread '...' panicked at src/...` to stderr, outside the sink, and `SEABRAID_LOG_OFF` does not
  silence it. The library never installs a panic hook (no `std::panic::set_hook` or `take_hook`
  anywhere in `src/`), so the hook belongs to the host process: a Rust host can call
  `std::panic::set_hook` to silence or redirect it, and a C or C++ host that wants quiet stderr
  must redirect the stream. Panic locations in the message are source paths of the build.
- `seabraid_log` lets adapters and providers report through the same sink, so one
  stream shows both Seabraid's and the adapter's failures.
- `seabraid_last_error` copies this thread's last failure text into `buffer`
  (NUL-terminated, truncated to `capacity`) and stores the size needed,
  including the NUL, in `*needed`. Call it with a null buffer to size it.
  The text persists until the next failure on the same thread: a successful call does not clear it, so read
  it right after a nonzero status. Each thread has its own text; a failure found by the thread that called
  `seabraid_collect` is not visible from the worker thread.
  Its own argument errors (a null buffer with a nonzero capacity, a misaligned `needed`) are
  returned but not recorded, so a bad call does not replace the message being asked for.
- `seabraid_version` returns a static NUL-terminated string, the package version
  (`MAJOR.MINOR.PATCH` plus any pre-release suffix). Do not free it, and do not compare it for
  compatibility: use the epoch and the SONAME (see [Layout epoch](#layout-epoch)).
- Messages carry status codes and short text only: no paths, environment
  values or personal data.

## Placement

```c
typedef struct seabraid_placement seabraid_placement;

int32_t seabraid_placement_open(const seabraid_placement_layout_v1 *layout,
                            const seabraid_placement_policy_v1 *optional_policy,
                            seabraid_placement **out);
int32_t seabraid_placement_close(seabraid_placement *placement);
int32_t seabraid_placement_plan(seabraid_placement *placement, uint64_t expected_epoch,
                            const double *heat, size_t heat_count,
                            seabraid_placement_ticket_v1 *out);
int32_t seabraid_placement_apply(seabraid_placement *placement, uint64_t ticket,
                             const seabraid_placement_provider_v1 *provider,
                             seabraid_placement_outcome_v1 *outcomes, size_t outcome_capacity,
                             seabraid_placement_applied_v1 *out);
int32_t seabraid_placement_snapshot(seabraid_placement *placement, uint8_t *executors,
                                size_t executor_count, seabraid_placement_snapshot_v1 *out);
int32_t seabraid_placement_quarantine(seabraid_placement *placement, uint64_t expected_epoch, uint8_t rung);
int32_t seabraid_placement_update(seabraid_placement *placement, uint64_t expected_epoch,
                              uint32_t mode, const uint8_t *owners, size_t owner_count);
```

Rungs: 0 is the lead card, `1 .. cards - 1` are helpers, and `cards` is the
verified fallback. Owners and heat are layer-major (`layer * experts + expert`).
The full protocol, outcome types and failure rules are in
[PLACEMENT-ABI.md](PLACEMENT-ABI.md). In short:

- `open` copies a layout the caller has verified (epoch 1) and an optional
  policy (`max_moves_per_round`, `ladder_boundaries`, `cooldown_rounds`,
  `promote_margin`, `max_helper_layer_share`); null policy uses defaults. The ranges, units and defaults are
  tabled in [PLACEMENT-ABI.md](PLACEMENT-ABI.md#policy); `max_moves_per_round` counts expert writes, so one swap is 2.
  No call changes a handle's policy.
- `plan` takes one window of per-cell heat for the stated epoch and replaces
  the pending ticket. It performs no provider work. The planner blends each window into a running per-cell heat
  (first window as given, then `0.75 * previous + 0.25 * window`), so it is not a function of one window
  ([Heat across windows](PLACEMENT-ABI.md#heat-across-windows)).
- `apply` runs the ticket through the engine's stage/replace callbacks at a
  drained fence and returns ordered outcomes (committed, rejected, restored,
  quarantined). A stale or unknown ticket returns `SEABRAID_BUSY_OR_STATE` before
  any I/O.
- Each `layer_slot_bytes` entry of an `apply` provider is 1 to 4294967295 bytes (2^32 - 1); a larger one is
  `SEABRAID_INVALID` before any I/O and the ticket stays pending. The library allocates one zero-filled buffer of that
  size per staged slot without touching its pages.
- During `apply`, each provider failure is exactly one log record: ERROR for a failure, timeout, unknown code,
  panic or short stage write, WARN for a clean refusal (no write began) and for the quarantine notice. `apply`
  itself returns `SEABRAID_OK` with the outcomes.
- `snapshot` exports `layers * experts` executor bytes and the epoch. Publish
  it at the same fence as the change.
- `quarantine` and `update` (`SEABRAID_PLACEMENT_ADOPT` / `SEABRAID_PLACEMENT_REBUILD`)
  record out-of-band loss or verified external moves.

### Spillover fill

```c
typedef struct seabraid_placement_fill_v1 {
    uint32_t abi_version;
    uint32_t layers;
    uint32_t experts;
    uint8_t cards;
    const uint64_t *slots;  size_t slot_count;
    const double *heat;     size_t heat_count;
} seabraid_placement_fill_v1;

int32_t seabraid_placement_fill(const seabraid_placement_fill_v1 *request,
                            uint8_t *owners, size_t owner_count);
```

Hottest-first spillover: the lead card fills first, then each helper in rung
order, and the rest go to the fallback rung `cards`. `slots` gives each card's
capacity in rung order, as cells counted across all layers together, `heat` is layer-major, and `owners` receives
`layers * experts` rung numbers, ready to pass as the layout of
`seabraid_placement_open`. Use it to compute the initial placement, load the
experts it names, verify them, and only then open the handle.

## Appendix: deprecated v1 and v2 entry points

These calls predate `seabraid_session_open` and `seabraid_session_submit`. They keep working unchanged
through the alpha and are removed at 1.0. A v1 or v2 submit is a batch v2 submit with F32 data, placement epoch 0
and no row compaction.

```c
SEABRAID_DEPRECATED int32_t seabraid_open(const seabraid_provider_v1 *provider, seabraid_session **out);
SEABRAID_DEPRECATED int32_t seabraid_open_reference(seabraid_session **out);
SEABRAID_DEPRECATED int32_t seabraid_open_multi(const seabraid_provider_v2 *provider, seabraid_session **out);
SEABRAID_DEPRECATED int32_t seabraid_prepare(seabraid_session *session, const seabraid_config_v1 *config);
SEABRAID_DEPRECATED int32_t seabraid_submit(seabraid_session *session, const seabraid_batch_v1 *batch,
                                            uint64_t *out_ticket);
SEABRAID_DEPRECATED int32_t seabraid_submit_routed(seabraid_session *session, const seabraid_batch_v1 *batch,
                                                   uint32_t source_modalities, uint64_t *out_ticket);
SEABRAID_DEPRECATED int32_t seabraid_submit_operation(seabraid_session *session, uint32_t operation_id,
                                                      const seabraid_batch_v1 *batch, uint32_t source_modalities,
                                                      uint64_t *out_ticket);
```

**Migrating.** Open with `seabraid_session_open` (`REFERENCE` replaces
`seabraid_open_reference`) and submit with `seabraid_session_submit`; `seabraid_collect`,
`seabraid_close` and the status calls stay. `SEABRAID_DEPRECATED` expands to nothing unless
the consumer defines `SEABRAID_WARN_DEPRECATED` (GCC, Clang and MSVC), so existing builds,
including `-Werror` ones, are unaffected until they opt in.

`seabraid_prepare` fixes the geometry (`seabraid_config_v1`) of a session opened with
`seabraid_open`, `seabraid_open_reference` or `seabraid_open_multi`, which carry no config of their own.
`seabraid_submit*` take a `seabraid_batch_v1` and otherwise behave as `seabraid_session_submit` does.
`seabraid_open` with a null provider returns `SEABRAID_INVALID`; `seabraid_open_reference` opens the
built-in reference transform on purpose. On every failed open `*out` is NULL.

### Batch v1 layout


```c
typedef struct seabraid_batch_v1 {
    uint32_t abi_version;                /* SEABRAID_ABI_VERSION */
    uint32_t tokens;
    const float *activations;  size_t activation_count;  /* tokens * hidden  */
    const int32_t *expert_ids; size_t expert_id_count;   /* tokens * top_k   */
    const float *weights;      size_t weight_count;      /* tokens * top_k   */
} seabraid_batch_v1;
```

Arrays are dense and token-major; counts are elements. The result is the
partial contribution of the experts this provider computed, as FP32; the
engine does the final merge.

### Providers v1 and v2

```c
typedef struct seabraid_provider_v1 {
    uint32_t abi_version;                 /* SEABRAID_ABI_VERSION */
    void *context;
    int32_t (*execute)(void *context, const seabraid_config_v1 *config,
                       const seabraid_batch_v1 *batch, float *output, size_t output_count);
} seabraid_provider_v1;

typedef struct seabraid_provider_v2 {
    uint32_t abi_version;                 /* SEABRAID_MULTI_PROVIDER_VERSION */
    uint32_t operation_count;
    uint32_t flags;                       /* SEABRAID_PROVIDER_PARTIAL_ROUTES only */
    void *context;
    int32_t (*execute)(void *context, uint32_t operation_id, const seabraid_config_v1 *config,
                       const seabraid_batch_v1 *batch, float *output, size_t output_count);
} seabraid_provider_v2;
```

- **v1** serves one operation. The legacy submit calls select operation zero.
- **v2** serves a dense table of operations (`0 .. operation_count - 1`)
  sharing one worker, one config and one buffer pool. An adapter typically maps
  an operation ID to a model layer. The set is stable until close.
- `seabraid_provider_v2` is frozen: `flags` holds `PARTIAL_ROUTES` or nothing, and any other bit,
  `COMPACT_ROWS` included, is refused with `SEABRAID_INVALID`. Row compaction is a provider v3 feature.
- Callback pointers expire at return. Finish all device work that touches
  them before returning, including on error. Fill every output element and
  return 0, or return nonzero (the session reports `SEABRAID_PROVIDER`).

The v1 and v2 providers follow the callback rules of provider v3 (see [Callback rules](#callback-rules)) and
the partial-routes rule (see [Partial routes](#partial-routes)).
