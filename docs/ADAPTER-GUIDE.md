# Adapter guide: putting Seabraid behind an engine

**Start with [ENGINE-INTEGRATION.md](ENGINE-INTEGRATION.md)**: the order of work, the process model and the
conformance checklist for an engine. This guide is the C mechanics it links to.

An adapter is the small piece of code in or beside an engine that sends routed
rows to a second device and takes partial sums back. This guide is for engine
authors. The reference for every call is the [ABI](ABI.md); the design is in
the [architecture](ARCHITECTURE.md).

What stays with the engine: model loading, the router, the scheduler, KV
state, batching, and the final merge. What Seabraid owns: the session, worker,
tickets, buffer reuse, validation, timeouts, and placement bookkeeping. What a
provider owns: the native kernels and weights on its device.

Dependencies point one way: the runtime knows no adapter, and an adapter knows
no add-on. The one exception is a build input, not an import: `adapters/strata`
builds against a Strata source tree that has had the patch series in
`compat/strata` (and `adapters/strata/strata-v7-seabraid.patch`) applied, which
you prepare yourself. Nothing in `src/`, `include/`, `cmake/`, `examples/`,
`native-tests/` or `tools/` builds from or depends on `compat`.

## 1. A minimal provider

A provider is one callback. It receives a validated, copied batch and writes
`tokens * hidden` floats. This sketch uses a placeholder for the vendor-native
kernel call.

```c
#include <seabraid/seabraid.h>

typedef struct { my_device *dev; } my_provider;

static int32_t my_execute(void *ctx, const seabraid_config_v1 *cfg,
                          const seabraid_batch_v2 *batch, float *out, size_t out_count) {
    my_provider *p = ctx;
    /* Run the experts this device holds for each row, weighted by batch->weights
       (batch->activations is F32 unless the provider accepts F16 or BF16).
       Finish all device work that touches batch/out before returning. */
    return my_device_run_experts(p->dev, cfg, batch, out, out_count) ? 0 : 1;
}

int main(void) {
    my_provider p = { open_device() };
    seabraid_config_v1 cfg = { SEABRAID_ABI_VERSION, /*hidden*/ 4096, /*experts*/ 128, /*max_tokens*/ 16, /*top_k*/ 8 };
    seabraid_provider_v3 prov = { SEABRAID_PROVIDER_V3_VERSION, sizeof prov, /*operations*/ 1, /*flags*/ 0,
                                  &cfg, &p, my_execute };
    seabraid_session *s;  uint64_t ticket;
    /* Refuse a library built for another struct layout before any call. */
    if (seabraid_abi_epoch() != SEABRAID_ABI_EPOCH) return 1;
    if (seabraid_session_open(&prov, &s)) return 1;   /* opens and prepares in one call */
    /* per layer: zero the batch, set abi_version and struct_size, then fill it */
    seabraid_batch_v2 b = { SEABRAID_BATCH_V2_VERSION, sizeof b, /*operation*/ 0, SEABRAID_SOURCE_TEXT,
                            /*placement_epoch: 0 = unversioned*/ 0, n_tokens, SEABRAID_FORMAT_ROUTED_F32,
                            acts, n_tokens * 4096, ids, n_tokens * 8, wts, n_tokens * 8 };
    if (seabraid_session_submit(s, &b, &ticket) == SEABRAID_OK)
        seabraid_collect(s, ticket, /*timeout_ms*/ 50, out, n_tokens * 4096);
    return seabraid_close(s);   /* or seabraid_close_timeout(s, ms) to bound a stuck callback */
}
```

See [`examples/minimal_provider.c`](../examples/minimal_provider.c) for a complete, buildable version, and
[`examples/adapter_walkthrough.c`](../examples/adapter_walkthrough.c) for a full provider v3 that uses partial
routes, `COMPACT_ROWS`, placement epochs, refusals, a device failure, session status and a log sink. Epoch 0 above
means "unversioned"; an engine with live placement versions every batch (see
[Versioned submits and stale epochs](#versioned-submits-and-stale-epochs)). To check the library at start-up
and at build time, compare the epoch and require the minor series:

```c
if (seabraid_abi_epoch() != SEABRAID_ABI_EPOCH) return 1;   /* header and library disagree */
```

```cmake
find_package(Seabraid 0.5 CONFIG REQUIRED)    # SameMinorVersion; also sets Seabraid_ABI_EPOCH
```

The epoch and the SONAME are the compatibility contract; the version string only labels the build
([ABI](ABI.md#layout-epoch)). Keep the
provider's `context` valid until `seabraid_close` returns (for the whole process
after a `seabraid_close_timeout` that returns `SEABRAID_TIMEOUT`, because the
detached worker may still call it). A null provider is invalid; use
`SEABRAID_PROVIDER_REFERENCE` (or `seabraid_open_reference`) when the built-in
demonstration transform is what you want. A C++ provider must catch every exception inside its callback.

### F16 and BF16 activations

A provider can take 16-bit activation rows without the engine converting them to F32. Set `SEABRAID_PROVIDER_ACCEPTS_F16`,
`SEABRAID_PROVIDER_ACCEPTS_BF16` or both in the `seabraid_provider_v3` flags, and submit with `seabraid_session_submit` and
`activation_format = SEABRAID_FORMAT_ROUTED_F16` or `SEABRAID_FORMAT_ROUTED_BF16` (provider v1 and v2 and their submit
calls are F32 only). What to expect:

- **Counts are elements.** `activation_count` is `tokens * hidden` for every format, never bytes. A 16-bit row of
  `hidden` elements is `2 * hidden` bytes, so odd `hidden` values are fine.
- **Your array must be 2-byte aligned.** An odd address is refused at submit with `SEABRAID_INVALID` for both 16-bit
  formats (F32 needs 4-byte alignment).
- **The provider sees 4-byte-aligned data.** The library copies the elements into its own buffer, so `batch->activations`
  is 4-byte-aligned whatever the caller's alignment. Read exactly `batch->activation_count` elements (`2 * count` bytes);
  when that is not a multiple of 4 the last word is zero padding, not data. Under `COMPACT_ROWS` the kept rows are
  repacked back to back, so they stay aligned too.
- **The library never converts.** The element bits reach the provider unchanged: NaN payloads, infinities, subnormals and
  signed zeros included. Activations are not inspected (only router weights must be finite). Decode them in the provider.
- **The output is F32.** `output` holds `tokens * hidden` floats whatever the input format.
- **An undeclared format is refused and changes nothing.** If the provider did not declare the flag for the format, or
  `activation_format` is not 1, 3 or 4, submit returns `SEABRAID_INVALID` and no buffer, epoch mark, counter or ticket
  moves. F32 is always accepted, even by a provider that declares both 16-bit flags.
- **Capabilities do not show it.** `seabraid_query_capabilities` reports the built-in F32 path (`input_format`
  `ROUTED_F32`, `output_format` `PARTIAL_F32`); it cannot advertise a provider's 16-bit support. The engine that writes the
  provider already knows which flags it set, so branch on that, not on the query.

## 2. Split a layer: partial routes

When two devices hold different experts of the same layer, open the provider
with `SEABRAID_PROVIDER_PARTIAL_ROUTES`. For each device, the engine sends the same
rows but replaces picks the device does not own with `SEABRAID_SKIP_EXPERT_ID` and
weight zero. The device computes only its picks; a row with no picks gives
zero. The engine sums the partial outputs (and its own local experts) into the
layer result.

Add `SEABRAID_PROVIDER_COMPACT_ROWS` (it requires `PARTIAL_ROUTES`) when a device would otherwise
compute many rows that are entirely skipped. The library then drops every row whose picks are **all**
`SEABRAID_SKIP_EXPERT_ID`: the provider sees only the kept rows, renumbered `0 .. tokens - 1` in their original
order, and `output_count` is `tokens * hidden` of those rows. The library writes each result row back to its
original position; dropped rows are exactly zero, and with no kept row the provider is not called at all.
A row with even one real pick is shipped whole, with its skipped picks still marked `SEABRAID_SKIP_EXPERT_ID`
and weight zero. The provider gets no kept-row-to-original-row map, so do not write a callback that needs one.
`config` and `batch` are library-owned copies, valid until the callback returns.

```c
for (int t = 0; t < tokens; t++)
  for (int k = 0; k < top_k; k++) {
    int e = routed_ids[t*top_k + k];
    bool mine = owner[layer*experts + e] == this_rung;
    ids[t*top_k + k] = mine ? e : SEABRAID_SKIP_EXPERT_ID;
    wts[t*top_k + k] = mine ? routed_wts[t*top_k + k] : 0.0f;
  }
```

The engine can submit to all helpers, compute its own share, then collect, so
the cards work at the same time.

## 3. Many layers on one worker

A provider declares `operation_count` operations. Map an operation ID to a
layer and set `batch.operation_id` before `seabraid_session_submit`. All operations
share one prepared geometry and one buffer pool; only one batch is in flight
per session. If your engine runs layers on several threads, open one session
per thread.

Each session runs its callback on one dedicated worker thread, the same thread for every call. A device
context that is bound to a thread (CUDA, HIP) must be created or made current from inside the callback, not on
the engine's thread, and it then stays valid from call to call. A call into the session (`seabraid_session_status`
included) from another thread while a call such as `seabraid_collect` is waiting returns
`SEABRAID_BUSY_OR_STATE`; see [Threads](ABI.md#threads).

## 4. The placement loop

For live placement the engine moves experts between cards while serving. The
loop is: fill, open, plan, apply at the fence, publish.

```c
/* 1. Initial placement, hottest first, then load and verify those experts.
 * slots[i] = cells card i can hold, counted across ALL layers; the second `cards` is slot_count (one entry per
 * card); the last argument is heat_count. */
seabraid_placement_fill_v1 fill = { SEABRAID_ABI_VERSION, layers, experts, cards, slots, cards, heat0, layers*experts };
seabraid_placement_fill(&fill, owners, layers*experts);
load_and_verify(owners);                           /* engine / provider work */

/* 2. Open the handle on the verified layout and export the first executor table. */
seabraid_placement_layout_v1 lay = { SEABRAID_ABI_VERSION, layers, experts, cards, owners, layers*experts };
seabraid_placement_open(&lay, NULL, &pl);
seabraid_placement_snapshot_v1 snap = { .abi_version = SEABRAID_ABI_VERSION, .struct_size = sizeof snap };  /* library-filled: set both */
seabraid_placement_snapshot(pl, exec, layers*experts, &snap);   /* route by exec[], keep snap.epoch */

/* 3. At your planning cadence: feed a window of heat, get a plan. */
seabraid_placement_ticket_v1 t = { .abi_version = SEABRAID_ABI_VERSION, .struct_size = sizeof t };
seabraid_placement_plan(pl, snap.epoch, window_heat, layers*experts, &t);

if (t.promotion_count) {
    stop_admission(); drain_old_epoch_work();                /* the engine fence */
    seabraid_placement_applied_v1 done = { .abi_version = SEABRAID_ABI_VERSION, .struct_size = sizeof done };
    /* outcomes[0] carries abi_version and struct_size; the library strides by that size. */
    seabraid_placement_apply(pl, t.ticket, &providers, outcomes, t.promotion_count, &done);
    if (done.epoch != snap.epoch) {                          /* something committed */
        seabraid_placement_snapshot(pl, exec, layers*experts, &snap);
        publish(exec, snap.epoch);                           /* still fenced */
    }
    resume_admission();
}
```

The `seabraid_placement_provider_v1` you pass to `apply` has two callbacks:
`stage` copies a whole verified slot out of a card, and `replace` writes a new
expert into a slot and acknowledges only when the bytes and the residency index
are both updated. Return the exact I/O status (`ACK`, `REFUSED`, `FAILED`,
`TIMEOUT`): the runtime restores or quarantines accordingly, and a quarantined
card's cells route to the fallback. Heat is yours to measure; counts of routed
picks per (layer, expert) over a window work. Planning respects cooldown, a
promotion margin and a move budget, but it does not judge each window alone: the
first window on a fresh handle is used as given, and every later one is blended
into a running per-cell heat (`0.75 * previous + 0.25 * window`), applied or not,
and apply does not reset it. A moved hot set therefore needs several windows
under the default policy before the first promotion, and one noisy window
cannot thrash. The policy fields (ranges, units, defaults) are tabled in
[PLACEMENT-ABI.md](PLACEMENT-ABI.md#policy); `max_moves_per_round` counts expert
writes, and one swap is 2. A policy cannot be changed on an open handle: see
[reopening](#reopening-a-placement-handle) below.

Do not apply a plan without draining first: a half-applied swap with live
dispatch is the failure this design exists to prevent. Details and failure
rules are in [PLACEMENT-ABI.md](PLACEMENT-ABI.md).

Read the outcomes after every apply. Each one names a `layer`, the expert that moved up (`promote`), the expert that moved
down (`displace`) and the upper rung (`upper_rung`; the lower rung is always `upper_rung + 1`). `committed_epoch` is
the epoch after a commit and zero otherwise. The epoch grows by 1 per committed swap and by 1 per quarantine event, and
not at all for a rejection or a restoration, so two commits take epoch 1 to 3. The library cannot give the owner table
back (a closed rung's cells read as the fallback in the snapshot), so keep your own copy of the owners and apply each
committed outcome to it; REBUILD needs that copy. [PLACEMENT-ABI.md](PLACEMENT-ABI.md#faults-and-their-outcomes) tables
which provider fault ends in which outcome.

### Reopening a placement handle

There is no setter for the policy, and a new handle starts at epoch 1. Sessions keep the highest epoch they accepted, so
after a close and reopen every batch stamped with the new handle's small epochs is refused as stale. Either reopen the
sessions with the handle, or stamp batches with an epoch the engine owns, for example
`(generation << 32) | snap.epoch` with `generation` raised on each reopen (the placement calls still take
`snap.epoch`). Both were run end to end; details in [PLACEMENT-ABI.md](PLACEMENT-ABI.md#reopening-and-the-epoch-mark).

### Versioned submits and stale epochs

A submit can carry the placement epoch the engine routed with, in `seabraid_batch_v2.placement_epoch`. It lets the
runtime refuse a batch stamped with an epoch lower than one the session has already accepted. It is a monotonic guard on
a number, not a check against the table: the runtime is not linked to the placement handle and cannot tell whether the
number came from the current snapshot. The fence and drain around apply are what keep routing correct. Because the
guard cannot see your table, have a provider that holds only some experts also check that it owns each pick that is not
`SEABRAID_SKIP_EXPERT_ID`, and fail the batch if not.

- **Pass the snapshot epoch with every batch.** Take `snap.epoch` from `seabraid_placement_snapshot`, route
  by that snapshot's `exec[]`, and put the same number in `placement_epoch`. Do not leave it 0 once you version.
- **0 is accepted only until the session has accepted a nonzero epoch.** After that a 0 is refused with
  `SEABRAID_INVALID` ("a versioning engine must version every batch"), so an unversioned batch cannot slip past
  the mark.
- **The mark is per session and only an accepted submit raises it.** An epoch below the highest one the
  session accepted returns `SEABRAID_BUSY_OR_STATE` and changes nothing: no buffer, no counter, no ticket.
  A refused batch does not raise the mark, and a later provider failure does not lower it. One accepted huge epoch
  (even `UINT64_MAX`) raises the mark for good: every lower epoch is stale until the session is closed.
- **On a stale refusal: re-snapshot, re-route, resubmit.** The placement changed under you. Call
  `seabraid_placement_snapshot` again, rebuild `ids` and `wts` from the new `exec[]` (the old rows may now name
  experts another device owns), and submit with the new `snap.epoch`. Resubmitting the old batch unchanged
  fails the same way.
- **Telling stale from other busy states.** `SEABRAID_BUSY_OR_STATE` also means a ticket is still outstanding
  or the session is closing. You know your own tickets, so with none outstanding it is a stale epoch;
  `seabraid_last_error` reads "placement_epoch N is stale: the session already accepted M". The library logs it at
  WARN.

## 5. Diagnostics

Install a sink once at startup, and report your own failures through it:

```c
static void on_log(void *ctx, uint32_t level, int32_t status, const char *msg) {
    engine_log(level, "seabraid[%d] %s", status, msg);
}
seabraid_set_log_sink(on_log, NULL, SEABRAID_LOG_INFO);
...
seabraid_log(SEABRAID_LOG_ERROR, SEABRAID_PROVIDER, "device 1 kernel launch failed");
```

After a failing call, `seabraid_last_error` returns this thread's failure text. It persists until the next
failure on that thread (a successful call does not clear it), so read it right after a nonzero status, on the
thread that made the call. `seabraid_version()` identifies the library at runtime. Do not put paths,
environment values or user data in messages.

The sink runs on whichever thread logs: the calling thread for the library's own records, and the session's
worker thread for a `seabraid_log` made inside your callback. Make it thread-safe.

Which event logs at which level (the default sink prints WARN and above to stderr):

| Event | Level | Status |
|---|---|---|
| A refused call: bad argument, shape, geometry or version | WARN | `SEABRAID_INVALID` |
| A busy session, an outstanding ticket, a stale placement epoch, a call from the provider callback | WARN | `SEABRAID_BUSY_OR_STATE` |
| The provider callback returned nonzero (logged at collect, with its return code in the text) | ERROR | `SEABRAID_PROVIDER` |
| A provider callback panicked, or the worker thread ended without a result | ERROR | `SEABRAID_PROVIDER` |
| A caught panic, or a call on a poisoned handle | ERROR | `SEABRAID_INTERNAL` |
| `seabraid_close_timeout` found a stuck callback and leaked the worker | ERROR | `SEABRAID_TIMEOUT` |
| A failed ticket nobody collected, found at close | by its code (ERROR for a provider failure) | as the failure |
| A `seabraid_collect` that times out (a zero-timeout poll is routine) | DEBUG | `SEABRAID_TIMEOUT` |
| Your own `seabraid_log` | the level you pass | the status you pass |

`min_level` filters by this table, so raising it to `SEABRAID_LOG_ERROR` hides every refusal and every stale
epoch (they are WARN), leaving only provider failures, internal errors and your own ERROR records. If a
submit returns a nonzero status and nothing is logged, check `min_level` before you suspect the library;
`seabraid_last_error` still has the text.

### Reading counters

`seabraid_session_status` copies the session's counters. It is a library-filled struct: set both header fields.

```c
seabraid_session_status_v1 st = { .abi_version = SEABRAID_ABI_VERSION, .struct_size = sizeof st };   /* the rest zero */
if (seabraid_session_status(s, &st) == SEABRAID_OK)
    engine_metrics(st.submitted, st.collected, st.provider_failures, st.wait_timeouts,
                   st.outstanding, st.rows_submitted, st.rows_shipped);
```

| Field | Counts |
|---|---|
| `submitted`, `rows_submitted` | accepted submits, and the sum of their `tokens` |
| `rows_shipped` | rows handed to the provider; below `rows_submitted` only when `COMPACT_ROWS` dropped rows |
| `collected` | successful collects |
| `provider_failures` | collects whose callback failed (or whose worker ended) |
| `wait_timeouts` | collect waits that timed out; one ticket can add several |
| `outstanding` | 1 while a ticket is in flight, else 0 |

Once every ticket is resolved, `submitted == collected + provider_failures`. A refused submit adds to nothing. Read
the counters from the thread that owns the session, between its calls. They are a snapshot, and a call from another
thread while a collect is waiting returns `SEABRAID_BUSY_OR_STATE` and writes nothing, so a separate metrics
thread cannot poll during a collect: copy the numbers into the engine's own metrics after each collect instead.

## 6. Failure contract

- **Open-time mismatch fails fast.** A wrong geometry, ABI version or
  descriptor returns `SEABRAID_INVALID` and nothing starts.
- **Errors after open fail open, and loudly.** A provider that returns nonzero
  gives `SEABRAID_PROVIDER` at `seabraid_collect`; the session counts it, logs it, and the engine decides
  whether to compute the picks locally. The ticket is consumed (submit again), `output` is untouched, `collected`
  does not move, `provider_failures` does, and the epoch mark stays raised. The callback's own return code
  is visible only in the log text and `seabraid_last_error`, never as the status. A timeout leaves the ticket
  pending; wait again or close.
- **Uncertain completion retires the device.** If a provider cannot say whether
  its device work finished, return a timeout. The Strata adapter then retires the
  helper and the engine routes future requests locally; placement quarantines the
  card.
- **No silent success.** Every outcome is a status or an outcome record.

## 7. The Python path

Python engines (and quick experiments) use the same runtime. With no native
library, `seabraid.Runtime()` falls back to a pure-Python reference provider and
warns; point it at a built library with `library_path` or `SEABRAID_LIBRARY`.

```python
import seabraid

seabraid.enable_logging()
print(seabraid.version())
with seabraid.Runtime() as rt:               # reference provider, no GPU needed
    layer = rt.prepare(hidden=4, experts=8, max_tokens=4, top_k=2)
    partial = layer.weave(rows, expert_ids, weights)   # flat sequences
```

`seabraid.ranking.rank(...)` orders cards (which is the lead, which are helpers)
from measurements, and `seabraid.placement.fill(...)` is the Python form of
`seabraid_placement_fill`. Inputs may be C-contiguous float32/int32 buffers
(`array.array`, `memoryview`, numpy) for speed; plain lists still work but are slow.

### Hosting a provider from Python

A Python engine (vLLM, SGLang, KTransformers) serves the experts itself by passing
a callable; the runtime keeps the queue, tickets and timeouts.

```python
def serve(operation_id, config, activations, expert_ids, weights, output):
    # views over runtime-owned memory: valid only during this call
    run_my_experts(operation_id, activations, expert_ids, weights, out=output)

with seabraid.Runtime(lib_path, provider=serve, operations=2, partial_routes=True) as rt:
    layer = rt.prepare(hidden=4096, experts=64, max_tokens=512, top_k=8)
    layer.weave(rows, ids, weights, operation=1, out=result_buffer)
```

- Arguments: `config` is a `Shape(hidden, experts, max_tokens, top_k)`; inputs are
  read-only float32, int32 and float32 buffers; `output` is writable float32 and
  must be fully written. With numpy importable they are numpy arrays over the same
  memory. Copy anything you keep; the memory expires when `serve` returns.
- Threading: with a native library `serve` runs on the runtime's worker thread
  (ctypes acquires the GIL); it must not call back into its own `Runtime`, and the
  Runtime must be closed before interpreter exit. Without one, the reference session
  calls `serve` on the caller's thread, so the API is the same everywhere.
- Failure: an exception is caught, logged once on logger `seabraid` (type only,
  never tensor data) and returned as `SEABRAID_PROVIDER`; `weave` raises
  `SeabraidError` with status 4 and the session stays usable.
- Finish device work before returning, as in the C contract.

## 8. Checklist

- Geometry (`hidden`, `experts`, `top_k`) matches the model; one prepared
  config per session.
- Provider finishes device work before returning, on every path.
- Partial routes use `SEABRAID_SKIP_EXPERT_ID` with zero weight.
- Placement is applied only at a drained fence; the new table is published at
  the same fence.
- Failures go to the log sink; nothing is swallowed.
- Every batch carries the snapshot epoch once you version; a stale-epoch refusal means re-snapshot, re-route,
  resubmit.
- The callback writes every output element and does not depend on `output` being pre-zeroed.
- Say what you tested and what you did not.
