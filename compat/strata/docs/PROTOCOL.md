# Strata integration engine<->helper channel: protocol as shipped in 0.0.3 (channel v6), plus v7 (section 7)

Status: DRAFT. This documents what the code does today, not a redesign. It is meant to become the base of a
protocol that engines other than Strata (first target: llama.cpp) can implement with a small client library while
the helper stays a separate process. Strata-specific review material stays in [STRATA-INTEGRATION.md](STRATA-INTEGRATION.md);
this file links to it instead of repeating it.

Source of truth is the code. Everything below cites it. Anything not confirmed from the code, the patch commit messages
or `compat/strata/docs/DESIGN-LOG.md` is marked UNCONFIRMED. Analysis (not read from code) is marked ANALYSIS.

Citation convention. `channel.hpp:N` and `expert_spec.hpp:N` are line numbers in the v6-era files that patches 0001, 0005
and 0007 produced (history: since 2026-11-19 the wire is a file of this repository, `adapters/strata/wire/strata/seabraid/`
`channel_layout.hpp` (the bytes), `channel.hpp` (the operations) and `expert_spec.hpp`, its one owner; no patch creates it).
To read the v6-era files, check out a commit before that date and apply 0001, 0005 and 0007 with `git apply --include='include/strata/seabraid/*'`.
Patch 0008 files are cited as `0008:ds4_helper_link.cpp:N` (line N of the file the patch creates). `helper.cpp` is in
`compat/strata/strata-ext/src/`. **Historical citations:** patches 0008/0009 and `helper_dsv41.cpp` (also `dsv41_plan_check.hpp`, `dsv41_exec.*`)
were dropped on 2026-11-17 with Strata's DeepSeek-V4.1 engine; every `0008:` and `helper_dsv41.cpp:` citation below describes the code as of Strata v0.2.13.1
and can be rebuilt from this repository's history, not from the current tree. Engine-side Qwen code is `0001:remote_experts.cpp`, cited by function.

## 1. Scope and transport

- One POSIX shared-memory segment per helper. Name from `SEABRAID_STRATA_SHM`, default `/seabraid0` (`helper.cpp:36`,
  `0001:remote_experts.cpp` `seabraid_open`). On Linux it appears as `/dev/shm/<name>`. Created with mode 0600
  (`channel.hpp:183`), so only the same user can attach.
- The helper creates (`Channel::create`, `channel.hpp:180`). The engine attaches (`Channel::connect`, `channel.hpp:209`).
  Names used: "helper" = the process that owns the GPU that holds experts; "engine" or "main" = the process that
  runs the model and calls the helper.
- `create` order: `shm_unlink(name)` (replaces any stale segment), `shm_open(O_CREAT|O_EXCL|O_RDWR)`, `ftruncate(total)`,
  `mmap(MAP_SHARED)`, zero `sizeof(Header)` bytes, set fields, release fence, store `magic` last (`channel.hpp:182-204`).
  The regions after the header are the zero pages `ftruncate` gives.
- Teardown: the creator's `unmap()` does `munmap` and `shm_unlink` (`channel.hpp:337`). A helper killed hard leaves the
  segment until the next `create` unlinks it. An engine that stays attached to a replaced segment keeps the old
  mapping; no field detects that (ANALYSIS).
- Host memory only. No GPU runtime is referenced in the header; helper and engine are built by different toolchains
  (nvcc+gcc, clang/hipcc) and include the same file (`channel.hpp:2-4`).
- Single helper, single engine. The segment names one helper and has no owner or lock field. Nothing stops a second
  engine attaching; its plan replaces the first (the helper logs "still held an earlier engine's plan; replacing it",
  `0001:remote_experts.cpp` `seabraid_open`, `0008:ds4_helper_link.cpp:113`). Strata allows three remote tiers but
  `SEABRAID_STRATA_SHM` names one (STRATA-INTEGRATION.md section 8, item 8).
- Platform: POSIX only. The header includes `<sys/mman.h>` and `<unistd.h>` and uses `__atomic` builtins; the
  `pause` instruction is used only under `__x86_64__` (`channel.hpp:27-32`). The dsv41 engine build has a Windows stub
  where `SEABRAID_STRATA_SHM` set means open fails fast with "SEABRAID_STRATA_SHM is set, but the helper channel is POSIX shared memory
  and this build is Windows" (`0008:ds4_helper_link.cpp:10-26`). The only tested ABI is little-endian x86-64. No
  endianness field exists. Other targets: UNCONFIRMED.
- Payload types: activations and results are float32, ids are int32, all host-endian.
- Waiting: both sides busy-wait on atomics through `Channel::wait(pred, timeout_s)` (`channel.hpp:317`): spin with
  `pause` for the first 2 ms, then sleep 20 us between checks; `timeout_s == 0` waits forever. No futex. The phase-1
  gate measured a 1-token round trip at p50 4.76 us, p99 6.30 us (DESIGN-LOG, under build load; not re-measured
  here; harness `compat/strata/tests/channel_pingpong.cpp`).

## 2. Header layout

`sizeof(Header)` = **2240** bytes, `alignof` 64 (`compat/strata/tests/channel_v6_golden.txt` line 1 `header_bytes 2240`;
`compat/strata/tests/channel_v6_test.cpp:84`; `channel.hpp:82`). Offsets and sizes in the table below were
checked mechanically against a clang `-Xclang -fdump-record-layouts` dump of the real header (x86-64 Linux ABI target, header rebuilt from the patches).
The `static_assert`s that pin the preamble are at `channel.hpp:140-147`: standard layout; `magic`@0, `version`@8,
`hdr_bytes`@12, `window_max_t`@16, `tier_max_ops`@20; 8-byte `uint64_t`; lock-free 8-byte atomics.
`RoleSpec` 32 B and `LayerSpec` 112 B/align 8 are asserted at `expert_spec.hpp:44,54-55`; `TierWireOp` 20 B/align 4
at `channel.hpp:76-77`.

Writer codes: **C** creator (helper, at `create`, before `magic`), **H** helper, **M** engine ("main").
Bytes not listed are padding: `alignas(64)` on `req_seq`, `resp_seq`, `layer`, `blob_req_seq`, `blob_resp_seq`,
`svc_n`, `tier_seq`, `tier_resp`, `epoch`, `req_epoch`, and the tail 2216..2240.

<!-- HEADER-TABLE-BEGIN -->
| Off | Size | Type | Writer | Field | Meaning |
|---:|---:|---|---|---|---|
| 0 | 8 | u64 | C | `magic` | `0x31445242414553` = bytes "SEABRD1\0". Plain store, written last (`channel.hpp:36,204`). |
| 8 | 4 | u32 | C | `version` | 6 (`channel.hpp:37`). Must equal exactly. |
| 12 | 4 | u32 | C | `hdr_bytes` | `sizeof(Header)` as the creator compiled it. |
| 16 | 4 | u32 | C | `window_max_t` | Max tokens per request; both sides must agree (Strata `STRATA_VERIFY_MAX_T`). |
| 20 | 4 | u32 | C | `tier_max_ops` | `kTierMaxOps` = 64 as the creator compiled it. |
| 24 | 8 | i64 | C | `hidden` | Activation width H. |
| 32 | 8 | i64 | C | `cap` | Pick capacity = `window_max_t` x picks per token. Sizes `ids`, `x`, `out`, `weights`. |
| 40 | 8 | i64 | C | `n_layers` | Layers the channel is sized for. |
| 48 | 8 | i64 | C | `n_expert` | Experts per layer; stride of `plan` and `held`. |
| 56 | 8 | i64 | C | `max_pairs` | `n_layers * n_expert`. Used to bound `plan_count` (dsv41 helper). |
| 64 | 4 | atomic u32 | H,M | `state` | State machine, section 5.1. Release store / acquire load. |
| 68 | 4 | i32 | M | `plan_auto` | 1 = size the plan to the helper's free memory; 0 = fixed slot count. |
| 72 | 8 | f64 | H | `free_gib` | Helper VRAM free before loading. Written before `kHelperUp`. |
| 80 | 8 | i64 | M | `plan_count` | Pairs in `plan`. |
| 88 | 8 | i64 | M | `plan_slots` | Slots the helper should open (both engines write the same value as `plan_count`). |
| 96 | 8 | i64 | H | `resident` | Experts the helper actually holds. |
| 104 | 8 | f64 | H | `gib` | VRAM those experts occupy. |
| 112 | 512 | char | H | `msg` | Reason string when `state == kFailed`. |
| 624 | 128 | char | H | `helper_name` | Device name, logged by the engine. |
| 768 | 8 | atomic u64 | M | `req_seq` | Layer-request doorbell. |
| 832 | 8 | atomic u64 | H | `resp_seq` | Layer-response doorbell. |
| 896 | 8 | i64 | M | `layer` | Layer of the request. |
| 960 | 8 | i64 | M | `n_tok` | Tokens in the request. |
| 1024 | 8 | i64 | M | `k` | Picks per token in the request. |
| 1032 | 4 | i32 | H | `ok` | 1 = request served. |
| 1036 | 256 | char | H | `err` | Error text when `ok == 0`. |
| 1344 | 8 | atomic u64 | M | `blob_req_seq` | Blob-read doorbell (v2). |
| 1408 | 8 | atomic u64 | H | `blob_resp_seq` | Blob-read response. |
| 1416 | 8 | i64 | C | `blob_max` | Bytes per blob slot (largest blob of any layer); 0 = no blob region. |
| 1424 | 4 | i32 | M | `blob_n` | Blobs in this read (<= 16). |
| 1428 | 4 | i32 | H | `blob_ok` | 1 = read succeeded. |
| 1432 | 64 | i32 x16 | M | `blob_layer` | Layer of blob i. |
| 1496 | 64 | i32 x16 | M | `blob_expert` | Expert of blob i. |
| 1560 | 16 | u8 x16 | H | `blob_hit` | 1 = blob i was found and copied. |
| 1600 | 8 | atomic u64 | H | `svc_n` | Layer requests served (v3, diagnostic). |
| 1608 | 8 | atomic u64 | H | `svc_ns` | Helper service time, seen to answered, summed (v3). |
| 1616 | 8 | atomic u64 | H | `svc_begin_ns` | Part of it spent in `begin()` staging/launch (v3; Qwen helper only). |
| 1664 | 8 | raw u64 | M | `tier_seq` | Tier-round doorbell (v4). `__atomic` builtins only. |
| 1728 | 8 | raw u64 | H | `tier_resp` | Tier-round ack (v4). |
| 1792 | 8 | raw u64 | M | `epoch` | Bumped by the engine after each tier round it adopted (v4). |
| 1856 | 8 | raw u64 | - | `req_epoch` | Initialised to 0; never written or read afterwards (v4). |
| 1864 | 4 | i32 | M | `tier_n` | Ops in the round (<= 64). |
| 1868 | 4 | i32 | M,H | `tier_ok` | M sets 0 at submit; H sets 1/0 at ack. Creator sets 1. |
| 1872 | 64 | u8 x64 | M,H | `tier_res` | Per-op `TierRes`; M sets pending, H sets the result. |
| 1936 | 256 | char | M,H | `tier_err` | M clears; H writes the first failure. |
| 2192 | 4 | i32 | H | `out_rows` | Rows the helper wrote to `out`. Echo, section 5.4. |
| 2196 | 4 | i32 | M,H | `out_mode` | `OutMode`. M writes the request, H overwrites with the echo. |
| 2200 | 4 | i32 | C | `engine` | `Engine` the helper serves (v6). Only the helper writes it. |
| 2204 | 4 | i32 | M | `spec_layers` | `LayerSpec` entries main wrote (0 = no spec). |
| 2208 | 4 | i32 | M | `spec_files` | Paths main wrote in the path table. |
| 2212 | 4 | i32 | H | `helper_pid` | Helper process id (dsv41 helper only; Qwen helper leaves 0). |
<!-- HEADER-TABLE-END -->

Field provenance: `create` initialises everything it lists at `channel.hpp:188-202` (zero fill, `hdr_bytes`, `tier_max_ops`,
`tier_ok = 1`, `out_mode = kOutPerPick`, `engine = kEngineQwen`). The dsv41 helper then sets `out_mode`, `helper_pid`,
`engine`, `free_gib`, `helper_name` before `kHelperUp` (`helper_dsv41.cpp:104-110`).

Constants (`channel.hpp:36-42`, `expert_spec.hpp:18-29`): `kBlobBatch` 16, `kDeviceBase` 1000 (Strata-internal: a
`remote_dev` value >= 1000 means "the helper"; not on the wire), `kTierMaxOps` 64, `kSpecMaxFiles` 64, `kSpecPathMax` 512,
`kRoles` 3. `Engine`: `kEngineQwen` 0, `kEngineDsv41` 1. `OutMode`: `kOutPerPick` 0, `kOutPerToken` 1.
`TierRes`: pending 0, applied 1, failed 2, rejected 3 (`channel.hpp:80`). `TierWireOp.kind`: load 1, swap 2 (`channel.hpp:44`).

## 3. Regions after the header

Each region starts on a 4096-byte boundary and is padded up to the next 4096 (`Layout::of`, `channel.hpp:151-173`). The
first region starts at `up(2240)` = 4096. The engine does not trust its own arguments for the layout: after the
geometry check it recomputes `Layout::of` from the header's own `hidden`, `cap`, `n_layers`, `n_expert`, `blob_max`
(`channel.hpp:242`). The installer mirrors the formula in `compat/strata/install/_plan_shm.py` and pins it to the golden file.

Size rule. Every size is a function of the compile-time window, never of prompt length: `cap = window_max_t x picks per
token`, fixed when the helper creates the channel (Qwen helper: `kWindowMaxT * 10`, `helper.cpp:77`; dsv41 helper:
`window * 8`, `helper_dsv41.cpp:99`). Rationale in the code: every region is host RAM, and host-memory pressure is the
measured failure mode (`channel.hpp:161-165`; DESIGN-LOG D4: a 16K-token chunk at top-10 would reserve about 3.4 GB). A prompt
chunk larger than the window therefore does not compute on the helper; it borrows expert bytes through the blob region (section 5.5).

| Order | Region | Bytes before page padding | Writer -> reader | Content |
|---:|---|---|---|---|
| 1 | `plan` | `n_layers * n_expert * 2 * 4` | M -> H | `plan_count` int32 pairs `(layer, expert)`, hottest first |
| 2 | `held` | `n_layers * n_expert` | H -> M | uint8 `[layer * n_expert + e]`, 1 = resident on the helper. Stride is the channel's `n_expert` |
| 3 | `ids` | `cap * 4` | M -> H | int32 `[t*k + i]` = expert id if the helper owns that pick, else -1 |
| 4 | `x` | `cap * hidden * 4` | M -> H | float32 `n_tok` rows of `hidden` (only `n_tok * hidden` used) |
| 5 | `out` | `cap * hidden * 4` | H -> M | float32 rows, shape by `out_mode` (section 5.4) |
| 6 | `blobs` | `16 * blob_max` | H -> M | `kBlobBatch` slots, slot i at `i * blob_max` |
| 7 | `tier_ops` | `64 * 20` | M -> H | `TierWireOp[64]` |
| 8 | `weights` | `cap * 4` | M -> H | float32 `[t*k + i]` router weight of pick i of token t (v6) |
| 9 | `spec` | `n_layers * 112 + 64 * 512` | M -> H | `LayerSpec[n_layers]`, then 64 NUL-terminated paths of 512 B (v6) |

`spec_path(i)` = spec start + `n_layers * 112` + `i * 512` (`channel.hpp:257`). Weights of picks the helper does not own
are 0.0f in both engines. A region with 0 bytes still has a start: dsv41 passes `blob_max = 0`, so `blobs` and `tier_ops`
share an offset.

Worked offsets (golden file rows 1 and 3; `compat/strata/tests/channel_v6_golden.txt`):

| Region | dsv41 (H 5120, window 16 x 8, L 43, E 384, blob 0) | Qwen IQ2_XS (H 2560, window 16 x 10, L 48, E 512, blob 1510400) |
|---|---:|---:|
| plan | 4096 | 4096 |
| held | 139264 | 200704 |
| ids | 159744 | 225280 |
| x | 163840 | 229376 |
| out | 2785280 | 1867776 |
| blobs | 5406720 | 3506176 |
| tier_ops | 5406720 | 27672576 |
| weights | 5410816 | 27676672 |
| spec | 5414912 | 27680768 |
| total | 5455872 | 27721728 |

## 4. Connection handshake

### 4.1 Helper start
1. Parse geometry (`hidden`, `n_layers`, `n_expert`, window), probe the card, compute `blob_max` (Qwen: max over layers of the blob
   size; dsv41: 0).
2. `create(...)` (section 1). Set `free_gib`, `helper_name`. dsv41 also sets `engine`, `out_mode`, `helper_pid`.
3. Release-store `state = kHelperUp`. The helper loads nothing until a plan arrives.

### 4.2 Engine connect
`Channel::connect` checks, in this order (`channel.hpp:209-246`). Each failure unmaps and returns the string shown.

| # | Check | Error string |
|---:|---|---|
| 1 | `shm_open(O_RDWR)` succeeds (any errno) | `seabraid helper channel <name> not found (start seabraid-helper first)` |
| 2 | `fstat` ok and size >= the engine's `sizeof(Header)` | `seabraid channel too small` |
| 3 | `mmap` | `mmap: <strerror>` |
| 4 | acquire fence, then `magic == kMagic` and `version == 6` | `seabraid channel: wrong magic/version (helper v<N>, main v6)` |
| 5 | `hdr_bytes == sizeof(Header)` and `tier_max_ops == 64` | `seabraid channel ABI mismatch: helper sizeof(Header)=<a> tier_max_ops=<b>; main sizeof(Header)=<c> tier_max_ops=<d> (the two processes were built with incompatible toolchains or flags)` |
| 6 | `hidden`, `cap`, `window_max_t` equal; `n_layers` / `n_expert` equal only if the caller passed a value > 0 | `seabraid channel geometry mismatch: helper H=<> cap=<> layers=<> experts=<> window=<>; main H=<> cap=<> window=<>` |
| 7 | recomputed `Layout.total` <= mapped size | `seabraid channel truncated` |

Not checked by `connect`: `blob_max`, `engine`, `max_pairs`, `state`. Check 4 comes before 5 so a v4 peer
(same `hdr_bytes`, section 7) is refused by `version`, not by size.

Retry behaviour belongs to the caller. Qwen `preflight` retries up to 600 x 100 ms, returning at once on an error containing
"geometry" or "magic", then waits up to 60 s for `state >= kHelperUp`. Qwen `seabraid_open` calls `connect` once. dsv41
`HelperLink::open` retries for `SEABRAID_STRATA_CONNECT_S` (default 60 s) only while the error contains "not found"
(`0008:ds4_helper_link.cpp:96-105`), then waits up to the remaining time for `state` in {`kHelperUp`, `kLoaded`, `kFailed`}.

### 4.3 Engine-side checks after connect
- Qwen (`seabraid_open`): the helper's identity is bound to a process first (patch 0019/0020): a helper that is dead or whose pid
  was reused opens as a closed card with one `seabraid: WARN: the helper on <name> is gone (<why>); this card starts closed` line,
  whatever `state` says. For a live helper, `kShutdown` (a graceful engine stop that the helper has not yet taken back to
  `kHelperUp`) is waited for, up to 30 s; then `state` must be `kHelperUp`, `kLoaded` or `kFailed`, else
  `seabraid: the helper is not ready for a plan (state <n>); restart seabraid-helper`. It never reads `engine`.
- dsv41 (`0008:ds4_helper_link.cpp:117-125`): `engine == kEngineDsv41`, else
  `seabraid engine mismatch: the helper serves engine <n>, this is dsv41 (1)`. The channel may be larger than the model:
  `hd->n_layers >= model layers` and `hd->n_expert >= model experts`, else
  `seabraid geometry mismatch: the helper is sized for <L> layers x <E> experts, this model has <l> x <e>` (it connects with
  `n_layers = n_experts = 0` wildcards).
- The engine must also confirm the blob slots fit its largest blob (Qwen): `seabraid: the helper's blob slots are smaller than this model's blobs`.

### 4.4 Describe, plan, load
1. Engine describes (dsv41 only; Qwen leaves the spec unused). Zero the spec region, write `LayerSpec[l]` for each MoE layer
   (non-MoE layers keep `hidden == 0`: "do not serve"), write the path table, set `spec_layers`, `spec_files`, `out_mode =
   kOutPerToken` (`0008:ds4_helper_link.cpp:129-144`). Layout in `expert_spec.hpp:37-60`:
   `RoleSpec{i32 file, i32 type, u64 off, u64 stride, u64 bytes}` x 3 roles (gate, up, down) + `i32 hidden, i32 ffn, f32 swiglu_limit,
   u32 flags`. Expert e of a role lives at `[off + e*stride, +bytes)` of path `file`. `flags` bit 0 `kSpecActQ8K` = round activations
   to Q8_K blocks before low-bit dots. Engine-side fail-fast: more than 64 files, a path >= 512 bytes, a role with bad file index or 0 bytes.
2. Engine plans: walk its ranking hottest-first, skip anything it or another tier holds, stop when the helper's `free_gib` less a
   512 MiB reserve is spent. Qwen: bytes per expert = blob size rounded to 256 (max blob when the pack is not native), counted only when `plan_auto`. dsv41: sum over the 3 roles of
   `bytes` rounded to 256 (`0008:ds4_helper_link.cpp:145-163`). Errors: `seabraid: no unclaimed expert fits the helper` (Qwen),
   `seabraid: no ranked expert fits the helper (<g> GiB free)` (dsv41). Write `plan`, `plan_count`, `plan_slots`, `plan_auto`.
3. Engine release-stores `state = kPlanReady`.
4. Helper acquire-loads `kPlanReady`, loads. Qwen: `RemoteExperts::open` from its own pack. dsv41: `check_header_for_plan`
   (below), reads each path, `pread`s the planned experts, uploads. Helper zeroes `held`, sets `held`, `resident`, `gib`, re-syncs
   `resp_seq = req_seq` and `blob_resp_seq = blob_req_seq`, then release-stores `kLoaded`. On error it writes `msg` and
   release-stores `kFailed`, then waits for another `kPlanReady`.
5. Engine waits for `kLoaded` or `kFailed`: Qwen 1800 s fixed, dsv41 `SEABRAID_STRATA_LOAD_S` (default 1800 s). `kFailed` ->
   `seabraid helper: <msg>`. Timeouts: `seabraid: the helper did not finish loading in 30 minutes` (Qwen),
   `seabraid: the helper did not finish loading its plan in time` (dsv41).
6. Engine copies `held` as its routing view and starts its sequence counters at the header's current `req_seq` / `blob_req_seq`.
   dsv41 fails with `seabraid: the helper loaded none of the planned experts` if `held` is empty. A helper may hold fewer
   experts than planned; the engine must route by `held`, not by its plan.

dsv41 helper refusals (`compat/strata/strata-ext/src/dsv41_plan_check.hpp:12-31`, strings exact; each becomes `msg` + `kFailed`):
`engine mismatch: main expects engine <n>, this helper serves dsv41 (1)`; `main wrote no usable spec (spec_layers=<n>, channel layers=<n>)`
(needs `0 < spec_layers <= n_layers`); `spec_files=<n> is outside 1..64`; `plan_count <n> exceeds the channel's pairs`;
`spec path <i> is empty or not NUL-terminated` (`helper_dsv41.cpp:41`). Because only the helper writes `engine`, the engine check cannot
fail in practice; a Qwen/dsv41 mix is caught by geometry (check 6) first. A plan naming a layer past `spec_layers` is refused by the executor
(`helper_dsv41.cpp:46`).

Crash detection while loading: dsv41 engine reads `helper_pid` and polls `kill(pid, 0)`; ESRCH gives `seabraid: the helper process (pid
<n>) died while loading its plan; see its log` (`0008:ds4_helper_link.cpp:170-177`). A helper that crashes never writes `kFailed`.

## 5. Per-layer request cycle

### 5.1 State machine (`State`, `channel.hpp:42`)
| Value | Name | Written by | Meaning |
|---:|---|---|---|
| 0 | `kNone` | creator | Header zeroed, helper not ready. |
| 1 | `kHelperUp` | helper | Probed, `free_gib` set, waiting for a plan. Also written by the helper after `kShutdown`. |
| 2 | `kPlanReady` | engine | Plan (and spec) written. Stays set while the helper loads. |
| 3 | `kLoaded` | helper | Plan loaded, `held` valid, serving requests. |
| 4 | `kFailed` | helper | Load refused, `msg` says why. |
| 5 | `kShutdown` | engine | Engine is done (destructor / `close()`). |

Legal transitions: `kNone -> kHelperUp` (H); `kHelperUp|kLoaded|kFailed -> kPlanReady` (M; a new plan from `kLoaded` or `kFailed`
is how a restarted engine replaces the helper's content); `kPlanReady -> kLoaded|kFailed` (H); `kLoaded -> kShutdown` (M);
`kShutdown -> kHelperUp` (H). There is no state for "helper gone": the creator unlinks the segment and the engine learns
only from timeouts or `helper_pid`. A Qwen engine that connects between `kShutdown` and the helper's reset sees `kShutdown`
and refuses; the dsv41 engine waits for `kHelperUp`/`kLoaded`/`kFailed` instead (section 4.2).

### 5.2 Request (engine -> helper)
One request in flight at a time. The engine does not overwrite an unanswered request (Qwen `seabraid_helper_begin` first finishes the
pending one; dsv41 `take` returns false while one is pending).
1. Write `ids` (-1 for picks the helper does not own), `x` (`n_tok` rows), `weights` (per-token mode), `layer`, `n_tok`, `k`, `out_mode`,
   clear `ok`/`err` (dsv41) and set `out_rows = -1` (both engines, so a stale count cannot pass the check).
2. Release-store `req_seq = ++seq`. Qwen sends nothing if the helper owns none of the picks.
3. Engine-side validity: Qwen `0 < n_tok*k <= cap`, `n_tok <= window`, `k == 10`, layer present; dsv41 `1 <= T <= 16`, every
   owned slot `< T*8`. Helper-side: dsv41 `n_tok >= 1`, `k >= 1`, `n_tok*k <= cap`
   (`request n_tok=<n> k=<k> does not fit the channel (cap <c>)`); Qwen per-token `1 <= n_tok <= window`.

### 5.3 Response (helper -> engine)
Helper acquire-loads `req_seq != seen`, reads the request, computes, then writes `out`, `out_rows`, `out_mode` (echo), `ok`, `err`
(on failure), service counters, and release-stores `resp_seq = req_seq`. The engine waits for `resp_seq == seq`
(acquire), reads `ok`, then validates the echo, then reads `out`. Qwen helper writes `out_rows` even on failure; the dsv41 helper writes 0
on failure (`helper_dsv41.cpp:162-164`).

### 5.4 Output modes and the echo
- `kOutPerPick` (0): `n_tok*k` rows, row i = the unweighted expert output of pick i; rows of picks not owned are untouched. The engine
  copies only the rows it knows it asked for. Qwen default. The Qwen engine does not check `out_rows` in this mode.
- `kOutPerToken` (1): `n_tok` rows, row t = sum over t's owned picks of `weights[t*k+i] * expert(x_t)`; a token with no owned
  pick gets a zero row (`expert_spec.hpp:25-29`). dsv41 always; Qwen only with `--remote-expert-opt` (patch 0010).
- Echo: the engine requires `out_mode == kOutPerToken` and `out_rows == n_tok` for a per-token request, else the request fails
  (`seabraid helper: answered out_mode <m>, <r> rows to a per-token request of <n> tokens` on Qwen; `the helper wrote <r> rows
  for a <n>-token request` on dsv41). Never partial or mixed data.

### 5.5 Prompt-path blob read (v2)
Engine writes `blob_layer[i]`, `blob_expert[i]` (<= 16), `blob_n`, release-stores `blob_req_seq`. Helper copies each held expert's
blob into slot i (`blob_hit[i] = 1`), sets `blob_ok`, release-stores `blob_resp_seq`. The engine copies `blob_bytes(layer)` from each
hit slot (`0001:remote_experts.cpp` `copy_cached_blobs`). Timeout 30 s: `seabraid: the helper did not answer a cache read within 30 s`.
The dsv41 helper answers every blob request with all-miss and `blob_ok = 0`; the dsv41 engine never sends one.

### 5.6 Memory ordering
| Publisher stores (plain) then release | Consumer acquire then reads |
|---|---|
| Creator: all header fields, release fence, plain `magic` | Engine: acquire fence, then reads `magic` and the rest (`channel.hpp:203,217`) |
| Helper: `free_gib`, `helper_name`, `helper_pid`, `engine`, then `state = kHelperUp` | Engine: acquire `state`, then reads them |
| Engine: `plan*`, spec region, `spec_*`, `out_mode`, then `state = kPlanReady` | Helper: acquire `state` in `wait`, then reads |
| Helper: `held`, `resident`, `gib` or `msg`, `resp_seq` sync, then `state = kLoaded`/`kFailed` | Engine: acquire `state`, then reads |
| Engine: request fields and regions, then `req_seq` | Helper: acquire `req_seq` |
| Helper: `out`, `out_rows`, `out_mode`, `ok`, `err`, `svc_*`, then `resp_seq` | Engine: acquire `resp_seq` |
| Blob read: same pair with `blob_req_seq` / `blob_resp_seq` | |
| Tier round: `tier_seq` / `tier_resp` via `__atomic_*` (`channel.hpp:281-307`); `epoch` release store / acquire load | section 6 |
The helper's serve loop reads `state` relaxed (`helper.cpp:151`, `helper_dsv41.cpp:141`); a `kPlanReady` seen there only exits the
loop, and the outer wait re-reads it with acquire. `req_seq`, `resp_seq`, `blob_*_seq`, `svc_*` are `std::atomic<uint64_t>` (v1-v3
fields); the v4 counters are raw `uint64_t` because `std::atomic`'s representation is not guaranteed equal across toolchains
(`channel.hpp:116-120`). `compat/strata/tests/channel_v4_race_test.cpp` checks the tier pair under ThreadSanitizer (one mapping, two threads).

### 5.7 Timeouts and failure behaviour (engine side)
| Wait | Qwen engine | dsv41 engine |
|---|---|---|
| connect / helper up | 60 s + 60 s (`preflight`) | `SEABRAID_STRATA_CONNECT_S`, 60 s |
| plan load | 1800 s fixed | `SEABRAID_STRATA_LOAD_S`, 1800 s, plus `helper_pid` liveness |
| layer response | 30 s, fixed: `seabraid: the helper did not answer layer <n> within 30 s` | `SEABRAID_STRATA_TIMEOUT_S`, 10 s |
| blob read | 30 s | not used |
| tier round | 60 s | not used |
| on a failed request | error returned to the caller; request fails | one stderr line, link marked dead, this layer's picks and all later helper picks computed on the engine's CPU lane |
Fail-open is possible only because the dsv41 engine still holds every expert in its own mapped model file. After open, a dsv41 link never
revives. Details and the policy question: STRATA-INTEGRATION.md section 2.7 and section 8, item 3.

## 6. Re-tiering and residency (v4/v5 ops; Qwen path only)

The dsv41 helper rejects every tier op: it sets each `tier_res[i] = kTierRejected`, writes `tier_err = "dsv41 helper does not support
live re-tiering"` and acks with `ok = false` (`helper_dsv41.cpp:62-67`); the dsv41 engine never submits one.

`TierWireOp` (20 bytes, `channel.hpp:65-77`): `u8 kind` (1 load, 2 swap), `u8 helper` (always 0), `u16 pad_`, `i32 layer_old`,
`i32 expert_old`, `i32 layer_new`, `i32 expert_new`. Load admits `(layer_new, expert_new)` into unclaimed capacity (old fields -1).
Swap fills `expert_new` into the slot of `expert_old` and replaces it; it is same-layer only because slot size depends on the layer and
a bigger expert into a smaller slot overruns the next slot (`channel.hpp:57-64`). There is no evict op (STRATA-INTEGRATION.md 2.2).

Round (`seabraid_retier`, `0001:remote_experts.cpp`; engine thread only):
1. Engine finishes any request in flight and closes admission (`bl_fenced_`). A request admitted during a round fails.
2. Engine writes `tier_ops[0..n)`, sets `tier_res[i] = pending`, `tier_n = n` (1..64), `tier_ok = 0`, clears `tier_err`, then release-stores
   `tier_seq = old + 1` (`tier_submit`). A bigger `n` is clamped to 64.
3. Helper: `tier_pending()` acquire-loads `tier_seq`; nonzero if `tier_seq > tier_resp`. It applies each op (`tier_apply`), writes `tier_res[i]` =
   applied(1) or failed(2), records only the first failure in `tier_err`, then does ONE device sync for the whole batch.
4. Helper writes `held[]` for every cell the round touched, from the cache's own answer (`publish_tier_residency`, `channel.hpp:365`), updates
   `resident`/`gib`, then `tier_ack(seq, ok)`: set `tier_ok`, release-store `tier_resp = seq`. The ack must come after the device work is done and
   after `held[]` is written, because a CPU release store says nothing about an outstanding device copy.
5. Engine waits for `tier_resp >= seq` (60 s: `seabraid retier: the helper did not finish tier round <n> within 60 s`), copies `tier_res`, adopts the whole
   `held[]` even if the round failed (a partly applied round still moved experts), and then runs the stale-view check.
6. Stale-view check (`tier_residency_consistent`, `channel.hpp:381`): replay, in order, the ops reported applied; every cell they touch must
   agree with `held[]`. Otherwise bump `epoch` and fail with `seabraid retier: helper residency is STALE after an applied round (<cell>)`.
7. Engine release-stores `epoch += 1`, reopens admission. A refused op gives `seabraid retier: the helper refused at least one op[: <tier_err>]`.
Engine must not publish a layer request between its last drained response and `tier_submit`; that is the caller's job.

`epoch` is advanced by the engine but no field it would be compared with is ever written, so it protects nothing today
(`req_epoch` is never stamped; STRATA-INTEGRATION.md section 7, item 2).

## 7. Version history

`kVersion` is exact-match (`channel.hpp:37,218`). The repo has one commit; the patch series (0001) first publishes the channel at v4.
Earlier layouts are not in the repo.

| Ver | Added | Evidence | Status |
|---:|---|---|---|
| 1 | Shared-memory segment, state machine, plan/load handshake, `req_seq`/`resp_seq` layer request with `ids`/`x`/`out`. | Protocol block `channel.hpp:6-9`; DESIGN-LOG phase-1 gate and "v0.0.2 built and measured". | Existence implied; exact layout and whether `magic`/`version` existed: UNCONFIRMED. |
| 2 | Prompt-path cache reads: `blob_req_seq`, `blob_resp_seq`, `blob_max`, `blob_n`, `blob_ok`, `blob_layer`, `blob_expert`, `blob_hit`, `blobs` region. | Comment `channel.hpp:105` ("v2"). | Field set CONFIRMED by comment; date UNCONFIRMED. |
| 3 | Helper service-time counters `svc_n`, `svc_ns`, `svc_begin_ns`. | Comment `channel.hpp:112`; DESIGN-LOG 2026-11-06 (b) "v3 channel". | CONFIRMED. |
| 4 | Negotiation preamble at fixed offsets (`hdr_bytes`, `window_max_t`, `tier_max_ops`); `TierWireOp`, tier round counters and the `tier_ops` region; `tier_n`, `tier_ok`, `tier_res`, `tier_err`; `out_rows`, `out_mode`; raw `__atomic` counters. Patch 0001 subject: "v4: negotiation preamble, layer requests, prompt-path blob reads, tier LOAD/SWAP ops". | `channel.hpp`, patch 0001 message; `compat/strata/tests/channel_v4_test.cpp`. | CONFIRMED. Whether v3 already had the preamble: UNCONFIRMED. |
| 4 (+0005) | `publish_tier_residency` and `tier_residency_consistent`: helper writes `held[]` after a round, engine refuses a contradicting round. No layout change. | Patch 0005 message (21 of 47 rounds refused before the fix, 2026-11-06 D7 arms). | CONFIRMED. |
| 5 | Reserved: "the neutral STRXHLP tag on the v4 layout". No code in this repo emits it. | `channel.hpp:37`; STRATA-INTEGRATION.md 2.8. | Contents UNCONFIRMED here (separate review, not in repo). |
| 6 | `engine`, `spec_layers`, `spec_files`, `helper_pid`; `weights` and `spec` regions; `expert_spec.hpp`; `OutMode`. | Patch 0007; `compat/strata/tests/channel_v6_test.cpp`; consumers 0008 (dsv41) and 0010 (Qwen `kOutPerToken`). | CONFIRMED. |
| 7 | The D4 prefill fields (`pf_*`) and the optional prefill region after `total`; `PrefillFormat`, `kPfRowsF16/F32`. | Patch 0013; `compat/strata/tests/pf_service_test.cpp`, `compat/strata/tests/channel_v6_test.cpp` (kVersion 7, golden header 2688); engine `remote_prefill_test`. | Built and host-tested; live runs on the earlier D4 builds are in DESIGN-LOG. |

Size note (measured with clang from the patches, ANALYSIS of the history): the v4 header from patches 0001+0005 is also 2240 bytes, and the four v6 fields
(offsets 2200..2216) sit in what was v4's tail padding. So `hdr_bytes` cannot tell v4 from v6; `version` is the only discriminator.

### v7 (patch 0013, D4 prompt reading, opt-in)

v7 is v6 plus fields APPENDED to the header and one optional region appended AFTER `Layout::of(...).total`; every v6 offset is where it was
(`static_assert(offsetof(Header, pf_tokens) > offsetof(Header, helper_pid))`), and `sizeof(Header)` grows from 2240 to **2688** (still under the
4 KiB page that starts the first region, so no region moves and `compat/strata/install/_plan_shm.channel_bytes()` is unchanged). `kVersion` is exact-match,
so a v6 helper and a v7 engine refuse each other at connect with the usual version message: rebuild both from the same series.

| Field (header) | Writer | Meaning |
|---|---|---|
| `pf_tokens`, `pf_k`, `pf_format` | helper (creator) | The prefill region's capacity in TOKENS per request, its pick slots per token, `kPfFp16`. `pf_tokens` 0 = no region (the default: `seabraid-helper` without `--prefill-tokens`) |
| `pf_layer`, `pf_ntok`, `pf_row_bytes` | engine | This request's layer, token count, and row element size (`kPfRowsF16` = 2 or `kPfRowsF32` = 4, declared per request) |
| `pf_req_seq` / `pf_resp_seq` | engine / helper | Raw uint64 + `__atomic` request pair, like the v4 tier counters. The helper answers EVERY request (`pf_service.hpp`), a failed one with `pf_ok` 0 |
| `pf_ok`, `pf_picks`, `pf_err[256]` | helper | 1 = every asked pick computed and every sum finite; echo of the picks computed (main compares); the reason on failure |
| `pf_svc_ns`, `pf_svc_n` | helper | Cumulative service time and requests (the cost model's measured rate) |

The region (`PrefillRegion::of(hidden, k, tokens)`) is sized by tokens, never by pairs: per token `(row_bytes + 2) * H + 8 * K` bytes
(fp16 rows 4H + 8K, f32 rows 6H + 8K; the first part has room for f32 rows). A chunk longer than `pf_tokens` goes as several requests. It is
`ftruncate`d, so /dev/shm commits a page only when it is first written. Engine switch: `STRATA_PREFILL_REMOTE_COMPUTE=1` (cost model per
chunk) or `=2` (every eligible chunk); unset = the lending path of v6, unchanged.

## 8. Engine-neutral vs Strata-specific

Classes: **(a)** shared, any engine needs it; **(b)** Strata-specific; **(c)** model or quant specific. The second engine's evidence is the dsv41
adapter (patch 0008, `helper_dsv41.cpp`): what it added or dropped relative to the Qwen path shows what a second engine actually needed.

| Item | Class | Evidence |
|---|---|---|
| Preamble (`magic`, `version`, `hdr_bytes`, `window_max_t`, `tier_max_ops`), connect refusals | (a) | dsv41 uses `connect` unchanged |
| `hidden`, `cap`, `n_layers`, `n_expert`, `max_pairs`; wildcard layers/experts | (a) | dsv41 added wildcard use (`0,0`) and `>=` checks |
| State machine, `msg`, `free_gib`, `helper_name`, `resident`, `gib` | (a) | both helpers |
| `plan` / `held`, hottest-first plan, 512 MiB reserve | (a) concept; the arithmetic is engine policy | both engines re-implement it |
| `req_seq`/`resp_seq`, `layer`, `n_tok`, `k`, `ids`, `x`, `out`, `ok`, `err` | (a) | both |
| `out_mode`, `out_rows` echo, `weights` | (a) | added for dsv41; Qwen adopted for `--remote-expert-opt` |
| `helper_pid` | (a) | added for dsv41 (a crashed helper never sets `kFailed`); the Qwen helper does not publish it |
| `svc_*` counters | (a) diagnostic; `svc_begin_ns` is (b) (Strata `begin()` staging) | dsv41 helper never adds `svc_begin_ns` |
| Spec region (`LayerSpec`, `RoleSpec`, paths) | (a) mechanism, (c) content | needed because dsv41's pack parser is private to the engine |
| `RoleSpec.type`, `swiglu_limit`, `kSpecActQ8K`, `kRoles = 3`, ggml type ids | (c) | DeepSeek MoE FFN, dp4a numerics |
| `Engine` enum (`Qwen`, `Dsv41`) | (b)/(c): names two model families | helper-written; only dsv41 reads it |
| Pick width: Qwen k = 10, dsv41 k = 8; window compile-time in both | (c) | `cap = window x k` |
| `kDeviceBase` 1000 (remote tier index) | (b) Strata internals | not on the wire |
| Blob region and prompt-path read | (b) | exists because Strata's prompt path reads remote cache bytes; dsv41 answers all-miss, `blob_max = 0` |
| Tier ops (load, same-layer swap), `held[]` republish, `epoch`, `req_epoch`, `TierWireOp.helper` | (b) | tied to `ExpertCache` semantics (no evict, per-layer slot size); dsv41 rejects every op |
| `plan_auto`, `plan_slots` | (b) | dsv41 always writes `plan_auto = 1`, `plan_slots = plan_count` |
| Fail-open vs fail-fast after open | (a) question, answers differ by engine | Qwen returns an error; dsv41 recomputes on CPU |
| `SEABRAID_STRATA_*` env names, timeouts 10/30/60/1800 s | (b) | each engine chose its own |

What llama.cpp would need that is not there (ANALYSIS, nothing here was tried on llama.cpp):
1. Runtime-negotiated maxima. `window_max_t` is a compile-time constant of the Strata kernels and must be equal on both sides. llama.cpp's batch
   sizes are runtime values and prefill batches are large; a channel sized `cap x hidden` at 512 tokens x 8 picks x H 4096 would be
   512*8*4096*4 B = 64 MiB each for `x` and `out` (ESTIMATE, arithmetic only).
2. A richer expert description. Three fixed roles (gate, up, down), one activation (SwiGLU with an optional clamp), float32 activations. Fused gate+up
   tensors, other activations, shared experts, router bias/normalisation, and bf16/f16 activations have no field.
3. An engine hook for submit/collect inside a layer. The protocol assumes the engine can publish a request mid-layer, compute its own
   experts, and join later; ggml's scheduler works at graph level (ANALYSIS, depends on llama.cpp internals not read here).
4. Static ownership without a cache: `held[]` can express it, but the ranking/plan policy and the tier ops assume a Strata-like expert cache.
5. Any liveness signal beyond `helper_pid` (which the Qwen helper does not publish).

## 9. Open questions for a public v1

1. Neutral tag instead of `"SEABRD1"`; reset numbering to 1 or keep v6/v5 history (STRATA-INTEGRATION.md section 8, item 2).
2. Capability exchange instead of exact version equality: the helper advertises supported engines/quant types/activations/output modes/max
   tokens, the engine selects. Today any new field needs a version bump and both sides rebuilt.
3. Replace compile-time `window_max_t` coupling and `hdr_bytes` equality with negotiated maxima and a header size that each side may
   read as "at least N bytes".
4. `engine` is written by the helper and only read by the engine; should the engine declare what it needs and the helper check it?
   Replace the enum by a string or capability id.
5. Define 8-byte sequence fields as plain aligned `u64` with acquire/release, not `std::atomic` (as v4 already does for the tier counters).
   Specify that `magic` is published with release semantics and read with acquire semantics (today a plain store, and the reader's fence comes
   before the read).
6. Creation race: between `ftruncate` and the `magic` store a reader can see `too small` or a zero `magic`; dsv41 retries only "not found" and
   Qwen `preflight` retries all but "geometry"/"magic" (ANALYSIS: window is microseconds; not measured).
7. Stale mapping after a helper replaces the segment: add a creation nonce so the engine can tell (ANALYSIS).
8. `req_epoch`: stamp and check, or remove. `epoch` as it stands protects nothing.
9. Uniform failure model: require `helper_pid` (or a heartbeat) from every helper; decide fail-open versus fail-fast; make the timeouts
   fields of the header or capabilities, not per-engine constants.
10. Is the blob region and the whole re-tier family a capability that a minimal helper may omit? The dsv41 helper already answers them with
    "no" instead of silence (`helper_dsv41.cpp:51-67`); that reply is the de-facto contract and should be written down.
11. Per-pick echo: check `out_rows` in per-pick mode too (the Qwen engine does not).
12. More than one helper and more than one engine per host: names, ownership marker, `TierWireOp.helper`.

## 10. Prior work: the llama.cpp RPC backend

Read from llama.cpp at 058df671 (2026-08-17), `ggml/include/ggml-rpc.h` and `ggml/src/ggml-rpc/ggml-rpc.cpp`.
Check it again before designing v1, both to borrow from it and to say what Strata integration adds.

- What it is: a whole ggml backend proxied to an `rpc-server` process over TCP. It works at the level of
  buffers, tensors and graphs (18 commands: alloc/free buffer, set/get/copy tensor, graph compute, device memory,
  HELLO, and others), not experts. The project's own README calls it a proof of concept and fragile.
- Framing: request = 1-byte command, 8-byte size, payload; response = 8-byte size, payload.
- Versioning: HELLO returns major.minor.patch (5.0.0 at that commit). The client accepts a server with the same
  major and a minor no newer than its own. Both sides exchange `conn_caps` in HELLO and upgrade the transport
  (RDMA when built with it) only if both advertise it. This is the shape open questions 2 and 3 ask for.
- Weight upload: a large tensor is sent as an FNV-1a hash first; the server loads it from its on-disk cache by
  hash and the bytes cross the wire only on a miss.
- Overlap with Strata integration (ANALYSIS, not measured): llama.cpp's `--override-tensor` can already place expert tensors
  on an RPC device, which gives a static split with experts on another GPU. What it does not have: an expert-level
  request (ids + router weights in, rows out), a resident set that changes at run time (section 6), or a
  shared-memory transport for a helper on the same host. Those are what a Strata integration protocol adds; the llama.cpp
  integration should be measured against `-ot` + RPC first.
