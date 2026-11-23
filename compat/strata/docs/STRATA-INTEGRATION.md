# Running Strata integration as a Strata plugin: what Strata would need to carry

Status: requirements list, written against the code in `compat/strata/patches/strata/` (series 0001-0006 and 0010-0020; the wire header 0001-0016 used to create is `adapters/strata/wire`) and `compat/strata/strata-ext/`,
pinned to Strata v0.2.15 (the commit in `compat/strata/patches/strata/STRATA_BASE`, its one owner; rebased from v0.2.13.1 (`628f74955`) on 2026-11-17, and from v0.2.12 on 2026-11-09 before that). Patches 0008 and 0009 (the DeepSeek-V4.1 helper link) were dropped in that rebase because Strata 0.2.14 removed its DeepSeek-V4.1 engine; every passage below that describes the DeepSeek helper, link or executor is history as of v0.2.13.1, and the code it describes was deleted from the add-on. Strata is MIT-licensed
(upstream: Niko1221/Strata). Nothing here has been proposed upstream yet.

Reading rules for this document:

- Every number is copied from `compat/strata/docs/DESIGN-LOG.md` with its hardware and sample size. "n=1" means one run or one text.
  A number with no stated sample size says so.
- "Verified" means a command was run and its output was read. "Unverified" means it was built, reasoned about, or
  not run at all. Where the code and an earlier design sketch disagree, the code is described and the disagreement
  is listed in section 7.
- Hardware named in measurements: Radeon AI PRO R9700 32 GB (gfx1201, HIP), GeForce RTX 3090 24 GB, RTX 3060 12 GB,
  RTX 4080 (all CUDA, sm_86 or sm_89), one Linux system with ~30 GB of RAM.

## 1. What Strata integration needs from an engine

Strata integration lets a GPU of a different make than the one the engine was built for hold and compute a share of a Mixture
of Experts model's routed experts. The engine (built for one vendor's runtime, e.g. HIP) keeps everything else:
attention, dense layers, KV cache, router, drafter, its own expert cache. A second process built for the other
vendor's runtime (e.g. CUDA) owns a disjoint set of experts in its own VRAM. For every MoE layer the engine (a)
decides which routed experts land on the helper, (b) hands the helper those activation rows, expert ids and router
weights through shared host memory, (c) keeps computing its own experts while the helper works, and (d) joins the
helper's partial sums into the layer output before continuing. Outside the layer path the engine also needs to
(e) tell the helper which experts to load, (f) borrow the helper's resident expert bytes while it reads a prompt,
(g) optionally move experts between the two caches while serving, and (h) survive the helper being absent, slow,
stale or dead. No P2P and no shared GPU runtime is involved; the only data path is a POSIX shared-memory region
that both runtimes read and write as ordinary host memory.

Why this belongs in the engine rather than beside it: the engine owns the router outputs, the cache residency,
the layer join and the maintenance threads that move experts. A helper cannot observe any of those from outside.
The engine-side surface is small (the channel header, a helper mode of the remote expert tier, about 50 lines of
wiring in the generate path). The helper program and every placement decision can stay outside the engine.

## 2. The interface Strata would own

### 2.1 Scope and names

This is a proposal for the engine-neutral surface. It is derived from what the code does today (names in the
right-hand column), not from a new design. Namespace `strata::expert_helper`; the current code uses `seabraid::`.

| Concept | Today in the series |
|---|---|
| `Channel` (shm region, versioned header, regions) | `adapters/strata/wire/strata/seabraid/channel_layout.hpp` + `channel.hpp` (Seabraid-owned; patches 0001-0016 created it in the Strata tree until 2026-11-19) |
| `ExpertSpec` (`LayerSpec`, `RoleSpec`, path table) | `adapters/strata/wire/strata/seabraid/expert_spec.hpp` (was patch 0007, dropped when the wire moved) |
| Engine side: `HelperClient` | `RemoteExperts::seabraid_open/begin/finish/retier` (Qwen-family engine, patches 0001, 0004, 0005, 0010; loss handling in 0019 and 0020). The DeepSeek `HelperLink` (patch 0008) was dropped 2026-11-17 |
| Helper side: `HelperServer` | `compat/strata/strata-ext/src/helper.cpp` and `helper_serve.cpp` (Qwen-family). `helper_dsv41.cpp` was removed with the DeepSeek engine |
| Placement policy | `compat/strata/strata-ext/src/tier_plan.cpp`, `retier_loop.cpp`, `card_bench.cpp`, patch 0006: stays outside Strata |

### 2.2 Types

```cpp
namespace strata::expert_helper {

// Which executor the helper runs. Declared by the helper at create time (section 7, item 3, says what is checked).
enum Engine : int32_t { kEngineQwen = 0, kEngineDsv41 = 1 };

// Shape of the rows a layer request returns (Header::out_mode, echoed by the helper in Header::out_rows).
//  kOutPerPick : n_tok*k rows, one UNWEIGHTED row per helper-owned pick; unowned rows untouched.
//  kOutPerToken: n_tok rows; row t = sum over t's helper-owned picks of weights[t*k+i] * expert(x_t);
//                a token with no owned pick gets a zero row.
enum OutMode : int32_t { kOutPerPick = 0, kOutPerToken = 1 };

// Channel state machine (Header::state). Written with release stores, read with acquire loads.
enum State : uint32_t { kNone, kHelperUp, kPlanReady, kLoaded, kFailed, kShutdown };

// One tier round op. 20 bytes, alignof 4, size-asserted: it is the only struct that crosses the process line as data.
// kTierLoad : admit into UNCLAIMED helper capacity and fill (layer_old/expert_old = -1).
// kTierSwap : fill expert_new into the slot of expert_old, then replace (SAME LAYER only: slot sizes differ per layer).
// There is no EVICT on the wire, deliberately: ExpertCache has no eviction, so a bare "mark non-resident" would
// strand its slot for good.
struct TierWireOp { uint8_t kind; uint8_t helper; uint16_t pad_; int32_t layer_old, expert_old, layer_new, expert_new; };
enum TierRes : uint8_t { kTierPending, kTierApplied, kTierFailed, kTierRejected };

// Engine-neutral description of one MoE layer (v6). Main describes, the helper only executes: the pack format and
// its parser belong to the engine, so a helper never re-parses it. 32-byte RoleSpec, 112-byte LayerSpec, both asserted.
struct RoleSpec  { int32_t file; int32_t type; uint64_t off, stride, bytes; };   // expert e = [off+e*stride, +bytes) of file
struct LayerSpec { RoleSpec role[3]; int32_t hidden, ffn; float swiglu_limit; uint32_t flags; };  // hidden==0: not served

struct Geometry { int64_t hidden, cap, n_layers, n_expert; uint32_t window_max_t; };  // both sides must agree
}
```

### 2.3 Calls (engine side)

All calls return `false` with an error string; none throws across the boundary. Signatures below are the
existing ones, generalised only by naming. `Channel` is the shared region; the engine wraps it.

```cpp
class HelperClient {                        // engine side; one per helper process
public:
  // OPEN. connect() tries once and refuses on a missing region or a magic/version/ABI/geometry mismatch, with a
  // message naming both sides; waiting for the helper is the caller's loop (today: Qwen preflight retries for ~60 s,
  // dsv41 for SEABRAID_STRATA_CONNECT_S on "not found" only). Then writes the spec (if the engine uses one), sends a plan,
  // and waits for kLoaded.
  bool open(const char* shm_name, const Geometry&, const SpecView* spec /*nullptr for Qwen*/,
            const std::vector<std::pair<int32_t,int32_t>>& ranked /*hottest first, minus what main holds*/,
            std::vector<uint8_t>& claimed, std::string& err);            // = RemoteExperts::seabraid_open, HelperLink::open

  // Does the helper own (layer, expert) right now? Cheap; read from several threads.
  bool holds(int64_t layer, int32_t expert) const;                        // reads the adopted held[] bitmap

  // SUBMIT one layer. Splits the picks into helper-owned and not, publishes only the owned ones, returns at once.
  // Engine thread only. x: n_tok rows of H floats; ids: n_tok*k; weights may be nullptr for kOutPerPick.
  bool submit(int64_t layer, const float* x, const int32_t* ids, const float* weights,
              int64_t n_tok, int64_t k, OutMode mode, std::string& err);  // = seabraid_helper_begin, HelperLink::take

  // COLLECT. Waits for the response, verifies the echo (out_rows, out_mode), merges rows into `out`.
  bool collect(float* out, std::string& err);                             // = seabraid_helper_finish, HelperLink::finish

  // PROMPT PATH. Copy resident expert blobs helper->engine through the blob slots, 16 per round trip.
  bool read_blobs(const int64_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n, uint8_t* hit, std::string&);

  // TRANSITION. Fence, apply 1..64 LOAD/SWAP ops, adopt the helper's published residency, move the epoch.
  // Engine thread only (checked at run time). Never overlaps a layer request.
  bool transition(const TierWireOp* ops, int32_t n, uint8_t* per_op_result,
                  std::vector<uint8_t>* claimed, std::string& err);       // = seabraid_retier (Qwen only today)

  void close();                                                           // publishes kShutdown; helper returns to kHelperUp
};
```

Helper side (`HelperServer`) is a loop over the same channel: `create` the region, publish `free_gib`, set
`kHelperUp`, wait for `kPlanReady`, load, publish `held[]`/`resident`/`gib`, set `kLoaded`, then serve `req_seq`
(layer requests), `blob_req_seq` (blob reads) and `tier_seq` (tier rounds) until `kShutdown`, `kPlanReady` again
(a restarted engine) or a signal. The helper is not part of the proposal's engine surface; Strata would ship the
reference helper as a build target (section 6, item 4) but any process that speaks the channel qualifies.

### 2.4 Lifecycle

1. Helper starts, creates the shm region, probes its card, publishes `free_gib`, `state = kHelperUp`.
2. Engine `open`: connect (retrying only "not found"), wait up to 120 s for the channel to leave `kNone`, then bind the
   helper's identity (`PeerWatch`: `helper_pid` and `helper_start`) BEFORE reading the state word (patch 0020). A helper that is
   gone, or whose pid was reused, opens as a closed card (code 6) with one WARN, whatever the state word says. For a live helper
   the state must be `kHelperUp`, `kLoaded` or `kFailed` (the last two mean an earlier engine died without `kShutdown`; the helper
   reloads on the new plan), or `kShutdown`, which is waited out for at most 30 s because a live helper returns to `kHelperUp`;
   any other state is refused. Then build the plan, `state = kPlanReady`, and wait up to 30 minutes for `kLoaded` or `kFailed`.
3. Plan: engine walks its ranking hottest-first and takes what its own caches do not hold, until the helper's
   free memory less a 512 MiB reserve is spent (256-byte blob rounding; both engines use the same arithmetic).
4. Serve rounds: per layer `submit` then `collect`; the engine computes its own experts between the two calls.
   Prompt reading additionally uses `read_blobs`.
5. Tier round (Qwen only today): drain any request in flight, close admission, `tier_submit`, wait up to 60 s for
   `tier_resp`, copy per-op results, adopt `held[]` unconditionally (a partly applied round still moved experts),
   check it against the ops reported applied, bump `epoch`, reopen admission. See 2.5.
6. Shutdown: engine destructor drains, sets `kShutdown`; helper sets `kHelperUp` and waits for a new plan.

### 2.5 The tier fence (the one subtle ordering rule)

A layer request names an expert by (layer, expert); the helper answers from whatever its slot holds at that
moment. A request that overlaps a swap is answered out of a slot between two experts: well-formed and
numerically wrong, and nothing in the wire detects it. So `transition` applies ops with no request outstanding
and none admitted until all of the following happened, in this order: helper applied every op, helper waited
for its device copies to finish (one stream sync per batch), helper wrote `held[]` from its cache's own answer
(`publish_tier_residency`), helper release-stored `tier_resp`; engine acquire-loaded it, adopted `held[]`,
checked it against the applied ops (`tier_residency_consistent`, refuses with "helper residency is STALE"),
bumped `epoch`. The bug class this closes was real: before patch 0005 `held[]` was written only at plan load, and
rows for a displaced expert were silently dropped (21 of 47 and 27 of 47 rounds refused in the first arms, per
DESIGN-LOG 2026-11-06/07). The ack is a CPU release store and says nothing about an outstanding device copy,
which is why the helper syncs before acking.

### 2.6 Threading rules (as implemented)

- Qwen engine path: `submit`/`collect`/`transition` run on the engine thread. `transition` checks the thread id
  against the one captured at `open` and refuses otherwise. `read_blobs` takes a mutex (the prompt path can be
  another thread). `holds` is read by the adaptive-tier thread; `held[]` changes only inside a fenced
  `transition`, which runs at a decode-window boundary when the adaptive thread has been joined.
- Loss detection: the `PeerWatch` thread sets one atomic and never acts; the engine thread closes the card. A loss seen on a
  stager thread is only recorded, because only the engine thread may rewrite the held bitmaps (patch 0019).
- (History: the DeepSeek engine path ran all link calls on one `serve_lane` thread; it was dropped with that engine.)
- Re-tier scheduling (Strata integration side): a planner thread consumes heat snapshots and produces plans through two
  single-slot mailboxes; the engine thread alone applies them. The planner never touches the channel.
- Helper: one serving thread, one GPU context.
- Shared counters: `req_seq/resp_seq`, `blob_*_seq`, `svc_*` are `std::atomic`; the v4 tier counters are raw
  `uint64_t` accessed only through `__atomic_*` builtins because the two sides are built by different toolchains
  and `std::atomic`'s representation is not guaranteed to match.

### 2.7 Failure contract

| Situation | Qwen-family path (as implemented, series 0001-0020) |
|---|---|
| Mismatch at open (magic, version, header size, tier op capacity, geometry, window) | fail fast, message names both sides |
| Helper cannot load its plan | `kFailed` plus message; the engine refuses to start |
| Helper dead or recycled at start (any channel state) | the card opens closed with one WARN; the engine starts (patches 0019 and 0020) |
| Helper exits, goes mute, errors or echoes wrongly after load | the card closes (codes 3 exited, 4 layer deadline of 30 s, 5 error or bad echo); the rows the helper owed are computed on the CPU and the request finishes (patch 0019) |
| A tier round times out, tears a slot or reports stale residency | the card closes for the session; its experts come from RAM or the drive (patch 0012). A refused round with untouched slots keeps the old verified ownership |
| Engine killed without `kShutdown` | the next engine's plan replaces the helper's; verified live (DESIGN-LOG 2026-11-06) |

Detection of a lost helper is the engine's own: a `PeerWatch` on the identity the helper wrote before `kHelperUp`, never a bare
pid probe, and nothing in it ever signals a process. Before patch 0019 a lost helper froze the engine for 30 s and ended it (shown
live in RESEARCH-LOG, run L1); the DeepSeek helper's fail-open to its CPU lane is history, as that engine was dropped.

### 2.8 Channel versioning, and why a neutral tag

- The first 24 bytes of the header are a negotiation preamble at fixed offsets: `magic` (0), `version` (8),
  `hdr_bytes` (12, `sizeof(Header)` as the creator compiled it), `window_max_t` (16), `tier_max_ops` (20). A peer
  that disagrees about everything after them can still parse these and refuse with a real message instead of
  reading shifted fields. They are never reordered or extended.
- New fields are appended after existing ones, never inserted. Every wire struct is explicitly padded and
  size-asserted, with golden sizes in `adapters/strata/wire_v10_golden.txt` (header 2752 bytes at v10; the v6 header was 2240).
- Version history as shipped in the series: the wire first carried v4 (negotiation preamble, tier ops; v2 blob reads
  and v3 service-time counters were folded in before it was published as a version), then v6 (adds `engine`,
  `spec_layers`, `spec_files`, `helper_pid`, the per-token output mode, the `weights` region and the `spec` region), v7 (the D4
  prefill region), v8 (`kTierSwapBlob`: the new expert's bytes in a blob slot, read back by the helper before it acks), v9
  (`kTierSwapFile`: a 72-byte file reference in a blob slot, and file fields in the header), v10 (`Header::helper_start`, the
  helper's identity) and v11 (the magic text becomes `SEABRD1`; no field moves). Each is an append or a retag and the layout totals did not change (`channel_layout.hpp:30-60`). Version 5 is reserved
  for "the v4 layout under a neutral tag" (below) and no code in the series emits it.
- Tag: the shipped magic is `"SEABRD1"` (wire v11; the tag before v11 spelled the name of the earlier add-on, and the change to
  SEABRD1 is the one thing v11 does, so an older helper is refused cleanly). A channel Strata owns should carry a
  neutral tag. A review branch prepared separately renames the header to `strata/core/expert_helper_channel.hpp`,
  tag `"STRXHLP"`, version 5 on the v4 layout, with a host test that a v5 peer connects and an old-tag or v4 peer is
  refused with a message saying whether the tag or the version differs. That branch is not part of
  `compat/strata/patches/strata/`; the series here is still `SEABRD1`/v11. The upstream sequence would be: land the neutral
  channel first (item 3 in section 6), then the later versions as appends on top of it.
- Region sizing rule: the shm size is a function of tokens, never of picks. Every region is host RAM, and the
  measured failure mode on this class of host is host-memory pressure evicting GPU queues (section 5.3). The
  tier-op block is sized by a constant (64 ops x 20 bytes), and a plain LOAD or SWAP carries only (layer, expert); the
  helper sources the bytes itself. A v8 swap carries the bytes in a blob slot and a v9 swap a file reference; the wire never
  carries a path.

### 2.9 What stays on the Strata integration side

- The helper process itself (CUDA/HIP executables and the quant kernels it links from Strata's tree).
- All placement policy: the ranking walk, hot-share interleave (patch 0006), per-card startup benchmark
  (`card_bench`), tier policy (`tier_plan`: decay, promotion margin, cooldown, per-layer helper-share ceiling), and
  the re-tier scheduler (`retier_loop`: cadence in windows, planner thread, mailboxes, flags).
- The installer, the pinned-patch build, configs and the KL-gate harness.
- Anything that names a card model, a vendor pairing or a workload.

## 3. Hook points, one row per patch

Categories: **seam** = generic hook the interface in section 2 needs; **fix** = bug fix that stands alone upstream;
**policy** = seabraid-specific behaviour that should not go upstream (it would move behind a seam); **diag** =
diagnostic or tuning, optional.

| # | Title (short) | Strata files / functions touched | What it adds | Category | Maps to |
|---|---|---|---|---|---|
| 0001 | External expert helper tier | `include/strata/seabraid/channel.hpp` (new until 2026-11-19; since then the patch puts Seabraid's `adapters/strata/wire` on the include path instead); `core/remote_experts.{hpp,cpp}`: `RemoteExperts::open/begin/finish/close/copy_cached_blobs`, new `seabraid_open/begin/finish`, `tier_apply/tier_sync`, `holds()`; `core/expert_source.cpp`: route tail-skip condition in `expert_pool_dispatch_multi`; `program/generate.cpp`: helper-tier wiring, RAM-copy exclusion, per-request timing line, `SEABRAID_STRATA_PTRACER`; top-level `CMakeLists.txt`: `SEABRAID_STRATA_SOURCE_DIR` | Channel header; helper mode of the remote tier (`remote_dev >= 1000` means "the helper process"); plan/load handshake; per-layer begin/finish; blob reads for the prompt path; service-time counters | **seam** (channel, helper mode, wiring). Parts to drop or rework: `SEABRAID_STRATA_PTRACER` (**diag**, Linux ptrace hook; removed by patch 0015), `SEABRAID_STRATA_SOURCE_DIR` (**policy**: replaced by an in-tree helper target), tail-skip change (see section 4.4) | `open/submit/collect/read_blobs` |
| 0002 | ExpertCache fill clamp | `core/expert_cache.{hpp,cpp}`: `fill_slot`, `fill_slot_blocking`, `fill_slot_queued`, `verify_slot`; `tests/core/fill_bytes_test.cpp` (not registered in CMake by this patch) | Clamp each fill to `slot_room(slot)`, refuse oversize loudly | **fix** | prerequisite of `transition` (swaps into sized slots), valuable alone |
| 0003 | `STRATA_DF2_WINDOW` | `core/dflash2.cpp` (drafter window) | Opt-in: drafter attends only the last N context rows (N >= 16 and < pack window) | **diag** (generic tuning knob, unrelated to the helper; separate PR if wanted) | none |
| 0004 | D7 call site: `--retier` | `CMakeLists.txt` (links `seabraid_retier`, defines `STRATA_HAVE_SEABRAID_RETIER`); `core/remote_experts.{hpp,cpp}`: `seabraid_retier`, `seabraid_held`, `seabraid_free_slots`; `program/generate.cpp`: flag parse, startup check, one `tick()` before `adapt_tick()` in the serial `--serve` loop, `--serve`-only guard | The fence and apply path; hooks for an external scheduler. Scheduling code lives outside Strata | **seam** (`transition`, one `tick` hook, accessors). The flags and scheduler stay seabraid-side | `transition` |
| 0005 | Helper publishes post-round residency | `channel.hpp`: `publish_tier_residency`, `tier_residency_consistent`; `remote_experts.cpp`: stale-view check in `seabraid_retier` | Helper writes `held[]` from cache truth before acking; engine refuses a round whose `held[]` contradicts the applied ops | **seam** (part of the `transition` contract; must land with 0004) | `transition` |
| 0006 | `SEABRAID_STRATA_HOT_SHARE` | `program/generate.cpp` (primary expert fill order, `hot_order`, disjointness check, `STRATA_DISJOINT_ADAPT` forced on); new `program/seabraid_hot_share.hpp`; test | f-paced interleave of the expert ranking between primary and helper; startup line counts experts in both caches and refuses if any | **policy** (a placement heuristic). The seam it needs: let a caller supply the primary's fill order instead of the pure ranking (section 6, item 10) | none today |
| 0007 | Channel v6 + `expert_spec.hpp` | `channel.hpp` (appended `engine`, `spec_layers`, `spec_files`, `helper_pid`; `weights` and `spec` regions); `expert_spec.hpp` (new) | Engine description region and per-token output mode | **seam** | `open(spec)`, `OutMode` |
| ~~0008~~ | **Dropped 2026-11-17** (Strata 0.2.14 removed its DeepSeek-V4.1 engine); as of v0.2.13.1: dsv41 helper link | `dsv41/ds4_helper_link.{hpp,cpp}` (new, the only dsv41 files that include the channel); `dsv41/ds4_model.cu`: `Impl::hlink`, `helper_open`, `cpu_pick_add`, hooks in `serve_lane` (take after `lane_take`, finish after the CPU lane's rows are in `rp.part`) and `cache_switch` (do not promote helper-held experts); `CMakeLists.txt`; test | Engine adapter for DeepSeek-V4.1-Flash; fail-open to the CPU lane; Windows stub that fails fast if the env var is set | **seam** (engine adapter; the blocks in `ds4_model.cu` are the engine's own hooks) | `HelperClient` for dsv41 |
| ~~0009~~ | **Dropped 2026-11-17**, see 0008; as of v0.2.13.1: `SEABRAID_STRATA_LINK_CHECK` | `dsv41/ds4_helper_link.{hpp,cpp}`, `ds4_model.cu` | Also computes every helper pick on the CPU lane and reports nrmse, max abs error, signed bias, relative bias and scale per layer and quant type; output stays the helper's | **diag** (qualification tool; optional, could be kept as a test-time flag) | none |
| 0010 | `--remote-expert-opt` across the helper process | `core/remote_experts.cpp` (`seabraid_helper_begin` writes router weights and asks `kOutPerToken`; `seabraid_helper_finish` requires the echo), `core/remote_expert_opt.{hpp,cu}` (`init`/`adapt`/`owns` skip or ask `holds()` for a helper-process remote; `accumulate` takes a rows pointer; `sum_rows()`; `init_host_only()`), `generate.cpp`, test | The helper returns n_tok weighted sums instead of one row per pick; `SEABRAID_STRATA_HOT_SHARE` no longer refuses `--remote-expert-opt` | **seam** (`OutMode::kOutPerToken` on the Qwen path) | `submit(mode)`, `collect` |
| 0011 | `STRATA_ROUTE_HIST` | `core/route_hist.hpp` (new), `core/expert_source.cpp`, `core/remote_experts.{hpp,cpp}`, `prefill/prefill.cpp`, `program/generate.cpp`; `tests/core/route_hist_test.cpp` | Opt-in per-request routed-expert histograms, and a card-1 own-work vs wait split in the decode timing line. Unset: one const-bool load per counting site and a byte-identical timing line | **diag** | none |
| 0012 | N-card ladder M3, engine side | `core/ownership_map.{hpp,cpp}`, `core/ownership_rungs.{hpp,cpp}` (new), `core/remote_experts.{hpp,cpp}`, `program/generate.cpp`; `tests/core/ownership_map_test.cpp` | The ownership map (one map, one epoch) with prepare/commit through two live rungs, behind `--retier-promote`. The existing `--retier` path gains fail-closed rounds: a round that times out, tears a slot or reports stale residency closes the helper card for the session, and a slot whose copy did not finish is quarantined | **seam** (the map and the fail-closed rule) | `transition` |
| 0013 | D4, `STRATA_PREFILL_REMOTE_COMPUTE` | `core/remote_prefill.{hpp,cpp}` (new), `seabraid/prefill_wire.hpp`, `prefill/d4_plan.hpp`, `prefill/d4_check.{hpp,cpp}`, `prefill/kernels.{hpp,cu}`, `core/remote_expert_opt.{hpp,cu}`, `cmake/strata_gpu_engine.cmake` | Opt-in: while a prompt is read, the expert card computes the experts it holds (channel v7 prefill regions sized by tokens, a compute-vs-lend cost model, a failure path on card 1). Unset: the lending path exactly as before | **seam** (opt-in; measured slower than lending on an 8.3k-token prompt, so off) | `submit` over a token-sized entry |
| 0014 | Validate channel geometry and the private handoff | `seabraid/prefill_wire.hpp`, `core/remote_experts.cpp`, `core/remote_prefill.cpp` | Checks shared region sizes before use and keeps the validated bounds; a Linux descriptor transport that seals the mapping extent before publish | **fix** (hardening) | `open` |
| 0015 | Remove `SEABRAID_STRATA_PTRACER` | `program/generate.cpp` | Removes the switch that let any process of the same user attach a debugger to the engine | **fix** (security) | none |
| 0016 | Bound every helper-written string | `core/remote_experts.cpp`, `core/remote_prefill.cpp`; `tests/core/seabraid_bounded_text_test.cpp` | Every engine-side read of a helper-written char array goes through `bounded_text` (at most cap-1 bytes, stops at NUL, printable ASCII only); the attach result is a structured `Attach` value | **fix** (hardening) | `open` |
| 0017 | One channel per helper process; blob swap | `core/remote_experts.{hpp,cpp}` | `SEABRAID_STRATA_SHM` as a comma-separated list, the position being the card; `seabraid_retier_blob` sends the new expert's bytes as one `kTierSwapBlob` op (wire v8); `tier_verify` reads a slot back before the helper acks | **seam** | `transition` |
| 0018 | Where an expert's bytes live; file reference | `core/expert_source.{hpp,cpp}`, `core/remote_experts.{hpp,cpp}`; `tests/core/file_expert_source_test.cpp` | `FileExpertSource::file_extents` and `seabraid_retier_file`, one `kTierSwapFile` op (wire v9) carrying a file reference; a result of `kTierBadSum` when the helper read other bytes than the manifest | **seam** | `transition` |
| 0019 | A lost helper closes its card, not the engine | `core/remote_experts.{hpp,cpp}`, `core/expert_source.cpp`, `core/remote_prefill.cpp`, `core/lost_rows.hpp` (new), `cmake/strata_gpu_engine.cmake`; `tests/core/seabraid_helper_loss_test.cpp`, `tests/core/lost_rows_test.cpp` | `PeerWatch` bound to the helper's identity (wire v10); every wait is `pred or lost`; on a loss the card closes and the rows the helper owed are computed on the CPU; the two `kill(pid, 0)` probes in the prompt path become `seabraid_closed()` | **seam** (the loss contract) | `open`, `collect` |
| 0020 | A dead helper at start opens its card closed | `core/remote_experts.cpp`; `tests/core/seabraid_helper_loss_test.cpp` | `seabraid_open` binds the identity before it reads the state word, so a helper dead at start opens closed whatever the channel state; a live helper in `kShutdown` is waited for, at most 30 s | **fix** (completes 0019) | `open` |

Patch subject lines carry "n/N" counters from the length of the series on the day each was written (0011 says 11/16, 0019 says 19/19); the file number is authoritative.

## 4. Standalone upstream bug fixes

These stand on their own, need no helper, and can go to Strata as plain fixes. Reproductions were run on the
pinned base unless marked otherwise.

### 4.1 HIP link break when `STRATA_PREFILL_MMQ=OFF` (v0.2.12)

- Reproduction (reproduced on unmodified `146df3c4`, per DESIGN-LOG 2026-11-07):
  configure a HIP build with the CMake defaults (no `-DSTRATA_PREFILL_MMQ=ON`), build target `strata`.
  The link fails with two undefined symbols: `fused::want_dm_f16` and `fused::dm_f16_used`.
- Cause: commit `6e31946c` calls both from `src/prefill/prefill.cpp` but did not add them to the stand-in
  namespace used when `STRATA_PREFILL_FUSED` is not defined (`#ifndef STRATA_PREFILL_FUSED`). Strata's own
  `setup.py` passes `PREFILL_MMQ`, `Q6K_EXPERTS` and `MMQ_KQUANTS`, so it does not see the break; a manual
  `cmake -DSTRATA_ENABLE_HIP=ON` does.
- Minimal fix: add the two stubs to that stand-in namespace, asking for nothing and returning `false`, so
  `moe_combine16` is never chosen and the FP32 combine runs (the pre-`6e31946c` behaviour).
- Evidence: the same default-options build with the stubs links `strata` with 0 undefined symbols; HIP and CUDA
  builds with `setup.py`'s options show no new failed targets and no new ctest failures against an unmodified
  control. Not run: GPU execution of the default-options HIP build.

### 4.2 `ExpertCache` fills not clamped to their own slot (patch 0002)

- Reproduction: open a sized cache (`open_sized`, the native path) whose slots differ in size per layer (this
  pack: 1510400 / 1305600 / 1177600 bytes per blob). `open_sized` sets the internal `blob_` to the LARGEST slot,
  and `fill_slot`, `fill_slot_blocking`, `fill_slot_queued` and `verify_slot` clamped to it. A request larger than
  its slot, or `bytes = 0` (the default argument), copies past the slot into the next expert's weights. There is
  no crash; the symptom is quality loss that does not repeat.
- Minimal fix: clamp to `slot_room(slot)` and refuse an oversize request with an error naming the slot. The
  arithmetic is a free function `fill_bytes_for(room, blob_max, bytes)` so a host test needs no GPU toolchain.
  Uniform caches are bit-identical (`slot_room == blob_`).
- Evidence: `tests/core/fill_bytes_test.cpp` passes on the fixed code, and with the old largest-slot clamp
  restored it reports 9 FAILED, exit 1 (review record). Register the test in Strata's CMake test list (patch 0002
  does not). Not run: any GPU execution of the fill path on v0.2.12.

### 4.3 Documentation / default mismatch worth a check (not a fix in the series)

`STRATA_EXPDEC_DOWN` is documented as "default 1 on gfx12" (`iq_kernels.hpp:125`) but defaults to 0 in code
(`iq_kernels.cu:4091`), found while reading for the 2026-11-07 log entry and not acted on. Unverified beyond that
reading: whether the code or the comment is the intent is upstream's call.

### 4.4 Behaviour change to review before proposing: route tail-skip beside remote tiers

Patch 0001 replaces `tail_rank > 0 && d.peer == nullptr && d.remote_count == 0` with
`tail_rank > 0 && d.peer == nullptr` and skips an expert only if no helper holds it. For in-process remote tiers
this means tail-skip now runs where it was previously off. It is not isolated from the helper in any
measurement (Strata integration configs run `STRATA_ROUTE_TAIL_SKIP=0`, same speed, exact routing). Treat as unverified and as
a separate, reviewable change, not a fix.

### 4.5 Observation, not a patch: draft-confidence sizing on mixed-vendor hosts

The DESIGN-LOG records that Strata's serve mode picked a draft confidence of 0.8 by sizing from the 3060, and that
setting `--draft-conf 0.35` gave +4-7% (RTX 3060 + R9700, one system, n not stated). This is a sizing heuristic
question for mixed-vendor hosts rather than a defect in the series.

### 4.6 Upstream findings on pristine Strata v0.2.15 (not ours)

Two ctest failures that reproduce on an unpatched v0.2.15 tree (the `STRATA_BASE` commit, no Strata integration or Seabraid patch), found while
rebasing on 2026-11-17. They are recorded here as upstream findings, not as defects of the series, and are not worked around.

- `conversation_copy_streams_test` on ROCm 7.13 / gfx1201 (R9700): "the first copy stream + event of the card took
  151552 KiB of device memory", against the test's bound of less than 16 MiB. Evidence: the pristine-tree HIP build and run on the
  box, and the patched-tree `ctest-hip.log` (same message), both in the 2026-11-17 chain report.
- `cuda_spill_guard` with nvcc 12.4 (sm_86): `check_cuda_spills` finds 3068 kernels, 469 using stack or local memory, 4 known from the
  baseline and 44 not allow-listed (`b6_mmvq_rows_kernel` and `native_mmvq_multi_kernel` instantiations, `bf16_f32_mmvf_multi_kernel`
  instantiations), on the pristine-tree CUDA library. The allow-list is upstream's; the compiler version is the variable.
- Not a failure but expected on this host: `ple_parity` fails everywhere because the model it needs is not installed.

## 5. Build requirements

### 5.1 Flags Strata integration passes (from `compat/strata/install/build.py`, `compat/strata/install/_build_rocm.py`)

| Flag | Applies to | Why |
|---|---|---|
| `-DSTRATA_PREFILL_MMQ=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON` | HIP builds | Match Strata's own `setup.py` HIP release configuration. Without `PREFILL_MMQ` a v0.2.12 HIP `strata` does not link (section 4.1). The other two select the fused/native expert paths the release build uses |
| `-DSTRATA_MMQ=ON` | HIP and CUDA builds | Since Strata v0.2.14 `STRATA_MMQ` defaults to OFF (Strata's own prompt kernels alone; through v0.2.13.1 every CUDA build had llama.cpp's MMQ prompt kernels). ON keeps the MMQ prompt path that patch 0013 (D4) hooks via `mmq_plan()`. On HIP, `STRATA_PREFILL_MMQ` builds the MMQ path only together with it. Strata's own release build passes `STRATA_PREFILL_MMQ=ON STRATA_MMQ=OFF`; we deliberately differ |
| `-DSTRATA_WINDOW_MAX_T=<N>`, same value in EVERY build | all builds, main and helper | `Channel::connect` refuses a `window_max_t` mismatch. Defaults differ per backend (CUDA 8, HIP 16), so per-vendor defaults would never connect. Also sizes the channel |
| `-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF` or the reverse | one tree per vendor | Strata builds ONE backend per binary; `CMakeLists.txt` says not to enable both. Mixed-vendor operation is therefore two processes, not one build |
| `-DCMAKE_HIP_ARCHITECTURES=<gfx> -DGPU_TARGETS=<gfx>` and `-DCMAKE_HIP_COMPILER=<clang++>` | HIP | The CMake HIP language needs the clang driver; the `hipcc` wrapper breaks its compiler probe, so it is never accepted |
| `-DCMAKE_CUDA_ARCHITECTURES`, `-DCMAKE_CUDA_HOST_COMPILER=<working gcc>` | CUDA | nvcc rejects a too-new default gcc; no fallback to it |
| `-DSTRATA_BUILD_TESTS=OFF` for installs, `ON` for qualification | all | Host tests (channel, tier, link) need `ON` |
| `-DSEABRAID_STRATA_SOURCE_DIR=<checkout>` | all | Today's mechanism to add the helper targets. Replaced by in-tree targets under this proposal |
| `-DSTRATA_GGML_DIR=<local copy>` | offline builds | Strata fetches llama.cpp at a pinned commit; air-gapped hosts supply a local copy |

The test `test_amd_main_plus_nvidia_helper_commands` fails with `PREFILL_MMQ` removed (installer suite, DESIGN-LOG
2026-11-07). Verified for CUDA 13.1 / sm_86;89: `strata` and `seabraid-helper` build and ctest `seabraid_*` 5/5 pass.
Verified 2026-11-09 on v0.2.13.1 (R9700 gfx1201 + RTX 3090 sm_86): HIP `strata` + `seabraid-helper` and CUDA `strata` +
`seabraid-helper` build from the series, `remote_experts_seabraid_opt_test`, `seabraid_hot_share_test` and `ds4_helper_link_test`
pass, and the Qwen IQ3_S engine runs on both cards (DESIGN-LOG 2026-11-09). A CUDA MAIN engine needs
`-DCMAKE_CXX_COMPILER`/`-DCMAKE_C_COMPILER` set to the CUDA host compiler too: on a gcc-15 host with CUDA 12.4 (gcc-13) the
link fails on `__cxa_call_terminate` otherwise. v0.2.13 vendors ggml (`third_party/strata-ggml`), so the `STRATA_GGML_DIR`
row above is only an override now. Not run: dsv41 on a GPU (model deleted).

### 5.2 Build constraints in the helper tree

- (Removed 2026-11-17 with Strata's DeepSeek engine.) The DeepSeek helper (`seabraid-helper-dsv41`) was built only in a CUDA tree (`STRATA_ENABLE_CUDA AND NOT
  STRATA_ENABLE_HIP`). It compiles Strata's `src/dsv41` quant sources itself, so it needs no `STRATA_BUILD_DSV41`.
  A HIP helper for DeepSeek does not exist.
- `-Wformat=2 -Werror=format` on the DeepSeek helper: a missing `printf` argument once crashed it at startup with
  an empty log, and `-Wall` alone let it through.
- Windows: the channel is POSIX shared memory. The DeepSeek link has a Windows stub that fails fast if the env var
  is set. Windows support for the Qwen path was not assessed.
- The channel header must compile under both nvcc + gcc and clang; it uses an inline-asm `pause`, not
  `<immintrin.h>`, because nvcc 12.4 with gcc 13 cannot parse the AMX intrinsics.

### 5.3 Runtime environment the measurements depended on

- A 30 GB host with a 17-22 GiB host expert arena stalled the AMD queue for tens of seconds under memory
  pressure. Measured directly (the engine process's KFD `evicted_ms` counter jumped 2,634 -> 41,465 ms, GPU idle
  at 3% / 17-22 W, run on R9700 + ROCm 7.13, n=1 observed stall plus repeated hangs, 9 vs 2-3 hangs stock vs Strata integration
  at expert-cache cap 4,000, none at cap 11,000). The cause is Strata plus ROCm on Linux (hipHostRegister memory
  as a userptr tracked by MMU notifiers, no XNACK on RDNA), independent of the helper; Strata integration makes it rarer
  because fewer tokens use the host tier. Mitigation env (opt-in, from an upstream Strata PR):
  `HSA_USERPTR_FOR_PAGED_MEM=0 HSA_SVM_FOR_PAGED_MEM=0 STRATA_HIP_ARENA_NONCOHERENT=1
  GPU_PINNED_MIN_XFER_SIZE=1048576`, which needs `ttm.pages_limit` raised (boot parameter).
- Helper env: `STRATA_REMOTE_ZEROCOPY=0` (set by the helper unless the user set it): 77.7 -> 65.3 us per layer
  request, +1% decode (RTX 3060, nsys; n not stated).
- Engine env needed for the measured wins on the Qwen path: `SEABRAID_STRATA_EXCLUDE_HELPER_RAM=1`,
  `STRATA_PREFILL_HELPER_SOURCE=1`, `STRATA_DISJOINT_ADAPT=1`, and `SEABRAID_STRATA_SHM=<name>` plus
  `--expert-cache-device1` to enable the helper tier.

## 6. Acquisition list (dependency order, one upstream PR each)

Each item is one upstream PR candidate. The measurements behind the statuses are kept, with hardware and build, in
[DESIGN-LOG.md](DESIGN-LOG.md) (entry 2026-11-20, "the upstream PR candidates"). Legend: **Measured** = backed by a number
in the DESIGN-LOG; **Built** = compiles and host tests pass; **Unverified** = neither. None of this has been proposed upstream.

1. **HIP link stubs** (section 4.1). Accept: configure HIP with defaults, build `strata` with 0 undefined symbols; delete the
   stubs and the two undefined symbols return. Status: link failure reproduced on unmodified `146df3c4` and the fixed build links
   (Verified); GPU run (Unverified).
2. **ExpertCache fill clamp and registering `fill_bytes_test`** (section 4.2). Accept: `ctest -R fill_bytes` passes and fails with
   the old clamp. Status: host test and sabotage (Verified); GPU fill on v0.2.12 (Unverified).
3. **Neutral channel header.** The header without engine code, under a neutral tag. Accept: a host test with no GPU covering
   preamble offsets, a peer of the current version connecting, an old-tag or older-version peer refused with a message that says
   which differs, an ABI mismatch of `hdr_bytes` or `tier_max_ops` refused, and the tier release/acquire race, each check
   sabotaged once (the repository records this for the v4 checks in `compat/strata/tests/SABOTAGE.md`). Status: the wire is now v11 under the
   `SEABRD1` tag and Seabraid owns it (section 7, item 1); the `STRXHLP` neutral-tag build exists only in a separate review
   (Built, not merged). Measured on R9700 + RTX 3060 and R9700 + RTX 3090 at v4: channel round trip p50 4.76 us, p99 6.30 us.
4. **Helper mode of the remote expert tier, wiring and a reference helper target** (`open/begin/finish/close`,
   `copy_cached_blobs`, `holds`, plan/load handshake; `STRATA_EXPERT_HELPER_SHM` and `STRATA_EXPERT_HELPER_EXCLUDE_RAM` in the generate
   path). Off by default; with the env var unset nothing new is constructed. Accept: a host test with an in-process fake helper on a
   real channel (`begin/finish`, `read_blobs`, a killed helper gives a timeout error and no hang), HIP and CUDA builds with no
   new warnings in touched files, and one same-day A/B against a single-vendor baseline with the KL gate (teacher-forced,
   `STRATA_LOGPOS`, top-32). Status: Measured on a 9f7d977 build (R9700 + RTX 3060: 8 of 10 prompts answered at cap 4,000 against 0
   of 10 stock, and a tie at cap 11,000; R9700 + RTX 3090: 105.6-105.9 tok/s, 13.5% of routed experts on the helper; KL gate mean
   0.0098 against a stock-vs-stock 0.0025, n=1 text). Since the series gained 0019 and 0020, a killed helper closes its card
   instead of ending the engine (section 7, item 5; shown on GPUs in RESEARCH-LOG 2026-11-20 entry on helper loss). Not run:
   Windows, other cards.
5. **Route tail-skip beside helper tiers** (section 4.4). Accept: a check that an expert some helper holds is never skipped and
   that with no helper the decision is unchanged; an A/B showing no change at the default `STRATA_ROUTE_TAIL_SKIP`. Status:
   Unverified (not isolated in any measurement).
6. **Tier transition**: `transition` fence, `tier_apply/tier_sync`, `seabraid_held()/seabraid_free_slots()`, `epoch`, and one
   engine-thread `tick` hook before `adapt_tick()` in the serial `--serve` loop. Depends on items 2, 3, 4. Accept: host tests
   (`tier_residency_test` fails 4 checks with the residency publish removed; a stale `held[]` is refused) and a hardware arm with a
   static-plan control and a KL gate. Status: Measured on R9700 + RTX 3060 (off 94.9 / 101.3 tok/s, on 99.3 / 100.3, 42 applied,
   0 failed, fence 4.3 ms average, 11 ms max; positive control KL 0.179 with STALE fired 109 times; n=1 per arm). Only within-layer
   swaps are implemented, with no cross-layer relocate and no wire EVICT. The ceiling is an admission gate, not an invariant (see
   the DESIGN-LOG ceiling-honesty entry).
7. **Per-token weighted output on the Qwen path** (patch 0010: `kOutPerToken` request/echo, router weights region,
   `RemoteExpertOpt` accepting a helper-process remote). Depends on 4. Accept: a host test with an in-process fake helper on a real
   channel, and sabotage of the weight, `out_rows` and `out_mode` checks (the patch records this). Status: Built (host test), and
   used by the private-helper live runs, which require `--remote-expert-opt` ([adapter README](../../../adapters/strata/README.md#opt-in-strata-source-integration)).
8. **Spec region and engine tag** (was patch 0007: `engine`, `spec_layers`, `spec_files`, `helper_pid`, `weights`, `spec`).
   Now the wire's `channel_layout.hpp` and `expert_spec.hpp`. Only the DeepSeek engine used the spec region, so this is no longer
   a candidate on its own; `engine` and `spec_layers` remain in the header (section 7, item 3).
9. **DeepSeek-V4.1-Flash adapter and executor.** Dropped on 2026-11-17 with Strata's DeepSeek engine (patches 0008 and 0009). The
   parity and smoke results are in the DESIGN-LOG entry above.
10. **Placement seam** replacing `SEABRAID_STRATA_HOT_SHARE` upstream: let the caller supply the primary cache's fill order (and `claimed`),
    keeping the startup check that the two caches are disjoint. The interleave heuristic itself stays out of Strata. Accept: a host
    test that a supplied order is honoured and that overlapping experts abort startup. Status: Unverified for speed. The motivation
    is measured: with IQ3_S all-VRAM the RTX 3090 held 45% of the experts (11,110 of 24,576) but did 13.6% of the expert work. On
    hardware (DESIGN-LOG 2026-11-09, runs 1 and 7) hot share did not pay with either card first.
11. **Optional, separable**: `STRATA_DF2_WINDOW` (patch 0003) as its own small PR (R9700, window 512 = 1.957 vs 2.110 ms per round at
    59.4% vs 59.9% acceptance, n not stated). `SEABRAID_STRATA_LINK_CHECK` (patch 0009) went with the DeepSeek engine.

Items 1 and 2 need no scope discussion. Items 3-10 depend on upstream agreeing that a second-process expert helper is in scope;
the project's contribution guide asks for that agreement first.

## 7. Where the code and the earlier design sketch disagree

The sketch (describe / read / submit(epoch) / collect / transition) was used as input. The code wins on each point. Points that
only concerned the DeepSeek helper (the `describe` and `read` calls, the DeepSeek output mode, the pid watch it published) are gone
with that engine; what remains:

1. **The wire is `SEABRD1`, v11, with a 2752-byte header.** `kMagic = "SEABRD1"` and `kVersion = 11`
   (`adapters/strata/wire/strata/seabraid/channel_layout.hpp:59-60`; header size in `adapters/strata/wire_v10_golden.txt`). v5 and the
   `STRXHLP` neutral tag are reserved in a comment and no code in the series emits them; `STRXHLP` exists only in a separate review
   (v4 layout).
2. **`epoch` is bumped but never enforced.** `Header::req_epoch` is initialised to 0 and not stamped by `seabraid_helper_begin`, and the helper
   never compares epochs, although the header comment says the helper rejects a request from an older epoch. The actual protection is
   the fence: no request in flight or admitted during a round (`adapters/strata/wire/strata/seabraid/channel.hpp:219-222`). So
   `submit(epoch)` in the sketch does not exist.
3. **`engine`, `spec_layers` and the spec region are unused by the Qwen path.** The Qwen helper opens the pack itself with
   `FileExpertSource` and takes only a ranked plan; Qwen main never reads `engine` (`channel_layout.hpp:214-216`). The only
   engine that used them was DeepSeek's. Two owners of the pack format do not arise today, because there is one engine.
4. **Output mode.** The Qwen path defaults to `kOutPerPick` and uses `kOutPerToken` only with `--remote-expert-opt` (patch 0010). Main
   requests the mode and the helper echoes it; a disagreement is an error.
5. **A lost helper is no longer an engine error (patches 0019 and 0020).** The sketch said failures fail open loudly and the first
   version of this table said a Qwen layer request returned an error after 30 s. Now the engine binds a `PeerWatch`
   (`peer_watch.hpp`: a pidfd opened first, then the `/proc` start time compared with `Header::helper_start`, wire v10) to the
   helper's identity. A helper that exited, a 30 s layer deadline, a helper error or a bad echo closes only that card, and the
   rows the helper owed are computed on the CPU, so the request in flight finishes (patch 0019; its close codes are 3 helper
   exited, 4 layer deadline, 5 helper error or bad echo, 6 gone at start). A helper that is already dead at start opens its card
   closed with one WARN, whatever state its last engine left in the channel (patch 0020). A live helper still in `kShutdown` is waited
   for, at most 30 s. A failed tier round also closes the card (patch 0012). Both patches were shown on GPUs on 2026-11-20:
   [RESEARCH-LOG](../../../docs/RESEARCH-LOG.md#2026-11-20--helper-loss-on-gpus-hl-hl2). Not shown on GPUs: a restart while a live helper
   is still in `kShutdown` (CPU test L12 only). A loss seen on a stager thread is only recorded, because only the engine thread
   may rewrite the held bitmaps.
6. **`kWindowMaxT x 10` is a pairs cap.** The DESIGN-LOG (2026-11-06) flags that sizing the channel by pairs would reserve about
   3.4 GB of shm for a 16K-token prompt chunk if the helper computed during prefill. The shipped channel does not do that (helper
   compute is decode-window only; prefill uses blob lending, or the opt-in D4 regions sized by tokens), but the sizing rule stands.

## 8. Open questions

1. Scope: does upstream want a second-process expert helper at all, or an in-process multi-runtime design? The draft proposal
   asks before sending code.
2. Version numbering: ship the channel as it is (v11 under `SEABRD1`), land the neutral tag as a rename, or reset the numbering
   since no external peer exists? (Preferred: reset.)
3. Should the engine check be a real handshake (main writes `engine`, the helper compares it with what it can run), and should
   `req_epoch` be stamped and checked, or removed from the header while no peer is built against it? With one engine left, removing
   `engine` and `spec_layers` is the simpler answer.
4. A helper of the same vendor as the engine, or on the same GPU, qualifies protocol and numerics but not the
   heterogeneous latency measurements. Which of those does upstream want as acceptance for item 4?
5. Cross-layer re-tier moves need a `relocate()` in `ExpertCache` and a wire EVICT; neither exists. Worth
   proposing, or leave within-layer swaps as the contract?
6. More than one helper: `SEABRAID_STRATA_SHM` takes up to three channels, because the engine's remote tier is limited to three remote
   devices (patch 0017, `kMaxChannels = 3`), and `TierWireOp::helper` is 0 within each channel. Two helper processes ran live in L1 and L2,
   the second on the main card's own GPU; a third physical card is not measured.
7. Supported matrix: the Qwen3.8-Flash-Next family (10 routed experts per token, hidden size from `kernels::cpu::H`). DeepSeek-V4.1-Flash
   (8 pick slots) went with Strata's engine for it. Other families need an adapter and an executor; no other family has been tried.
8. Hot-share and tier policy defaults (item 10 and the re-tier ceiling) were chosen on one system; the ceiling gates
   admissions rather than restoring an invariant, and the heat signal is volatile at the settings tried
   (over-ceiling layer count swung 4 -> 45 -> 12 -> 45 -> 21 across 44 rounds in one smoke). An A/B is needed
   before any default is proposed.
9. Mixed-vendor main/helper numerics differ in reduction order. The measured KL is about 4x the run-to-run floor on
   one text (n=1) and below the 0.03-0.035 Strata accepts for its own default fast paths. Upstream may want a
   larger text set before accepting that as the bar.

Resolved since the first version: whether the Qwen path should recompute a failed helper request instead of returning an error
(patch 0019 does, from the engine's own source of the expert, RAM or the drive, and closes the card); whether the Qwen helper
should adopt the spec region, and whether a CUDA-only DeepSeek helper is acceptable (both went with the DeepSeek engine).
