# Options catalog

This is the one list of Seabraid's tweaks, each offered as an opt-in option. Every row names the owner, the exact
switch, the default, which engines can use it today, and the measured evidence. Nothing else in the repository
keeps a second copy of these facts; other pages link here.

[How it works](HOW-IT-WORKS.md#8-who-owns-what) explains who owns what. The test `tests/test_options_catalog.py`
checks this table (see [How this table is checked](#how-this-table-is-checked)).

## Reading the table

- **Owner.** `library field` is a field of a Seabraid ABI struct, set by whoever opens the handle. `engine flag` is a
  command-line flag of the engine (Strata) or of a patch in `compat/strata/patches/strata`. `engine env` is an
  environment variable read by the engine, a patch or the helper. `adapter env` is read by `adapters/strata`.
  `binding env` is read by the Python package. `patch` is a patch file that is applied or not applied.
- **Default.** Every default is `off` except a correctness fix, which says `on (correctness fix)`. For a tuning
  value, `off` means the option does nothing until the feature that uses it is switched on, and the value shown is what
  it takes then.
- **Engines.** `Strata` means Strata v0.2.15 with the patches in `compat/strata/patches/strata` and
  `adapters/strata/strata-v7-seabraid.patch`. `ggml adapter` is `adapters/ggml`. `Python` is the `bindings/python`
  package.
- **Evidence.** A link to the measured row, or `not measured`. A link to a log entry is not a claim that the option
  wins; read the entry.

## Placement policy (live expert moves)

These run only with `--retier on --retier-promote seabraid`
([how to run it](../adapters/strata/README.md#running-the-placement-driver---retier-promote-seabraid)). The v1 fields
are in `seabraid_placement_policy_v1`, the v2 fields add to `seabraid_placement_policy_v2`
([ABI](PLACEMENT-ABI.md#policy)).

| Option | What it does | Owner | Switch | Default | Engines today | Evidence |
|---|---|---|---|---|---|---|
| Planning cadence | How many windows pass between planning rounds. | engine flag | `--retier-every` | off (128 windows when the driver runs) | Strata; not ggml adapter, not Python | [S8 run](RESEARCH-LOG.md#2026-11-18--first-matched-arm-run-of-the-live-placement-publisher-s8): not passed at the default interval |
| Moves per round | Most expert writes per round; one swap writes 2. | library field, engine flag | `max_moves_per_round`, `--retier-moves` | off (4 when the driver runs) | Strata via `--retier-moves`; Python | [S8 run](RESEARCH-LOG.md#2026-11-18--first-matched-arm-run-of-the-live-placement-publisher-s8) |
| Promote margin | A challenger must be this many times hotter than the expert it replaces. | library field, engine flag | `promote_margin`, `--retier-margin` | off (1.25 when the driver runs) | Strata via `--retier-margin`; Python | [S8 run](RESEARCH-LOG.md#2026-11-18--first-matched-arm-run-of-the-live-placement-publisher-s8): placement never settled at the default margin |
| Cooldown rounds | Rounds before an expert that moved may move again. | library field, engine flag | `cooldown_rounds`, `--retier-cooldown` | off (8 when the driver runs) | Strata via `--retier-cooldown`; Python | not measured |
| Helper layer share | Most of one layer's heat a receiving helper may hold after a swap (not applied at the top boundary or to the initial fill). | library field, engine flag | `max_helper_layer_share`, `--retier-ceiling` | off (0.08 when the driver runs) | Strata via `--retier-ceiling`; Python | not measured |
| Ladder boundaries | Which card boundaries may swap; 0 means all. The Strata patch opens the policy with 0 in `--retier-promote seabraid` mode. | library field | `ladder_boundaries` | off (0, all boundaries, when the driver runs) | Strata (fixed at 0 by the patch, no flag); Python | not measured |
| Fallback boundary | Plans live swaps between the last helper card and the engine's own copy (RAM or NVMe); 0 is the v1 behaviour. The flag takes 0 or 1 with `--retier-promote seabraid` and is refused without it (add-on `retier_loop`, v7 ladder patch). | library field, engine flag | `fallback_boundary`, `--retier-fallback` | off (0) | Strata via `--retier-fallback` (the patch opens policy v2); Python | not measured |
| Max hops per round | Lets one expert climb up to this many adjacent boundaries in one round; 1 is the v1 behaviour. The flag takes 1 to 64 with `--retier-promote seabraid`; `--retier-moves` must cover 2 writes per hop (add-on `retier_loop`). | library field, engine flag | `max_hops_per_round`, `--retier-hops` | off (1) | Strata via `--retier-hops` (the patch opens policy v2); Python | not measured |

## Re-tiering and residency

| Option | What it does | Owner | Switch | Default | Engines today | Evidence |
|---|---|---|---|---|---|---|
| Re-tier | Turns on run-time moves of experts between the main and helper card. | engine flag | `--retier` | off | Strata (named channel and private helper, see the adapter README for what each accepts); not ggml adapter | [S8 run](RESEARCH-LOG.md#2026-11-18--first-matched-arm-run-of-the-live-placement-publisher-s8) |
| Legacy promotion | With `--retier on`, promotes the helper card's hot experts to the main card (`on` or `legacy`). | engine flag | `--retier-promote` | off | Strata, named channel only (the private helper refuses it, exit 2) | [S8 run](RESEARCH-LOG.md#2026-11-18--first-matched-arm-run-of-the-live-placement-publisher-s8) |
| Seabraid placement driver | With `--retier on`, hands the planning to Seabraid (`--retier-promote seabraid`); live promotion stays experimental. | engine flag | `--retier-promote` | off | Strata, named channel and private helper | [S8 run](RESEARCH-LOG.md#2026-11-18--first-matched-arm-run-of-the-live-placement-publisher-s8): mean decode did not pass, static placement is the supported baseline |
| Residency publishing | The helper reports its residency after each re-tier round so the engine routes from the true table. | patch | `compat/strata/patches/strata/0005-seabraid-retier-helper-publishes-post-round-residency.patch` | off (only runs with `--retier`) | Strata | not measured |
| N-card ladder | Engine-side ownership map and promotion across the card ladder (patch 0012), selected by `--retier-promote`. | patch | `compat/strata/patches/strata/0012-N-card-ladder-M3-engine-side-ownership-map-promotion.patch` | off | Strata, main card plus one helper (a second helper tier is refused) | not measured |
| Installer ladder | Installer option that turns the pyramid on: passes `--retier on --retier-promote seabraid --retier-fallback --retier-hops --retier-moves --adapt-every 64 --adapt-swaps 8` for the plan's helper count, and refuses a non-positive value of your own for either adapt flag. | installer option | `--ladder` | off (live placement has not beaten static placement) | Installer (`ConfigOptions.ladder`, `ladder_fallback`); Strata engine | [S8 run](RESEARCH-LOG.md#2026-11-18--first-matched-arm-run-of-the-live-placement-publisher-s8): not passed |
| Ladder engine env | Strata's RAM and NVMe adaptation, which the bottom of the pyramid needs; the installer sets both with `--ladder`. | engine env | `STRATA_RAM_ADAPT` | off (set to 4 by `--ladder`) | Strata | not measured |
| Ladder engine env 2 | Second RAM adaptation switch set by `--ladder` (value 1). | engine env | `STRATA_RAM_ADAPT2` | off (set to 1 by `--ladder`) | Strata | not measured |
| Installer file moves | Installer option: with `--ladder`, builds or verifies the pack's checksum manifest, then passes `--retier-file-moves 1` to the engine and `--file-io auto` to every helper; refuses without `--ladder`, when `--retier-file-moves` is also given by hand, or when the manifest cannot be built. | installer option | `--file-moves` | off (file moves have not been measured) | installer (add-on) | not measured |
| File-reference moves | The engine sends a helper a reference into the model file instead of the expert's bytes; a refused or unavailable reference falls back to bytes. | engine flag | `--retier-file-moves` | off (0) | Strata (add-on `retier_loop` flag, v7 patch commit 3 "file moves"; the engine side is implemented, not yet shown on GPUs); needs `--retier-promote seabraid` and a fresh `expert-sums.v1` (without one it turns itself off with one warning) | not measured |
| Helper file I/O | How `seabraid-helper` serves a by-reference move from the model file. `auto` (the helper's own default): direct I/O where every file allows it, else buffered; if the model-file table cannot be opened, one warning and file moves stay off for that helper. `direct` or `buffered`: the same, but a mode that cannot be honoured is a launch failure. `off`: no table is opened. `gds` is named but refused ("not served by this build"). | engine flag | `--file-io` | off (does nothing until `--retier-file-moves 1`; the helper's own default is `auto`, so with no flag it still opens the model-file table at launch; `--file-moves` passes `auto` explicitly) | Strata `seabraid-helper` (`adapters/strata/wire/helper_options.hpp`) | not measured |
| Pack checksum tool | Writes `expert-sums.v1` beside a pack (and checks it with `--verify`); the proof that a file read is the expert the engine meant. The installer runs it only with `--file-moves`. | tool | `seabraid-sums` | off (run by `--file-moves`) | add-on tool (`compat/strata/strata-ext/src/seabraid_sums.cpp`) | not measured |
| Qualification heat | Path of a scripted heat bias for placement qualification runs; empty means none. | adapter env | `SEABRAID_PLACEMENT_QUAL` | off | Strata placement driver | not measured |
| Move ledger and latency trace | Path of a JSON-lines file that records every expert move (both experts, rungs, source, bytes, stage and write time), every round, each helper's up/down events and per-window latency (submit to collect, time blocked in collect). A write or open error logs one warning and turns it off; it never changes serving. Read it with `compat/strata/tools/trace_report.py`. | adapter env | `SEABRAID_TRACE` | off | Strata placement driver and Qwen helper tier | not measured |
| Trace window size | How many collects each trace latency window covers (1 to 1048576). | adapter env | `SEABRAID_TRACE_EVERY` | off (4096 when the trace runs) | Strata Qwen helper tier | not measured |
| Helper channel list | Comma-separated channels, one per helper process; the plan allows at most 3 helpers (`MAX_HELPERS`). | engine env | `SEABRAID_STRATA_SHM` | off | Strata (patch 0017) | not measured |
| Second and third helper slots | Expert slots for helper rungs 2 and 3, one `--expert-cache-deviceN` per helper (the installer writes them). | engine flag | `--expert-cache-device2` | off | Strata (upstream flag, written by the installer; not defined in this repo's patches) | not measured |
| Compact rows | Hands a provider only the rows a card keeps, not every row of the batch. Not a placement policy field (it is not in `include/seabraid/placement.h`): it is a batch and session setting (`BatchOptions::compact_rows` in `src/batch.rs`, declared by a provider with `SEABRAID_PROVIDER_COMPACT_ROWS`, which needs partial routes). The flag sets it (0 or 1) for the private Qwen helper's session; the Strata v7 patch passes it to the launch (`adapters/strata/engine_flags.cpp`). | library field, engine flag | `compact_rows`, `--seabraid-compact-rows` | off | Strata private helper via `--seabraid-compact-rows` (`PrivateQwenLaunch::compact_rows` in `adapters/strata/private_qwen.hpp`); Python | not measured |
| Heat counts for planning | Strata's own cache cadence; also the heat counts Seabraid plans from (both above 0 for the driver). | engine flag | `--adapt-every` | off (`--ladder` sets 64) | Strata | not measured |
| Heat swaps per round | Number of cache swaps per adaptive round. | engine flag | `--adapt-swaps` | off (`--ladder` sets 8) | Strata | not measured |
| Hot-share fill | Fills the primary cache by an f-paced interleave so the helper gets a share of hot experts. | engine env | `SEABRAID_STRATA_HOT_SHARE` | off | Strata (patch 0006, Qwen path) | [measured, no win](../compat/strata/README.md): helper time per request roughly doubled |

## Prompt reading and quality

| Option | What it does | Owner | Switch | Default | Engines today | Evidence |
|---|---|---|---|---|---|---|
| D4 remote prefill | The helper card computes its own experts while a prompt is read, instead of lending them. | engine env | `STRATA_PREFILL_REMOTE_COMPUTE` | off | Strata (patch 0013) | [v0.2.15 verification](RESEARCH-LOG.md#2026-11-17--verification-on-strata-v0215): read an 8.3k-token prompt slower than lending (n=2) |
| D4 token threshold | Prompt length the helper accepts for remote compute; used with the env above. | engine flag | `--prefill-tokens` | off | Strata `seabraid-helper` flag (patch 0013) | [v0.2.15 verification](RESEARCH-LOG.md#2026-11-17--verification-on-strata-v0215) |
| Weighted output | The helper returns one router-weighted row per token instead of one row per pick. | engine flag | `--remote-expert-opt` | off (required for the private Qwen helper) | Strata (patch 0010) | [verification table](../README.md#whats-verified) |
| Route histograms | Writes per-request routed-expert histograms for `compat/strata/tools/route_hist_report.py`. | engine env | `STRATA_ROUTE_HIST` | off | Strata (patch 0011) | not measured |
| Fill clamp fix | ExpertCache clamps fills to the slot, not to the largest expert. | patch | `compat/strata/patches/strata/0002-ExpertCache-clamp-fills-to-the-slot-not-to-the-large.patch` | on (correctness fix) | Strata | not measured |
| Channel hardening | Validates channel geometry and the private handoff before use. | patch | `compat/strata/patches/strata/0014-seabraid-validate-channel-geometry-and-private-handoff.patch` | on (correctness fix) | Strata | not measured |
| Bounded helper strings | Reads of helper-written text are bounded and control-stripped. | patch | `compat/strata/patches/strata/0016-seabraid-bound-helper-written-strings.patch` | on (correctness fix) | Strata | not measured |
| Draft window | Drafter attends only the last N context rows instead of the pack's window. | engine env | `STRATA_DF2_WINDOW` | off | Strata (patch 0003) | not measured |

## Cards, memory and paths

| Option | What it does | Owner | Switch | Default | Engines today | Evidence |
|---|---|---|---|---|---|---|
| Helper expert slots | Expert slots on the helper card; enables the helper tier together with its channel. | engine flag | `--expert-cache-device1` | off | Strata (patch 0001) | [verification table](../README.md#whats-verified) |
| Copy share | Share of uncached experts copied to the GPU instead of computed by the CPU. | engine flag | `pcie_frac_auto` | off | Strata upstream flag `--pcie-frac`; it is parsed in Strata's own source, not in this repository (only the parsed field shows in patch 0004 context and the payoff arms use the flag) | not measured |
| Drop helper RAM copy | Drops the helper's experts from the engine's RAM copy. Left at off, they stay there and long prompts read faster. | engine env | `SEABRAID_STRATA_EXCLUDE_HELPER_RAM` | off | Strata (patch 0001) | [long prompts](RESEARCH-LOG.md#2026-11-17--long-prompts-on-the-private-helper-tier) |
| Private Qwen helper | The helper executable for the private Seabraid tier. | adapter env | `SEABRAID_QWEN_HELPER` | off | Strata via `adapters/strata`; not ggml adapter | [verification table](../README.md#whats-verified) |
| Library path | Which native library the Python binding loads. | binding env | `SEABRAID_LIBRARY` | off (library found by the default search) | Python | not measured |

## Not found in code

`--expert-cache-device3` is not written by name anywhere (the installer builds `--expert-cache-device%d`); the
`--expert-cache-device2` row stands for it. `--pcie-frac` is not defined in this repository (see its row). The fill clamp,
channel hardening and residency publishing are patch files applied with the series, not options.

## How this table is checked

`tests/test_options_catalog.py` fails when a switch is not found in tracked source outside `docs/` (a path must be a
tracked file; `compat/strata/docs/` is excluded too), a row lacks an owner, default or evidence, or a default other
than `off` is not marked `correctness fix`. To add a tweak, add one row here; no other page lists settings.
