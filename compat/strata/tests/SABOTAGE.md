# channel_v4_test: recorded sabotage runs

House rule: a guard nobody has watched FAIL is not known to work. Each check below was disabled, the test was
rebuilt and run, the specific expected failure was confirmed, and the guard was restored.

Build line used throughout (Linux, g++):

```
g++ -std=c++17 -Wall -Wextra -O1 -Iinclude tests/seabraid/channel_v4_test.cpp -o /tmp/channel_v4_test -lrt
```

These runs were recorded in the Strata-fork layout, where the test lived at `tests/seabraid/`; the paths in the outputs
below are kept as recorded. In this repo the file is `tests/channel_v4_test.cpp`; its header gives the current build line.

Baseline, before any sabotage:

```
channel_v4_test: all checks passed
exit=0
```

## Sabotage 1 — disable the ABI check in `Channel::connect()`

`if (h->hdr_bytes != (uint32_t) sizeof(Header) || h->tier_max_ops != (uint32_t) kTierMaxOps) {`
replaced with `if (false) {`.

```
FAIL: connect REFUSES a mismatched sizeof(Header) (tests/seabraid/channel_v4_test.cpp:103)
FAIL: and says it is an ABI mismatch (tests/seabraid/channel_v4_test.cpp:104)
FAIL: connect REFUSES a mismatched tier_max_ops (tests/seabraid/channel_v4_test.cpp:112)
channel_v4_test: 3 FAILED
exit=1
```

This is the check that matters most. Without it, a helper built by nvcc + gcc and a main engine built by clang
that disagree on `Header` padding both map the same bytes and read shifted fields: a LOAD lands on the wrong
expert and a layer silently loses a contribution. The failure mode is wrong answers, not a crash, so it can only
be caught here.

## Sabotage 2 — make `tier_done()` always true

`return __atomic_load_n(&header()->tier_resp, __ATOMIC_ACQUIRE) >= seq;` replaced with `return true;`.

```
FAIL: not done before the helper acks (tests/seabraid/channel_v4_test.cpp:63)
FAIL: a stale ack does not satisfy a new round (tests/seabraid/channel_v4_test.cpp:87)
channel_v4_test: 2 FAILED
exit=1
```

Both halves of the fence are covered: a round that has not been acked must not read as done, and an ack for an
earlier round must not satisfy a later one. Either would let the main engine read `held[]` and `tier_res` while
the helper was still mutating them.

## Sabotage 3 — change `TierWireOp`'s layout

An `int32_t extra_pad_` added to the struct, simulating one side compiling a different definition.

```
include/strata/seabraid/channel.hpp:74:34: error: static assertion failed: TierWireOp must be 20 bytes in both builds
```

This one fails at COMPILE time, in every translation unit that includes the header — which is the point. The
runtime checks above catch a mismatch between two already-built processes; this catches it before either binary
exists, in both the nvcc + gcc helper tree and the clang HIP main tree. It is the guard that actually protects
the cross-layer swap path, where a wrong field offset means writing an expert's blob into a slot sized for a
different layer.

## Restored

```
channel_v4_test: all checks passed
exit=0
```

## Not covered by this test, and why

- **GPU completion before ack.** `tier_ack()` must only be called once the helper's device work is done. A CPU
  release-store says nothing about an outstanding device copy, so no host-side test can prove this. It is a
  contract documented at the call site and it needs an epoch-stamped trace on the test system to verify — see the
  ordering contract comment in `channel.hpp`.
- **Main-side admission control.** The lock-step `req_seq`/`resp_seq` pair bounds requests already PUBLISHED,
  not future ones, so "no other producer publishes during the fence" is the caller's responsibility and is not
  testable from the channel alone. `epoch` exists so that a request which slips through is rejected loudly
  rather than silently losing an expert; the rejection path belongs to the helper and is tested there.
- **Real cross-toolchain layout.** This test runs one build, so it simulates a mismatch by poking `hdr_bytes`.
  The genuine check is the `static_assert` set in `channel.hpp`, which fires in every translation unit that
  includes it — including the nvcc + gcc helper and the clang HIP main.

# retier_loop_test / tier_plan_test (D7 call site, 2026-11-06): recorded sabotage runs

Build: `g++ -std=c++17 -O1 -pthread -I<strata-d7>/include -Istrata-ext/include tests/retier_loop_test.cpp strata-ext/src/retier_loop.cpp strata-ext/src/tier_plan.cpp -lrt`
(each guard was edited out in a copy, the test rebuilt, the named failure seen, the original left untouched).
Baseline: `tier_plan_test: all checks passed`, `retier_loop_test: all checks passed`; 8/8 clean under -fsanitize=thread.

| Sabotage | Failing check |
|---|---|
| owner-thread guard in `tick()` replaced by `if (false)` | `[engine-thread hand-off] !ret && f.calls.load() == before && lp.stats().refused_off_thread == 1` |
| zero-op no longer clears its ops | `[zero-op] f.calls.load() == 0` |
| failed ops not fed to `note_failed` | `[note_failed feedback] seen.size() >= 2 && seen[0] == seen[1]` |
| `to_wire` pairs an EVICT across layers | `[to_wire] r.wire.size() == 1 && r.wire[0].kind == kTierLoad && r.dropped.size() == 1 ...` |
| `same_layer_swaps` incumbent filter removed | `[same-layer swaps] o4.size() == 2 && ... o4[1].layer == 0` (first attempt of this sabotage was NOT caught; the o4 case was added because of it) |
| `retier_check` empty-heat/adapt-flag test removed | 3 FAILED in `[retier_check]` (names both flags, "not allocated", adapt-swaps 0) |
| `tick()` ignores mode off | `[off] !lp.thread_started() && lp.stats().windows == 0 ...` |

# channel_v6_test / dsv41_exec_parity: recorded sabotage runs (2026-11-07)

(History: `dsv41_exec_parity` and the dsv41 geometry rows of `channel_v6_test` were removed on 2026-11-17 with Strata's DeepSeek-V4.1 engine; the runs below were recorded before that.)

1. Layout::of with the spec region placed BEFORE weights (patched copy of channel.hpp, `-I` override; Strata tree untouched):
   `channel_v6_test: 8 FAILED` (tier_ops/weights/spec "packed to its page" and "documented order" checks, geometries unit and dsv41).
2. Golden `header_bytes 2240` -> `2112`: `FAIL: sizeof(Header) equals the golden header_bytes ...` / `channel_v6_test: 1 FAILED`.
3. `install/_plan_shm.py` `CHANNEL_HEADER_BYTES` 2240 -> 2112: `FAILED tests/install/test_plan.py::test_v6_formula_is_pinned_to_layout_of_through_the_golden_file`.
4. dsv41_exec_parity built with `-DPARITY_SABOTAGE_NO_CLAMP` (reference drops the swiglu clamp): COMPILES (exit 0); NOT RUN, it needs a free GPU.

# ownership_map_test (Strata tests/core, M3) / tier_ladder_test (M3b) / tier_residency_test re-run: recorded sabotage runs (2026-11-09)

Each guard was edited out in a COPY of its source with
sed, the copy checked to differ from the original (a sabotage that edits nothing would "pass"), the test rebuilt against
the copy, run, and the clean build re-run at the end.
Baseline: `ownership_map_test: all checks passed (0 failures)`, `tier_ladder_test: all checks passed (0 failures)`,
`tier_plan_test` and `retier_loop_test` (after the shared-validation refactor): all checks passed.

| Sabotage (ownership_map.cpp) | Failing check (first lines) |
|---|---|
| lower-card failed sync / timeout no longer closes it (the old reopen-on-timeout put back) | `[lower card's sync fails] p.state == kClosed && closed(1)`, `executor(1, 3) == fallback()`, `world_ok()`; `[lower card times out: fail closed] executor(l, e) != 1` |
| `executor()` ignores closed rungs | `[upper card tears, restore fails] executor(2, 1) == fallback()`, `world_ok()` (a torn slot routed to) |
| `commit()` ignores a moved epoch | `[the map moved before commit] !commit(...) && state == kRolledBack` (2 FAILED) |
| the upper card is not given the dropped expert back after the lower card fails | `[lower card refuses] card[0].holds(1, 0)...`, `[lower card's sync fails] ... == source(1, 0)`, `world_ok()` (5 FAILED) |
| an upper-card timeout is restored (written again) instead of closed | `[upper card times out] state == kClosed && closed(0) && replaces == 1`, `world_ok()` (2 FAILED) |

| Sabotage (tier_ladder.cpp) | Failing check |
|---|---|
| challengers from every rung but the target (the two-set planner's blindness to card 1) | `[no challenger from a higher rung] m.from == rung[i]`, `!(m.expert <= 1 && m.to == 1)`, `s.of(0, 0) == 0 ...` |
| incumbents from every layer (no same-layer limit) | `[same-layer limit] mv.empty()`, the per-step pairing check `mv[k - 1].layer == m.layer` (4 FAILED) |
| a pair spends one move of budget | `[throttle] mv.size() <= cfg.max_moves_per_round`, `mv.empty()` (4 FAILED) |
| no promotion margin | `[promote card 2 -> card 1] ... s.of(1, 1) == 1`, `[margin and cooldown] a.step({10, 12}).empty()` (5 FAILED) |
| no cooldown | `[margin and cooldown] round - last[i] > cooldown_rounds`, `s.step({100, 1}).empty()` (4 FAILED) |
| `ladder_boundaries` ignored | `[ladder_boundaries] s.step(h, false).empty()` (1 FAILED) |

tier_residency_test with `publish_tier_residency` emptied (patched copy of channel.hpp via `-I` override): `FAIL (4 failures)`:
`[publish -> consistent]` x2, `[A->B then B->C]` x2, the same 4 checks as the 2026-11-07 record.

Not host-testable here (need a GPU): PrimaryRung / RemoteRung (Strata ownership_rungs.cpp), the helper's quarantine of a
torn slot (RemoteExperts::tier_apply / tier_sync) and main's close-on-timeout / close-on-kTierFailed in seabraid_retier.
They are exercised only by the live A/B 5 arms (arms_ab5_promotion.txt).

## 2026-11-14 channel geometry and request capacity

The installed Clang cross-compiler evaluated all six v7 golden assertions, including the 2688-byte header.
The Linux named-shm declarations were supplied for syntax checking because the Android sysroot omits them.

Replacing `tokens <= cap / picks` with `tokens <= cap` in a scratch copy of the channel header made the assertion
for a four-token, nine-pick request against a 32-row channel fail with `reject wider than the channel`.
The same assertions passed with the original guard in place. The shipped source and the test used by CMake
retain the guard.

The native Linux fixture has yet to run. It exchanges a synthetic request through a sealed descriptor passed
at exec, checks descriptor isolation and seal enforcement, then starts a fresh mapping after a helper exits
mid-request. Its Linux runtime result and seal-check sabotage are pending.

## 2026-11-19 pyramid track C: the blob swap round and the retier flags (box, CPU only)

Each guard was removed in a scratch copy of `strata-ext/include/strata/seabraid/tier_round.hpp` (or `retier_loop.cpp`), the host
tests rebuilt, the result read, and the file restored; afterwards `ctest` ran 15/15 green.

| Guard removed | Result |
|---|---|
| the read-back of a blob-written slot (`c.verify(...)` skipped) | `seabraid_blob_swap_test` and `seabraid_channel_blob_swap_test` both FAIL (a corrupted slot is ACKed as applied) |
| `publish_tier_residency` fed the raw ops instead of the plain-swap view | both FAIL (held[] keeps the displaced expert) |
| the blob slot index range check | both FAIL (slot 16 reaches the cache) |
| the blob size vs slot size check | `seabraid_blob_swap_test` FAILS (1 of 2; the channel test's blobs always fit) |
| the verify step removed from `run_tier_round` | both FAIL |
| `check_pyramid` removed from `retier_check` | `seabraid_retier_loop_test` FAILS |
| the `helpers <= 1` pass in `retier_check_helpers` removed | `seabraid_retier_loop_test` FAILS |

Not host-testable (needs a GPU): `RemoteExperts::tier_verify` (an `ExpertCache::verify_slot` over the real arena) and
`seabraid_retier_blob`'s round with the real helper. The 0017 patch is proved to apply and to compile (syntax) only.

## 2026-11-19 FR-H: the file reader, the file swap round, the sums tool (box, CPU only, ext4 on NVMe, O_DIRECT arms ran)

Each guard was edited out in `~/wt/fr-h` (one at a time), the host tests rebuilt, the result read, and the file restored; afterwards
`ctest` ran 20/20 green.  `SEABRAID_STRATA_FR_REQUIRE_DIRECT=1` makes a skipped direct arm a failure.

| Guard removed | Result |
|---|---|
| sum check in `FileTable::fetch` (always "applied") | `seabraid_file_fetch_test` and `seabraid_channel_file_swap_test` FAIL (a flipped byte is served) |
| `file_ref_valid` in `FileTable::read` | both FAIL (SEGFAULT: the helper read outside its table) |
| direct window floored to the block (`start = offset`) | `seabraid_file_fetch_test` FAILS (EINVAL on the unaligned O_DIRECT read) |
| short direct read refused | `seabraid_file_fetch_test` FAILS (a shrunk file served stale bytes) |
| buffered EOF (`pread` = 0) refused | `seabraid_file_fetch_test` FAILS |
| `O_NOFOLLOW` on the table's open | `seabraid_file_fetch_test` FAILS (a symlink is opened) |
| direct reads through the buffered fd | `seabraid_file_cache_test` FAILS (pages resident after a "direct" move) |
| direct/buffered byte counters swapped | `seabraid_file_fetch_test` and `seabraid_file_cache_test` FAIL |
| `tier_round_shape_ok` in `serve_tier_channel` | `seabraid_channel_file_swap_test` FAILS (a file op beside another op is run) |
| read-back of file ops (`carries_blob` -> blob only) | `seabraid_channel_file_swap_test` FAILS (a torn slot is ACKed) |
| `kTierBadSum` reported as `kTierRejected` | `seabraid_channel_file_swap_test` FAILS |
| `as_swap` mapping of `kTierSwapFile` | `seabraid_channel_file_swap_test` FAILS |
| same-layer rule of `op_refusal` | `seabraid_blob_swap_test` and `seabraid_channel_file_swap_test` FAIL |
| `verify_sums` comparison | `seabraid_sums_test` FAILS |
| `write_atomic` rename | `seabraid_sums_test` FAILS (.tmp left, no manifest) |
| sums indexed `[e*layers+l]` | `seabraid_sums_test` FAILS |
| `--retier-hops` vs `--retier-moves` check (`retier_check`) | `seabraid_retier_loop_test` FAILS |
| an `open(` planted on the serve path of `file_fetch.cpp` | `test_file_fetch_open_rule.py` FAILS (self-test inside it) |

Not host-testable (needs a Strata build with a GPU runtime): the Strata-side `file_extents` test in 0018
(`test_file_extents_canonical`), `RemoteExperts::seabraid_retier_file`, `seabraid-sums` and `seabraid-helper` themselves.  0018 and the
helper/tool TUs are proved to apply and to compile (`-fsyntax-only -Wall -Wextra`) only.

## 2026-11-19 FZFIX: two fuzz findings in the helper tier round (box, CPU only)

Both findings (fuzz harness fz2, seeds 0x30c5dbe6fb647ddd KQ and 0x2012c9cd21ef2cca KD) reproduced on 2cc3316: the new unit tests
failed there (10 checks, including residency-consistent) before the fix.  Each guard was then edited out of the box copy, the host
tests rebuilt, the result read, and the file restored; afterwards `ctest` ran 20/20 green.

| Guard removed | Result |
|---|---|
| `quarantine_refusal` (quarantined destination expert refused before the slot write) | `seabraid_blob_swap_test` FAILS (5 checks: applied without read-back, round not residency-consistent) |
| `duplicate_destination` refusal in `run_tier_round` | `seabraid_blob_swap_test` FAILS (5 checks: both ops written, round not residency-consistent) |

## 2026-11-19 SOLFIX: five findings from an independent source review of the file moves (box, CPU only, cores 10-13, ext4 on NVMe)

Red at 3d5375f first.  Finding 1: the new `channel_blob_swap_test` compiled against the 3d5375f `tier_round.hpp` FAILS 6 checks (a
peer rewriting the op mid-round skips the read-back of a torn slot; an inflated Header n_expert publishes held[] at the wrong stride;
a blob op past the mapped layers reaches the cache).  Findings 4 and 2 call APIs the fix adds, so a scratch reproduction against the
3d5375f API stood in: B's op (layer 1, 3 -> 5) carrying C's (expert 7) extents and C's valid sum came back kTierApplied with C's
bytes in B's slot and held(1,5)=1; a link planted at `expert-sums.v1.tmp` made `write_atomic` overwrite its target and leave the
manifest a symlink.  Findings 3 and 5 were confirmed by reading the 3d5375f code (a second path open with no inode check; stamps
taken by `stat(path)` after the scan; `std::istreambuf_iterator` to EOF in `seabraid-sums --verify` and in main's
`PackSums::pack_file`); their red is the sabotage below.  Each guard was then edited out of the box copy (one at a time), the tests
rebuilt and run, and the file restored; afterwards compat `ctest` ran 20/20 with `SEABRAID_STRATA_FR_REQUIRE_DIRECT=1` and root `ctest -j1` 46/46.

| Guard removed | Result |
|---|---|
| `file_ref_mismatch` in `FileTable::fetch` (the wire ref is not compared with the cell's own) | `seabraid_file_fetch_test` FAILS 48, `seabraid_channel_file_swap_test` FAILS 18 |
| the wire's extents and sum used instead of the helper's own (`own = wire`) | `seabraid_file_fetch_test` FAILS 12, `seabraid_channel_file_swap_test` FAILS 16 (B's op applied with C's bytes) |
| `fresh_for` in `bind_manifest` (a stale or foreign manifest is bound) | `seabraid_file_fetch_test` FAILS 3 |
| `serve_tier_channel` runs on `ch.tier_ops()` (shared) instead of its copy | `seabraid_channel_blob_swap_test` FAILS 2 (torn slot ACKed) |
| `serve_tier_channel` takes geometry from the Header, not `ch.limits()` | `seabraid_channel_blob_swap_test` FAILS 4 |
| dev/inode check between the buffered and the O_DIRECT open | `seabraid_file_fetch_test` FAILS 1 (a swapped path is opened) |
| inode check in `FileTable::unchanged` | `seabraid_file_fetch_test` FAILS 1 (a replaced file passes) |
| `write_atomic` back to a fixed `path.tmp` opened `O_CREAT\|O_TRUNC` | `seabraid_sums_test` FAILS 4 (the planted link's target is overwritten) |
| `read_manifest` bound (reads 128 MB instead of kMaxManifestBytes + 1) | `seabraid_file_fetch_test` FAILS 1; `strata_file_moves_test` FAILS (pack_file read the whole 64 MB file) |
| main asks the engine for the extents of `old_e` instead of the op's `new_e` | `strata_file_moves_test` FAILS |

Not host-testable (needs a Strata build with a GPU runtime): `seabraid-helper` itself (the launch-time `bind_cells` in helper.cpp) and
`seabraid-sums` (`still_same` after the scan).  Both TUs and helper_serve.cpp compile clean with `-fsyntax-only -Wall -Wextra` against
the patched Strata tree (0001..0018 + the v7 mbox, `git am` clean).
