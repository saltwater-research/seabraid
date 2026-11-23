# Capped-overflow test

Runs a model that does NOT fit the cards plus a small RAM budget, so the experts that fit nowhere else genuinely live on the SSD, and
proves from counters (not from flags) that the SSD was read. It uses a model already on the machine (Qwen3.8-Flash-Next IQ3_S, 48 layers x
512 experts, about 47 GiB of experts) and shrinks capacity instead of looking for a bigger model.

**Status: built and tested against synthetic logs and a planted procfs/cgroup tree, dry-run on the target box, and the cgroup mechanism
checked on the box (below). No engine was started and no GPU was touched while writing it. Every number that is not a command, a flag or
a quoted measurement is an ESTIMATE or a design choice, not a result.**

## The trap, and how it is closed

Linux serves "SSD" reads from the page cache. A run whose file tier was told to read the SSD can finish without the SSD being touched, and
the test would report an overflow that never happened. Two things cause it, both measured on the target box on 2026-11-19 (a 300 MB
read of a model file, disk bytes taken from /proc/diskstats around each read):

| Read | Disk MB read | What it shows |
|---|---|---|
| in a scope with `MemoryMax=64M`, cold | 310 | the cold read |
| the same read again in a new `MemoryMax=64M` scope | 260 | the limit charges page cache: 300 MB cannot stay cached in 64 MB, so the second read goes to the disk again |
| the same read, no limit | 190 | pages the two dead scopes left behind (about 124 MB resident, up to one limit each) served part of it for free |
| the same read again, no limit | 0 | without a limit the whole range is cached: this is the lie |

So the harness does three things, and a start that fails any of them is INVALID (never a pass):

1. **A cgroup v2 memory limit that charges page cache.** Every capped engine, helper and W7 capture starts under
   `systemd-run --user --scope -p MemoryMax=<bytes> -p MemorySwapMax=0`. The limit is engine RAM budget + engine overhead + a small page-cache
   allowance (`capacity` block of `arms.json`), so the OS cannot hold more than that allowance of the SSD reads. Delegation works on the
   box: `python overflow.py --probe-cgroup` starts a 64 MiB scope and reads `memory.max` and `memory.swap.max` back (measured: 67108864 and
   0). The memory and cpu controllers are delegated to the user manager; the **io controller is not**, so there is no per-cgroup `io.stat`
   and the disk proof uses `/proc/diskstats` and `/proc/<pid>/io` instead.
2. **Eviction before every capped start.** A cache page is charged to whoever first read it; the dead scopes above leave pages behind that
   the next start reads for free. The model files are evicted with `posix_fadvise(DONTNEED)` (no privilege; measured: a 124 MB residue
   went to 0) and `fincore` must then show at most `gates.max_resident_mb` of them still cached, or the start does not run.
3. **Counters, read back.** Per counted decode request the data gate needs the engine's file-tier bytes above zero AND the model's disk
   counter AND the engine's own `read_bytes` to rise by at least `gates.min_device_frac` of those bytes; `memory.max` and `memory.swap.max`
   read from the cgroup of the engine's own pid must equal the plan; the engine's own startup lines (slot count, RAM budget, experts left
   for the files) must match the plan; no OOM kill. Details and WHY of each: `ov_gate.py` docstring.

`unbuffered=1` (`--var unbuffered=1`, sets `STRATA_UNBUFFERED_LOAD=1`) makes the engine read the file tier with O_DIRECT, which no cache can
serve: the strictest variant of the same test and a cross-check of the cgroup mechanism.

## Design (the WHY of each choice)

| Arm | What it is | Capacity (ESTIMATE inputs) | SSD share of experts |
|---|---|---|---|
| **R** | uncapped reference: both cards, `--expert-cache auto`, no file tier, no limit | all experts on the cards | about 0% |
| **O1** | main card only | 3000 main slots + 6 GiB RAM budget (3137 blobs) | 75.0% = 18439 blobs = 35.26 GiB |
| **O2** | O1 plus the helper card | + 3000 helper slots | 62.8% = 15439 blobs = 29.53 GiB |
| **O3** | O2 plus a second helper process (static placement) | + 3000 more | 50.6% = 12439 blobs = 23.79 GiB |
| **O3L** | O3 plus the live ladder (opt-in, `--arms ...,O3L`) | same as O3 | same as O3; experts move SSD -> RAM -> helper2 -> helper1 -> main |

The arithmetic (`ov_cap.py`, printed by `--dry-run`): 24576 expert blobs = 47.00 GiB, one blob 1.96 MiB. Per capped arm, main slots +
helper slots + RAM-budget blobs must leave at least `capacity.min_file_share` (40%) of all blobs on the SSD, otherwise the plan refuses to
build ("this cap does not overflow, it would test nothing"). Every capped arm holds that margin (35.0, 22.8 and 10.6 points). Memory limit
of the engine: 6.00 RAM + 8.00 overhead (ESTIMATE until a first run reports `memory.peak`) + 2.00 page cache = 16.00 GiB; each helper 6.00 GiB;
swap 0. The slot counts are small on purpose (the main card could hold about 24 GiB of experts): capacity is capped to force the overflow.

`experts_gib` is the owner's figure, not a measurement of the pack. The data gate compares it with the engine's own `RAM budget ... of the
C GiB of experts the GPU cache does not hold` startup line and refuses a start more than 10% off, so a wrong input is caught, not trusted.

**What is matched across O1-O3** (so the arms differ only in the rungs they add): model, quantization, `--max-context`, the same main slots,
the same RAM budget, the same memory limit, the same request stream (payoff's fixed greedy decode prompt and long-prefill text), the same
disk. Checked from the engine's own log lines, not assumed. **Reference:** R is the uncapped run all-VRAM; the capped arms' output is
compared with it.

**Quality (W7, decode path).** Each arm's `force_check.py score --path decode` capture (placement-qual's W7: the engine decodes a FIXED
continuation through the real decode windows, with the file tier and helper engaged) is compared with the reference's by `tools/kl.py`
with placement-qual's thresholds (read from its `arms.json`, one owner). The reference is captured twice by default: the second is the
report-only noise floor. A prefill KL (W4) is deliberately not run: placement-qual found that the tier never runs in a prefill. The
arm's engine config JSON must carry exactly the capacity flags the plan prints; the capture is refused before the window is spent if it
does not (`ov_capture.forced_cfg_problems`).

**Speed.** Same decode (200 tokens) and prefill (about 3500 tokens, distinct first line) requests as `../payoff`, 1 warm-up + 3 counted rounds.
Reported with n and spread; **no ranking between arms**: three requests cannot support one. A 3500-token prefill touches nearly every
expert, so it is also the heaviest SSD read of the run.

**Two helpers (O3).** The engine takes `SEABRAID_STRATA_SHM` as a comma list, one channel per helper process, position = card id, at most three
(`adapters/strata/wire/strata/seabraid/channel_names.hpp`, patch 0017), and `--expert-cache-device1` / `--expert-cache-device2` give each its
slots. The runner starts two helper processes (each in its own memory-limited scope) on `SEABRAID_STRATA_SHM=/seabraid-ovf1,/seabraid-ovf2`. Nothing
refuses O3 up front: a start is INVALID only if the engine log does not show both channels up (one `seabraid: helper .. on channel` line per
helper). Both helpers run on the card named by `--var gpu` (`--var gpu2` names another), so their slots share that card's VRAM.

**The live ladder (O3L).** The arm's flags and env are imported from the installer's one owner, `install/_ladder.py` (`ladder_args(2)`:
`--retier on --retier-promote seabraid --retier-fallback 1 --retier-hops 3 --retier-moves 6`; `LADDER_ENGINE_ENV`: `STRATA_RAM_ADAPT=4`,
`STRATA_RAM_ADAPT2=1`), and the installer's own `refusal()` runs on them; no flag string is copied here. O3L differs from O3 by exactly
those. Its data gate also needs, from the engine log, the placement driver's `placement move L/E lower->upper src=card|fallback bytes=N
verified=1` lines (rung 0 is the main card, rung r the r-th helper, `src=fallback` the RAM/SSD fallback rung): at least one move with
`src=fallback` onto a helper rung, a hop onto EACH helper rung, and no stale-placement record (`stale placement epoch` or `helper residency
is STALE`). A ladder whose flags were accepted but never moved anything is INVALID, not slow. `_ladder` carries `--adapt-every 64 --adapt-swaps 8`
(the engine refuses `--retier` without them); `--var ladder_extra_args=...` is appended after and, being last, overrides them (not validated).

## Files

| File | Owns |
|---|---|
| `overflow.py` | the CLI, the per-start loop, `--probe-cgroup` |
| `arms.json` | arms, capacity inputs, gates, the window ESTIMATE |
| `ov_cap.py` | the capacity arithmetic and the "must overflow" rule |
| `ov_cfg.py` | config expansion, the memory-limited commands, the plan order, the ESTIMATE |
| `ov_engine.py` | the engine and helper processes of a start, the per-request counter bracketing |
| `ov_measure.py`, `ov_capture.py` | the speed start and the W7 capture, each with its witnesses |
| `ov_proc.py`, `ov_cache.py` | procfs/cgroup/disk readers; eviction and residency of the model files |
| `ov_gate.py`, `ov_parse.py` | the data gate and the extra startup-line formats |
| `ov_report.py`, `ov_paths.py` | SUMMARY text and criteria rows; the reused-module path hook |

Reused, not copied: `qual_engine` (readiness, stop by tag), `qual_work`, `qual_report`, `qual_decode.pool`, `qual_cfg`, payoff's
`pay_measure` request loop and `pay_parse` log formats, `pay_gate.run_kl_py`, `tools/kl.py`. Tests: `compat/strata/tests/overflow/`.

## How to run it in a GPU window

A live run needs both cards and the box's production services stopped (the owner's call); this harness starts and stops only its own
processes (found by an exact per-start tag) and never touches a service, a GPU or a lock file.

**Before the window (any machine, CPU only):**

```
cd compat/strata/scripts/overflow
python overflow.py --out /tmp/x --dry-run --arms R,O1,O2,O3,O3L --var gpu=0 --var rocm=<dir> --var pack=<dir> --var native=<gguf>
python overflow.py --probe-cgroup                        # on the box: the memory limit really is delegated to a user scope
cd ../.. && python -m pytest -q tests/overflow
```

`--dry-run` starts nothing, reads no model file and creates no file. Read the arithmetic and each command: the capped arms differ only in
the helper flags, and every capped command starts with the `systemd-run` scope.

**Preconditions the harness cannot check for you:** a launcher that follows the placement-qual contract (`launcher RUNDIR [flags]`, log in
`RUNDIR/strata.log`, prints `ready: http`, honours `NOHELPER=1`) whose base config contains none of the flags `arms.json` owns; the
helper binary that matches the engine; the W7 inputs (`force_check`, `forced_set`, one engine config JSON per arm, `--var` names in
`arms.json`); about 31 GB of RAM free; the model on an SSD (a rotational model disk is refused unless `--var allow_rotational=1`: a model
copy on the box's second, spinning disk would measure the wrong thing).

**Shakedown first (ungated, not reportable):** `python overflow.py --out <dir>/shake --arms O1 --kl off <vars>`. It proves the launcher, the
cgroup readback and the counters on one capped start. Read `<dir>/shake/SUMMARY.txt`: `memory.peak` (replace `engine_overhead_gib` with
it plus margin), the disk/process/engine MB per decode request, and any INVALID reason, before spending the full window.

**The run:** `python overflow.py --out <dir>/run1 <vars>` (default arms R,O1,O2,O3). Exit 0 = every arm PASS (or UNGATED with `--kl off`);
1 = an arm is INVALID or FAIL, or a step failed (read SUMMARY.txt); 2 = bad config, nothing started. Then read `SUMMARY.txt` and
`chain_report.html`, append the run to the account's runs ledger, release the window.

**GPU window ESTIMATE (nothing here is measured; from the `estimate` block: load 3-8 min per start, decode 1.5-30 tok/s, prefill 40-800
tok/s, W7 2400 forced tokens):**

| Selection | Starts | Window ESTIMATE |
|---|---|---|
| shakedown `--arms O1 --kl off` | 1 | 0.1 h to 0.4 h |
| `--arms R,O1 --kl off` (ungated) | 2 | 0.1 h to 0.8 h |
| default `--arms R,O1,O2,O3` (gated) | 9 | 0.7 h to 4.5 h |
| `--arms R,O1,O2,O3,O3L` (gated, with the ladder) | 11 | 0.8 h to 5.5 h |
| ladder shakedown `--arms O3L --kl off` | 1 | 0.1 h to 0.4 h |

The wide upper end is the decode speed under heavy SSD overflow, which nobody has measured; replace the block with the shakedown's
`chain.tsv` seconds before booking the full window.

## What it cannot show

- It does not rank arms by speed (3 counted requests per phase), and it says nothing about drives other than the one the model sits on.
- The disk counter is machine-wide; it is trusted only together with the engine's own `read_bytes` and the engine's file-tier bytes, and
  the box's root disk is shared with everything else running there.
- Only O3L exercises live moves; O1-O3 are the static fill order (cards, RAM, SSD). O3L's W7 capture proves its quality against the reference,
  but its move lines come from the speed start's engine log only (the capture has no per-request log).
- O3 and O3L have not run on a real engine: the two-channel startup lines and the move lines are checked against the adapter's source, not against a log.
- `engine_overhead_gib` and `experts_gib` are ESTIMATES; the first live run replaces the first, and the engine's own startup lines check
  the second.
- A cgroup limit that is too small OOM-kills the engine; the gate reports it with the remedy, and the arm is INVALID, not slow.
