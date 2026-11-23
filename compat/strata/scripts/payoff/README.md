# Second-GPU payoff harness

Measures when a second GPU pays off for a model that does not fit on the main card, by running the same requests through
three arms over a sweep of expert-cache caps and reporting where (if anywhere) the second card starts to win. This is
the harness for decision 2 of the audit: "which comparison should establish when the second GPU pays off?"

**Status: built and tested against synthetic logs only. No engine was started and no GPU was touched while writing it.
Every number in this file that is not a command or a flag is an ESTIMATE or a design choice, not a result.**

## Design (the WHY of each choice)

**The question.** For a model whose experts do not all fit in the main card's VRAM, three things can serve the experts
the main card lacks: the CPU computing them where they sit in RAM, the main GPU pulling them over PCIe and computing them,
or a second GPU holding them. The comparison is only fair if everything else is identical, so the harness fixes it and
varies one thing per arm.

| Arm | What serves the experts the main cache lacks | Flags it owns (on top of the shared ones) | Why this flag |
|---|---|---|---|
| **A** local RAM experts | the CPU pool, in place, from RAM | `--pcie-frac 0`, env `NOHELPER=1` | `--pcie-frac` is "the share of the experts missing from VRAM that are copied to the GPU instead of computed by the CPU" (Strata `docs/DETAILS.md`); 0 sends every miss to the CPU. An explicit value always wins over Strata's automatic share. |
| **B** host weights transferred | the main GPU, after a PCIe copy per use | `--pcie-frac 1`, env `NOHELPER=1` | 1 sends every miss over PCIe. The log is checked afterwards (see "Data gate"): a launcher that dropped or clamped the flag would make A and B the same arm. |
| **C** second GPU holds experts | the seabraid helper on the second card (static placement) | `--pcie-frac 0`, `--expert-cache-device1 auto`, env `SEABRAID_STRATA_SHM=<channel>`, the helper process | `--pcie-frac 0` so that whatever neither card holds is CPU-computed exactly as in A: A-vs-C then differs by the helper alone, and A-vs-B by the transfer alone. |

`STRATA_PCIE_BALANCE=1` (the other PCIe knob Strata documents) is CUDA, one GPU, and not combined with `--pcie-frac`; the
main card here is HIP, so it is not an arm. It is named so nobody wonders.

**What is matched** (and by what, so a mismatch is detectable rather than assumed away):

| Matched | By | Checked by |
|---|---|---|
| model, quantization | one `pack` / `native` var for every start; the helper command is read from `placement-qual/arms.json` (one owner) | same files for all arms by construction |
| usable memory | the same `--expert-cache N` on the main card in every arm (the cap), the same `--resident-budget-gib` RAM copy, the same `--max-context`. The only memory an arm adds is what it is being tested for: the second card's VRAM in C | the engine's own `expert cache N slots` startup line is parsed; a cap the engine shrank for one arm only makes the point "NOT MATCHED" and undecided |
| requests | one fixed greedy decode prompt (`placement-qual/prompts/w1_decode.txt`, 400 tokens) and one fixed long prefill text (about 3,500 tokens, a distinct first line per request so no prefix is reused), same order in every arm | each prefill request's reused-token count is parsed; a prefix that came from the prompt cache makes that request unusable |
| output quality | teacher-forced KL against a baseline, per text, with `tools/kl.py` (its own thresholds) | a failing or unrunnable gate keeps that arm's speed out of the point |

**Why prefill and decode are separate requests.** One request cannot report both cleanly: a long prompt with a long answer
mixes two regimes. A decode request (short prompt, 400 tokens) gives the decode tok/s of the engine's own `prompt` line;
a prefill request (long prompt, `max_tokens` 1) gives the prefill tok/s of the same line. Both come from the engine's clock,
the same for every arm.

**Why the three time components.** The crossover is only useful if it can be explained. For each decode request the harness
records, in milliseconds for the whole request (the parts overlap, so they do not sum to `decode_ms`): *staging* (the
`stage` term of the decode timing line plus the host's "staging+launching" ms toward a helper), *merge wait* (the host's
"waiting for it" ms: the stall at the merge with the helper's answer; arm C only, A and B have no merge partner), *local
compute* (the CPU-pool term), *GPU-reach wait* (the engine's own wait term, kept apart because it is a different wait) and
*io wait* (decode time blocked on file-tier reads, the confound when RAM holds less than the model).

**Why a sweep of caps, and why the statement stops at the measured range.** The caps span from "the model fits on main"
to "heavy overflow". The crossover is reported as the boundary between two ADJACENT measured caps or as "no crossover inside
the measured range". Nothing is interpolated between caps and nothing is extrapolated beyond them.

**Why the quality baseline is arm A at the largest cap.** At the largest cap nearly nothing is missed, so A there is the
closest thing to "all experts on the GPU" that the same model gives. Every other start is compared with it; the baseline's
own repeat start (default 2 starts) gives the report-only noise floor.

**Why the arm order rotates per cap point and one start per arm per point.** Only one model can own the GPUs at a time, so
arms cannot run side by side. One engine start per arm per point is what the brief asks for; rotating the order (A,B,C then
B,C,A then C,A,B ...) keeps slow drift (heat, page cache) from lining up with one arm. Inside a start, decode and prefill
requests alternate for the same reason.

## What it reports

For every cap point: a table (arm, valid starts, quality gate, decode and prefill tok/s with mean, sd and n), the time
components per decode request with their share of `decode_ms`, and the second-GPU comparison. Per metric the second-GPU
arm is compared with each rival using `qual_stats.welch_bounds` (placement-qual owns it, including the t quantile): the
one-sided 95% lower and upper bound of the difference, as percent of the rival's mean.

| Status at a point | Rule |
|---|---|
| `pays` | the lower bound is above `+gates.margin_pct` (3%) against EVERY rival |
| `loses` | the upper bound is below `-margin` against at least one rival |
| `tie` | neither |
| `undecided` | an arm has no valid start, fewer than 2 samples, or the arms did not get the same cap |

Then one sentence per metric, for example: "decode: within the measured caps 2500..11000, the boundary lies between cap
8000 (tie) and cap 6000 (pays); each side is measured, nothing inside the gap is."

**Limits, stated so they are not rediscovered:**

- The samples are requests inside one engine start per arm per point, so the bounds cover request-to-request noise only,
  not start-to-start variance. `--starts 2` or more adds starts (all their valid requests are pooled).
- Arms A and B read the RAM copy and, past it, the file tier; with about 31 GB of RAM and a 33 GB model the page cache is
  part of the answer. `io wait` and the OS read MB per request are recorded so that cost is visible, not hidden.
- On AMD, experts held in RAM need the GTT route (the four `HSA_*` and `GPU_PINNED_MIN_XFER_SIZE` variables in
  `arms.json`, applied to every arm so they stay matched); see `docs/DESIGN-LOG.md` for the memory conditions and the
  hang that was seen without it. A hang costs one request timeout (`limits.request_timeout_s`, 900 s) and two consecutive
  failed requests end the start.
- The runner has no heat guard (`tools/arms.sh` has one). Watch the temperatures yourself.
- `b_pcie_min_share` is how the harness knows `--pcie-frac 1` took effect: from the decode timing line, the PCIe terms
  over (CPU experts + PCIe). It has never been observed on this hardware.

## Files

| File | Owns |
|---|---|
| `payoff.py` | the CLI and the per-start loop |
| `arms.json` | the three arms, the sweep, the gates, the window ESTIMATE (a placement-qual schema plus `sweep`, `gates`, `estimate`) |
| `pay_cfg.py` | loading and expanding the config, the run plan (arm rotation), `--dry-run` commands, the window estimate |
| `pay_parse.py` | the engine and helper log line formats (source named in its docstring) |
| `pay_measure.py` | what one start sends and what it keeps per request |
| `pay_gate.py` | the data gate: the arm really ran as that arm, the KL gate |
| `pay_analyze.py`, `pay_render.py` | the statistics, the per-point status, the crossover sentence, the text tables |
| `pay_paths.py` | where the reused placement-qual modules and `tools/kl.py` are |

Reused, not copied: `qual_engine` (engine process lifetime by exact tag and pid), `qual_work` (the loopback HTTP request,
the teacher-forced rows, the long-prompt builder), `qual_report` (the recorded chain and HTML), `qual_stats` (confidence
bounds), `qual_cfg` (config loading), `tools/kl.py` (the KL gate). Tests: `compat/strata/tests/payoff/`.

## How to run it in a GPU window

A live run needs both cards to itself: the window is the maintainers' call. This harness does not start, stop or reconfigure
anything except its own engine and helper processes (found by an exact per-start tag, never a pattern, never a service
manager) and never touches the GPU lock file.

**Before the window (any machine, CPU only):**

```
cd compat/strata/scripts/payoff
python payoff.py --out /tmp/x --dry-run                      # every engine and helper command with its env, the request plan, the ESTIMATE
python payoff.py --out /tmp/x --dry-run --var gpu=0 --var rocm=<dir> --var pack=<dir> --var native=<gguf>   # the same with real values filled in
cd ../.. && python -m pytest -q tests/payoff                 # the harness's own tests (no engine)
```

`--dry-run` starts nothing, sends nothing and creates no file. Read the printed commands: each arm must differ from the
others only in the flags of the design table above, and every start of one cap point must show the same `--expert-cache`.

**Preconditions the harness cannot check for you:**

1. The maintainers have opened the window: production stopped, both cards idle, the machine's GPU lock held by you.
2. The launcher (`--var launcher=<script>`) follows the placement-qual contract: `launcher RUNDIR [engine flags]` starts
   the engine server on `QUAL_PORT`, writes the engine log to `RUNDIR/strata.log`, prints a line containing `ready: http`
   on its stdout, and honours `NOHELPER=1` by leaving the helper flag out. Its base config must NOT contain
   `--expert-cache`, `--expert-cache-device1`, `--pcie-frac`, `--resident-budget-gib`, `--max-context`, `--short-read` or
   `--stats`: this harness owns those seven.
3. The helper binary (`--var helper=<binary>`) is the build that matches the engine; `pack` and `native` name the model files.
4. About 30 GB of RAM free, and the GTT route's kernel setting from `docs/DESIGN-LOG.md` if the main card is AMD.

**Shakedown (about a third of the window, ungated, not reportable):**

```
python payoff.py --out <dir>/shake --caps 11000,6000,4000 --kl off --var gpu=... --var rocm=... --var pack=... --var native=...
```

It proves the launcher contract and that the log lines parse: open `<dir>/shake/results.json` and confirm that
`prefill_tps`, `decode_tps`, `windows` and (arm C) `merge_wait_ms` are numbers, not null, and that SUMMARY.txt shows
`matched` at each cap and no `INVALID` line. Fix the parser or the launcher before spending the full window.

**The run:**

```
python payoff.py --out <dir>/run1 --var gpu=... --var rocm=... --var pack=... --var native=...
```

Watch `<dir>/run1/chain.tsv` (one row per step) and `nvidia-smi` / the AMD tool for temperature. Exit 0 = every step ran
and no quality gate failed; 1 = a step failed or a gate failed (read SUMMARY.txt); 2 = bad config, nothing started.
Afterwards read `SUMMARY.txt` (tables, crossover) and `chain_report.html`, append the run's rows to the account's runs
ledger, and delete the lock.

**Time (ESTIMATE, nothing here is measured):** from `arms.json` `estimate`: an engine load of 3 to 8 minutes per start,
decode 20 to 105 tok/s, prefill 150 to 1,500 tok/s.

| Selection | Starts | GPU window ESTIMATE |
|---|---|---|
| full: 5 caps x 3 arms, speed + quality, baseline repeated | 31 | 2.1 h to 6.2 h |
| `--kl off` (UNGATED, shakedown only) | 15 | 1.1 h to 3.4 h |
| shakedown above (3 caps, `--kl off`) | 9 | 0.6 h to 2.0 h |
| `--starts 2 --kl off` | 30 | 2.1 h to 6.8 h |

The load time dominates: replace the estimate block with the shakedown's measured `chain.tsv` seconds before booking the
full window.
