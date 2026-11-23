# Strata integration: module ownership

One concept, one owner. Paths are relative to `compat/strata`. Dependencies point one way: detect, plan, config,
build and services, then the setup command. Nothing in `install/` imports the setup command, and nothing in
`strata-ext/` imports the installer.

| Module | Owns | Depends on |
|---|---|---|
| `install/seabraid_setup.py` | command line, prompts, the order the stages run in, exit statuses | every stage below |
| `install/detect.py`, `_detect_*.py` | what the machine is: GPUs, toolkits, RAM, filesystems, holders | `_probe`, `_log` |
| `install/_probe.py` | the three probe primitives (run a command, read a file, parse an int) and why each returned nothing | `_log` |
| `install/plan.py`, `_plan_*.py` | card roles, tier sizes, shared-memory sizing; card ranking comes from `seabraid.ranking` | detect output, `bindings/python` |
| `install/_bench*.py`, `_capture.py` | optional measurement: card bench, lead-job probe, bounded process capture | `plan`, `build`, `_http` |
| `install/config.py` | the Strata engine config and helper arguments | `plan` |
| `install/build.py`, `_build_rocm.py`, `_rocm.py` | fetching Strata at `STRATA_BASE`, applying `patches/strata/*.patch` in sorted order, one build tree per vendor | `plan` |
| `install/services.py` | systemd unit text, hardening list, install steps, health probe | `plan`, `_exposure`, `_http`, `_log` |
| `install/_exposure.py` | "is this bind address exposed, and is there a valid API-key file" | none |
| `install/_http.py` | the installer's only HTTP client; never uses a proxy | none |
| `install/_log.py` | log configuration and the no-full-paths rule for installer logs | none |
| `patches/strata/` | the engine-side changes as a `git format-patch` series (0001-0006 and 0010-0020; 0007 was only the wire header, now `adapters/strata/wire`; 0008 and 0009 were dropped with Strata's DeepSeek-V4.1 engine) | the commit in `STRATA_BASE` |
| `strata-ext/src/helper.cpp` | the expert helper process (Qwen path) and its error reporting | Strata headers, the shared-memory channel header from patch 0001 |
| `strata-ext/src/log_text.hpp` | the no-full-paths rule for helper logs | none |
| `strata-ext/src/card_bench*.cpp` | per-card benchmark line format, parsing and tier ordering | none |
| `strata-ext/src/tier_*.cpp`, `retier_loop.cpp`, `retier_help.cpp`, `_retier_ladder.cpp` | tier policy, live re-tiering rounds | `card_bench` |
| `tools/` | the KL gate, replay and histogram tools, load tools | a running server |
| `tools/_loopback.py` | the tools' only HTTP client; never uses a proxy | none |
| `configs/` | measured example engine configs | none |
| `tests/` | host-only checkers: channel ABI, tier policy, card bench, log hygiene, installer | the modules above |
| `scripts/placement-qual/` | Live-qualification runner (S6) for Seabraid placement on Strata: runs DESIGN.md section 5 and records the chain | `tools/kl.py` (KL gate, subprocess), `tools/_loopback.py` (HTTP client) |
| `scripts/payoff/` | Measures when a second GPU pays off for a model that does not fit on the main card, running the same requests through three arms over a sweep of expert-cache caps | `scripts/placement-qual/` (`qual_cfg`, `qual_engine`, `qual_work`, `qual_stats`, `qual_report`), `tools/kl.py` (via `pay_gate`) |
| `scripts/overflow/` | Runs a model that does not fit the cards plus a small RAM budget, so the experts that fit nowhere else live on the SSD, and proves from counters that the SSD was read | `scripts/payoff/` (`pay_gate`, `pay_measure`, `pay_parse`, `pay_analyze`), `scripts/placement-qual/` (`qual_cfg`, `qual_decode`, `qual_engine`, `qual_work`, `qual_stats`), `tools/kl.py` (via `pay_gate`) |

The helper wire contract (`adapters/strata/wire`: the channel layout and operations, the v6 spec region and
`helper_options.hpp`, the controller-to-helper command line) is owned by `adapters/strata` at the repository root. The
helper, the host tests and (through patch 0001's include path) the patched Strata engine include it and never define
it: runtime <- adapters <- compat <- patched Strata. `strata-ext/seabraid.cmake` is the one place compat names that
directory; `tests/test_wire_owner.py` fails on a second copy or a patch that recreates it.

## Checkers

| Rule | Checker |
|---|---|
| No full paths in helper logs | `tests/log_text_test.cpp` |
| No full paths in installer logs | `tests/install/test_hardening.py` (`scrub`, `describe`, probe logging) |
| No proxy for local HTTP; no `shell=True`; every `curl` has `--noproxy` | `tests/install/test_hardening.py` (class scans) |
| Exposed bind needs a private key file | `tests/install/test_exposure.py` |
| Unit text cannot be injected into | `tests/install/test_hardening.py`, `tests/install/test_services.py` |
| Prompts never guess | `tests/install/test_hardening.py` |
| Patch series applies in order | `install/build.py` applies `*.patch` sorted; `tests/install/test_build.py` |
| A lost helper (exit, deadline, error, stale identity) closes its card and never ends the engine | Strata tests `seabraid_helper_loss_test` (L1-L9, real forked helpers) and `lost_rows_test` (L10) carried by patch 0019, and L11 (a graceful stop's `kShutdown` channel with a dead helper opens closed) and L12 (a live helper in `kShutdown` is waited for) carried by patch 0020; `tests/test_patch_0019.py` and `tests/test_patch_0020.py` pin the patches; no numeric-pid probe: `checks/check_pid_probes.py` (Piece A) |
