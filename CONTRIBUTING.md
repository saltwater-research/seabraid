# Contributing to Seabraid

**This project is not accepting outside contributions (pull requests) at this time.** Issues and bug reports are
welcome, and so are private security reports (see [SECURITY.md](SECURITY.md)). Seabraid is alpha software and may be
updated or changed at any time without notice; use it at your own risk.

The rest of this file is for building and testing locally, and for when contributions open. Please read
[MAP.md](MAP.md) first; it says which module owns which concept.

## Sign-off (DCO)

This is the rule that will apply once contributions open. Every commit must then carry a Developer Certificate of Origin sign-off:

```sh
git commit -s
```

That adds a `Signed-off-by:` line and certifies that you wrote the change or
have the right to submit it under the project's license
(<https://developercertificate.org/>). By contributing you agree your work is
licensed as described in [NOTICE](NOTICE): Apache-2.0 for the runtime,
Apache-2.0 OR MIT for `include/seabraid/*.h`, and MIT for patch files that modify
MIT-licensed upstream code. The sign-off applies to new contributions; the existing
patch files carry no `Signed-off-by` line and no license text, and NOTICE alone states their license.

## Build and test

Run what your change touches, and say in the pull request which you ran.

Install the test tools first, at the versions CI pins. Use a virtual environment (stock
Ubuntu refuses a system-wide `pip install`); inside it `python` works, outside it the
command is `python3`:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install pytest PyYAML==6.0.3 numpy==2.4.6 lizard==1.26.1
```

Without them `python -m pytest tests -q` still runs and exits 0: a missing extra turns tests into skips, not errors,
and `checks/run_all.py` reports `check_complexity` as failed (it needs `lizard`). Run `python -m pytest tests -q -rs`
to see each skip's reason. The causes are: `lizard` not installed (14 tests in `tests/test_complexity.py`), PyYAML or
numpy not installed (the workflow and provider tests), no built `libseabraid.so` in `target/release` or `target/debug`
(and `SEABRAID_LIBRARY` unset; 1 test in `tests/test_no_egress.py`, the link-level egress gate: CI builds the release library first, sets `SEABRAID_LIBRARY` to it and sets `SEABRAID_REQUIRE_EGRESS_LINK_TEST=1`, which turns that skip into a failure; set both yourself to run it), `cmake` or a POSIX shell missing (the native-check
tests), and a source tarball with no git checkout (3 tests). A
skip is not a pass.

To test the CMake 3.22 floor: `python3 -m venv v && v/bin/pip install cmake==3.22.6`, then
`SEABRAID_CMAKE=$PWD/v/bin/cmake` steers `native-tests/installed/check_sdk.sh`, `check_layouts.sh` and
`pytest tests/test_native_checks.py tests/test_build_config.py` (recipe in the header of
`native-tests/installed/check_sdk.sh`).

```sh
# Rust core
cargo test --offline

# Native build, C ABI fixtures, Python linked tests
cmake -S . -B build/seabraid -DSEABRAID_BUILD_NATIVE_TESTS=ON
cmake --build build/seabraid --config Release
ctest --test-dir build/seabraid -C Release --output-on-failure
# The Strata fixtures need a Strata checkout and are not built by
# default: add -DSEABRAID_BUILD_STRATA_ADAPTER=ON -DSTRATA_SOURCE=<strata checkout>
# to the configure line. CI does not run them.

# Python binding and reference provider (CI also runs this with the native library:
# cargo build --release, then SEABRAID_LIBRARY="$PWD/target/release/libseabraid.so" before the command)
python -B -m unittest discover -s bindings/python

# Repository hygiene (identity, paths, addresses)
python -m pytest tests -q

# Every checker in checks/ is a gate. CI runs them all through one script, with CI's arguments
# (check_error_paths --strict-runtime); it prints one line per checker and fails if any fails:
python checks/run_all.py
# Not in CI or run_all: it needs the word to forbid, and fails on any hit
python checks/check_names.py --forbid <word>

# Strata integration (installer, tools, host tests)
cd compat/strata && python -m pytest tests
```

The load simulator (`tools/sim`, POSIX only) is built and tested by CI with
`-DSEABRAID_BUILD_TOOLS=ON` added to the CMake configure line above. The README Quickstart
shows how to build it, run it and sweep it with `tools/sim/matrix.sh`. A C consumer of the
installed SDK is `examples/consumer/CMakeLists.txt`; the README has it and the plain `gcc` line.
Install the SDK outside the source tree (`--prefix` somewhere else): one inside the checkout is scanned by the repository checks.

To replay this page and the README Quickstart as a newcomer would (a fresh tree, a clean environment, every step timed
and recorded), run `python3 scripts/journey/run.py --tree <fresh copy of the repository> --out <record directory outside it>`;
add `--offline` where there is no network (those steps are recorded as skipped, and the documented stop without
`lizard` is a warning, not a failure). It exits non-zero on any other failing step and writes `REPORT.md` and `index.tsv`
to the record directory. The steps are data in `scripts/journey/steps.toml`; a test keeps each command equal to the one in
the docs, so change the docs and that file together. CI runs it on a `git archive` of each commit (job `journey`).

The GPU paths need hardware and toolkits that CI will not have. If you changed
one, report what you ran it on, and say plainly what you could not run. An
untested path is labelled untested everywhere it is documented.

## Design rules

These are enforced by review and, where noted, by checkers.

- **One owner per concept.** A constant, threshold, algorithm or schema is
  defined in one place. If you need it elsewhere, import it or pass it in.
- **One-way dependencies.** The core never imports an adapter, an engine or a
  binding. An adapter is a leaf that knows both sides.
- **Files stay at or under 500 lines; split at 400.** `checks/check_size.py` fails a file past 400
  unless `checks/size_ratchet.json` lists it at its length (it may only shrink; `--write-ratchet` lowers
  the rows). Functions stay at
  cyclomatic complexity 10 or below, in every code directory (`checks/check_complexity.py`; the
  known exceptions are `checks/complexity_allow.txt` and `checks/complexity_ratchet.txt`, and they only shrink). CI pins lizard 1.26.1 for this count
  (`checks/check_complexity.py`); a local run under any other lizard version is
  advisory only, because versions count some constructs differently. Install
  `lizard==1.26.1` to get the same numbers CI will.
- **Every error is logged.** Failures reach the log sink (`seabraid_log`,
  `seabraid.enable_logging()`) with a status code and a short message. Nothing is
  swallowed; a fail-open path says so loudly.
- **No network code.** The runtime opens no sockets and talks to no service.
  Shared memory between processes on one machine is the only transport.
- **No paths, environment values or personal data in logs, errors or
  tracked files.** `checks/check_no_local_refs.py` fails on machine names,
  home paths and private addresses.
- **Presentation is thin.** Adapters translate; they do not compute policy.
- **A rule names its checker.** If you add a rule, add the test or lint that
  fails when it is broken, and show that it fails.

## Pull requests

- Keep a change to one concern. Explain the why in the commit message.
- Add a test that fails before the change and passes after it.
- Do not strengthen a claim in the docs beyond its evidence. Measurements
  carry their hardware, run count and caveats; negative results are welcome.
- Do not commit secrets, model weights, logs, or machine-specific values.

## Before tagging

Follow the [Linux alpha release checklist](docs/RELEASING.md#before-you-run-it) before tagging.

## Conduct and security

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
