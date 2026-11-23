# Seabraid user guide

How to use the project, by who you are. If you write an inference engine and want to
put Seabraid behind it, read [ENGINE-INTEGRATION.md](ENGINE-INTEGRATION.md) instead.

> **Alpha: use at your own risk.** The ABI can change before 1.0. Nothing is published
> to PyPI or any package index; you build from this repository. Everything here was run
> on Linux x86_64 only. Where a step was not run on real hardware, this guide says so.

| You are | Read |
|---|---|
| Someone who wants to run a MoE model on an NVIDIA card plus an AMD card | [Section 1](#1-run-a-moe-model-on-an-nvidia-and-an-amd-card) |
| A [Strata](https://github.com/Niko1221/Strata) user: this release ships with a Strata plug-in to try | [Section 1](#1-run-a-moe-model-on-an-nvidia-and-an-amd-card), paths A and B |
| A Python user (scripts, a Python engine, experiments) | [Section 2](#2-python-user) |
| A C or C++ user of the SDK | [Section 3](#3-c-and-c-user) |
| Stuck on an error | [Section 4](#4-troubleshooting) |

## What exists

The pieces and their evidence are in the [README](../README.md#why-engines-benefit) table and
[STATUS](STATUS.md). Only the first two serve a real model across two vendors today, both for Strata: the
[Strata integration](../compat/strata/README.md) (measured on one machine, mostly n=1) and the
[Strata adapter](../adapters/strata/README.md) on the new runtime (live runs on one machine, 2026-11-16 and 2026-11-17).
The [status report](STATUS.md#private-alpha7-rehearsal) covers the private alpha.7 installer and serving rehearsals. The [ggml adapter](../adapters/ggml/MAP.md) is fixtures only: the hook
appends ggml ops and does not call the Seabraid runtime, so **no expert split inside llama.cpp has been run**. Other
engines: [ENGINE-INTEGRATION.md](ENGINE-INTEGRATION.md).

## 1. Run a MoE model on an NVIDIA and an AMD card

### 1.1 What you need

- Linux. Windows is out of scope for the add-on and for the Strata adapter.
- The Strata engine at the pinned commit: v0.2.15, `c77572806996349b49995e4f98d41b5294d36a21`
  (file `compat/strata/patches/strata/STRATA_BASE`). Which models run is Strata's business.
- ROCm with `hipcc` for the AMD card and CUDA with `nvcc` for the NVIDIA card.
  Supported cards are bounded by Strata's kernels: NVIDIA sm_80 and newer, AMD RDNA3 and RDNA4.
- `git`, `cmake`, a C++17 compiler and Python 3. For the new runtime path also Rust 1.79+ and
  Linux 5.4+ (pidfd).
- Strata's data directory with one installed pack (Strata's own `setup.py` makes it).
- Host RAM for experts that live outside the cards. On AMD, experts in RAM need the GTT route or the
  GPU can stall for seconds to minutes; keeping the model in VRAM avoids it.

The idea behind both paths: the faster card runs the main engine, the other card runs a small
helper process built for its own vendor. They talk through shared memory once per layer. Each
card computes the experts it holds. Measured: a tie with stock when the model fit on the main
card; 8 of 10 classification prompts answered against 0 of 10 for stock at a smaller expert
cap (different metrics; two points, not a threshold).

### 1.2 Path A: the Strata integration (the measured one)

This is the path that has the speed numbers. Work from its directory:

```sh
cd compat/strata
python3 install/seabraid_setup.py --plan-only      # detect cards and toolkits, print a plan, change nothing
python3 install/seabraid_setup.py --pack NAME --yes   # build, write services, start them
```

`--plan-only` exits non-zero when it finds a blocker (for example an AMD card but no ROCm
root: it names the fix, `--rocm-path`). That exit is the signal, not a crash. Other useful flags
are `--dry-run`, `--no-build`, `--no-start`, `--port N` (default 8080), `--rocm-path DIR`,
`--cuda-path DIR`, `-v`, and `--bench` (loads each card to rank main versus helpers).
`python3 install/seabraid_setup.py --help` lists all of them. The installer fetches Strata,
checks out the pinned commit, applies the patch series with `git am`, builds, writes one systemd
user service per helper plus the server, and waits for `/health`.

What it needs from you: the toolkits above, a Strata data directory, and network access to
fetch Strata (the installer is the one place that does; the Seabraid runtime itself never
opens a connection). To build by hand, follow "Manual build" in the
[add-on README](../compat/strata/README.md#manual-build).

For file-reference moves, enable both `--ladder` and `--file-moves`. Setup uses `seabraid-sums` to verify the pack's `expert-sums.v1` or rebuild it when needed.
Add `--no-start` to write the units and leave the services stopped. The IQ3_S rehearsal reached this point with a fixture hardware plan, real pack data and a temporary unit directory.
A full unattended installation through service startup remains unverified.

### 1.3 Path B: the Strata adapter on the new runtime (opt-in)

Use this when you want the new runtime's behaviour: a helper loss that fails one request and
leaves the engine running, and live placement. It is opt-in and never the default. In short:

1. Build Strata at the pin, apply the whole series, then the opt-in patch:

   ```sh
   git clone https://github.com/Niko1221/Strata strata && cd strata
   git checkout "$(tr -d '\r' < /path/to/seabraid/compat/strata/patches/strata/STRATA_BASE)"
   git am /path/to/seabraid/compat/strata/patches/strata/*.patch
   git am /path/to/seabraid/adapters/strata/strata-v7-seabraid.patch
   ```

   Configure with `-DSTRATA_SEABRAID_SOURCE_DIR=/path/to/seabraid` and the flags in the add-on README.
   Build the helper for the other vendor's card the same way, with the same window setting.
2. Set `SEABRAID_QWEN_HELPER` to the helper's absolute path. The launcher refuses a helper that is
   not owned by you or root, is group- or world-writable (run `chmod g-w` on it), or is setuid.
3. Add to the engine's own command line: `--serve --remote-expert-opt` and one explicit
   `--expert-cache-device1 N`. The slot count `N` is a number you derive from the helper card's free
   VRAM. The formula is in the [adapter README](../adapters/strata/README.md#opt-in-strata-source-integration).
4. Put `CUDA_VISIBLE_DEVICES` (NVIDIA helper) or `HIP_VISIBLE_DEVICES` (AMD helper) in the engine's
   own environment. The helper inherits only an allow-listed environment and uses the visible
   ordinal zero.
5. For live placement add `--retier on --retier-promote seabraid` (and `--adapt-every` and
   `--adapt-swaps` above 0). It is refused with a message and exit 2 if combined with `--batch` above 1,
   a layer split or `--peer-device`. More than one helper (up to three, `SEABRAID_STRATA_SHM` list), moves to and
   from RAM and NVMe (`--retier-fallback`) and multi-rung climbs (`--retier-hops`) are what the add-on
   installer's `--ladder` switches on; `--file-moves` adds file-reference moves and needs `--ladder`. All of it is opt-in; it ran
   once on real cards ([live run L1, 2026-11-19](RESEARCH-LOG.md#2026-11-19--live-run-l1-the-pyramid-on-real-cards)), with decode speed load-contaminated (4.2 to 7.8 tok/s, SSD tier starved), so a speed against static placement is **not yet shown on GPUs**.

Do not run this path and Path A on the same engine. For this tier keep the named helper, hot share,
split stages and batch serving off; of the re-tier modes only `--retier-promote seabraid` is accepted.

The adapter's tests need no GPU, only a Strata source tree (`cmake -S adapters/strata -B build/strata-adapter -DSTRATA_SOURCE=/path/to/strata`, then build and `ctest`).

### 1.4 Path C: llama.cpp through the ggml adapter (a fixture, not a deployment)

The adapter computes routed SwiGLU on a ggml backend, and an opt-in patch adds a hook to llama.cpp's
Qwen3 MoE graph. On a loaded model the hook is bit-identical to stock llama.cpp on CPU, CUDA, HIP
and Vulkan: it changes nothing and does not send experts to another card. Use it to test the adapter,
not to run a model faster. Build steps: [`adapters/ggml/MAP.md`](../adapters/ggml/MAP.md). The fixture build is:

```sh
cmake -S adapters/ggml -B build/seabraid-ggml -DSEABRAID_GGML_TESTS=ON -Dggml_DIR=/path/to/ggml/package
cmake --build build/seabraid-ggml
ctest --test-dir build/seabraid-ggml --output-on-failure
```

### 1.5 How to check it is working

For Path A:

- The installer ends with `seabraid is serving <pack> on <host>:<port>` and a decode tokens-per-second
  figure from a fixed test chat.
- `systemctl --user status seabraid-server.service seabraid-helper-0.service` shows both running.
- `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/health` prints `200`
  (use your `--port`). Chat requests go to `/v1/chat/completions`.
- `nvidia-smi` and `rocm-smi` show memory in use on both cards.
- The measured baseline is a tie with stock when the model fits, and a gain only when it does not
  fit. If your model fits on one card, expect no speed-up.

For Path B:

- A helper process is a child of the engine, started by the engine: `ps --ppid <engine pid>`.
- Records from the runtime reach stderr with the prefix `seabraid:` (`seabraid-strata:` for the
  adapter). With `--retier-promote seabraid` a `seabraid: INFO: placement round=...` line appears per round.
- To see every expert move, set `SEABRAID_TRACE=/path/trace.jsonl` and run
  `python compat/strata/tools/trace_report.py /path/trace.jsonl`: one row per hop, plus where the time goes.
- A helper that dies mid-request gives one error for that request, and the next is served from the
  local tiers (records: helper exit with signal, collect status, tier retired, local fallback).

The quality gate is KL against the same engine on one card: `compat/strata/tools/kl_gate.sh`
(needs two configs and a text file you supply; the script names what is missing).

### 1.6 How to stop it

- Path A: `systemctl --user stop seabraid-server.service seabraid-helper-0.service`
  (one `seabraid-helper-N.service` per helper). Run the helper unit with `Restart=on-failure`;
  without patch 0019 a dead helper froze the engine 30 s, then it exited (live run L1). With 0019 and 0020 only that
  card closes and the request finishes on the CPU ([runs HL, HL2](RESEARCH-LOG.md#2026-11-20--helper-loss-on-gpus-hl-hl2)).
- Path B: stop the engine. Close sends SIGTERM to the helper, gives it 300 ms, then SIGKILL, and reaps it.
  If the engine itself dies, the helper exits on its own (a pidfd watch on the controller).
- Kill exact process ids only: `kill $(pgrep -f ...)` can match its own shell.
- Redeploying an install made before 2026-11-21: run the installer again; names changed. Units `seabraid-server.service`
  and `seabraid-helper-N.service`, binaries `seabraid-helper` and `seabraid-sums`, channels `/seabraid0` and `/seabraid1`,
  switches `SEABRAID_STRATA_SHM`, `SEABRAID_STRATA_HOT_SHARE`, `SEABRAID_STRATA_EXCLUDE_HELPER_RAM` (all `SEABRAID_STRATA_*`).
  The wire is v11: rebuild engine and helper from the same series, an older helper is refused.

### 1.7 Known limits

- Linux only. Strata v0.2.15 only. No model fingerprint in the helper protocol: match the helper's
  pack to the engine's model yourself. It helps when the model overflows the biggest card (a tie when it fits).
- Every speed figure is one machine, small n. The DeepSeek-V4.1 figures are historical (Strata 0.2.14
  removed that engine) and its quality gate failed.
- Live placement did **not** pass its first matched-arm run (host page-migration stalls hit every arm,
  placement does not settle at the default margin). Treat `--retier-promote seabraid` as experimental.
- Hot-share and D4 remote prefill are opt-in with no measured win. Windows handle handoff and
  multiple helpers on the new runtime: code exists, no GPU run yet. Other engines (vLLM, SGLang,
  KTransformers, ik_llama.cpp, Project Maya): no adapter.

## 2. Python user

The package has no dependencies; numpy is used if present.

### 2.1 Install and run

`pip install seabraid` is not available. From the repository root (stock Ubuntu needs a venv):

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install .
python examples/quickstart.py
```

`pip install .` downloads `setuptools>=77` into an isolated build environment. A venv on Python 3.12
or newer has no setuptools of its own. Offline, make the venv with the system packages and skip the
isolation (works when the system already has `setuptools>=77`):

```sh
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --no-build-isolation --no-index .
python examples/quickstart.py
```

Expected output of the quickstart (pure Python, no warning):

```text
seabraid 0.5.0a8
provider=reference result=[2.5, 5.0]
stale epoch refused: status 2
card order: RX 7900 XTX > RTX 3060 (basis: Basis(card1='lead', rest='rows'))
```

### 2.2 Reference versus native

| You write | What runs |
|---|---|
| `Runtime(reference=True)` | The built-in demonstration transform. Pure Python without a library, the native engine with one. Silent: you asked for it |
| `Runtime()` with no library | The pure-Python reference provider, with one `RuntimeWarning` and one `seabraid` log warning |
| `Runtime(library_path, provider=fn)` | Your Python callable serves the experts; the native runtime keeps queue, tickets and timeouts |
| `Runtime()` with `SEABRAID_LIBRARY` set | Refused with `ValueError`: a native library never picks a transform for you |

The reference provider is an API demonstration and test transform, not model math. To run the same
code on the native engine, build the library and name it with an absolute path:

```sh
cargo build --offline
SEABRAID_LIBRARY="$PWD/target/debug/libseabraid.so" python examples/quickstart.py
```

Native output has one extra first line, `seabraid: WARN: seabraid_session_submit: placement_epoch 1 is stale ...`.
It is intended: the example submits an old epoch on purpose to show it is refused.

A platform wheel carries the library so no source build is needed on the target machine:

```sh
python scripts/build_wheel.py                       # needs cargo, setuptools>=77 and (Linux) readelf
python3 -m venv .venv-wheel
.venv-wheel/bin/pip install --no-index dist/seabraid-*.whl
.venv-wheel/bin/python -c "import seabraid; print(seabraid.bundled_library())"
```

`bundled_library()` returns the path of the shipped library, or `None` on the pure package.
Install the Linux x86-64 wheel in a fresh venv on glibc 2.34 or newer so pip picks up the bundled library.
For experimental Windows core builds, see the [source build instructions](../README.md#experimental-windows-core-runtime).

### 2.3 The calls you will use

```python
import seabraid

with seabraid.Runtime(reference=True) as rt:
    layer = rt.prepare(hidden=4, experts=8, max_tokens=4, top_k=2)
    rows    = [1.0, 2.0, 3.0, 4.0]              # tokens * hidden, row-major float32
    ids     = [0, 5]                            # tokens * top_k expert ids
    weights = [0.6, 0.4]                        # tokens * top_k router weights
    print(layer.weave(rows, ids, weights))      # one call: submit and collect; prints [3.0, 6.0, 9.0, 12.0]
```

- Inputs are flat. A C-contiguous float32 or int32 buffer (`array.array`, `memoryview`, numpy) is used
  without per-element work; plain lists work but are slow.
- `seabraid.enable_logging()` forwards the native library's log records to the `seabraid` logger
  (it needs a native library; with none it warns and returns `False`). `seabraid.version()` names the build.
- `layer.weave(..., operation=i)` picks a prepared operation; `placement_epoch=` tags the batch.
- Errors are `seabraid.SeabraidError` with `.operation`, `.status` and `.detail`. Status 1 invalid,
  2 busy or stale epoch, 3 timeout, 4 provider failure, 5 internal.
- After `os.fork()` the child cannot use an inherited native session: open a new `Runtime` there.
- To serve experts from Python (a Python engine), pass `provider=fn`; the contract is in the
  [adapter guide](ADAPTER-GUIDE.md#hosting-a-provider-from-python).

### 2.4 Ranking and placement helpers

`seabraid.ranking.rank(cards, bench_lead, bench_rows, fits_main=...)` orders cards: which is the lead,
which are helpers, with the basis it used. Cards are any objects with `key`, `name`, `arch`, `free`,
`total` and `pcie`; [`examples/quickstart.py`](../examples/quickstart.py) shows a full call. It never
raises for a bad measurement; it drops all cards to a proxy order and says so in the basis. The rules
are untested on hardware.

`seabraid.placement.fill(layers, experts, slots, heat, library_path=None)` is the Python form of
`seabraid_placement_fill`. `slots` is one capacity per card, lead first. `heat` and the result are
layer-major. A result of `0..cards-1` names a card; the value `cards` means the fallback (host RAM or
the model file). It needs a native library:

```python
import seabraid
owners = seabraid.placement.fill(layers=2, experts=4, slots=[3, 2],
                                 heat=[9, 1, 5, 2, 8, 7, 3, 4], library_path="/abs/path/libseabraid.so")
print(owners)   # [0, 2, 1, 2, 0, 0, 2, 1]: card 0 holds 3 cells, card 1 holds 2, the other 3 go to the fallback
```

Only `fill` is bound in Python. The live placement handle (`open`, `plan`, `apply`, `snapshot`) is C only.

### 2.5 Run the Python tests

Run `python -m pytest -q tests bindings/python/tests` (install `pytest` first, see CONTRIBUTING). Native tests skip unless `SEABRAID_LIBRARY` is set.

## 3. C and C++ user

### 3.1 Build and install the SDK

You need CMake 3.22+, a C11 compiler and Rust 1.79+. If your shell cannot find
`cargo` after a standard `rustup` install, run `export PATH="$HOME/.cargo/bin:$PATH"`.

```sh
cmake -S . -B build/seabraid -DSEABRAID_BUILD_NATIVE_TESTS=ON
cmake --build build/seabraid
ctest --test-dir build/seabraid --output-on-failure
cmake --install build/seabraid --prefix /path/to/seabraid-sdk
```

Install outside the source tree so repository checks exclude the SDK. Add `--config Release` for a multi-config generator.
The `Runtime` and `Development` components both install licenses and notice under `share/doc/seabraid` by default.
Projects that vendor the public C headers can choose Apache-2.0 or MIT. The runtime uses Apache-2.0. See [NOTICE](../NOTICE) for the MIT-licensed upstream patches.

### 3.2 Use it from your project

```cmake
find_package(Seabraid 0.5 CONFIG REQUIRED)       # set CMAKE_PREFIX_PATH to the SDK directory
target_link_libraries(my_app PRIVATE Seabraid::runtime)
```

[`examples/consumer/CMakeLists.txt`](../examples/consumer/CMakeLists.txt) is a complete project. The
smallest useful program checks that the header and the library agree before it calls anything else:

<!-- compilable -->
```c
#include <seabraid/seabraid.h>
#include <stdio.h>

int main(void)
{
    if (seabraid_abi_epoch() != SEABRAID_ABI_EPOCH) { /* a library from another ABI epoch must not be driven */
        fprintf(stderr, "seabraid: header and library disagree on the ABI epoch\n");
        return 1;
    }
    printf("seabraid %s\n", seabraid_version());
    return 0;
}
```

Save it as `main.c` next to this `CMakeLists.txt`, then configure with the SDK on the path:

```cmake
cmake_minimum_required(VERSION 3.22)
project(hello LANGUAGES C)
find_package(Seabraid 0.5 CONFIG REQUIRED)
add_executable(hello main.c)
target_link_libraries(hello PRIVATE Seabraid::runtime)
```

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/seabraid-sdk
cmake --build build
./build/hello
```

Without CMake, point `gcc` at the installed SDK. The `-Wl,-rpath` is required because the library's
SONAME is `libseabraid.so.0.5`:

```sh
gcc -std=c11 -I/path/to/seabraid-sdk/include main.c -L/path/to/seabraid-sdk/lib \
    -Wl,-rpath,/path/to/seabraid-sdk/lib -lseabraid -o hello
```

An installed program needs `INSTALL_RPATH` set to `$ORIGIN/../lib` (`cmake --install` drops the build-tree rpath); the consumer example does this.

### 3.3 Provider examples to start from

| File | What it shows |
|---|---|
| [`examples/minimal_provider.c`](../examples/minimal_provider.c) | A complete provider (21 lines of callback) and its driver: open, submit, collect, close. Prints `output = [2.5, 5]` |
| [`examples/adapter_walkthrough.c`](../examples/adapter_walkthrough.c) | Provider v3 with partial routes, `COMPACT_ROWS`, placement epochs, refusals, a device failure, session status and a log sink. Prints `ok` last |
| [`examples/consumer`](../examples/consumer/CMakeLists.txt) | Both of the above built against an installed SDK with `find_package` |

Every C call returns a status; `seabraid_last_error` says why a nonzero one failed. A null provider is
invalid: `SEABRAID_PROVIDER_REFERENCE` selects the demonstration transform on purpose. A C++ provider
must catch every exception inside its callback. The full call reference is [ABI.md](ABI.md); the
design of a provider and the placement loop is [ADAPTER-GUIDE.md](ADAPTER-GUIDE.md).

## 4. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error while loading shared libraries: libseabraid.so.0.5: cannot open shared object file` (exit 127) | The library's SONAME is `libseabraid.so.0.5`. Link with `-Wl,-rpath,<sdk>/lib`, or set `INSTALL_RPATH` to `$ORIGIN/../lib` for an installed program. A bare `cargo build` writes only `libseabraid.so`, so for a program linked against `target/debug` run `ln -sf libseabraid.so target/debug/libseabraid.so.0.5` and set `LD_LIBRARY_PATH=target/debug`; the CMake build makes that symlink for you |
| `No module named 'setuptools'` or `pip install .` fails to build | Python 3.12+ venvs have no setuptools. `pip install "setuptools>=77"` in the venv. Offline: `--system-site-packages` and `pip install --no-build-isolation --no-index .` |
| `pip install` refused: "externally-managed-environment" | PEP 668. Make a venv first (`python3 -m venv .venv && . .venv/bin/activate`) |
| `cargo: command not found` after installing Rust (or `python: command not found` outside a venv) | `export PATH="$HOME/.cargo/bin:$PATH"`; stock Ubuntu has `python3` only until a venv is active |
| `ValueError` from a bare `Runtime()` with `SEABRAID_LIBRARY` set | A native library needs `reference=True` or `provider=fn` |
| A `RuntimeWarning` about the reference provider | Expected for a bare `Runtime()` with no library. Pass `reference=True` to say you mean it |
| `seabraid: WARN: ... placement_epoch 1 is stale` in the native quickstart | Intended: the example submits an old epoch to show the refusal |
| `SEABRAID_BUSY_OR_STATE` from a submit | A ticket is still outstanding, the session is closing, or a stale epoch. With no ticket out it is a stale epoch: re-snapshot, re-route, resubmit with the new epoch. `seabraid_last_error` gives the text |
| A submit fails and nothing is logged | `min_level` of the log sink hides WARN. Lower it, or read `seabraid_last_error` |
| `seabraid_session_status` returns busy from a metrics thread | Any call on a session from another thread during a blocking collect is refused. Read counters from the owning thread, between calls |
| Ctrl-C does nothing in a test or a script started with `cmd &` or `nohup` | A background shell starts the job with SIGINT ignored, so Python installs no handler. Stop it with `kill <pid>`; a test that needs SIGINT installs the handler itself (`test_a_native_interrupt_during_collect_leaves_the_session_usable`) |
| `pytest` skips most native tests | `SEABRAID_LIBRARY` is unset. Build with `cargo build --offline` and set it to the absolute path |
| Helper refused to start (Path B) | The helper must be an absolute path to a regular file you or root own, not group- or world-writable, not setuid. Run `chmod g-w` on it. It needs `/proc` and, preferably, Linux 5.4+ |
| Installer stops: "BLOCKERS" | `--plan-only` names the fix for each, such as `--rocm-path` or `--cuda-path` |
| An engine on Path A freezes about 30 s then exits | The named-channel helper died on a tree without patch 0019. Apply the full series (through 0020), run the helper unit with `Restart=on-failure`, or use Path B |

## 5. Reporting problems

Open an issue. Report a vulnerability privately, never as an issue: see [SECURITY.md](../SECURITY.md).
Contributions (pull requests) are closed for now: see [CONTRIBUTING.md](../CONTRIBUTING.md).
