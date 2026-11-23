# Sabotage record

Every checker in `checks/` has been watched failing.  `tests/test_sabotage.py` copies the tree, proves the checker is
green on the copy, plants ONE violation, and requires exit code 1 plus the output line below.  Regenerate this file
with `SABOTAGE_REPORT=checks/SABOTAGE.md python -m pytest tests/test_sabotage.py`; `test_sabotage_doc_lists_every_case`
fails if a case is missing from it.  Only the table between the marker comments is regenerated; the sections below it are hand-written and are kept.  Line numbers in the last column are those of the recording run.

<!-- sabotage-table:begin -->
| case | checker | planted | failing output line |
|---|---|---|---|
| refs-home-path | check_no_local_refs.py | a developer home-directory path appended to README.md | README.md:214:10: home-path: shape |
| refs-private-ip | check_no_local_refs.py | a private LAN address in a comment in src/lib.rs | src/lib.rs:24:9: private-ip: shape |
| refs-gpu-uuid | check_no_local_refs.py | a GPU UUID (GPU-xxxxxxxx-xxxx-...) in a CUDA_VISIBLE_DEVICES line appended to README.md | README.md:270:22: gpu-uuid |
| refs-dated-owner | check_no_local_refs.py | a dated person attribution in parentheses (the word owner, a comma, a year) appended to README.md | README.md:413:13: dated-owner |
| refs-internal-vocab | check_no_local_refs.py | the maintainers' machine named as a place for a run (the word box before run) in a comment in src/lib.rs | src/lib.rs:40:20: internal-vocab |
| placeholder-marker | check_no_placeholders.py | docs/leak.md: a line with an unfilled fill-in marker, as .zenodo.json once had for the paper DOI | docs/leak.md:2: unfilled placeholder |
| compat-edge-import | check_compat_edges.py | src/leak.py: `import compat.strata` (the core importing the add-on) | src/leak.py:1:1: compat reference from a core scope (move the code under compat/strata/ or drop the reference) |
| compat-edge-path-join | check_compat_edges.py | adapters/ggml/leak.py: a path join that names the compat directory | adapters/ggml/leak.py:1:24: compat reference from a core scope (move the code under compat/strata/ or drop the reference) |
| compat-edge-allow-row-rots | check_compat_edges.py | an allow row for a file that has no compat reference | checks/compat_edges_allow.txt: row for 'src/lib.rs' matches no compat reference; delete it |
| egress-rust | check_no_egress.py | src/leak.rs: `use std::net::TcpStream;` | src/leak.rs:1: egress: std::net |
| egress-c | check_no_egress.py | adapters/ggml/leak.c: a `socket(` call | adapters/ggml/leak.c:1: egress: socket( |
| egress-python | check_no_egress.py | bindings/python/seabraid/leak.py: `import urllib.request` | bindings/python/seabraid/leak.py:1: egress: import urllib |
| egress-cmake | check_no_egress.py | `file(DOWNLOAD ...)` appended to adapters/ggml/CMakeLists.txt | adapters/ggml/CMakeLists.txt:35: egress: file(DOWNLOAD |
| egress-compat-non-loopback | check_no_egress.py | compat tool that opens http://example.invalid (not loopback, not allow-listed) | compat/strata/tools/leak.py:2: egress: http://example.invalid/x |
| egress-allow-row-rots | check_no_egress.py | the allow-listed Strata URL removed from build.py, so its ALLOW row matches nothing | unused ALLOW row: compat/strata/install/build.py github\.com/Niko1221/Strata |
| errors-cpp-catch | check_error_paths.py | examples/leak.cpp: an empty `catch (...)` block (--strict-runtime; examples/ is a runtime scope with no files today) | examples/leak.cpp:1: catch-without-log |
| errors-rust-discard | check_error_paths.py | examples/leak.rs: `let _ = g();` (--strict-runtime) | examples/leak.rs:1: let-underscore-call |
| errors-py-silent | check_error_paths.py | examples/leak.py: `except Exception: pass` (--strict-runtime) | examples/leak.py:3: silent-except |
| errors-ratchet-new-file | check_error_paths.py | a NEW compat file with a bare `except:` (ratchet baseline 0) | compat/strata/tools/leak.py: 1 hit(s), baseline 0 (ratchet: may shrink, never grow) |
| errors-ratchet-grows | check_error_paths.py | a second swallowed error appended to a baselined compat file (ratchet grows 1 to 2) | compat/strata/tests/install/test_probe.py: 2 hit(s), baseline 1 (ratchet: may shrink, never grow) |
| spdx-missing | check_spdx.py | a new src/new.rs with no SPDX line | src/new.rs:1: expected Apache-2.0, found missing |
| spdx-header-wrong | check_spdx.py | include/seabraid/seabraid.h downgraded from `Apache-2.0 OR MIT` to `Apache-2.0` | include/seabraid/seabraid.h:1: expected Apache-2.0 OR MIT, found Apache-2.0 |
| spdx-too-late | check_spdx.py | SPDX identifier on line 6 (outside the first 5 lines) | src/late.rs:1: expected Apache-2.0, found missing |
| size-501-lines | check_size.py | src/big.py with 501 lines | src/big.py: 501 lines (limit 500) |
| size-excepted-file-grows | check_size.py | an excepted 501-line file grown to 521 lines | src/big.py: 521 lines (ceiling 501) |
| version-cargo | check_version.py | Cargo.toml version changed to 0.5.1 | Cargo.toml: found 0.5.1, expected <VERSION> (read from the VERSION file) |
| version-cargo-lock | check_version.py | Cargo.lock seabraid entry left at 0.5.1 while VERSION and Cargo.toml moved on | Cargo.lock: found 0.5.1, expected <VERSION> (read from the VERSION file) |
| version-cmake | check_version.py | CMake project VERSION hard-coded to 0.6.0 instead of read from VERSION | CMakeLists.txt: found 0.6.0, expected 0.5.0 |
| version-pyproject-literal | check_version.py | pyproject.toml with literal version 9.9.9 | pyproject.toml: found 9.9.9, expected <VERSION> (read from the VERSION file) |
| version-pyproject-dynamic-wrong-file | check_version.py | pyproject.toml dynamic version reading a file other than VERSION | pyproject.toml: dynamic version must read the VERSION file, found OTHER |
| abi-offset-moved | check_abi_append.py | src/layout/abi_layout.txt: seabraid_batch_v1.tokens moved from offset 4 to 8 | src/layout/abi_layout.epoch1.txt: seabraid_batch_v1.tokens: offset moved 4 -> 8 |
| abi-grows-without-struct-size | check_abi_append.py | src/layout/abi_layout.txt: seabraid_config_v1 (no struct_size field) grown from 20 to 24 bytes | src/layout/abi_layout.epoch1.txt: seabraid_config_v1: size grew 20 -> 24 but the struct has no .struct_size field to tell an old caller apart |
| abi-v2-offset-moved | check_abi_append.py | src/layout/abi_layout.txt: seabraid_batch_v2.tokens (a v2 row, absent from the pre-2026-11-19 v1-only snapshot) moved from offset 24 to 28 | src/layout/abi_layout.epoch1.txt: seabraid_batch_v2.tokens: offset moved 24 -> 28 |
| abi-epoch-bumped-without-snapshot | check_abi_append.py | SEABRAID_ABI_EPOCH raised to 2 in the header with no abi_layout.epoch2.txt (the old checker skipped silently) | src/layout: no snapshot for epoch 2 (have: epoch1); freeze abi_layout.epoch2.txt with the bump |
| links-missing-file | check_links.py | a relative link to docs/NO-SUCH-DOC.md appended to README.md | README.md:279: broken link (docs/NO-SUCH-DOC.md): no such file or directory |
| links-missing-anchor | check_links.py | a link to docs/ABI.md#no-such-heading (the file exists, the heading does not) appended to README.md | README.md:279: broken link (docs/ABI.md#no-such-heading): no heading or anchor '#no-such-heading' in docs/ABI.md |
| names-in-content | check_names.py | `<probe>_open()` in a comment in src/lib.rs (forbidding a probe name that occurs nowhere else) | src/lib.rs:27:4: forbidden name '<probe>' |
| names-in-path | check_names.py | include/<probe>/x.h (name in a directory path) | include/<probe>/x.h: forbidden name '<probe>' in the path |
| names-camel-prefix | check_names.py | `const int k<Probe> = 1;` appended to src/lib.rs (a letter before the name; the old rule let it through) | src/lib.rs:27:12: forbidden name '<Probe>' |
| names-upper-suffix | check_names.py | `const int <PROBE>X = 1;` appended to src/lib.rs (a letter after the name) | src/lib.rs:27:11: forbidden name '<PROBE>' |
| names-in-readme | check_names.py | the name in README.md (outside the old runtime-only scan) | README.md:270:1: forbidden name '<Probe>' |
| names-allow-row-rots | check_names.py | a checks/names_allow.txt row whose glob matches no file | checks/names_allow.txt: glob 'gone/*.txt' matches no file |
| helper-tier-ram-rule-reverted | check_helper_tier.py | the 2026-11-17 bug put back: the v7 patch no longer replaces 0001's `seabraid()` RAM-copy rule | compat/strata/patches/strata/0001-External-expert-helper-tier-an-out-of-process-helper.patch:1016: seabraid() asks for the named-channel helper only; use process_tier() or add a row to checks/helper_tier_allow.txt saying why the private helper is excluded: if (remote_experts[(size_t) r].seabraid() && !bl_exclude) continue; |
| helper-tier-stale-row | check_helper_tier.py | an allow row whose `seabraid()` call site does not exist | checks/helper_tier_allow.txt:8: fragment matches 0 call sites, needs exactly 1: 'gone.seabraid()' |
| pid-probe-kill-readded | check_pid_probes.py | the D4 bug put back: a `::kill(helper_pid, 0)` liveness probe added to the v7 patch (the sites 0013 had are explained rows; a new one is not) | adapters/strata/strata-v7-seabraid.patch:1941: kill: use the PeerWatch pidfd identity / bl_wait / wire pidfd.hpp, or add a row to checks/pid_probes_allow.txt saying why: if (::kill(helper_pid, 0) != 0) return false; |
| pid-probe-kill-in-repo-cpp | check_pid_probes.py | adapters/strata/leak_probe.cpp: a `::kill(helper_pid, 0)` probe in repo C++ | adapters/strata/leak_probe.cpp:1: kill: use the PeerWatch pidfd identity / bl_wait / wire pidfd.hpp, or add a row to checks/pid_probes_allow.txt saying why: bool alive(int helper_pid) { return ::kill(helper_pid, 0) == 0; } |
| pid-probe-second-pidfd-open | check_pid_probes.py | adapters/strata/leak_pidfd.cpp: a second `SYS_pidfd_open` syscall outside wire pidfd.hpp | adapters/strata/leak_pidfd.cpp:1: pidfd_open: use the PeerWatch pidfd identity / bl_wait / wire pidfd.hpp, or add a row to checks/pid_probes_allow.txt saying why: int f(int p) { return (int) ::syscall(SYS_pidfd_open, p, 0); } |
| pid-probe-bare-wait | check_pid_probes.py | a `Channel::wait(` line added to src/core/remote_experts.cpp by the v7 patch that is not the single bl_wait wrapper | adapters/strata/strata-v7-seabraid.patch:1942: bare-wait: use the PeerWatch pidfd identity / bl_wait / wire pidfd.hpp, or add a row to checks/pid_probes_allow.txt saying why: if (!seabraid::Channel::wait([&] { return done(); }, 30)) return false; |
| pid-probe-stale-row | check_pid_probes.py | an allow row whose pid-probe site does not exist | checks/pid_probes_allow.txt:29: fragment matches 0 sites, needs exactly 1: 'gone.kill(1)' |
<!-- sabotage-table:end -->


## check_no_egress.py evasions (tests/test_no_egress.py)

The 2026-11-17 rewrite scans the whole tracked tree.  Each row below is a planted evasion in a throwaway tree that the old
checker passed and the current one must fail (`python -m pytest tests/test_no_egress.py`).  The failing output line is always
`<path>:<line>: egress: <matched text>` (`egress-binary` for the NUL-file case).

| case | planted at |
|---|---|
| include-socket-h | src/x.c |
| include-curl | src/x.c |
| include-netdb | src/x.c |
| include-arpa | src/x.c |
| include-netinet | src/x.c |
| rust-cfg-use | src/x.rs |
| rust-tokio-net | src/x.rs |
| rust-reqwest | src/x.rs |
| c-star-line | src/x.c |
| c-global-socket | src/x.c |
| c-global-connect | src/x.c |
| c-sendto | src/x.c |
| c-sendmsg | src/x.c |
| c-recvfrom | src/x.c |
| c-getaddrinfo | src/x.c |
| c-gethostbyname | src/x.c |
| c-system-curl | src/x.c |
| c-popen-wget | src/x.c |
| c-execv-ssh | src/x.c |
| c-dlopen-curl-lib | src/x.c |
| c-url-ftp | src/x.c |
| c-url-ws | src/x.c |
| c-url-wss | src/x.c |
| rust-command-curl | src/x.rs |
| rust-unixstream | src/x.rs |
| rust-udpsocket | src/x.rs |
| rust-tcplistener | src/x.rs |
| py-dunder-import | bindings/python/seabraid/x.py |
| py-import-module | bindings/python/seabraid/x.py |
| py-from-http-import-client | bindings/python/seabraid/x.py |
| py-urllib3 | bindings/python/seabraid/x.py |
| py-httpx | bindings/python/seabraid/x.py |
| py-aiohttp | bindings/python/seabraid/x.py |
| py-requests | bindings/python/seabraid/x.py |
| py-websocket | bindings/python/seabraid/x.py |
| py-after-semicolon | bindings/python/seabraid/x.py |
| py-import-list | bindings/python/seabraid/x.py |
| py-subprocess-curl | bindings/python/seabraid/x.py |
| py-smtplib | bindings/python/seabraid/x.py |
| localhost-in-trailing-comment | compat/strata/tools/x.py |
| localhost-lookalike-host | compat/strata/tools/x.py |
| loopback-lookalike-ip | compat/strata/tools/x.py |
| compat-api-without-url | compat/strata/tools/x.py |
| runtime-loopback-url | src/x.rs |
| unscanned-build-rs | build.rs |
| unscanned-root-cmake | CMakeLists.txt |
| unscanned-cmake-dir | cmake/x.cmake |
| unscanned-tools | tools/x.py |
| unscanned-native-tests | native-tests/x.sh |
| unscanned-scripts | scripts/x.py |
| unscanned-examples-sh | examples/x.sh |
| extensionless-sh | scripts/run |
| extensionless-python | scripts/serve |
| patch-added-include | adapters/x.patch |
| patch-added-python | p/x.patch |
| binary-with-nul | docs/blob.dat |
| patch-context-not-scanned | negative control: unchanged and removed patch lines must stay clean |
| localhost-same-line | a loopback URL plus a remote URL on one line: only the remote one is a hit |
| unreadable-file | a tracked file that cannot be read is an `egress-unreadable` hit |
| link-gate-parser | planted `connect` import in `nm -D --undefined-only` output is flagged |
| built-library-link-gate | Linux only: libseabraid.so must import none of socket/connect/bind/listen/accept/getaddrinfo/gethostbyname/sendto/recvfrom |

## check_complexity.py (2026-11-17)

Run on temp trees with lizard 1.26.1; each exits 1. The other rules (empty reason, growth past the
ceiling, duplicate row, row for a missing function) are covered by `tests/test_complexity.py`.

| Planted | Output |
|---|---|
| a function with 11 `if` branches (CCN 12), empty allow file | `src/mod.py:1: f CCN 12 > limit 10` |
| the row `src/mod.py f 16 <why>` | `checks/complexity_allow.txt:1: ceiling 16 for f must be 11..15` |
| the row `src/mod.py f 12 <why>` with `f` reduced to `return x` | `stale row for src/mod.py f (CCN is now 1 <= 10); delete it` |

## R8 gates (2026-11-19)

Each row is asserted by the named test file, which damages one thing in a temp tree or a copy of this tree and requires exit 1.

| Checker | Planted | Output | Test |
|---|---|---|---|
| check_complexity.py | CCN 12 function in `compat/strata/strata-ext/`, `native-tests/`, `scripts/`, `tests/`, `examples/`, `checks/`, `include/` or `bindings/python/tests/` | `<path>:1: f CCN 12 > limit 10` | tests/test_complexity.py |
| check_complexity.py | a code file in a directory no `SCOPES` entry covers | `newdir/x.py: code file outside SCOPES` | tests/test_complexity.py |
| check_complexity.py | a ratchet row whose function grew, or shrank without the row being lowered | `CCN 19 > limit 18 (grew past its row)`; `lower the row to 17` | tests/test_complexity.py |
| check_size.py | a 401-line file with no ratchet row; a ratcheted file that grew; a `.yml`/`.toml`/`.json` over 400 | `over the 400-line split point and not in checks/size_ratchet.json`; `past its ratchet of 450` | tests/test_size_ratchet.py |
| check_size_budgets.py | a header budget of 10 lines on an 11-line file; an unreadable budget line; a known-false row that grew | `src/a.py: 11 lines, header budget 10`; `cannot read`; `past its row of 12` | tests/test_size_budgets.py |
| check_version.py | a stale version in CITATION.cff, README, SECURITY, ROADMAP, TECHNICAL-NOTE, USER-GUIDE or ABI.md; a stray old pre-release token in any doc | `docs/NEW-NOTE.md:3: stale version 0.0.1-alpha.9` | tests/test_pins_and_versions.py |
| check_pins.py | a lizard, numpy or PyYAML pin that differs from ci.yml in CONTRIBUTING, steps.toml, this file or `LIZARD_PIN`; an MSRV or CMake floor mention that differs from Cargo.toml or the root CMakeLists.txt; a new sub-project below the root floor | `<file>:<line>: lizard <other>, but .github/workflows/ci.yml installs <pinned>`; `cmake_minimum_required 3.10, root requires 3.22` | tests/test_pins_and_versions.py |
| check_error_paths.py | a swallowed error in `tools/`, `scripts/` (strict) or `native-tests/` (ratchet) | `scripts/y.py:4: silent-except`; `native-tests/t.cpp: 1 hit(s), baseline 0` | tests/test_error_path_scopes.py |
| check_map.py | a new `src/` module not in MAP.md; a renamed module's old row; a checker missing from the `checks/` row; an unsafe-allow attribute outside the listed files | `src module src/newmod.rs`; `is named but does not exist`; `outside the places MAP.md lists` | tests/test_map_checker.py |
