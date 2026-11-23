# Security policy

Seabraid is alpha software (0.5.0-alpha.8). It may be updated or changed at any
time without notice. Use it at your own risk.

## Reporting a vulnerability

Report privately through the code host's private vulnerability reporting: open
the repository's **Security** tab and choose **Report a vulnerability**. Please
do not open a public issue for a suspected vulnerability. Include the version,
the platform, what you did, and what you expected. We will acknowledge a report
and tell you what we plan to do about it. This is a small project; there is no
fixed response-time guarantee.

## Supported versions

Only the latest alpha release is supported. The C ABI can change before 1.0.

## Scope

In scope:

- memory-safety or lifetime bugs in the Rust core and the C ABI
  (`include/seabraid/*.h`), including the provider callbacks and the placement API;
- the Strata adapter's helper process handling: the sealed descriptor handoff,
  geometry validation, the pidfd watchdog, and helper retirement;
- the Python binding's argument validation;
- the installer under `compat/strata/install`, which runs build commands.

Out of scope: vulnerabilities in Strata, llama.cpp, ggml, CUDA, ROCm or driver
code (report those to their owners), and model-quality questions.

## Threat model

Trusted: processes running as the same user (UID) and the process environment;
the engine; the Seabraid runtime library.

Partly trusted: helper processes, which run a vendor runtime and third-party
engine code. A compromised or buggy helper may return wrong numbers, which
affects output quality. It must not be able to corrupt engine memory, read
engine memory beyond the shared region. The engine must retain responsibility
for stopping and collecting a retired helper. The Strata adapter (`adapters/strata`) enforces this with:

- a sealed memfd descriptor handed to the helper, so the shared region has a
  fixed size;
- geometry read once at attach and kept as a snapshot;
- every helper-written field compared against that snapshot, never used
  directly as an index or a size;
- replies copied out of the shared region before they are judged;
- a pidfd watch and a worker reserved before launch. Retirement allows up to
  300 ms for SIGTERM and 20 ms after SIGKILL in the caller. A slow exit or a
  refused wait transfers collection to the worker. A refused SIGKILL is retried.
  If the kernel keeps denying the operation, the worker holds the pidfd for the
  life of the engine. The host must keep the adapter loaded while it is reaping;
- the helper executed through a pinned inode (the vetted file descriptor), not a
  path that can be swapped after the check;
- exact environment names for loader, device, kernel and channel settings.
  The server authentication variable `STRATA_API_KEY` and unlisted variables are
  excluded in normal and debug launches. The child receives allowed values as supplied.

Out of scope: a malicious engine, root, other users sharing the GPU, and side
channels.

Network: the runtime makes no network calls. `checks/check_no_egress.py` scans
the tree for network APIs as a tripwire, not a proof, and a link-level test
(Linux) checks that a built `libseabraid.so` has no undefined socket symbols.
The Strata integration installer is different: it clones upstream Strata from a
pinned commit, builds it and runs its server. That builds and runs third-party
code on your machine. The helper services the installer generates are
sandboxed with systemd: no network address families (only `AF_UNIX` and
`AF_NETLINK` for driver IPC), read-only system and home directories, and
`NoNewPrivileges`.

Fuzzing: random-input runs of the placement ABI, the helper's move handling and the helper-list parser are recorded in
[STATUS](docs/STATUS.md). Not yet done: fuzzing of the channel attach path; Windows (no handle handoff, no helper hardening).

## What the runtime does with your data

Beyond the no-network rule above, the runtime collects no data: no
telemetry, no update check and no crash upload. Providers and adapters receive
activations and weights only as pointers inside one process or one shared-memory
mapping on the same machine. A checker (`checks/check_no_local_refs.py`) keeps
machine-specific names and paths out of the tracked files; log text is
restricted to status codes and short messages, with no paths, environment
values or personal data.
