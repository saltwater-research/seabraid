# Releasing

How to publish a version of this project to GitHub, PyPI and crates.io, and what to do after. The script is
`scripts/release/publish.py`; this guide is the human half. Everything outward (making a repository public, uploading
to a registry, creating accounts, submitting a paper) is **your** action: the script does it only when you run it with
`--publish`, and then only after you type a confirmation for each step.

The previous alpha.7 working snapshot has private installer, mixed-GPU and package rehearsal
evidence. For the release candidate, use a fresh private work directory, identify the annotated
export tag and its commit, and verify the packages built from that export. Record the
mixed-GPU smoke against that same revision. The [status report](STATUS.md#private-alpha7-rehearsal)
describes how the snapshot runs were performed.

## Release in one command

For the owner. The final name is Seabraid (decided 2026-11-21). Everything below runs on the Linux build host (the desktop has no MSVC and
Git's `link.exe` shadows the linker, so a real run refuses there; dry runs work anywhere and print the plan).

1. **Rename**: already done. The tree carries the final name Seabraid; `scripts/rename.py --from OLD --to NAME` remains for a future rename.
2. **Verify** the commit (the test machine and a GPU run, recorded in `docs/RESEARCH-LOG.md`). The coordinator writes the
   verified full 40-character SHA as the first word of a line in `~/.release/VERIFIED` (override with `--verified-file` or
   `RELEASE_VERIFIED_FILE`). A real run refuses when HEAD is not that commit.
3. **Preflight** (dry run): `python scripts/release/publish.py all --name seabraid --github-owner saltwater-research`. On Linux it runs every
   stage for real except the uploads and prints the exact commands. Read the `[WARN]` lines: each says what a real run would
   refuse. Record and attach the official crates.io proof using the instructions below before the first publication.
4. **Publish**: `python scripts/release/publish.py all --name seabraid --github-owner saltwater-research --publish`, as the first
   public action once every "Ready to place" item is green, including the mixed-GPU smoke on the exact final sha.
   **Owner decision (2026-11-21): "Wait until package is ready": no `register` run and no empty 0.0.0 placeholders.** The names
   are secured by this real alpha publish (`all` checks that each name is free before uploading; crates.io forbids name
   squatting and PyPI can reclaim names under PEP 541). Note: `publish.py register` exists in the tool for a future name and
   is a local-only preview. Live name registration requires a separate sealed-artifact workflow.

What the prompts ask. Each outward target prints its exact commands and effect, then waits for you to type the target and
the version exactly, for example `pypi 0.5.0-alpha.8`; anything else (or end of input) stops the run before that target.
The order is fixed: `repo` (public GitHub repository, one commit, tag), `release` (wheels, binding sdist, full tagged source, tested crate, `SHA256SUMS`, notes),
`pypi`, `crates`. Before the first prompt a real run checks that the network answers, that HEAD is the verified commit, that
the name is final, that the dev remote is never the public target, that the public repository does not exist yet
(a fresh one has no history), and that the PyPI and crates.io names are free or already this tool's own (a name that exists at any
version and was not published by this tool refuses before anything is public). A dry run asks no registry and stays offline-capable. For a real run, including TestPyPI, read the Linux, GPU, claims and final-name checklist and type `reviewed <full verified SHA>`. The saved sign-off covers that revision and that artifact receipt.

Credentials come only from the environment or the tools' own logins (`gh auth login`, `~/.pypirc` or `TWINE_*`,
a release-scoped `CARGO_REGISTRY_TOKEN` for the checksum-bound Cargo provider). Credentials stay out of command arguments, logs and receipts.

What has no command. The tree keeps network code out on purpose (`checks/check_no_egress.py`), so these are listed at the
end of the run and in `scripts/release/post_publish.md`, for you to do in a browser: the Zenodo DOI (switch the repository
on before the release exists, or upload the full tagged source archive, the GitHub release's source tarball of the tag or `git archive` of the tag, as a Software record with `.zenodo.json`; the sdist is Python binding source only and would be an incomplete record), the Software Heritage and
Wayback Machine requests, the GitHub settings, and arXiv. npm and conda are not targets.

Resume using the same work directory. The publisher checkpoints repository creation, the verified main branch push, and the annotated tag push separately. On retry it checks the fetch and push addresses, branch commit and tag object on the remote. A conflicting or missing recorded ref requires human review. If the process lost the result of repository creation before its receipt reached disk, establish ownership separately before proceeding.

A Cargo probe runs outside any project with a fresh registry cache. An unreachable index or ambiguous output leaves availability unknown and stops publication. An existing crate version without this release's receipt also stops publication. Private preparation and exact package tests run offline.

### Record the official crates.io dry run

After Linux preparation, use a clean checkout of its **annotated export tag** on a trusted online host. Keep the publisher and proof file outside that checkout. Use a new output filename in an existing directory. The recorder runs Cargo against the official index in a fresh Cargo home and target, checks its successful stop-before-upload diagnostic, and hashes the archive Cargo generated.

```text
python -B scripts/release/publish.py crates-io-proof --export <checkout> --tag <tag> --output <proof.json>
```

On a desktop without the build toolchain, add `--tested-crate <path-to-Linux-dist-crate>`. This runs Cargo with `--no-verify`, compares its generated bytes to the supplied crate, and records the skipped build. Linux must still package, unpack and test the exact crate. An offline or loopback rehearsal cannot satisfy the official registry gate.

Bring the proof back through a trusted private channel. The JSON is an operator record and carries no registry signature. To attach it without uploading or rebuilding, run the following command against the existing prepared work directory, using the same name, owner and artifact options as the build.

```text
python scripts/release/publish.py --name <name> --github-owner <owner> --work <work> --only publish --crates-io-proof <proof.json>
```

The flat and `all` commands also accept `--crates-io-proof` during preparation or when consuming an existing build. The publisher verifies the export commit, tag name and object, crate filename and SHA-256 against the build receipt and unpacked-crate test digest. A resumed publish rechecks the saved proof. Running a fresh export creates a new tag object. If a changed publisher is included in a new export, prepare that export and record a fresh proof. The earlier tag and its proof can still be inspected with a separately staged publisher.

## Ready to place (all must be green before the owner gets the commands)

The public repository is https://github.com/saltwater-research/seabraid: a fresh repository, one squashed commit. **The owner
creates it and pushes; agents never do.** It is not placed until it is completely ready and no other change needs to occur.
`--name` is the product name (seabraid) and `--github-owner` the account that owns the public repository (default
`saltwater-research`).

a. Final name gate: `python checks/check_names.py --forbid RETIRED` passes for each retired name listed in `rel_guard.PROVISIONAL` (the temporary project name and the old add-on name, spelled in halves there), and `rel_guard` refuses each as `--name`.
b. Full box verification of the exact head, with its sha in `~/.release/VERIFIED` (`rel_guard.verified_problems`).
c. Mixed-GPU smoke on the exact final sha: R9700 main + RTX 3090 helper, one request per card order passes the decode-path KL gate, and one mid-request helper kill that the engine survives (run by the coordinator on the test machine; recorded in docs/RESEARCH-LOG.md).
d. On Linux x86-64, run the full root pytest suite (`python -m pytest -q tests`), the `compat/strata` pytest suite, and `python checks/run_all.py` (16 checkers).
e. Secret and identity scan of the export: `rel_secrets.scan_tree` plus the export stage's own scan.
f. CHANGELOG version heading and CITATION.cff `date-released` set to the tag date (preflight gate `CITATION.cff date-released is today`).
g. Metadata URLs filled in the export, no `@REPO@` left (export stage check).
h. Docs and preprint state only verified results (`docs/STATUS.md`).
i. `python scripts/release/publish.py all --name seabraid --github-owner saltwater-research` dry run on the owner's Linux host with 0 FAIL.
j. A matching official crates.io dry-run proof and Linux `twine check`.
k. The private repository pushed at that exact sha.
l. The export frozen: no commits planned after it; if anything changes, the whole list is re-run.

## Redeploying an older install

The add-on's names changed on 2026-11-21 (the Strata integration is `compat/strata` now), so an install made before that date does
not match what the installer writes today.  Run the installer again and stop the units it replaces by the names the old install
printed.  The new names: units `seabraid-server.service` and `seabraid-helper-N.service` (one per helper), binaries `seabraid-helper`
and `seabraid-sums`, shared-memory channels `/seabraid0` and `/seabraid1`, engine and helper switches `SEABRAID_STRATA_*` (for
example `SEABRAID_STRATA_SHM`, `SEABRAID_STRATA_HOT_SHARE`, `SEABRAID_STRATA_EXCLUDE_HELPER_RAM`), log prefix `seabraid:`.  The
helper wire is v11 (the magic text is `SEABRD1`): an engine and a helper built before it refuse each other, so rebuild both from
the same patch series.  A unit file written by an older installer is never edited in place and never deleted by this repository.

## The rules the script enforces

- **The final product name is Seabraid (decided 2026-11-21); the retired temporary name never ships.** You pass the name
  with `--name`; the script renames the export with `scripts/rename.py`, then runs `checks/check_names.py` with the
  old name forbidden, and refuses to continue if the old name is still anywhere (a binary file that holds it, for
  example). If `--name` is the tree's own name (Seabraid), no rename happens; the export is still checked that the retired name is gone.
- **The private dev repository keeps its full history.** The first public version goes to a fresh repository: one
  commit of the scrubbed tree (hardware models, tests and benchmarks stay; machine paths, addresses, tokens and personal
  addresses go), authored and committed by whoever runs the release, under their own git identity (`git config user.name` and `user.email` of the dev repository; owner decision, 2026-11-24: each member or associate is responsible for their own releases). The email must be a GitHub noreply address (`<id>+<login>@users.noreply.github.com`, see github.com/settings/emails) or the project contact, because a public commit's email is permanent; the script refuses anything else. Also turn on GitHub's "Keep my email addresses private" so web merges do not expose a personal address. No trailers, UTC timestamps, and the tag `v<VERSION>`.
- **A dry run is the default.** Without `--publish` every stage runs for real up to the uploads (checks, export,
  builds, `twine check`, `cargo package`, `cargo publish --dry-run`), then the publish stage only prints the commands it
  would run and stops.
- **No credential is ever stored or printed.** The tools use your existing logins. The script reports only whether one
  seems to exist, and masks token shapes and the values of `*TOKEN*` / `*PASSWORD*` variables in everything it prints.

## Accounts to create (once, by you)

| Account | Why | Notes |
|---|---|---|
| GitHub (user or organisation) | the public repository and the release | the owner you pass as `--github-owner`; log in the CLI with `gh auth login` |
| PyPI | the Python package | turn on two-factor authentication before anything else; PyPI requires it for uploads |
| TestPyPI | the rehearsal upload | a **separate** account and separate token from PyPI |
| crates.io | the Rust crate | sign in with GitHub; verify the email address on the account, or publishing is refused |
| Zenodo | a DOI per release | sign in with GitHub and switch the repository on (see below) |
| arXiv | the paper | needs an endorsement for `cs.DC` if you have not submitted there before (see below) |

Software Heritage and the Wayback Machine need no account for the "save" actions in the checklist.

## Tokens and where they live

Create each token with the narrowest scope the site offers. Never paste a token into chat, a commit, an issue or a shell
history line you cannot clear.

- **PyPI:** account settings, API tokens. The first upload of a new project needs an account-wide token; once the
  project exists, replace it with a token scoped to that project (or use trusted publishing from CI later). Put it in
  `~/.pypirc` (permissions `600`):
  ```
  [pypi]
  username = __token__
  password = <the token>

  [testpypi]
  username = __token__
  password = <the TestPyPI token>
  ```
  or set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD` in the shell for this one session. The script runs twine with
  `--non-interactive`, so a missing credential fails instead of prompting.
- **crates.io:** supply a narrowly scoped upload token in `CARGO_REGISTRY_TOKEN` through your local secret manager. The publisher starts Cargo with an empty credential store and authorizes the tested crate through its checksum-bound credential provider.
- **GitHub:** `gh auth login` (browser or device flow). `gh auth status` must succeed; the script checks it before the
  first step.

Revoke or re-scope each token after the release.

## Deciding the name

The PyPI and crates.io names and every uploaded version are permanent, so decide before the first upload.

1. Check it is free: `python -m pip index versions seabraid` (no such distribution means free), `cargo search NAME --limit 5`,
   and the GitHub owner/repository address. a dry run never contacts a registry, and
   `gh repo create` and the uploads fail loudly if a name is taken.
2. The name must be lower-case letters and digits, 3 to 21 characters (the rule `scripts/rename.py` applies).
3. Pass it as `--name NAME` on every run.

## Before you run it

- The dev branch (`main` unless you pass `--dev-branch`) is clean and committed.
- `VERSION`, `Cargo.toml`, `Cargo.lock` and the CMake version agree (`python checks/check_version.py`).
- `CHANGELOG.md` has a section for the version with its date (the script warns when it is missing or says "unreleased";
  the GitHub release notes are that section).
- The Linux test-machine verification and a GPU run of this version are recorded in `docs/RESEARCH-LOG.md`. The script cannot prove
  this. The real run records the human checklist against the verified full SHA and the sealed artifact receipt.
- Before tagging, qualify the candidate on Linux x86-64. Run the Linux suites on the local build host and record
  the revision, test environments, results and skips.
- Optional portability checks include `windows`, `macos` and `wheels-other`.
  Run them through `workflow_dispatch` to evaluate the experimental Windows core or investigate macOS portability.

## Run the dry run

```
python scripts/release/publish.py --name seabraid --github-owner saltwater-research
```

Stages, each printing `[PASS]`, `[FAIL]`, `[WARN]`, `[SKIP]` or `[MANUAL]`:

1. **preflight**: clean tree, branch, version agreement, final-name gate, package metadata, `python checks/run_all.py`,
   `pytest`, secret and identifier scan, the manual list.
2. **export**: `git archive HEAD` into `<work>/export`, rename, the same gates again on the export, fill the repository
   address, prove only that placeholder changed, then one commit and the tag.
3. **build**: build the platform wheel and binding sdist, run `twine check`, install each exact wheel in a fresh offline venv, then load the shipped native library and weave a batch. Build the locked crate, unpack and test its archive offline, and create the full source archive from the gated export commit. Seal all four distributions in `SHA256SUMS` with their release notes and the wheel/crate test digests. An offline Cargo publish rehearsal can also use a populated index.
4. **publish**: prints the four steps and the checklist; changes nothing.

`<work>` defaults to a temporary directory named after the name; pass `--work DIR` (outside the repository) to keep it.
Pick stages with `--only export,build` or `--skip preflight`; `<work>/state.json` records which stages passed for which
commit, and the publish stage refuses to run if the dev branch moved or the export does not match what was gated.
`--skip-gate local-checks|pytest|manual` exists for quick re-runs and is refused with `--publish`.
The build receipt records the checks skipped during preflight and export, even when those
stages ran in separate invocations. It binds the stage records to the source revision,
export commit and annotated tag as each becomes available. An interrupted stage must finish
successfully before it counts again. Older receipts without this history still support a
dry-run preview. Before a first or resumed publication, rerun the complete preparation with
all checks enabled if the records are missing, skipped or mismatched.

When reviewing the full source archive, retain `LICENSE`, `LICENSE-MIT` and `NOTICE`.
The runtime uses Apache-2.0. Public C headers offer Apache-2.0 or MIT, and the upstream
patches listed in [NOTICE](../NOTICE) carry MIT terms.

For this Linux-only alpha, prepare the Linux x86-64 wheel on the Linux build host. Check its platform tag and test
that exact wheel in a fresh environment before uploading it.

## Rehearse on TestPyPI, then publish

```
python scripts/release/publish.py --name seabraid --github-owner saltwater-research --publish --test-pypi
python -m venv /tmp/rehearsal && /tmp/rehearsal/bin/python -m pip install --index-url https://test.pypi.org/simple/ --no-deps seabraid==VERSION
python scripts/release/publish.py --name seabraid --github-owner saltwater-research --publish
```

Each `--publish` step prints exactly what it will run and what it cannot undo, then waits for you to type its token.
Anything else (or end of input) skips that step and stops the run; completed steps are remembered, so a rerun resumes.

| Step | You type | What happens |
|---|---|---|
| repo | `OWNER/NAME` | Create the repository, push the exact main commit, then push the annotated tag. Each command has its own checkpoint. |
| release | `v<VERSION>` | Attach the sealed wheels, binding sdist, full tagged source, tested crate and `SHA256SUMS`. Use the full tagged source for an archive deposit. |
| pypi | `pypi:NAME` | `twine upload` of the wheels (the sdist only with `--upload-sdist`) |
| crates | `crates:NAME` | Reproduce the locked crate, then run Cargo with an isolated registry configuration and a credential provider bound to the crate checksum. |
| testpypi (with `--test-pypi`) | `testpypi:NAME` | `twine upload --repository testpypi`; nothing else runs |

**Switch Zenodo on before the release step** (next section), or the tag gets no DOI. To stop after the repository, run with
`--skip-step release --skip-step pypi --skip-step crates`, do the Zenodo switch, then run again without the skips.

Cargo packages its input again during publication. Immediately before that command the publisher compares a fresh offline repack against the sealed crate. Cargo then presents the actual upload name, version, index and SHA-256 to its version-one credential provider. The provider supplies the owner's token only for the matching publish operation. The publisher records completion only when Cargo succeeds and the provider leaves a matching authorization witness. A mismatch, an unsupported protocol or missing witness stops the operation. Registry availability and acceptance still require an owner-operated online run.

WHY the sdist is not uploaded to PyPI by default: setuptools cannot build the Rust library from an sdist, so installing
it would give a package that looks like the engine and is not. It goes into the GitHub release as the readable source.

## What cannot be undone

- A public repository is public the moment it exists; deleting it later does not recall clones, forks or caches.
- A PyPI version and filename can be yanked but never removed or reused. Same for a crates.io version.
- A Zenodo record is permanent and its DOI is never reused; a published arXiv version stays.
- The first commit's author and date are in every clone.

## Zenodo

1. Sign in at zenodo.org with GitHub, open the GitHub settings page, and switch the new repository on. Zenodo only
   archives releases created **after** the switch.
2. Creator: use the organisation, not a person. `CITATION.cff` lists `Saltwater Research LLC` as an organisation
   (`name:`, no family or given name), which Zenodo reads for the record's creators. After the first archive, open the
   record's edit page and check the creator is typed **Organizational** with the name `Saltwater Research LLC`; if
   Zenodo made it a person, change it there (and fix `creators` in the `.zenodo.json` that is already in the repository, which
   lists `{"name": "Saltwater Research LLC"}`, in a pull request so the next releases are right). I could not verify Zenodo's
   current behaviour from here: look.
3. Copy the DOI into `CITATION.cff` and the README in a pull request.

## arXiv

arXiv shows the submitter's real name on the abstract page's submission history, whatever the paper's author line says:
there is no way to submit anonymously or as an organisation. The author line may name `Saltwater Research LLC`. A first
submission to `cs.DC` needs an endorsement from an established author in that category (request it from the submit
page or from the endorsement help on arxiv.org); do this days ahead, not on the day. Cite the Zenodo DOI and the tag.

## After publishing

The script prints the checklist (also in `scripts/release/post_publish.md`): private vulnerability reporting, branch
protection, Zenodo, Software Heritage "save code now", Wayback Machine snapshots, an install check, revoking tokens and
arXiv, each with the exact address for this repository.

## If something fails

The step's output follows its `[FAIL]` line with tokens masked. Fix the cause and rerun the same command: finished
steps are skipped. If the work directory is wrong, delete it and start again; the dev repository is never modified.
