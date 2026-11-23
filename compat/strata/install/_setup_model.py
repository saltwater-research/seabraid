# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Get the model, by asking Strata's own setup for it -- and then find what it
produced.

OWNER: this file owns the Strata-setup invocation and the layout rules for a
Strata data directory.  It defines no type that config.py already owns; it
returns a ModelPaths.

WHY SUBPROCESS AND NOT IMPORT.  Strata's setup.py is 11,018 lines, does work at
module level, and is fetched at build time, so importing it would couple us to
a moving target and run its startup for free.  We shell out to it for the model
download, the pack and the tokenizer -- the part worth reusing, and the part we
must not reimplement -- and do nothing else with it.

The discovery functions take a directory listing rather than touching the disk,
so the layout rules are tested against a recorded tree with no model present.
"""

from __future__ import annotations

import dataclasses
import os
import re
import struct
import subprocess
from typing import Callable, Optional, Sequence

from .config import ModelPaths


class ModelSetupError(Exception):
    """Strata's setup could not be run, or produced nothing we can find."""


# One filename family and a complete 1..N sequence are required: counting only the first and last
# silently under-sizes multi-shard models. Strata itself discovers PLE by its tensor, not its position.
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)

def shard_order(names: Sequence[str]) -> list:
    """GGUF shards in index order; non-shards and other files dropped."""
    found = []
    for n in names:
        m = _SHARD.search(n)
        if m:
            found.append((int(m.group(1)), n))
    return [n for _i, n in sorted(found)]


def pick_draft_pack(names: Sequence[str]) -> Optional[str]:
    """The drafter pack to use, or None.

    A ``.pack`` with no ``.pack.ok`` beside it is an unfinished download, and
    pointing the engine at one fails at load with a corrupt-file error that
    reads nothing like "this never finished".  The selective-quant build is
    preferred when both are complete, because it is the one the measured runs
    used.
    """
    complete = {n[: -len(".ok")] for n in names if n.endswith(".pack.ok")}
    packs = [n for n in names if n.endswith(".pack") and n in complete]
    if not packs:
        return None
    for p in sorted(packs):
        if "selq4" in p:
            return p
    return sorted(packs)[0]


def run_strata_setup(
    strata_dir: str,
    args: Sequence[str],
    python: str = "python3",
    echo: Callable[[str], None] = print,
    run=subprocess.run,
) -> None:
    """Invoke Strata's setup.py for the model only.

    ``--no-start`` is always appended: this installer starts the engine itself,
    through systemd units with the helper ordered first, and letting Strata
    start one too would leave two servers contending for the same card.
    """
    setup = os.path.join(strata_dir, "setup.py")
    if not os.path.exists(setup):
        raise ModelSetupError("no setup.py in %s -- fetch Strata first" % strata_dir)
    cmd = [python, setup, *args]
    if "--no-start" not in cmd:
        cmd.append("--no-start")
    echo("  running Strata setup: %s" % " ".join(cmd))
    result = run(cmd, cwd=strata_dir, check=False)
    if getattr(result, "returncode", 1) != 0:
        raise ModelSetupError(
            "Strata setup exited %s. Its output is above; this installer has not "
            "changed anything." % getattr(result, "returncode", "?")
        )


def discover(
    data_dir: str,
    pack_name: str,
    exe: str,
    cwd: str,
    log: str,
    model_name: Optional[str] = None,
    listdir: Callable[[str], list] = os.listdir,
    isdir: Callable[[str], bool] = os.path.isdir,
    isfile: Callable[[str], bool] = os.path.isfile,
) -> ModelPaths:
    """Find what Strata's setup produced, under a Strata data directory.

    Layout, as the reference install has it:
        <data>/packs/<pack>/        tokenizer, dense.bin, index.txt, ...
        <data>/models/<VARIANT>/    the GGUF shards
        <data>/dflash2/             the drafter pack and draft_vocab.bin

    Nothing here is guessed: a path that is not present comes back None and the
    caller decides whether it can proceed without it.
    """
    j = os.path.join
    pack = j(data_dir, "packs", pack_name)
    if not isdir(pack):
        raise ModelSetupError(
            "no pack at %s. Strata's setup makes this; run the model step first." % pack
        )
    tokenizer = j(pack, "tokenizer")
    if not isdir(tokenizer):
        raise ModelSetupError("the pack at %s has no tokenizer directory" % pack)

    models_root = j(data_dir, "models")
    variant = _pick_variant(models_root, pack_name, listdir, isdir)
    shards = _inventory(j(models_root, variant), listdir(j(models_root, variant)), isfile)
    native = shards[0]
    # With --native the pinned Strata engine finds the shard containing per_layer_token_embd.weight.
    ple = None

    draft_model = draft_vocab = None
    dflash = j(data_dir, "dflash2")
    if isdir(dflash):
        names = listdir(dflash)
        pick = pick_draft_pack(names)
        if pick:
            draft_model = j(dflash, pick)
        if "draft_vocab.bin" in names:
            draft_vocab = j(dflash, "draft_vocab.bin")

    # Strata selects a distinct profile for Coder; a fresh checkout may not exist until build completes.
    profile = _profile_path(model_name or pack_name, cwd)

    return ModelPaths(
        exe=os.path.abspath(exe),
        cwd=os.path.abspath(cwd),
        pack=pack,
        native=native,
        tokenizer=tokenizer,
        model_name=model_name or pack_name,
        log=log,
        ple_gguf=ple,
        draft_model=draft_model,
        draft_vocab=draft_vocab,
        expert_profile=os.path.abspath(profile) if isfile(profile) else None,
    )


def _complete_shards(ordered: list[str]) -> bool:
    """Validate names as one 1..N set, rather than counting unrelated GGUFs."""
    matches = [_SHARD.search(n) for n in ordered]
    first = matches[0]
    prefix = ordered[0][:first.start()].lower()
    count = int(first.group(2))
    if count == 0 or len(ordered) != count:
        return False
    return all(int(m.group(1)) == i and int(m.group(2)) == count and n[:m.start()].lower() == prefix
               for i, (n, m) in enumerate(zip(ordered, matches), 1))


def _ordered_model_files(folder: str, names: list[str]) -> list[str]:
    """Unsharded model or a complete single-family set, never a partial size estimate."""
    if len(names) == 1 and not _SHARD.search(names[0]):
        return names
    ordered = shard_order(names)
    if not ordered or len(ordered) != len(names) or not _complete_shards(ordered):
        raise ModelSetupError("incomplete or mixed GGUF shard set under %s" % folder)
    return ordered


def _inventory(folder: str, names: Sequence[str], isfile: Callable[[str], bool]) -> list[str]:
    """One complete GGUF set; refusing mixed families prevents incorrect model sizing."""
    ggufs = sorted(n for n in names if n.lower().endswith(".gguf"))
    paths = [os.path.join(folder, n) for n in _ordered_model_files(folder, ggufs)]
    if not all(isfile(p) for p in paths):
        raise ModelSetupError("GGUF file is missing under %s" % folder)
    return paths

def model_shards(native: str) -> list[str]:
    """Reconstruct all shards from the first native filename for the planner."""
    folder = os.path.dirname(native)
    try:
        paths = _inventory(folder, os.listdir(folder), os.path.isfile)
    except OSError as exc:
        raise ModelSetupError("cannot inventory GGUF files under %s: %s" % (folder, exc)) from exc
    if os.path.normcase(paths[0]) != os.path.normcase(native):
        raise ModelSetupError("native path is not the first shard: %s" % native)
    return paths


def _profile_path(model_name: str, work: str) -> str:
    # Strata setup_config.py selects fam["profile"]; its coder family alone overrides the default.
    filename = "expert-profile-coder.bin" if "coder" in model_name.lower().split("-") else "expert-profile.bin"
    return os.path.abspath(os.path.join(work, "strata", "data", filename))


def _check_profile(path: str, layers: int, experts: int) -> None:
    """Check the pinned engine's STRP header and extent before it reaches a service."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(24)
            size = os.fstat(fh.fileno()).st_size
    except OSError as exc:
        raise ModelSetupError("cannot read expert profile %s: %s" % (path, exc)) from exc
    if len(header) != 24:
        raise ModelSetupError("expert profile %s has a truncated header" % path)
    magic, version, nl, ne, slots, ranked = struct.unpack("<4s5I", header)
    if magic != b"STRP" or version != 1 or (nl, ne) != (layers, experts):
        raise ModelSetupError("expert profile %s has the wrong format or model dimensions (want %dx%d)" % (path, layers, experts))
    if ranked > slots or size < 24 + 4 * ranked:
        raise ModelSetupError("expert profile %s has a truncated or invalid ranking" % path)


def resolve_profile(paths: ModelPaths, work: str, *, require_exists: bool = True) -> ModelPaths:
    """Re-resolve after the Strata checkout is built; dry runs render the path without inventing a file."""
    path = _profile_path(paths.model_name, work)
    if require_exists:
        from ._plan_model import ModelError, read_model_facts
        try:
            facts = read_model_facts(model_shards(paths.native))
            _check_profile(path, facts.layers, facts.experts)
        except ModelSetupError:
            raise
        except (OSError, ValueError, ModelError) as exc:
            raise ModelSetupError("cannot verify model or expert profile: %s" % exc) from exc
    return dataclasses.replace(paths, expert_profile=path)

def _pick_variant(models_root: str, pack_name: str, listdir, isdir) -> str:
    """The models/ subdirectory matching this pack.

    Matched case-insensitively with separators removed, because the pack is
    named one way and the model directory another for the same thing.  An
    ambiguous match refuses rather than picking: the wrong one is a different
    quantisation, which would load and be quietly worse.
    """
    if not isdir(models_root):
        raise ModelSetupError("no models directory at %s" % models_root)
    want = re.sub(r"[^a-z0-9]", "", pack_name.lower())
    names = [n for n in listdir(models_root) if isdir(os.path.join(models_root, n))]
    hits = [n for n in names if re.sub(r"[^a-z0-9]", "", n.lower()) == want]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ModelSetupError(
            "no model directory under %s matches the pack %r (saw: %s)"
            % (models_root, pack_name, ", ".join(sorted(names)) or "nothing")
        )
    raise ModelSetupError(
        "the pack %r matches several model directories (%s); pass --model-dir"
        % (pack_name, ", ".join(sorted(hits)))
    )


def list_packs(data_dir: str, listdir=os.listdir, isdir=os.path.isdir) -> list:
    """Every pack already installed, so the CLI can offer them."""
    root = os.path.join(data_dir, "packs")
    if not isdir(root):
        return []
    return sorted(n for n in listdir(root) if isdir(os.path.join(root, n)))
