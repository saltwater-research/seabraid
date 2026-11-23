#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid one-click setup: detect this machine, plan an install, build it, run it.

Run it either way:
    python3 install/seabraid_setup.py
    python3 -m install.seabraid_setup

OWNER     : this file owns the command line, the interactive flow, and the order
            the stages run in.  It decides NOTHING a stage owns -- it does not
            rank cards, size tiers or choose knob values; it asks the stage that
            owns each and prints what came back.
CONTRACT  : main(argv) -> exit status.  0 done, 1 blocked by the machine,
            2 a probe could not describe the host, 3 a stage refused,
            130 the operator declined.
FAILURE   : fail-fast and LOUD.  Every refusal names the flag or the command
            that clears it.  Nothing on the host changes before the plan has
            been shown and accepted, and --yes answers the questions without
            suppressing anything printed above them.
            Import direction: detect <- plan <- config <- build/services <- here.

Linux first; Windows is out of scope and says so rather than half-working.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

MIN_PYTHON = (3, 10)        # config.py uses dataclass(slots=True), which does not exist before 3.10


def check_python(version=None, minimum=MIN_PYTHON):
    """2 (message on stderr) when ``version`` is older than ``minimum``, else None.  WHY first: on an old Python the later
    imports die with a TypeError naming a dataclass, not the fix."""
    version = tuple(sys.version_info[:3]) if version is None else tuple(version)
    if version >= minimum:
        return None
    msg = "seabraid_setup: Python %d.%d or newer is required, this is %s. Run it with python3.%d (or newer)." % (
        minimum[0], minimum[1], ".".join(str(v) for v in version[:3]), minimum[1])
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__" and check_python():
    raise SystemExit(2)

# The stages import each other relatively, which needs a package, and the user
# will type the direct form -- so the package is established here, in the single
# entry point, rather than by asking anyone to remember -m.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "install"

from ._exposure import ExposureError, cleartext_warning, require_key_for_host  # noqa: E402
from ._log import configure as configure_logging, scrub  # noqa: E402
from ._setup_stages import lead_stage, finish, print_cleartext as _print_cleartext  # noqa: E402,F401
from ._setup_support import PromptAborted, Prebuilt, Refused, ask_choice, ask_yes, engine_for, find_prebuilt  # noqa: E402,F401
from .detect import DetectionRefused, detect_host, format_report  # noqa: E402

_log = logging.getLogger(__name__)

DEFAULT_DATA = "~/.local/share/strata/Strata-data"
DEFAULT_WORK = "~/.cache/seabraid"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="seabraid_setup.py",
        description="Run one MoE model across GPUs of different vendors.",
        epilog="With no options: detect the machine, propose a plan, and ask before changing anything.",
    )
    stage = ap.add_argument_group("how far to go")
    stage.add_argument("--detect-only", action="store_true", help="print what this machine is, and stop")
    stage.add_argument("--plan-only", action="store_true", help="also propose a plan, change nothing, and stop")
    stage.add_argument("--no-build", action="store_true", help="use binaries a previous run already built")
    stage.add_argument("--no-start", action="store_true", help="write the services but do not start them")
    stage.add_argument("--bench", action="store_true",
                       help="measure each card to rank main vs helpers (briefly LOADS every card; default off: ranks by the VRAM/bandwidth heuristic)")
    stage.add_argument("--bench-lead", action="store_true",
                       help="rank card 1 by MEASURED lead-job ms/window: starts the whole engine once per candidate card (top 3), ~70-100 s each with a big model; "
                            "separate from --bench because it is minutes, not seconds, and needs the engine built (default off)")
    stage.add_argument("--dry-run", action="store_true", help="say what would be written, write nothing")

    what = ap.add_argument_group("what to install")
    what.add_argument("--data-dir", default=DEFAULT_DATA, help="Strata data directory (default: %(default)s)")
    what.add_argument("--pack", metavar="NAME", help="which installed pack to run; asked when there are several")
    what.add_argument("--work-dir", default=DEFAULT_WORK, help="where Strata is fetched and built (default: %(default)s)")
    what.add_argument("--port", type=int, default=8080, help="server port (default: %(default)s)")
    what.add_argument("--host", default="127.0.0.1",
                      help="bind address (default: %(default)s, this PC only). Anything else needs --api-key-file")
    what.add_argument("--api-key-file", metavar="PATH",
                      help="file holding the API key clients must send (one line, 16+ characters, mode 600, owned by you); "
                           "required when --host is not this PC. The key is read by the service at start-up, never written into a unit")
    what.add_argument("--max-context", type=int, metavar="N", help="override the context the plan sized VRAM for")
    what.add_argument("--streams", type=int, default=1, metavar="N",
                      help="concurrent streams the plan is for: picks which lead-job measurement ranks card 1 (1, or 4 and up) (default: %(default)s)")
    what.add_argument("--main-card", metavar="VENDOR:INDEX", help="force this card (e.g. nvidia:0) to be card 1, whatever the ranking says")
    what.add_argument("--admission", action="store_true",
                      help="drop cards after card 1 that would hold no experts that otherwise sit in RAM/NVMe (rule R3a) or make a layer slower (R3b)")
    what.add_argument("--ladder", action="store_true",
                      help="turn on the expert pyramid's live moves between every card (--retier on --retier-promote seabraid); off by default "
                           "because live placement has not yet beaten static placement in a measurement")
    what.add_argument("--file-moves", action="store_true",
                      help="with --ladder: let a helper load an expert straight from the model file instead of main copying its bytes "
                           "(--retier-file-moves 1, helper --file-io auto); builds <pack>/expert-sums.v1 with seabraid-sums first. "
                           "Off by default: its speed is unmeasured")
    what.add_argument("--jobs", type=int, metavar="N", help="parallel build jobs (default: every core)")

    where = ap.add_argument_group("things the search cannot guess")
    where.add_argument("--rocm-path", metavar="DIR", help="a ROCm root containing bin/hipcc")
    where.add_argument("--cuda-path", metavar="DIR", help="a CUDA root containing bin/nvcc")

    ap.add_argument("--allow-root", action="store_true",
                    help="run as root anyway (not recommended: it clones and builds third-party code as root)")
    ap.add_argument("--yes", action="store_true", help="accept the recommended answer at every prompt")
    ap.add_argument("-v", "--verbose", action="store_true", help="log the installer's diagnostics (probe failures, refusals) to stderr")
    ap.add_argument("--strict", action="store_true", help="with --detect-only: exit non-zero if anything blocks")
    return ap


def _fail(message: str, status: int) -> int:
    print("\nseabraid_setup: %s" % message, file=sys.stderr)
    # INFO, not ERROR: the console already shows the full message above; -v adds this scrubbed line to the diagnostic trail.
    _log.info("exit %d: %s", status, scrub(message))
    return status


# ----------------------------------------------------------------------- flow


def refuse_root(allow_root: bool, geteuid=getattr(os, "geteuid", None)):
    """2 (after saying why) when running as root without --allow-root, else None.  WHY: it builds third-party code and writes
    user units; under sudo that code runs as root.  Windows has no geteuid (Linux-only installer), so the check is skipped there."""
    if geteuid is None or geteuid() != 0:
        return None
    if allow_root:
        _log.warning("running as root because --allow-root was passed: third-party code will be built as root")
        return None
    return _fail("running as root (sudo?) would build third-party code as root and write the services into root's home. "
                 "Run it as your normal user, or pass --allow-root if you really mean it.", 2)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # The lead probe writes files and launches engines, which violates plan-only's no-change contract.
    if args.plan_only and args.bench_lead:
        return _fail("--plan-only cannot be combined with --bench-lead (the probe writes files and starts engines).", 2)
    configure_logging(args.verbose)          # once, here: every other module only asks logging for its own logger
    refused = refuse_root(args.allow_root)
    if refused is not None:
        return refused
    note = cleartext_warning(args.host)
    if note:                                 # exposed bind: the key travels in clear (see _exposure.CLEARTEXT_NOTE)
        _log.warning("--host %s: %s", args.host, note)
    try:                                     # before anything is probed or built: an exposed server needs its key now
        require_key_for_host(args.host, args.api_key_file)
    except ExposureError as exc:
        return _fail(str(exc), 2)
    try:
        return _run(args)
    except PromptAborted as exc:             # one place for "a question had no answer": status 130, nothing changed
        return _fail(str(exc), 130)


def _fail_refused(exc: Refused) -> int:
    return _fail(exc.message, exc.status)


def _parse_main_card(spec):
    """(vendor, index) from --main-card VENDOR:INDEX, or None when it is not given."""
    if not spec:
        return None
    vendor, _, idx = spec.partition(":")
    if not idx.isdigit() or vendor not in ("nvidia", "amd"):
        raise Refused("--main-card wants VENDOR:INDEX with vendor nvidia or amd, e.g. nvidia:0 (got %r)" % spec, 2)
    return (vendor, int(idx))


def _choose_model(args, data_dir: str, work: str):
    """(pack, paths, model facts) for the pack the operator picked; refuses with status 3 when the model cannot be read."""
    # Imported here, not at module scope, so --detect-only keeps working on a
    # tree where the later stages have not landed.
    from . import _setup_model as M  # noqa: PLC0415
    from . import plan as P  # noqa: PLC0415

    packs = M.list_packs(data_dir)
    if not packs:
        raise Refused(
            "no model packs under %s. Install one with Strata's own setup first:\n"
            "    python3 <strata>/setup.py --no-start\n"
            "then run this again. seabraid does not reimplement the download." % data_dir, 3)
    pack = args.pack or ask_choice("Which model?", packs, args.yes)
    if pack not in packs:
        raise Refused("no pack named %r under %s (have: %s)" % (pack, data_dir, ", ".join(packs)), 3)
    try:
        paths = M.discover(data_dir, pack, exe="", cwd=work, log=os.path.join(work, "%s.log" % pack))
        model = P.read_model_facts(M.model_shards(paths.native))
        if paths.expert_profile:
            paths = M.resolve_profile(paths, work)
    except (M.ModelSetupError, P.ModelError) as exc:
        raise Refused(str(exc), 3) from exc
    if not paths.expert_profile and (args.no_build or args.bench_lead):
        # Fail at plan time.  A fresh build fetches Strata (and its profile) later, so only the stages that need the checkout
        # to exist already (--no-build, the lead probe) can judge it now.
        from .config import EXPERT_PROFILE_MISSING  # noqa: PLC0415
        raise Refused(EXPERT_PROFILE_MISSING, 3)
    return pack, paths, model


def _measure(args, report, model, paths, work: str):
    """The card measurements: none unless asked (--bench loads every card), plus the lead-job probe under --bench-lead."""
    # Measure, then plan.  WHY here: only this stage knows both the detected cards and the model's files; the measurement
    # owner (_bench) and the ranking owner (plan) are asked, not second-guessed.  The helper must already be built.
    from . import _bench as BN  # noqa: PLC0415
    from . import build as B  # noqa: PLC0415
    from . import plan as P  # noqa: PLC0415

    # Opt-in: measuring loads every card, and no card is loaded unless the user asked for it.
    bench = BN.measure(
        report, P.candidates(report), find_prebuilt(work).helper_exes, paths.pack, paths.native, model.layers, model.experts,
        B.Runner(dry_run=args.dry_run)) if args.bench else BN.skipped(BN.NOT_OPTED_IN)
    if args.bench_lead:
        bench = lead_stage(args, report, model, paths, work, bench, find_prebuilt(work))
    return bench


def _plan_stage(args, report, model, paths, work: str):
    """Measure, then plan, and show the plan."""
    from . import plan as P  # noqa: PLC0415

    main_card = _parse_main_card(args.main_card)
    bench = _measure(args, report, model, paths, work)
    plan = P.make_plan(report, model, os.path.dirname(paths.native), bench=bench.rates, bench_why=bench.lead_why or bench.why,
                       opts=P.PlanOptions(streams=args.streams, main_card=main_card, admission=args.admission, assumed_max_context=args.max_context),
                       bench_lead=bench.lead_for(args.streams))
    print("\n" + plan.explain() + "\n")
    return plan


def _detect(args):
    """The host report, printed before anything is judged: a refusal that hides the readout cannot tell anyone what to fix."""
    try:
        report = detect_host(rocm_path=args.rocm_path, cuda_path=args.cuda_path)
    except DetectionRefused as exc:
        raise Refused(str(exc), 2) from exc
    print(format_report(report))
    return report


def _global_blockers(report):
    """The planner handles card-specific exclusions; an unscoped blocker still stops the whole plan."""
    return [b for b in report.blockers if b.kind not in ("gpu", "toolkit") or not b.ref]

def _run_stages(args) -> int:
    report = _detect(args)
    if args.detect_only:
        return 1 if (args.strict and report.blockers) else 0
    # Device-local blockers are passed to the planner to exclude only affected cards.
    blockers = _global_blockers(report)
    if blockers:
        raise Refused("%d global blocker(s) above stop a plan being made. Each one names its fix." % len(blockers), 1)
    data_dir = os.path.expanduser(args.data_dir)
    work = os.path.expanduser(args.work_dir)
    pack, paths, model = _choose_model(args, data_dir, work)
    plan = _plan_stage(args, report, model, paths, work)
    if args.plan_only:
        return 0 if plan.ready else 1
    if not plan.ready:
        raise Refused("the plan above is not ready, so nothing was changed.", 1)
    if not ask_yes("Build and install this?", args.yes):
        raise Refused("nothing was changed.", 130)
    return _finish(args, report, plan, paths, pack, work)


def _run(args) -> int:
    """The flow in order; every stage refuses by raising Refused, and this is the one place that turns it into a status."""
    try:
        return _run_stages(args)
    except Refused as exc:
        return _fail_refused(exc)


def _finish(args, report, plan, paths, pack: str, work: str) -> int:
    """Build, configure, install (see _setup_stages.finish); a refusal becomes its exit status here."""
    try:
        return finish(args, report, plan, paths, pack, work)
    except Refused as exc:
        return _fail_refused(exc)


if __name__ == "__main__":
    raise SystemExit(main())
