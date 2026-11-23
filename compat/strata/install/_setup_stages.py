# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The two long stages of the setup flow: the lead-job probe, and build-configure-install.

OWNER     : this module owns what those two stages DO, step by step.  seabraid_setup.py owns when they run and how a refusal
            becomes an exit status; it passes in everything environmental (args, the prebuilt binaries), so a test that
            patches seabraid_setup's find_prebuilt still steers the probe.
CONTRACT  : lead_stage(...) -> BenchResult; finish(...) -> exit status.  Both raise Refused (never exit) to stop.
FAILURE   : fail-fast; a refusal names the flag or command that clears it.  Nothing is written before the plan was accepted.
            Import direction: _setup_support <- here <- seabraid_setup; the stage modules (build, config, services, plan)
            are imported inside the functions so --detect-only works on a tree where they have not landed.
"""

from __future__ import annotations

import copy
import dataclasses
import os

from ._exposure import cleartext_warning, read_key_file
from ._setup_support import Refused, ask_yes, engine_for, find_prebuilt


def key_args(args) -> tuple:
    """(render_units kwargs, check_health kwargs) for --api-key-file; both empty without it, so the default call is unchanged."""
    if not args.api_key_file:
        return {}, {}
    return {"api_key_file": os.path.abspath(args.api_key_file)}, {"api_key": read_key_file(args.api_key_file)}   # absolute: systemd needs it


def print_cleartext(host: str) -> None:
    """The summary's one-sentence TLS advice, only for an exposed bind."""
    note = cleartext_warning(host)
    if note:
        print("\n  !! %s" % note)


# ------------------------------------------------------------- lead-job probe


def _lead_skip_reason(args, klass, pre) -> str:
    """Why the probe cannot run ("" when it can), in the order the operator can fix them."""
    if klass is None:
        return "covers 1 stream or 4 and up; --streams %d has no measured class" % args.streams
    if args.dry_run:
        return "was not run (dry-run)"
    if not pre.engine or not pre.helper_exes:
        return "needs the engine and seabraid-helper built (first install: build, then rerun with --no-build --bench-lead)"
    return ""


def _best_first(report, streams: int) -> tuple:
    """The candidate cards, best-first by the heuristic, so the cap (MAX_CANDIDATES) drops the least likely lead cards."""
    from . import plan as P  # noqa: PLC0415
    from seabraid.ranking import rank  # noqa: PLC0415

    cands = P.candidates(report)
    by_key = {c.key: c for c in P.adapt_report(report)["cards"]}
    order, _ = rank([by_key[(c.vendor, c.index)] for c in cands], None, None, lambda c: True, streams, None)
    return tuple(next(c for c in cands if (c.vendor, c.index) == o.key) for o in order)


def _probe_libs(report, cands) -> dict:
    """{vendor: LD_LIBRARY_PATH} for each candidate vendor's toolkit ("" when the toolkit was not found)."""
    from . import plan as P  # noqa: PLC0415
    from ._build_rocm import rocm_lib_dirs  # noqa: PLC0415

    kits = P.adapt_report(report)["kits"]
    return {v: ":".join(rocm_lib_dirs(tuple(kits[v].lib_dirs), os.path.isdir)) if kits.get(v) is not None else ""
            for v in {c.vendor for c in cands}}


def _probe_config(args, report, plan, paths, work: str, probe_dir: str, port: int, card, log: str):
    """The engine config with `card` first, written under probe_dir; refused as PlacementError so one bad card skips, not aborts."""
    from . import _bench_lead as BL  # noqa: PLC0415
    from . import config as C  # noqa: PLC0415

    engine = engine_for(work, card.vendor)
    if not engine:
        raise BL.PlacementError("no %s engine built under %s/strata (build-%s*/strata); not starting another backend's binary"
                                % (card.vendor, work, card.vendor))
    try:
        cfg = C.make_config(report, plan, dataclasses.replace(paths, exe=engine, log=log), C.ConfigOptions(port=port, max_context=args.max_context))
        return C.write_config(cfg, probe_dir, "lead-%s%d.json" % (card.vendor, card.index))
    except C.ConfigRefused as exc:
        raise BL.PlacementError("the plan with this card first is not usable: %s" % str(exc).splitlines()[0]) from exc


def _placement_for(args, report, model, paths, work: str, probe_dir: str, port: int):
    """The callable that turns a candidate card into a BL.Placement (the plan and config with that card first)."""
    from . import _bench_lead as BL  # noqa: PLC0415
    from . import plan as P  # noqa: PLC0415

    def placement(card):
        opts = P.PlanOptions(streams=args.streams, main_card=(card.vendor, card.index),
                             admission=args.admission, assumed_max_context=args.max_context)
        plan = P.make_plan(report, model, os.path.dirname(paths.native), opts=opts)
        log = os.path.join(probe_dir, "engine-%s%d.log" % (card.vendor, card.index))
        cfg = _probe_config(args, report, plan, paths, work, probe_dir, port, card, log)
        others = tuple(BL.Other(r.vendor, r.gpu_index, r.uuid, tuple(cfg.helper_args[r.helper_index]), cfg.helper_env.get(r.helper_index, {}))
                       for r in plan.helpers())
        return BL.Placement(cfg.cfg_json_path, cfg.server_env, others, log)

    return placement


def _print_lead_results(results) -> None:
    for key, r in results.items():
        print("  %s gpu%d first: %s" % (key[0], key[1], ", ".join("%d stream(s) %.1f ms/window" % (c, v) for c, v in r.ms.items()) or "UNMEASURED"))


def lead_stage(args, report, model, paths, work: str, bench, pre):
    """Measure the lead job per candidate card 1 (see _bench_lead) and fold it into `bench`; any failure leaves the heuristic.

    `pre` is the prebuilt binaries, passed in (not looked up here) so the host decides where they come from."""
    from . import _bench_lead as BL  # noqa: PLC0415
    from . import _bench_lead_run as BR  # noqa: PLC0415
    from . import services as S  # noqa: PLC0415
    from seabraid.ranking import lead_class  # noqa: PLC0415

    klass = lead_class(args.streams)
    why = _lead_skip_reason(args, klass, pre)
    if why:
        return dataclasses.replace(bench, lead_why="the lead-job probe " + why)
    cands = _best_first(report, args.streams)
    probe_dir = os.path.join(work, "lead-probe")
    port = args.port + 17           # never the port a running service holds
    placement = _placement_for(args, report, model, paths, work, probe_dir, port)
    lp = BL.LeadPlan(S.pick_python(os.path.join(work, "strata")), os.path.join(work, "strata", "serve", "server.py"), port,
                     pre.helper_exes, _probe_libs(report, cands), placement)
    n = min(len(cands), BL.MAX_CANDIDATES)
    print("\nlead-job probe: %d cold engine start(s), about 70-100 s each with a large model, with %s first in turn; logs in %s"
          % (n, ", ".join(c.name for c in cands[:n]), probe_dir))
    os.makedirs(probe_dir, exist_ok=True)
    results = BL.measure_lead(cands, lp, [klass], BR.ProcessLeadRunner())
    _print_lead_results(results)
    lead, why = BL.to_lead_ms(results, cands)
    return dataclasses.replace(bench, lead_ms=lead, lead_why=why)


# ------------------------------------------------------ build, configure, install


def _binaries(args, plan, project: str, work: str) -> tuple:
    """(engine, {vendor: helper}, strata_dir): the prebuilt ones under --no-build, else a fresh build."""
    from . import build as B  # noqa: PLC0415

    if args.no_build:
        vendor = plan.main().vendor
        engine = engine_for(work, vendor)
        if not engine:
            raise Refused("--no-build, but no %s engine under %s. Drop --no-build to build one." % (vendor, work), 3)
        _, helper_exes = find_prebuilt(work)
        print("--no-build: using %s" % engine)
        return engine, helper_exes, os.path.join(work, "strata")
    try:
        # dry_run EXPLICITLY (now a required argument; it used to default to True, so a bare B.Runner() made a REAL install print every build command as
        # "[dry-run] ...", build nothing, and carry on: run() returns 0 in dry-run,
        # build_all only raises on non-zero, and BuildResult carries computed PATHS that
        # nothing existence-checks -- so cfg.json ended up pointing at a binary that was
        # never built and the units were installed anyway.
        built = B.build_all(plan, project, work, B.Runner(dry_run=args.dry_run), jobs=args.jobs,
                            file_moves=getattr(args, "file_moves", False))
    except B.BuildError as exc:
        raise Refused("the build failed: %s" % exc, 3) from exc
    return built.server_exe, dict(built.helper_exes), built.strata_dir


def _require_helpers(plan, helper_exes: dict) -> None:
    missing = {r.vendor for r in plan.helpers()} - set(helper_exes)
    if missing:
        raise Refused("no helper binary for %s; the plan needs one per helper vendor." % ", ".join(sorted(missing)), 3)


def _manifest_for(args, paths, model, strata_dir: str) -> None:
    """--file-moves: refuse early (before minutes of hashing) when the ladder is off, then make the pack's manifest fresh."""
    import subprocess  # noqa: PLC0415

    from . import _file_moves as FM  # noqa: PLC0415

    why = FM.refusal(bool(getattr(args, "ladder", False)), ())
    if why:
        raise Refused(why, 3)
    try:
        state = FM.ensure_manifest(
            paths.pack, FM.find_tool(strata_dir), lambda argv: subprocess.run(list(argv)).returncode, args.dry_run,
            native=paths.native, layers=model.layers, experts=model.experts,
        )
    except FM.FileMovesRefused as exc:
        raise Refused(str(exc), 3) from exc
    print("\nfile moves: %s %s (%s)" % (FM.MANIFEST, {FM.FRESH: "is fresh", FM.BUILT: "was built",
                                                     FM.WOULD_BUILD: "would be built or verified by seabraid-sums"}[state], paths.pack))


def _config_for(args, report, plan, paths, work: str):
    """The engine config for this plan, written unless --dry-run, with its warnings shown and confirmed."""
    from . import config as C  # noqa: PLC0415

    opts = C.ConfigOptions(port=args.port, host=args.host, max_context=args.max_context, ladder=bool(getattr(args, "ladder", False)),
                         file_moves=bool(getattr(args, "file_moves", False)))
    try:
        cfg = C.make_config(report, plan, paths, opts)
    except C.ConfigRefused as exc:
        raise Refused(str(exc), 3) from exc
    # The path is always carried, written or not: services needs an absolute cfg.json path to build a
    # unit, and a dry run that hands it an empty string is refused before it can print anything.
    cfg = (C.write_config(cfg, work) if not args.dry_run
           else dataclasses.replace(cfg, cfg_json_path=C.config_path(work)))
    print("\n" + cfg.explain())
    # The warnings surface HERE and nowhere else: services prints only its own
    # health output, and the GTT one is the difference between a slow host and a
    # frozen card.  A warning must not die inside the stage that produced it.
    for w in cfg.warnings:
        print("\n  !! %s: %s\n     fix: %s" % (w.knob, w.what, w.fix))
    if cfg.warnings and not ask_yes("\nContinue despite the warning(s) above?", args.yes):
        raise Refused("stopped at the warnings; nothing was started.", 130)
    return cfg


def _install(args, plan, cfg, strata_dir: str, helper_exes: dict, extra: dict) -> str:
    """Render and install the units; returns the unit directory."""
    from . import services as S  # noqa: PLC0415

    unit_dir = os.path.expanduser("~/.config/systemd/user")
    try:
        units = S.render_units(plan, cfg, strata_dir, helper_exes, bind=args.host, **extra)
        S.install_units(units, unit_dir, apply=not args.dry_run, start=not args.no_start)
        # FIX (2026-11-09 install): --no-start used to pass apply=False, which wrote NOTHING while the message below claimed
        # the units were in place.  Its help says "write the services but do not start them": so write, do not start.
    except S.ServiceError as exc:
        raise Refused(str(exc), 3) from exc
    return unit_dir


def _announce(args, cfg, paths, pack: str, unit_dir: str, health_kwargs: dict, plan) -> int:
    """The closing step: say what was written, or wait for the service and say that it serves."""
    from . import services as S  # noqa: PLC0415

    if args.dry_run:
        print("\n--dry-run: nothing was written or started.")
        return 0
    if args.no_start:
        print("\n--no-start: units are in %s. Start them with:\n    systemctl --user start %s"
              % (unit_dir, S.SERVER_UNIT))
        return 0
    try:
        health = S.check_health(cfg.port, host=args.host, model=pack, plan=plan, **health_kwargs)
    except S.ServiceError as exc:
        raise Refused("%s\nThe engine log is at %s" % (exc, paths.log), 1) from exc
    print("\nseabraid is serving %s on %s:%d%s"
          % (pack, args.host, cfg.port,
             "" if health.decode_tok_s is None else "  (%.1f tok/s decode)" % health.decode_tok_s))
    print_cleartext(args.host)
    return 0


def _available_flags(args, plan):
    """Keep the usable primary when an optional helper feature has nowhere to run."""
    if plan.helpers() or not getattr(args, "ladder", False):
        return args
    effective = copy.copy(args)
    effective.ladder = effective.file_moves = False
    print("\nNo usable helper in the plan. Running the primary without ladder or file moves.")
    return effective


def _engine_paths(paths, engine: str, work: str, dry_run: bool):
    """Resolve a first-run profile only after Strata's checkout exists."""
    from ._setup_model import ModelSetupError, resolve_profile  # noqa: PLC0415

    paths = dataclasses.replace(paths, exe=engine)
    if paths.expert_profile:
        return paths
    try:
        return resolve_profile(paths, work, require_exists=not dry_run)
    except ModelSetupError as exc:
        raise Refused(str(exc), 3) from exc


def finish(args, report, plan, paths, pack: str, work: str) -> int:
    """Build, configure, install.  Separated only to keep main() readable."""
    args = _available_flags(args, plan)
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    engine, helper_exes, strata_dir = _binaries(args, plan, project, work)
    _require_helpers(plan, helper_exes)
    paths = _engine_paths(paths, engine, work, args.dry_run)
    if getattr(args, "file_moves", False):
        _manifest_for(args, paths, plan.model, strata_dir)
    cfg = _config_for(args, report, plan, paths, work)
    units_kwargs, health_kwargs = key_args(args)
    unit_dir = _install(args, plan, cfg, strata_dir, helper_exes, units_kwargs)
    return _announce(args, cfg, paths, pack, unit_dir, health_kwargs, plan)
