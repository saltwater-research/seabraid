# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for install/_setup_model.py and the install/seabraid_setup.py CLI.

The directory layouts here mirror the reference install exactly
(packs/<pack>/tokenizer, models/<VARIANT>/*.gguf, dflash2/*.pack), built in a
temp tree, so no model and no GPU are needed.

Run:  python3 -m pytest tests/install/test_setup_cli.py -q
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from install import _setup_model as M  # noqa: E402
from install import seabraid_setup as CLI  # noqa: E402


def touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("x")


class TestDraftPack(unittest.TestCase):
    def test_an_unfinished_download_is_never_chosen(self):
        """A .pack with no .pack.ok beside it fails at load with a corrupt-file
        error that reads nothing like 'this never finished'."""
        self.assertIsNone(M.pick_draft_pack(["d.q4_0.pack"]))

    def test_the_selective_quant_build_is_preferred(self):
        names = ["d.q4_0.pack", "d.q4_0.pack.ok", "d.q4_0-selq4.pack", "d.q4_0-selq4.pack.ok"]
        self.assertEqual("d.q4_0-selq4.pack", M.pick_draft_pack(names))

    def test_a_complete_plain_pack_is_used_when_it_is_all_there_is(self):
        self.assertEqual("d.q4_0.pack", M.pick_draft_pack(["d.q4_0.pack", "d.q4_0.pack.ok"]))

    def test_an_incomplete_selq4_loses_to_a_complete_plain_one(self):
        names = ["d.q4_0.pack", "d.q4_0.pack.ok", "d.q4_0-selq4.pack"]
        self.assertEqual("d.q4_0.pack", M.pick_draft_pack(names))


class TestDiscover(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="seabraid-data-")
        d = self.data = self.tmp.name
        touch(os.path.join(d, "packs", "iq3_s", "index.txt"))
        os.makedirs(os.path.join(d, "packs", "iq3_s", "tokenizer"))
        touch(os.path.join(d, "models", "IQ3_S", "Q-GSQ-IQ3_S-00001-of-00002.gguf"))
        touch(os.path.join(d, "models", "IQ3_S", "Q-GSQ-IQ3_S-00002-of-00002.gguf"))
        touch(os.path.join(d, "dflash2", "q.q4_0-selq4.pack"))
        touch(os.path.join(d, "dflash2", "q.q4_0-selq4.pack.ok"))
        touch(os.path.join(d, "dflash2", "draft_vocab.bin"))

    def tearDown(self):
        self.tmp.cleanup()

    def _discover(self, pack="iq3_s"):
        return M.discover(self.data, pack, exe="/b/strata", cwd="/b", log="/b/x.log")

    def test_finds_everything_the_reference_install_has(self):
        p = self._discover()
        self.assertTrue(p.native.endswith("00001-of-00002.gguf"))
        self.assertIsNone(p.ple_gguf)  # Strata locates the PLE tensor across all native shards.
        self.assertTrue(p.tokenizer.endswith(os.path.join("iq3_s", "tokenizer")))
        self.assertTrue(p.draft_model.endswith("selq4.pack"))
        self.assertTrue(p.draft_vocab.endswith("draft_vocab.bin"))

    def test_the_pack_name_matches_the_model_dir_case_insensitively(self):
        """The pack is iq3_s and the model directory IQ3_S for the same thing."""
        self.assertIn("IQ3_S", self._discover().native)

    def test_a_missing_pack_is_refused_with_the_path(self):
        with self.assertRaises(M.ModelSetupError) as c:
            self._discover("nope")
        self.assertIn("nope", str(c.exception))

    def test_a_pack_without_a_tokenizer_is_refused(self):
        os.rmdir(os.path.join(self.data, "packs", "iq3_s", "tokenizer"))
        with self.assertRaises(M.ModelSetupError):
            self._discover()

    def test_an_ambiguous_model_directory_refuses_rather_than_picking(self):
        """The wrong one is a different quantisation: it would load, and be
        quietly worse."""
        os.makedirs(os.path.join(self.data, "models", "iq3s"))
        touch(os.path.join(self.data, "packs", "iq3s", "index.txt"))
        os.makedirs(os.path.join(self.data, "packs", "iq3s", "tokenizer"))
        touch(os.path.join(self.data, "models", "iq3s", "a-00001-of-00001.gguf"))
        with self.assertRaises(M.ModelSetupError) as c:
            self._discover("iq3s")
        self.assertIn("several", str(c.exception))

    def test_a_single_shard_model_has_no_separate_ple_file(self):
        for n in os.listdir(os.path.join(self.data, "models", "IQ3_S")):
            os.remove(os.path.join(self.data, "models", "IQ3_S", n))
        touch(os.path.join(self.data, "models", "IQ3_S", "a-00001-of-00001.gguf"))
        self.assertIsNone(self._discover().ple_gguf)

    def test_only_scoped_device_blockers_can_be_excluded(self):
        from install.detect import Blocker
        report = types.SimpleNamespace(blockers=(
            Blocker("gpu:amd:0", "bad card", "fix it"),
            Blocker("toolkit:sycl", "unsupported card", "fix it"),
            Blocker("fs", "probe failed", "fix it"),
            Blocker("gpu", "GPU probe failed", "fix it"),
        ))
        self.assertEqual(["fs", "gpu"], [b.where for b in CLI._global_blockers(report)])
    def test_plan_only_rejects_lead_probe_before_detection(self):
        with mock.patch.object(CLI, "detect_host", side_effect=AssertionError("must not probe")):
            self.assertEqual(2, CLI.main(["--plan-only", "--bench-lead"]))
    def test_list_packs(self):
        self.assertEqual(["iq3_s"], M.list_packs(self.data))

    def test_list_packs_on_a_machine_with_none(self):
        self.assertEqual([], M.list_packs(os.path.join(self.data, "absent")))


class TestRunStrataSetup(unittest.TestCase):
    def test_no_start_is_always_appended(self):
        """This installer starts the engine itself, through units with the
        helper ordered first; a second server would contend for the card."""
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return type("R", (), {"returncode": 0})()

        with tempfile.TemporaryDirectory() as tmp:
            touch(os.path.join(tmp, "setup.py"))
            M.run_strata_setup(tmp, ["--yes"], echo=lambda s: None, run=fake_run)
        self.assertIn("--no-start", seen["cmd"])

    def test_a_missing_setup_py_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(M.ModelSetupError):
                M.run_strata_setup(tmp, [], echo=lambda s: None)

    def test_a_nonzero_exit_is_refused_and_says_nothing_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            touch(os.path.join(tmp, "setup.py"))
            with self.assertRaises(M.ModelSetupError) as c:
                M.run_strata_setup(tmp, [], echo=lambda s: None,
                                   run=lambda *a, **k: type("R", (), {"returncode": 7})())
        self.assertIn("has not changed anything", str(c.exception))


class TestPrompts(unittest.TestCase):
    def test_yes_takes_the_recommendation(self):
        self.assertTrue(CLI.ask_yes("go?", yes=True))

    def test_yes_takes_the_first_choice(self):
        self.assertEqual("a", CLI.ask_choice("which?", ["a", "b"], yes=True))

    def test_a_single_choice_is_not_asked_about(self):
        self.assertEqual("only", CLI.ask_choice("which?", ["only"], yes=False))


class TestFindPrebuilt(unittest.TestCase):
    def test_finds_the_engine_and_a_helper_per_vendor(self):
        with tempfile.TemporaryDirectory() as tmp:
            touch(os.path.join(tmp, "strata", "build-amd", "strata"))
            touch(os.path.join(tmp, "strata", "build-nvidia", "seabraid", "seabraid-helper"))
            engine, helpers = CLI.find_prebuilt(tmp)
        self.assertTrue(engine.endswith("strata"))
        self.assertEqual(["nvidia"], list(helpers))

    def test_an_empty_tree_yields_nothing_rather_than_a_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(("", {}), CLI.find_prebuilt(tmp))


class TestParser(unittest.TestCase):
    def test_binds_to_this_pc_only_by_default(self):
        """A model server reachable from the LAN by default is not a default."""
        self.assertEqual("127.0.0.1", CLI.build_parser().parse_args([]).host)

    def test_nothing_destructive_is_the_default(self):
        a = CLI.build_parser().parse_args([])
        self.assertFalse(a.yes)
        self.assertFalse(a.no_build)
        self.assertFalse(a.dry_run)


class TestFinishEndToEnd(unittest.TestCase):
    """Drive seabraid_setup._finish with the REAL config stage and stubbed build/services.

    This exists because the first end-to-end run on the test system crashed here and no test saw it:

        paths = C.ModelPaths(**{**paths.__dict__, "exe": engine})
        AttributeError: 'ModelPaths' object has no attribute '__dict__'

    Every other test exercised a stage in isolation, so the line that JOINS them was the only
    unexercised code on the path -- and it was the line that failed. config is deliberately real here;
    stubbing it would reintroduce exactly the blind spot.
    """

    class Args:
        no_build = True
        dry_run = True
        no_start = False
        port = 8080
        host = "127.0.0.1"
        api_key_file = None
        max_context = None
        jobs = None
        yes = True

    @staticmethod
    def _plan():
        from install import plan as P

        card = P.Role("main", 0, "amd", "R9700", "gfx1201", 0, None, 8 << 30)
        helper = P.Role("helper", 0, "nvidia", "3060", "sm_86", 0, 0, 4 << 30)
        return P.Plan(
            model=P.ModelFacts(layers=48, experts=512, expert_bytes=1 << 35, dense_bytes=1 << 32,
                               hidden=2560),
            mode="all-vram",
            roles=(card, helper),
            builds=(),
            tiers=P.TierSizes(vram_bytes={}, ram_bytes=0, nvme_bytes=0, nvme_mount=None),
            assumed_max_context=131072,
        )

    @staticmethod
    def _report():
        from install import detect as D

        return D.HostReport(
            gpus=(D.Gpu(0, "amd", "R9700", "gfx1201", 32 << 30, 31 << 30, 17179869184,
                        "0000:02:00.0", 4, 8, 4, 8, None, "kfd-sysfs"),
                  D.Gpu(0, "nvidia", "3060", "sm_86", 12 << 30, 11 << 30, None,
                        "0000:03:00.0", 3, 8, 3, 8, "595", "nvidia-smi", "GPU-abc")),
            toolkits=(D.Toolkit("amd", "rocm", "/opt/r", "7", "/opt/r/bin/hipcc"),
                      D.Toolkit("nvidia", "cuda", "/usr", "12.4", "/usr/bin/nvcc", "/usr/bin/gcc-13")),
            ttm_pages_limit_bytes=17179869184,
        )

    def _run(self, work):
        from install import config as C

        paths = C.ModelPaths(
            exe="", cwd=work, pack=os.path.join(work, "pack"),
            native=os.path.join(work, "m-00001-of-00001.gguf"),
            tokenizer=os.path.join(work, "pack", "tokenizer"),
            model_name="iq2_xs", log=os.path.join(work, "x.log"), expert_profile=os.path.join(work, "ep.bin"))

        rendered = {}

        class Svc:
            SERVER_UNIT = "seabraid-server.service"
            ServiceError = RuntimeError

            @staticmethod
            def render_units(plan, cfg, strata_dir, helper_exes, bind="127.0.0.1"):
                # The stub enforces what the REAL services enforces.  The first version accepted
                # anything, so a dry run handing over an empty cfg_json_path passed here and was
                # refused on the test system by services.py:64 -- a stub more permissive than the thing it
                # stands in for is a test that agrees with you instead of checking you.
                if not os.path.isabs(str(cfg.cfg_json_path)):
                    raise RuntimeError(
                        "cfg.json must be an absolute path for systemd, got %r" % cfg.cfg_json_path)
                rendered["cfg"] = cfg
                rendered["helpers"] = dict(helper_exes)
                return {"seabraid-helper@0.service": "unit"}

            @staticmethod
            def install_units(units, unit_dir, apply=False, start=True):
                rendered["apply"], rendered["start"] = apply, start
                return []

            @staticmethod
            def helper_unit_name(n):
                return "seabraid-helper@%d.service" % n

        class Bld:
            BuildError = RuntimeError

        # Both sys.modules AND the package attribute.  `from . import services` resolves the attribute
        # once any other test has imported the real submodule, so patching sys.modules alone leaves the
        # real services running -- which is exactly what happened: this test passed on its own and failed
        # in the full suite, an order-dependent pass that is worth less than no test.
        import install as pkg

        with mock.patch.dict(sys.modules, {"install.services": Svc, "install.build": Bld}),                 mock.patch.object(pkg, "services", Svc, create=True),                 mock.patch.object(pkg, "build", Bld, create=True):
            from install import seabraid_setup as B

            rc = B._finish(self.Args(), self._report(), self._plan(), paths, "iq2_xs", work)
        return rc, rendered

    def test_finish_runs_the_whole_path_without_crashing(self):
        with tempfile.TemporaryDirectory() as work:
            touch(os.path.join(work, "strata", "build-amd", "strata"))
            touch(os.path.join(work, "strata", "build-nvidia", "seabraid", "seabraid-helper"))
            rc, rendered = self._run(work)
        self.assertEqual(0, rc, "a dry run of the full path must succeed")
        self.assertTrue(rendered["cfg"].cfg_json["exe"].endswith("strata"),
                        "the built engine must reach cfg.json")
        self.assertFalse(rendered["apply"], "--dry-run must not touch systemd")

    def test_no_start_applies_the_write_but_not_the_start(self):
        """--no-start used to pass apply=False (write nothing) yet print 'units are in ...'."""
        with tempfile.TemporaryDirectory() as work:
            touch(os.path.join(work, "strata", "build-amd", "strata"))
            touch(os.path.join(work, "strata", "build-nvidia", "seabraid", "seabraid-helper"))
            self.Args.dry_run, self.Args.no_start = False, True
            try:
                rc, rendered = self._run(work)
            finally:
                self.Args.dry_run, self.Args.no_start = True, False
        self.assertEqual(0, rc)
        self.assertTrue(rendered["apply"], "--no-start must still write the units")
        self.assertFalse(rendered["start"], "--no-start must not start them")

    def test_a_dry_run_reports_the_path_it_would_write_without_writing_it(self):
        """systemd needs an absolute cfg.json path to build a unit, so a dry run that reports an empty
        one is refused before it can print anything -- which is how the test system failed."""
        with tempfile.TemporaryDirectory() as work:
            touch(os.path.join(work, "strata", "build-amd", "strata"))
            touch(os.path.join(work, "strata", "build-nvidia", "seabraid", "seabraid-helper"))
            rc, rendered = self._run(work)
            path = rendered["cfg"].cfg_json_path
            self.assertEqual(0, rc)
            self.assertTrue(os.path.isabs(path), path)
            self.assertFalse(os.path.exists(path), "a dry run must not write the file it names")

    def test_a_missing_helper_binary_is_refused_not_ignored(self):
        """The plan has an NVIDIA helper, so a build with no NVIDIA helper binary cannot be served."""
        with tempfile.TemporaryDirectory() as work:
            touch(os.path.join(work, "strata", "build-amd", "strata"))
            rc, _ = self._run(work)
        self.assertEqual(3, rc)

    def test_no_engine_at_all_is_refused(self):
        with tempfile.TemporaryDirectory() as work:
            rc, _ = self._run(work)
        self.assertEqual(3, rc)


class TestRealInstallActuallyBuilds(unittest.TestCase):
    """A real install must hand the builder a Runner that RUNS commands, not one that prints them.

    build.Runner defaults dry_run=True so tests can read .commands, which turned the single
    real call site into a silent no-op when it omitted the flag: run() returns 0 in dry-run,
    build_all raises only on a NON-ZERO return, and BuildResult carries computed PATHS that nothing
    existence-checks.  A real install therefore printed "[dry-run] cmake ...", built nothing, wrote a
    cfg.json pointing at a binary that never existed, and installed the units anyway -- exiting 0.

    The assertion is on the Runner the builder RECEIVES, because that is the thing that was wrong.
    Asserting on the exit code could not catch it: the broken path exits 0 and looks like success.
    """

    class Args:
        no_build = False          # take the BUILD branch; TestFinishEndToEnd uses --no-build
        dry_run = False           # a real install
        no_start = False
        port = 8080
        host = "127.0.0.1"
        api_key_file = None
        max_context = None
        jobs = None
        yes = True

    @staticmethod
    def _stubs(work, seen):
        from install import build as RealB

        class Bld:
            BuildError = RuntimeError
            Runner = RealB.Runner          # the REAL class, so the REAL default applies

            @staticmethod
            def build_all(plan, project, work_dir, runner, jobs=None, file_moves=False):
                seen["dry_run"] = runner.dry_run
                return RealB.BuildResult(
                    strata_dir=os.path.join(work_dir, "strata"),
                    server_exe=os.path.join(work_dir, "strata", "build-amd", "strata"),
                    helper_exes={"nvidia": os.path.join(
                        work_dir, "strata", "build-nvidia", "seabraid", "seabraid-helper")},
                    build_dirs={}, jobs=jobs or 1)

        class Svc:
            SERVER_UNIT = "seabraid-server.service"
            ServiceError = RuntimeError

            @staticmethod
            def render_units(plan, cfg, strata_dir, helper_exes, bind="127.0.0.1"):
                return {"seabraid-helper@0.service": "unit"}

            @staticmethod
            def install_units(units, unit_dir, apply=False, start=True):
                return []

            @staticmethod
            def helper_unit_name(n):
                return "seabraid-helper@%d.service" % n

            @staticmethod
            def check_health(port, host="127.0.0.1", model=None, **kw):
                # A real install does not stop at installing units -- it starts them and polls
                # /health.  The stub has to reach this far precisely BECAUSE the bug lived on the
                # non-dry path: every pre-existing _finish test ran with dry_run=True and returned
                # before here, which is how a silent no-op install survived a green suite.
                seen["health_checked"] = True
                return types.SimpleNamespace(decode_tok_s=12.5)

        return Bld, Svc

    def _drive(self, args):
        from install import config as C

        seen = {}
        with tempfile.TemporaryDirectory() as work:
            paths = C.ModelPaths(
                exe="", cwd=work, pack=os.path.join(work, "pack"),
                native=os.path.join(work, "m-00001-of-00001.gguf"),
                tokenizer=os.path.join(work, "pack", "tokenizer"),
                model_name="iq2_xs", log=os.path.join(work, "x.log"), expert_profile=os.path.join(work, "ep.bin"))
            Bld, Svc = self._stubs(work, seen)

            # Both sys.modules AND the package attribute, for the same reason the sibling test
            # documents: `from . import build` resolves the attribute once any other test has
            # imported the real submodule, so patching sys.modules alone leaves the real one in.
            import install as pkg

            with mock.patch.dict(sys.modules, {"install.services": Svc, "install.build": Bld}), \
                    mock.patch.object(pkg, "services", Svc, create=True), \
                    mock.patch.object(pkg, "build", Bld, create=True):
                from install import seabraid_setup as B

                B._finish(args, TestFinishEndToEnd._report(), TestFinishEndToEnd._plan(),
                          paths, "iq2_xs", work)
        return seen

    def test_a_real_install_hands_the_builder_a_live_runner(self):
        seen = self._drive(self.Args())
        self.assertIn("dry_run", seen, "build_all was never reached, so this proved nothing")
        self.assertFalse(
            seen["dry_run"],
            "a real install handed the builder a dry-run Runner: every build command would be "
            "printed and none executed, then the install would continue as if it had built")

    def test_a_dry_run_still_gets_a_dry_runner(self):
        """The other direction, so the fix cannot be a hardcoded False."""

        class Args(self.Args):
            dry_run = True

        seen = self._drive(Args())
        self.assertIn("dry_run", seen, "build_all was never reached, so this proved nothing")
        self.assertTrue(seen["dry_run"], "--dry-run must not execute build commands")


if __name__ == "__main__":
    unittest.main()
