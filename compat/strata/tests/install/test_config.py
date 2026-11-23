# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for install/config.py.

The plan and the detect report are LOCAL duck-typed stand-ins, so no GPU and no
Linux are needed.  The decisive test is not a stand-in though: it asserts the
generated cfg.json against ``configs/example-iq3s-tiered.json``, which is a
configuration that was actually run and measured, so a knob that drifts away
from the verified launch fails here.

Run:  python3 -m pytest tests/install/test_config.py -q
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from install import config as C  # noqa: E402

GIB = 1 << 30


# ------------------------------------------------------------- stand-ins


@dataclass(frozen=True)
class Gpu:
    index: int
    vendor: str
    gtt_total_bytes: Optional[int] = None
    uuid: Optional[str] = None


@dataclass(frozen=True)
class Toolkit:
    vendor: str
    lib_dirs: tuple = ()


@dataclass(frozen=True)
class Report:
    gpus: tuple = ()
    toolkits: tuple = ()
    ttm_pages_limit_bytes: Optional[int] = None

    def toolkit_for(self, vendor):
        return next((t for t in self.toolkits if t.vendor == vendor), None)


@dataclass(frozen=True)
class Role:
    kind: str
    gpu_index: int
    vendor: str
    device_id: int
    helper_index: Optional[int] = None

    @property
    def shm(self):
        return None if self.helper_index is None else "/seabraid%d" % self.helper_index


@dataclass(frozen=True)
class Tiers:
    ram_bytes: int = 0
    nvme_bytes: int = 0
    vram_bytes: Mapping[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Model:
    layers: int = 48
    experts: int = 512


@dataclass(frozen=True)
class Plan:
    mode: str = "tiered"
    roles: tuple = ()
    tiers: Tiers = field(default_factory=Tiers)
    model: Model = field(default_factory=Model)
    blocked: tuple = ()
    refusals: tuple = ()
    assumed_max_context: Optional[int] = None

    @property
    def ready(self):
        return self.mode != "infeasible" and not self.blocked and not self.refusals and bool(self.roles)

    def main(self):
        return next(r for r in self.roles if r.kind == "main")

    def helpers(self):
        return tuple(r for r in self.roles if r.kind == "helper")

    def explain(self):
        return "plan: " + self.mode


ROCM = "/opt/rocm-x"
HOST_GTT = 17179869184  # the test system, card and ttm cap alike


def host_report():
    """A two-GPU host: AMD main (gfx1201) plus one NVIDIA helper."""
    return Report(
        gpus=(
            Gpu(index=0, vendor="nvidia", uuid="GPU-0a1b2c3d-1111-2222-3333-444455556666"),
            Gpu(index=0, vendor="amd", gtt_total_bytes=HOST_GTT),
        ),
        toolkits=(Toolkit("amd", (ROCM + "/lib",)), Toolkit("nvidia", ("/usr/lib64",))),
        ttm_pages_limit_bytes=HOST_GTT,
    )


def host_plan(ram_gib=10, mode="tiered", **kw):
    kw.setdefault("assumed_max_context", 32768)
    return Plan(
        mode=mode,
        roles=(
            Role("main", 0, "amd", 0),
            Role("helper", 0, "nvidia", 0, helper_index=0),
        ),
        tiers=Tiers(ram_bytes=int(ram_gib * GIB)),
        **kw,
    )


def host_paths():
    return C.ModelPaths(
        exe="/opt/b/strata", cwd="/opt/b", pack="/data/packs/iq3_s",
        native="/data/models/a-00001-of-00002.gguf", tokenizer="/data/packs/iq3_s/tokenizer",
        model_name="qwen-iq3-s", log="/var/log/seabraid.log",
        ple_gguf="/data/models/a-00002-of-00002.gguf",
        draft_model="/data/dflash2/d.pack", draft_vocab="/data/dflash2/draft_vocab.bin",
        expert_profile="/opt/b/strata/data/expert-profile.bin",
    )


def build(report=None, plan=None, paths=None, opts=None):
    return C.make_config(report or host_report(), plan or host_plan(), paths or host_paths(),
                         opts or C.ConfigOptions(), isdir=lambda p: False)


# ----------------------------------------------------------------- tests


class TestRefusal(unittest.TestCase):
    def test_a_blocked_plan_is_refused_not_configured(self):
        """The whole point of a blocked tier is that nobody approved it."""
        plan = replace(host_plan(), blocked=("ram",))
        with self.assertRaises(C.ConfigRefused):
            build(plan=plan)

    def test_an_infeasible_plan_is_refused(self):
        with self.assertRaises(C.ConfigRefused):
            build(plan=replace(host_plan(), mode="infeasible"))

    def test_a_refused_tier_never_becomes_a_resident_budget(self):
        plan = replace(host_plan(), refusals=("no room",))
        with self.assertRaises(C.ConfigRefused):
            build(plan=plan)


class TestAgainstTheMeasuredConfig(unittest.TestCase):
    """The generated config against one that was really run."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(REPO, "configs", "example-iq3s-tiered.json"), encoding="utf-8") as fh:
            cls.ref = json.load(fh)
        cls.out = build()
        cls.args = cls.out.cfg_json["args"]

    def _value(self, flag, args=None):
        args = self.args if args is None else args
        return args[args.index(flag) + 1] if flag in args else None

    def test_every_tuning_flag_matches_the_measured_run(self):
        for flag in ("--spec", "--draft-conf", "--kv", "--expert-cache", "--prefill",
                     "--draft", "--draft-fallback", "--expert-cache-device1"):
            self.assertEqual(
                self._value(flag, self.ref["args"]), self._value(flag),
                "%s drifted from the measured configuration" % flag,
            )

    def test_resident_experts_is_present_as_a_bare_flag(self):
        self.assertIn("--resident-experts", self.args)

    def test_the_ram_tier_becomes_resident_budget_gib(self):
        self.assertEqual("10", self._value("--resident-budget-gib"))

    def test_draft_conf_is_reasserted_in_user_args(self):
        """#5109: drafting.apply sizes the draft from the wrong card on a mixed
        host, so the value has to be repeated where it cannot be recomputed."""
        self.assertEqual(self.ref["user_args"], self.out.cfg_json["user_args"])

    def test_engine_env_matches(self):
        for key, value in self.ref["env"].items():
            self.assertEqual(value, self.out.cfg_json["env"].get(key), key)


class TestAgainstTheAllVramConfig(unittest.TestCase):
    """An all-VRAM plan against ``configs/example-iq3s-all-vram.json``, the IQ3_S launch measured on the
    R9700 + RTX 3090 system (2026-11-07). It differs from the tiered reference only by the RAM budget, so the
    installer must drop exactly that and keep every tuning flag."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(REPO, "configs", "example-iq3s-all-vram.json"), encoding="utf-8") as fh:
            cls.ref = json.load(fh)
        cls.out = build(plan=host_plan(ram_gib=0, mode="all-vram"))
        cls.args = cls.out.cfg_json["args"]

    def _value(self, flag, args=None):
        args = self.args if args is None else args
        return args[args.index(flag) + 1] if flag in args else None

    def test_every_tuning_flag_matches_the_measured_run(self):
        for flag in ("--spec", "--draft-conf", "--kv", "--expert-cache", "--prefill",
                     "--draft", "--draft-fallback", "--expert-cache-device1"):
            self.assertEqual(
                self._value(flag, self.ref["args"]), self._value(flag),
                "%s drifted from the measured configuration" % flag,
            )

    def test_no_ram_budget_without_a_ram_tier(self):
        self.assertNotIn("--resident-budget-gib", self.ref["args"])
        self.assertNotIn("--resident-budget-gib", self.args)

    def test_resident_experts_is_kept(self):
        """Kept as a bare flag, as measured: it pins whatever little the caches leave over."""
        self.assertIn("--resident-experts", self.ref["args"])
        self.assertIn("--resident-experts", self.args)

    def test_engine_env_matches_apart_from_the_gtt_route(self):
        """The measured run kept the GTT route env (one variable at a time against the tiered run). The
        installer adds it only with a RAM tier (config.py), where it prevents a KFD hang; without pinned host
        memory it has nothing to route, so its absence here is by rule, not drift."""
        for key, value in self.ref["env"].items():
            if key in C.GTT_ROUTE_ENV:
                continue
            self.assertEqual(value, self.out.cfg_json["env"].get(key), key)


class TestGttRoute(unittest.TestCase):
    def test_amd_main_with_a_ram_tier_gets_the_route(self):
        """Without it the KFD userptr path can freeze the card, so this is a
        hang being prevented, not a speed-up."""
        env = build().cfg_json["env"]
        for key, value in C.GTT_ROUTE_ENV.items():
            self.assertEqual(value, env.get(key), key)

    def test_no_ram_tier_means_no_route(self):
        env = build(plan=host_plan(ram_gib=0, mode="all-vram")).cfg_json["env"]
        self.assertNotIn("HSA_USERPTR_FOR_PAGED_MEM", env)

    def test_nvidia_main_never_gets_the_amd_route(self):
        plan = Plan(roles=(Role("main", 0, "nvidia", 0),), tiers=Tiers(ram_bytes=8 * GIB),
                    assumed_max_context=32768)
        env = build(plan=plan).cfg_json["env"]
        self.assertNotIn("HSA_SVM_FOR_PAGED_MEM", env)



class TestDeviceMasks(unittest.TestCase):
    def test_nvidia_helper_is_pinned_by_uuid(self):
        """CUDA_DEVICE_ORDER does not default to PCI bus order, so an ordinal is
        read against an ordering that depends on which cards are present."""
        env = build().helper_env[0]
        self.assertEqual("GPU-0a1b2c3d-1111-2222-3333-444455556666", env["CUDA_VISIBLE_DEVICES"])
        self.assertEqual("PCI_BUS_ID", env["CUDA_DEVICE_ORDER"])

    def test_an_nvidia_card_without_a_uuid_falls_back_to_the_ordinal(self):
        report = replace(host_report(), gpus=(Gpu(0, "nvidia"), Gpu(0, "amd", HOST_GTT)))
        self.assertEqual("0", build(report=report).helper_env[0]["CUDA_VISIBLE_DEVICES"])

    def test_nvidia_primary_mask_is_preserved_when_the_server_launches_the_engine(self):
        report = replace(host_report(), gpus=(Gpu(0, "nvidia", uuid="GPU-first"),
                                              Gpu(1, "nvidia", uuid="GPU-second")))
        plan = Plan(roles=(Role("main", 1, "nvidia", 1),), assumed_max_context=32768)
        result = build(report=report, plan=plan)
        self.assertEqual("cuda", result.cfg_json["backend"])
        self.assertEqual([1], result.cfg_json["gpu"])
        self.assertEqual("GPU-second", result.cfg_json["env"]["CUDA_VISIBLE_DEVICES"])
        self.assertEqual("GPU-second", result.server_env["CUDA_VISIBLE_DEVICES"])

    def test_amd_main_mask_goes_to_the_server(self):
        self.assertEqual("0", build().server_env["HIP_VISIBLE_DEVICES"])

    def test_the_server_env_carries_the_seabraid_switches(self):
        env = build().server_env
        for key, value in C.SERVER_ENV.items():
            self.assertEqual(value, env[key], key)
        self.assertEqual("/seabraid0", env["SEABRAID_STRATA_SHM"])


class TestHelperArgs(unittest.TestCase):
    def test_helper_takes_the_shm_name_on_the_command_line(self):
        """helper.cpp reads no environment at all, so --shm is the only way."""
        args = build().helper_args[0]
        self.assertEqual("/seabraid0", args[args.index("--shm") + 1])

    def test_layers_and_experts_come_from_the_model_not_a_constant(self):
        plan = replace(host_plan(), model=Model(layers=61, experts=256))
        args = build(plan=plan).helper_args[0]
        self.assertEqual("61", args[args.index("--layers") + 1])
        self.assertEqual("256", args[args.index("--experts") + 1])

    def test_helper_is_always_device_zero_inside_its_own_mask(self):
        args = build().helper_args[0]
        self.assertEqual("0", args[args.index("--device") + 1])


class TestContext(unittest.TestCase):
    def test_a_plan_with_no_context_is_refused_rather_than_defaulted(self):
        """Stage 2 reserves VRAM for the KV cache, so a context invented here
        would silently disagree with the tier sizing."""
        with self.assertRaises(C.ConfigRefused):
            build(plan=host_plan(assumed_max_context=None))

    def test_the_plan_owns_the_context(self):
        out = build(plan=host_plan(assumed_max_context=40960))
        self.assertEqual("40960", out.cfg_json["args"][out.cfg_json["args"].index("--max-context") + 1])

    def test_an_explicit_context_beyond_the_plan_warns(self):
        out = build(plan=host_plan(assumed_max_context=32768),
                    opts=C.ConfigOptions(max_context=131072))
        self.assertEqual(["max-context"], [w.knob for w in out.warnings])



class TestLibDirs(unittest.TestCase):
    def test_rocm_sysdeps_is_added_only_when_it_exists(self):
        """A TheRock install keeps its own libstdc++ there; leaving it off gives
        a symbol error at start that looks nothing like a missing ROCm."""
        out = C.make_config(host_report(), host_plan(), host_paths(), C.ConfigOptions(),
                            isdir=lambda p: p.endswith("rocm_sysdeps/lib"))
        self.assertIn(ROCM + "/lib/rocm_sysdeps/lib", out.server_env["LD_LIBRARY_PATH"])

    def test_absent_sysdeps_is_not_invented(self):
        self.assertEqual(ROCM + "/lib", build().server_env["LD_LIBRARY_PATH"])


class TestWrite(unittest.TestCase):
    def test_write_returns_the_path_it_used(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = C.write_config(build(), tmp)
            self.assertTrue(os.path.isabs(out.cfg_json_path))
            with open(out.cfg_json_path, encoding="utf-8") as fh:
                self.assertEqual(out.cfg_json["port"], json.load(fh)["port"])


if __name__ == "__main__":
    unittest.main()
