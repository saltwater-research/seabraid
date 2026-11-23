# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 parsed against SYNTHETIC sample output, with no hardware.

Every fixture in ``fixtures/`` is a hand-built sample in the shape the real tool or kernel interface prints (one AMD
gfx1201 node and one NVIDIA sm_86 card, typical desktop mounts), so these tests exercise the text the probes meet.  The
values are made up: no real machine's serial, UUID, PCI slot, driver build or exact RAM size is stored.  A fixture
carries only the fields a parser reads plus a little realistic noise.

Re-sample only when a tool's SHAPE changes (a new driver's columns, a different kernel's sysfs), never for different
values.

Run:  python3 -m pytest tests/ -q

NOT ``unittest discover``.  It silently collects only the unittest-style classes
and MISSES every pytest-style test function in this directory -- 136 of 203 on
this branch, so it reports a green suite while 67 tests never ran.  That is the
same "the check never ran" failure this file's own guards exist to catch, so the
gate is pytest.  ``unittest ... -k <name>`` is still the right tool for driving
ONE test during a sabotage, where the count is the point.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
sys.path.insert(0, REPO)

from install import _detect_gpu as dg  # noqa: E402
from install import _detect_pci as dpci  # noqa: E402
from install import _detect_host as dh  # noqa: E402
from install import _detect_toolkit as dt  # noqa: E402
from install import detect  # noqa: E402

GIB = 1024 ** 3


def fixture(name: str) -> str:
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as fh:
        return fh.read()


class TestNvidia(unittest.TestCase):
    def test_parses_the_recorded_card(self):
        rows = dg.parse_nvidia_smi(fixture("sample-nvidia-smi.csv"))
        self.assertEqual(1, len(rows))
        g = rows[0]
        self.assertEqual("sm_86", g["arch"])
        self.assertEqual(12 * GIB, g["vram_total_bytes"])
        self.assertEqual("0000:03:00.0", g["pci_bus_id"])

    def test_carries_the_uuid(self):
        """CUDA_DEVICE_ORDER does not default to PCI bus order, so pinning a
        helper to a card by ordinal is ambiguous; the UUID is not."""
        g = dg.parse_nvidia_smi(fixture("sample-nvidia-smi.csv"))[0]
        self.assertTrue(g["uuid"].startswith("GPU-"), g["uuid"])

    def test_keeps_max_and_current_link_apart(self):
        """The recorded card is idle: gen 3 x16 capable, running at gen 1 x8.
        Collapsing these to one pair would rank cards by how idle they are."""
        g = dg.parse_nvidia_smi(fixture("sample-nvidia-smi.csv"))[0]
        self.assertEqual(3, g["pcie_gen_max"])
        self.assertEqual(1, g["pcie_gen_current"])
        self.assertEqual(16, g["pcie_width_max"])
        self.assertEqual(8, g["pcie_width_current"])

    def test_not_available_fields_become_none_not_zero(self):
        row = dg.parse_nvidia_smi(
            "0, Some Card, [N/A], [Not Supported], 00000000:01:00.0, "
            "[N/A], [N/A], [N/A], [N/A], [N/A], [N/A], [N/A]"
        )[0]
        self.assertIsNone(row["vram_total_bytes"])
        self.assertIsNone(row["arch"])
        self.assertIsNone(row["driver_version"])

    def test_short_or_junk_lines_are_skipped(self):
        self.assertEqual([], dg.parse_nvidia_smi("garbage\n\n1,2,3"))


class TestBusId(unittest.TestCase):
    def test_eight_digit_domain_becomes_four(self):
        self.assertEqual("0000:0a:00.0", dpci.normalise_bus_id("00000000:0A:00.0"))

    def test_four_digit_domain_is_unchanged(self):
        self.assertEqual("0000:02:00.0", dpci.normalise_bus_id("0000:02:00.0"))

    def test_nonsense_is_passed_through_lowercased_not_invented(self):
        self.assertEqual("n/a", dpci.normalise_bus_id("N/A"))
        self.assertIsNone(dpci.normalise_bus_id(""))


class TestGfxDecode(unittest.TestCase):
    def test_recorded_card(self):
        self.assertEqual("gfx1201", dg.gfx_from_target_version(120001))

    def test_step_is_hex(self):
        """90010 is gfx90a.  Decimal would give gfx9010, which compiles for
        nothing and is the reason this lives in one function."""
        self.assertEqual("gfx90a", dg.gfx_from_target_version(90010))

    def test_other_known_families(self):
        self.assertEqual("gfx1100", dg.gfx_from_target_version(110000))
        self.assertEqual("gfx942", dg.gfx_from_target_version(90402))

    def test_absent_is_none(self):
        self.assertIsNone(dg.gfx_from_target_version(0))
        self.assertIsNone(dg.gfx_from_target_version(-1))


class TestKfd(unittest.TestCase):
    def test_cpu_node_is_recognisable_as_not_a_gpu(self):
        props = dg.parse_kfd_properties(fixture("sample-kfd-node0-properties.txt"))
        self.assertEqual(0, props["simd_count"])
        self.assertEqual(0, props["gfx_target_version"])

    def test_gpu_node(self):
        props = dg.parse_kfd_properties(fixture("sample-kfd-node1-properties.txt"))
        self.assertGreater(props["simd_count"], 0)
        self.assertEqual(120001, props["gfx_target_version"])
        self.assertEqual(129, props["drm_render_minor"])


class TestAmdFromFakeSysfs(unittest.TestCase):
    """The AMD probe end to end against a /sys tree holding the test system's values.

    This is the test that matters most: a typical AMD-only host has neither amd-smi
    nor rocminfo, so sysfs is the only AMD probe there is.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="seabraid-sysfs-")
        root = self.tmp.name
        self.kfd = os.path.join(root, "kfd")
        self.drm = os.path.join(root, "drm")
        for node, name in ((0, "sample-kfd-node0-properties.txt"), (1, "sample-kfd-node1-properties.txt")):
            d = os.path.join(self.kfd, str(node))
            os.makedirs(d)
            with open(os.path.join(d, "properties"), "w", encoding="utf-8") as fh:
                fh.write(fixture(name))

        # A PCI-address directory plus a symlink is what sysfs really looks
        # like, and it is what lets the probe recover the bus id.  Windows
        # allows neither a ":" in a name nor an unprivileged symlink, so there
        # the device directory is made directly and the bus-id assertion skips.
        os.makedirs(os.path.join(self.drm, "renderD129"))
        link = os.path.join(self.drm, "renderD129", "device")
        self.linked = True
        try:
            # root port (gen4 x8) -> the card's own switch (gen5 x16) -> the card (gen5 x16), which is the
            # the test system's actual topology and the reason the card's own link overstates it.
            self.port = os.path.join(root, "pci", "0000:00:01.0")
            self.switch = os.path.join(self.port, "0000:01:00.0")
            self.pci = os.path.join(self.switch, "0000:02:00.0")
            os.makedirs(self.pci)
            for d, speed, width in ((self.port, "16.0 GT/s PCIe", "8"),
                                    (self.switch, "32.0 GT/s PCIe", "16")):
                for n, v in (("max_link_speed", speed), ("max_link_width", width),
                             ("current_link_speed", speed), ("current_link_width", width)):
                    with open(os.path.join(d, n), "w", encoding="utf-8") as fh:
                        fh.write(v + "\n")
            os.symlink(self.pci, link)
        except (OSError, NotImplementedError):
            self.linked = False
            self.pci = link
            os.makedirs(self.pci, exist_ok=True)

        for name, value in (
            ("mem_info_vram_total", "34208743424"),
            ("mem_info_vram_used", "59912192"),
            ("mem_info_gtt_total", "17179869184"),
            ("max_link_speed", "32.0 GT/s PCIe"),
            ("max_link_width", "16"),
            ("current_link_speed", "32.0 GT/s PCIe"),
            ("current_link_width", "16"),
            ("device", "0x7551"),
        ):
            with open(os.path.join(self.pci, name), "w", encoding="utf-8") as fh:
                fh.write(value + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_one_gpu_and_skips_the_cpu_node(self):
        rows = dg.amd_gpus(sys_drm=self.drm, sys_kfd=self.kfd)
        self.assertEqual(1, len(rows), "the CPU KFD node must not be reported as a GPU")
        self.assertEqual("gfx1201", rows[0]["arch"])

    def test_reads_memory_and_link_from_the_drm_device(self):
        g = dg.amd_gpus(sys_drm=self.drm, sys_kfd=self.kfd)[0]
        self.assertEqual(34208743424, g["vram_total_bytes"])
        self.assertEqual(34208743424 - 59912192, g["vram_free_bytes"])
        self.assertEqual(17179869184, g["gtt_total_bytes"])

    def test_decodes_the_link_rate_into_a_generation(self):
        """sysfs states GT/s and never a generation, so the mapping is ours."""
        if not self.linked:
            self.skipTest("this filesystem allows neither ':' in a name nor symlinks")
        g = dg.amd_gpus(sys_drm=self.drm, sys_kfd=self.kfd)[0]
        self.assertIn(g["pcie_gen_max"], (4, 5))

    def test_reports_the_upstream_link_not_the_cards_own(self):
        """The test system's R9700 says gen5 x16 -- true, but that is its link to its own internal
        switch.  The root port above it is gen4 x8, and that is what limits traffic to host memory.
        Reading the card's own link overstates it 4x, and card ranking reads this field."""
        if not self.linked:
            self.skipTest("this filesystem allows neither ':' in a name nor symlinks")
        g = dg.amd_gpus(sys_drm=self.drm, sys_kfd=self.kfd)[0]
        self.assertEqual(4, g["pcie_gen_max"], "the root port generation must win")
        self.assertEqual(8, g["pcie_width_max"], "the root port width must win")

    def test_recovers_the_bus_id_through_the_render_node(self):
        if not self.linked:
            self.skipTest("this filesystem allows neither ':' in a name nor symlinks")
        g = dg.amd_gpus(sys_drm=self.drm, sys_kfd=self.kfd)[0]
        self.assertEqual("0000:02:00.0", g["pci_bus_id"])


class TestRocmDiscovery(unittest.TestCase):
    def test_a_stray_library_does_not_become_a_root(self):
        """Regression, found by running the probe on the test system: ldconfig there
        lists a distro libamdhip64 under /usr/lib, and the search returned that
        directory as a ROCm root -- which SILENCED the blocker that should have
        told the user where to point.  A root without a compiler is not a root.
        """
        self.assertIsNone(dt.rocm_root_from_ldconfig(fixture("sample-ldconfig-amdhip.txt")))

    def test_a_directory_without_the_compiler_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(dt.is_rocm_root(tmp))
            os.makedirs(os.path.join(tmp, "bin"))
            with open(os.path.join(tmp, "bin", "hipcc"), "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\n")
            self.assertTrue(dt.is_rocm_root(tmp))

    def test_none_is_not_a_root(self):
        self.assertFalse(dt.is_rocm_root(None))


class TestCuda(unittest.TestCase):
    def test_version_from_recorded_output(self):
        self.assertEqual("12.4", dt.parse_nvcc_version(fixture("sample-nvcc-version.txt")))

    def test_unparseable_version_is_none(self):
        self.assertIsNone(dt.parse_nvcc_version("nvcc: not a real thing"))


class TestSycl(unittest.TestCase):
    def test_only_level_zero_gpu_devices_count(self):
        text = (
            "[opencl:cpu][opencl:0] Intel(R) OpenCL, 13th Gen Core 3.0\n"
            "[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Arc A770 3.0\n"
            "[level_zero:gpu][level_zero:0] Intel(R) Level-Zero, Arc A770 1.3\n"
        )
        rows = dg.parse_sycl_ls(text)
        self.assertEqual(1, len(rows), "the same card must not be counted once per backend")


class TestMeminfo(unittest.TestCase):
    def test_recorded_values(self):
        total, avail = dh.parse_meminfo(fixture("sample-proc-meminfo.txt"))
        self.assertEqual(33554432 * 1024, total)
        self.assertGreater(avail, 0)
        self.assertLess(avail, total)

    def test_available_is_preferred_over_free(self):
        """MemFree excludes reclaimable page cache; sizing a RAM tier off it
        under-uses the machine by gigabytes."""
        total, avail = dh.parse_meminfo("MemTotal: 100 kB\nMemFree: 10 kB\nMemAvailable: 80 kB\n")
        self.assertEqual(80 * 1024, avail)

    def test_falls_back_to_free_when_available_is_absent(self):
        _total, avail = dh.parse_meminfo("MemTotal: 100 kB\nMemFree: 10 kB\n")
        self.assertEqual(10 * 1024, avail)


class TestMounts(unittest.TestCase):
    def test_read_only_mounts_are_dropped(self):
        """A desktop mounts dozens of squashfs images read-only.  Listing them
        made the first report dozens of lines long and unreadable; the sample keeps six."""
        rows = dh.parse_mounts(fixture("sample-proc-mounts.txt"))
        self.assertEqual([], [r for r in rows if r[2] == "squashfs"])
        self.assertLess(len(rows), 15, "the report must stay scannable by hand")

    def test_real_storage_is_kept(self):
        mounts = {m for _d, m, _f in dh.parse_mounts(fixture("sample-proc-mounts.txt"))}
        self.assertIn("/", mounts)
        self.assertIn("/srv/data", mounts)

    def test_dev_shm_is_kept_because_the_channel_lives_there(self):
        rows = dh.parse_mounts(fixture("sample-proc-mounts.txt"))
        shm = [r for r in rows if r[1] == "/dev/shm"]
        self.assertEqual(1, len(shm))
        self.assertEqual("tmpfs", shm[0][2], "it must be identifiable as not storage")

    def test_pseudo_filesystems_are_dropped(self):
        rows = dh.parse_mounts("proc /proc proc rw 0 0\nsysfs /sys sysfs rw 0 0\n")
        self.assertEqual([], rows)


class TestRotational(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="seabraid-block-")
        self.sys = self.tmp.name
        for disk, rota in (("sda", "1"), ("nvme0n1", "0")):
            q = os.path.join(self.sys, "class/block", disk, "queue")
            os.makedirs(q)
            with open(os.path.join(q, "rotational"), "w", encoding="utf-8") as fh:
                fh.write(rota + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _partition(self, disk: str, part: str) -> None:
        d = os.path.join(self.sys, "class/block", disk, part)
        os.makedirs(d)
        with open(os.path.join(d, "partition"), "w", encoding="utf-8") as fh:
            fh.write("1\n")
        os.symlink(d, os.path.join(self.sys, "class/block", part))

    def test_whole_disks(self):
        self.assertIs(True, dh.rotational_for("/dev/sda", self.sys))
        self.assertIs(False, dh.rotational_for("/dev/nvme0n1", self.sys))

    def test_partition_resolves_to_its_parent_disk(self):
        try:
            self._partition("sda", "sda1")
        except (OSError, NotImplementedError):
            self.skipTest("this filesystem does not allow symlinks")
        self.assertIs(True, dh.rotational_for("/dev/sda1", self.sys))

    def test_unresolvable_devices_stay_unknown_rather_than_guessing_ssd(self):
        """A wrong False here puts a 50 GB model on a hard disk, so every device
        whose chain cannot be followed must answer None."""
        for dev in ("/dev/mapper/vg-root", "/dev/dm-0", "/dev/md0", "/dev/loop3", "tmpfs"):
            self.assertIsNone(dh.rotational_for(dev, self.sys), dev)

    def test_absent_device_is_unknown(self):
        self.assertIsNone(dh.rotational_for("/dev/sdz", self.sys))


class TestTtmAndOsRelease(unittest.TestCase):
    def test_pages_are_converted_to_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "module/ttm/parameters")
            os.makedirs(d)
            with open(os.path.join(d, "pages_limit"), "w", encoding="utf-8") as fh:
                fh.write("4194304\n")
            got = dh.probe_ttm_pages_limit(tmp)
        self.assertEqual(17179869184, got, "the test system's cap, in bytes rather than pages")

    def test_missing_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(dh.probe_ttm_pages_limit(tmp))

    def test_os_pretty_name(self):
        self.assertEqual("Example Linux 1.0", dh.parse_os_release('PRETTY_NAME="Example Linux 1.0"\n'))


class TestHostReport(unittest.TestCase):
    def _fs(self, mount, rotational=False):
        return detect.Filesystem(
            mount=mount, device="/dev/x", fstype="ext4",
            total_bytes=GIB, free_bytes=GIB, rotational=rotational,
        )

    def test_require_ok_is_silent_without_blockers(self):
        detect.HostReport().require_ok()

    def test_require_ok_raises_and_names_the_fix(self):
        r = detect.HostReport(
            blockers=(detect.Blocker(where="toolkit:rocm", what="no ROCm", fix="pass --rocm-path"),)
        )
        with self.assertRaises(detect.DetectionRefused) as caught:
            r.require_ok()
        self.assertIn("--rocm-path", str(caught.exception))

    def test_filesystem_holding_picks_the_longest_prefix(self):
        r = detect.HostReport(filesystems=(self._fs("/"), self._fs("/srv/data", True)))
        self.assertEqual("/srv/data", r.filesystem_holding("/srv/data/models/x.gguf").mount)
        self.assertEqual("/", r.filesystem_holding("/srv/x.gguf").mount)

    def test_filesystem_holding_does_not_match_a_partial_name(self):
        r = detect.HostReport(filesystems=(self._fs("/"), self._fs("/srv/data")))
        self.assertEqual("/", r.filesystem_holding("/srv/data2/x").mount)


class TestPcieChain(unittest.TestCase):
    def test_the_narrowest_hop_wins_not_the_first(self):
        links = [(5, 16, 5, 16), (4, 8, 4, 8)]        # card behind a switch, then the root port
        self.assertEqual((4, 8, 4, 8), dpci.narrowest_link(links))

    def test_width_is_weighed_against_generation_not_compared_alone(self):
        """gen5 x8 carries more than gen3 x16, so the comparison has to be bandwidth."""
        self.assertEqual((3, 16, 3, 16), dpci.narrowest_link([(5, 8, 5, 8), (3, 16, 3, 16)]))

    def test_hops_that_do_not_state_both_are_skipped_not_assumed_wide(self):
        self.assertEqual((4, 8, None, None), dpci.narrowest_link([(None, 16, None, None), (4, 8, None, None)]))

    def test_no_usable_hop_is_none(self):
        self.assertIsNone(dpci.narrowest_link([]))
        self.assertIsNone(dpci.narrowest_link([(None, None, None, None)]))

    def test_the_walk_stops_at_the_first_non_pci_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = os.path.join(tmp, "0000:00:01.0", "0000:02:00.0")
            try:
                os.makedirs(dev)
            except OSError:
                self.skipTest("this filesystem does not allow ':' in a name")
            for d, speed, width in ((os.path.dirname(dev), "16.0 GT/s PCIe", "8"),
                                    (dev, "32.0 GT/s PCIe", "16")):
                for n, v in (("max_link_speed", speed), ("max_link_width", width)):
                    with open(os.path.join(d, n), "w", encoding="utf-8") as fh:
                        fh.write(v + "\n")
            links = dpci.pci_chain_links(dev)
        self.assertEqual(2, len(links), "both hops, and nothing above the non-PCI parent")


class TestRocmFromTool(unittest.TestCase):
    """A TheRock tarball unpacked outside every conventional location and surfaced by symlinking its tools
    into ~/.local/bin -- which is how the test system does it, and what the first version missed."""

    def test_follows_a_symlinked_tool_back_to_its_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = os.path.join(tmp, "rocm-7")
            os.makedirs(os.path.join(real, "bin"))
            for name in ("hipcc", "amd-smi"):
                with open(os.path.join(real, "bin", name), "w", encoding="utf-8") as fh:
                    fh.write("#!/bin/sh\n")
            shim = os.path.join(tmp, "localbin", "amd-smi")
            os.makedirs(os.path.dirname(shim))
            try:
                os.symlink(os.path.join(real, "bin", "amd-smi"), shim)
            except (OSError, NotImplementedError):
                self.skipTest("this filesystem does not allow symlinks")
            got = dt.rocm_root_from_tool(which=lambda t: shim if t == "amd-smi" else None)
        self.assertEqual(real, got)

    def test_a_tool_whose_root_has_no_compiler_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "bin"))
            tool = os.path.join(tmp, "bin", "rocminfo")
            with open(tool, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\n")
            self.assertIsNone(dt.rocm_root_from_tool(which=lambda t: tool if t == "rocminfo" else None))

    def test_nothing_on_path_is_none(self):
        self.assertIsNone(dt.rocm_root_from_tool(which=lambda t: None))

if __name__ == "__main__":
    unittest.main()
