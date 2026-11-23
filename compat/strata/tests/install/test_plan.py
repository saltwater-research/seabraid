# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for install/plan.py.  The detect report is a LOCAL stand-in (duck-typed, same field names as detect's
final contract); no GPU, no Linux needed.  Sizes are in GiB via G()."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field, replace
from typing import Optional

import pytest

from install import _plan_shm as SHM
from install import plan as P

G = lambda x: int(x * (1 << 30))  # noqa: E731


@dataclass(frozen=True)
class Gpu:
    index: int
    vendor: str
    name: str
    arch: Optional[str]
    vram_total_bytes: Optional[int]
    vram_free_bytes: Optional[int]
    gtt_total_bytes: Optional[int] = None
    pcie_gen_max: Optional[int] = 4
    pcie_width_max: Optional[int] = 16
    pcie_gen_current: Optional[int] = 1
    pcie_width_current: Optional[int] = 1
    uuid: Optional[str] = None


@dataclass(frozen=True)
class Toolkit:
    vendor: str
    kind: str
    root: Optional[str]
    version: Optional[str]
    compiler: Optional[str]
    host_compiler: Optional[str] = None
    host_compiler_tried: tuple = ()
    lib_dirs: tuple = ()


@dataclass(frozen=True)
class Filesystem:
    mount: str
    free_bytes: Optional[int]
    rotational: Optional[bool]
    fstype: str = "ext4"


@dataclass(frozen=True)
class Blocker:
    where: str          # detect's grammar: "<kind>:<ref>" (toolkit:rocm, gpu:<vendor>:<index>, fs:<mount>) or bare "fs"
    what: str
    fix: str

    @property
    def kind(self) -> str:
        return self.where.split(":", 1)[0]

    @property
    def ref(self) -> str:
        return self.where.split(":", 1)[1] if ":" in self.where else ""


@dataclass(frozen=True)
class Host:
    gpus: tuple
    toolkits: tuple
    filesystems: tuple
    ram_total_bytes: Optional[int] = G(30)
    ram_available_bytes: Optional[int] = G(20)
    ttm_pages_limit_bytes: Optional[int] = G(15.13)
    memlock_bytes: Optional[int] = 8 << 20
    blockers: tuple = ()


R9700 = Gpu(0, "amd", "R9700", "gfx1201", G(32), G(31.5), gtt_total_bytes=G(15.13))
RTX3060 = Gpu(0, "nvidia", "RTX 3060", "sm_86", G(12), G(11.5), pcie_gen_max=3)
KITS = (Toolkit("amd", "rocm", "/opt/r", "7.13", "/opt/r/llvm/bin/clang++", lib_dirs=("/opt/r/lib",)),
        Toolkit("nvidia", "cuda", "/opt/c", "12.4", "/opt/c/bin/nvcc", host_compiler="/usr/bin/gcc-13"))
FS = (Filesystem("/", G(44), False), Filesystem("/srv/data", G(29), True, "ext4"),
      Filesystem("/dev/shm", G(15), None, "tmpfs"))


def host(**kw) -> Host:
    base = dict(gpus=(R9700, RTX3060), toolkits=KITS, filesystems=FS)
    base.update(kw)
    return Host(**base)


def model(expert=50.0, dense=3.0, layers=48, experts=512, hidden=2560, blob=1510400) -> P.ModelFacts:
    return P.ModelFacts(layers, experts, G(expert), G(dense), hidden, blob)


def test_all_vram_roles_builds_and_no_hardcoded_shapes():
    pl = P.make_plan(host(), model(expert=33, layers=7, experts=13), "/models/m.gguf")
    assert pl.mode == "all-vram" and pl.ready
    m, h = pl.main(), pl.helpers()
    assert (m.vendor, m.arch, h[0].vendor, h[0].shm, h[0].device_id) == ("amd", "gfx1201", "nvidia", "/seabraid0", 0)
    assert pl.tiers.ram_bytes == 0 and pl.tiers.nvme_bytes == 0
    assert sum(pl.tiers.vram_bytes.values()) == G(33)
    assert (pl.model.layers, pl.model.experts) == (7, 13)
    by = {b.vendor: b for b in pl.builds}
    assert by["amd"].targets == ("strata", "seabraid-helper") and by["amd"].archs == ("gfx1201",)
    assert by["nvidia"].targets == ("seabraid-helper",) and by["nvidia"].archs == ("sm_86",)
    assert by["nvidia"].host_compiler == "/usr/bin/gcc-13"


def test_two_cards_build_the_helper_binary_for_every_vendor_so_bench_can_time_the_main_card():
    pl = P.make_plan(host(), model(expert=33, layers=7, experts=13), "/models/m.gguf")
    assert all("seabraid-helper" in b.targets for b in pl.builds)


def test_nvidia_only_single_card_builds_strata_only():
    pl = P.make_plan(host(gpus=(RTX3060,), filesystems=FS), model(expert=5), "/models/m.gguf")
    assert [b.targets for b in pl.builds] == [("strata",)] and pl.helpers() == ()


def test_tiered_ram_then_nvme_on_model_disk_with_gtt_blocked():
    pl = P.make_plan(host(), model(), "/data/models/m.gguf")      # lives on "/" (NVMe)
    assert pl.mode == "tiered"
    assert pl.tiers.ram_bytes == G(12)                               # 20 avail - max(8, 25% of 30)
    assert pl.tiers.nvme_mount == "/" and pl.tiers.nvme_bytes > 0
    (b,) = pl.blocked
    assert b.tier == "ram" and "ttm.pages_limit" in b.remedy and not pl.ready
    assert "BLOCKED ram" in pl.explain() and "not applied" in pl.explain()


def test_gtt_uses_smaller_of_card_gtt_and_ttm_and_refuses_unknown():
    big = G(30)
    ok = P.make_plan(host(gpus=(replace(R9700, gtt_total_bytes=big), RTX3060), ttm_pages_limit_bytes=big), model(), "/m.gguf")
    assert ok.ready and not ok.blocked
    small_ttm = P.make_plan(host(gpus=(replace(R9700, gtt_total_bytes=big), RTX3060), ttm_pages_limit_bytes=G(10)), model(), "/m.gguf")
    assert small_ttm.blocked
    unknown = P.make_plan(host(gpus=(replace(R9700, gtt_total_bytes=big), RTX3060), ttm_pages_limit_bytes=None), model(), "/m.gguf")
    assert unknown.blocked and "unknown" in unknown.blocked[0].reason


GTT_IQ3S = 17179869184      # 16 GiB, a synthetic cap


def test_gtt_margin_admits_the_verified_iq3s_run_and_blocks_a_15_5_gib_tier():
    amd = replace(R9700, gtt_total_bytes=GTT_IQ3S)
    roomy = dict(gpus=(amd, RTX3060), ttm_pages_limit_bytes=GTT_IQ3S, ram_total_bytes=G(40), ram_available_bytes=G(30))
    cards = sum(P.make_plan(host(**roomy), model(expert=60), "/data/m.gguf").tiers.vram_bytes.values())    # what the two cards hold
    exact = lambda tier: P.ModelFacts(48, 512, cards + tier, G(3), 2560, 1510400)  # noqa: E731
    ok = P.make_plan(host(**roomy), exact(10 * (1 << 30)), "/data/m.gguf")      # exactly a 10 GiB RAM tier
    assert ok.tiers.ram_bytes == G(10) and ok.ready and not ok.blocked
    big = P.make_plan(host(**roomy), exact(int(15.5 * (1 << 30))), "/data/m.gguf")       # 15.5 GiB tier needs 20.5 GiB of GTT
    assert big.tiers.ram_bytes == int(15.5 * (1 << 30)) and big.blocked and not big.ready and "ttm.pages_limit" in big.blocked[0].remedy


def test_plan_exposes_kv_reserve_and_assumed_context():
    pl = P.make_plan(host(), model(expert=33), "/m.gguf")
    o = P.PlanOptions()
    assert pl.mode == "all-vram" and pl.assumed_max_context == 131072 and o.assumed_max_context is None
    tiered = P.make_plan(host(), model(), "/data/m.gguf")
    assert tiered.mode == "tiered" and tiered.assumed_max_context == 32768
    assert pl.kv_reserve_bytes == {("amd", 0): P.main_reserve(o, G(3), None, 131072), ("nvidia", 0): o.helper_reserve_bytes}
    assert tiered.kv_reserve_bytes[("amd", 0)] == P.main_reserve(o, G(3), None, 32768)      # the KV part follows the context
    for m in (model(expert=33), model()):                       # explicit value overrides in either mode
        assert P.make_plan(host(), m, "/data/m.gguf", opts=P.PlanOptions(assumed_max_context=65536)).assumed_max_context == 65536


def test_nvidia_main_needs_no_gtt():
    pl = P.make_plan(host(gpus=(RTX3060,)), model(expert=20), "/m.gguf")
    assert not pl.blocked and pl.tiers.ram_bytes > 0


def test_large_model_on_rotational_disk_refuses_and_names_nothing_to_move_to():
    pl = P.make_plan(host(), model(), "/srv/data/models/m.gguf")
    assert pl.mode == "infeasible" and not pl.ready
    why = " ".join(pl.refusals)
    assert "rotational" in why and "/srv/data" in why
    assert "largest non-rotational filesystem is / with 44 GiB" in why     # 53 GiB * 1.1 does not fit
    assert pl.tiers.nvme_bytes == 0


def test_rotational_disk_recommends_nvme_fs_that_fits():
    h = host(gpus=(replace(R9700, vram_free_bytes=G(25)), RTX3060), ram_available_bytes=G(9))
    pl = P.make_plan(h, model(expert=36.5, dense=1), "/srv/data/m.gguf")
    assert pl.mode == "infeasible" and "move the model to /" in pl.refusals[0]


def test_model_fitting_vram_plus_ram_is_fine_on_rotational_disk():
    pl = P.make_plan(host(), model(expert=45), "/srv/data/m.gguf")
    assert pl.mode == "tiered" and pl.tiers.nvme_bytes == 0 and pl.tiers.nvme_mount is None


def test_unknown_rotational_never_gets_a_tier():
    fs = (Filesystem("/", G(44), None), FS[2])
    pl = P.make_plan(host(filesystems=fs), model(), "/m.gguf")
    assert pl.mode == "infeasible" and "unknown type" in pl.refusals[0]


def test_longest_prefix_mount_wins():
    fs = (Filesystem("/", G(44), True), Filesystem("/fast", G(100), False), FS[2])
    pl = P.make_plan(host(filesystems=fs), model(), "/fast/models/m.gguf")
    assert pl.tiers.nvme_mount == "/fast"


def test_rocm_blocker_refuses_amd_role_and_names_fix():
    bl = Blocker("toolkit:rocm", "ROCm root not found", "set ROCM_PATH to your ROCm install")
    pl = P.make_plan(host(blockers=(bl,)), model(expert=5), "/m.gguf")
    assert pl.main().vendor == "nvidia"
    reason = dict(pl.excluded)[("amd", 0)]
    assert "set ROCM_PATH to your ROCm install" in reason and "ROCm root not found" in reason
    assert "set ROCM_PATH" in pl.explain()


def test_cuda_without_host_compiler_is_refused_listing_tried():
    kits = (KITS[0], replace(KITS[1], host_compiler=None, host_compiler_tried=("gcc", "gcc-13")))
    pl = P.make_plan(host(toolkits=kits), model(expert=5), "/m.gguf")
    assert pl.main().vendor == "amd" and pl.helpers() == ()
    assert "gcc, gcc-13" in dict(pl.excluded)[("nvidia", 0)]


def test_ranking_uses_max_pcie_not_current():
    a = Gpu(0, "nvidia", "A", "sm_86", G(12), G(11.5), pcie_gen_max=3, pcie_width_max=8, pcie_gen_current=4, pcie_width_current=16)
    b = Gpu(1, "nvidia", "B", "sm_86", G(12), G(11.5), pcie_gen_max=4, pcie_width_max=16, pcie_gen_current=1, pcie_width_current=8)
    pl = P.make_plan(host(gpus=(a, b)), model(expert=5), "/m.gguf")
    assert pl.main().gpu_index == 1 and pl.ranking == "heuristic"


def test_card_bench_overrides_heuristic():
    pl = P.make_plan(host(), model(expert=5), "/m.gguf", bench_lead={("amd", 0): 26.7, ("nvidia", 0): 24.0})
    assert pl.ranking == "bench" and pl.main().vendor == "nvidia"
    partial = P.make_plan(host(), model(expert=5), "/m.gguf", bench_lead={("amd", 0): 26.7, ("nvidia", 0): None})
    assert partial.ranking == "heuristic" and partial.main().vendor == "amd"
    rows_only = P.make_plan(host(), model(expert=5), "/m.gguf", bench={("amd", 0): 90000.0, ("nvidia", 0): 148383.0})
    assert rows_only.ranking == "heuristic" and rows_only.rest_ranking == "bench" and rows_only.main().vendor == "amd"


def test_unusable_vendor_and_unknown_vram_are_excluded_with_reasons():
    arc = Gpu(0, "intel", "Arc", "xe", G(16), G(16))
    odd = Gpu(0, "nvidia", "X", "sm_86", G(12), None)
    pl = P.make_plan(host(gpus=(R9700, arc, odd)), model(expert=5), "/m.gguf")
    ex = dict(pl.excluded)
    assert "no seabraid build" in ex[("intel", 0)] and "free VRAM unknown" in ex[("nvidia", 0)]


def test_channel_bytes_against_the_box_measurements():
    # REAL facts of the IQ2_XS GGUF used in testing (read with read_model_facts): L=48, E=512, H=2560, blob_max=1,510,400.
    f = (48, 512, 2560, 1510400)
    # MEASURED: `ls -la /dev/shm/seabraid0` of a window-16 helper on channel v3 (no tier-op block) = 27,672,576 B.
    assert P.channel_bytes(*f, 16, tier_ops=False, version=4) == 27_672_576
    # window 16, v4 = the v3 figure + one 4 KiB page for the 64 x 20 B tier-op block (patch 0001 Layout::of).
    assert P.channel_bytes(*f, 16, version=4) == 27_672_576 + 4096 == 27_676_672
    # DERIVED, not measured here: window 8 (cap 80).  It equals the 26,038,272 B once seen from a CUDA-default (window 8) compile.
    assert P.channel_bytes(*f, 8, version=4) == 26_038_272
    assert P.channel_bytes(48, 512, None, 1510400, 8) is None and P.channel_bytes(48, 512, 2560, None, 8) is None
    assert P.channel_bytes(*f, 0) is None
    # version 5 is the v4 layout under the neutral STRXHLP tag (an upstream branch): it must size exactly like v4
    assert P.channel_bytes(*f, 16, version=5) == P.channel_bytes(*f, 16, version=4)
    # v6 adds two page-aligned regions after the tier ops: weights (cap floats) and the spec (layers * 112 + 64 * 512).
    assert P.channel_bytes(*f, 16) == 27_676_672 + 4096 + 40960 == 27_721_728
    assert P.channel_bytes(*f, 8) == 26_038_272 + 4096 + 40960 == 26_083_328
    assert P.channel_bytes(48, 512, 2560, 0, 8) is None, "a Qwen caller that does not know blob_max still gets None"


GOLDEN = __import__("pathlib").Path(__file__).resolve().parents[1] / "channel_v6_golden.txt"


def _golden():
    head, rows = None, []
    for line in GOLDEN.read_text().splitlines():
        f = line.split()
        if f and f[0] == "header_bytes":
            head = int(f[1])
        elif f:
            rows.append(tuple(int(v) for v in f))
    return head, rows


def test_v6_formula_is_pinned_to_layout_of_through_the_golden_file():
    # tests/channel_v6_golden.txt is written by seabraid_layout_print (the real compiler + the real Layout::of) and
    # tests/channel_v6_test.cpp checks it from the C++ side, so a Layout::of change fails there and a mirror drift fails here.
    head, rows = _golden()
    assert SHM.CHANNEL_HEADER_BYTES == head == 2752
    assert len(rows) >= 3
    for hidden, window, k, layers, experts, blob, total in rows:
        assert P.channel_bytes(layers, experts, hidden, blob, window, cap_pairs_per_token=k) == total, (hidden, window, k)


def test_window_is_the_main_backends_default_and_propagates_to_helpers():
    amd_main = P.make_plan(host(), model(expert=33), "/m.gguf")                       # the test system: HIP main + CUDA helper
    assert amd_main.main().vendor == "amd" and amd_main.window_max_t == 16 and amd_main.shm_bytes_per_helper == 27_721_728
    nv_main = P.make_plan(host(), model(expert=33), "/m.gguf", bench_lead={("amd", 0): 2.0, ("nvidia", 0): 1.0})
    assert nv_main.main().vendor == "nvidia" and nv_main.window_max_t == 8 and nv_main.shm_bytes_per_helper == 26_083_328
    over = P.make_plan(host(), model(expert=33), "/m.gguf", opts=P.PlanOptions(window_max_t=12))
    assert over.window_max_t == 12 and over.shm_bytes_per_helper == P.channel_bytes(48, 512, 2560, 1510400, 12)


def test_plan_checks_shm_free_against_the_sum_over_helpers():
    pl = P.make_plan(host(), model(expert=33), "/m.gguf")
    assert not hasattr(pl, "shm_chunk_tokens")
    two = Gpu(1, "nvidia", "N2", "sm_86", G(12), G(11.5))
    fs = lambda free: (FS[0], Filesystem("/dev/shm", free, None, "tmpfs"))  # noqa: E731
    ok = P.make_plan(host(gpus=(R9700, RTX3060, two), filesystems=fs(2 * 27_721_728)), model(expert=33), "/m.gguf")
    assert ok.mode != "infeasible" and len(ok.helpers()) == 2
    short = P.make_plan(host(gpus=(R9700, RTX3060, two), filesystems=fs(2 * 27_721_728 - 1)), model(expert=33), "/m.gguf")
    assert short.mode == "infeasible" and "/dev/shm" in short.refusals[0] and "2 helper" in short.refusals[0]
    assert P.make_plan(host(filesystems=(FS[0],)), model(expert=33), "/m.gguf").mode == "infeasible"
    assert P.make_plan(host(), model(expert=33, blob=None), "/m.gguf").mode == "infeasible"


def test_same_index_on_two_vendors_gets_two_distinct_entries():
    """Gpu.index is per vendor: amd 0 and nvidia 0 coexist.  A mapping keyed by index alone would merge them."""
    assert R9700.index == RTX3060.index == 0
    pl = P.make_plan(host(), model(expert=33), "/m.gguf")
    v = pl.tiers.vram_bytes
    assert set(v) == {("amd", 0), ("nvidia", 0)} and v[("amd", 0)] != v[("nvidia", 0)]
    assert set(pl.kv_reserve_bytes) == set(v) and pl.kv_reserve_bytes[("amd", 0)] != pl.kv_reserve_bytes[("nvidia", 0)]
    lines = [l for l in pl.explain().splitlines() if "experts on card" in l]
    figs = [l.split("experts on card: ")[1].split(" of ")[0] for l in lines]
    assert len(figs) == 2 and figs[0] != figs[1], lines
    assert "amd gpu0" in lines[0] and "nvidia gpu0" in lines[1]
    excl = P.make_plan(host(blockers=(Blocker("toolkit:rocm", "no rocm", "fix it"),)), model(expert=5), "/m.gguf")
    assert list(dict(excl.excluded)) == [("amd", 0)]


def test_gpu_blocker_matches_only_its_card_and_other_kinds_do_not_leak():
    on_nv = Blocker("gpu:nvidia:0", "driver too old", "update the NVIDIA driver")
    pl = P.make_plan(host(blockers=(on_nv,)), model(expert=5), "/m.gguf")
    assert pl.helpers() == () and "update the NVIDIA driver" in dict(pl.excluded)[("nvidia", 0)]
    other = P.make_plan(host(blockers=(Blocker("fs:/srv/data", "x", "y"),)), model(expert=5), "/m.gguf")
    assert len(other.helpers()) == 1 and other.excluded == ()


def test_fs_blocker_fix_is_carried_into_the_nvme_refusal():
    fs = (Filesystem("/", G(44), None), FS[2])
    bl = Blocker("fs:/", "cannot tell if / is rotational", "run lsblk as root once")
    pl = P.make_plan(host(filesystems=fs, blockers=(bl,)), model(), "/m.gguf")
    assert "run lsblk as root once" in pl.refusals[0]


def test_arch_taxonomy_carries_the_generic_pragma():
    from seabraid import ranking as R
    src = open(R.__file__, encoding="utf-8").read().splitlines()
    line = next(l for l in src if l.startswith("_ARCH_CLASS"))
    assert line.endswith("# seabraid-generic: vendor-wide taxonomy, not this machine")


def test_ram_unknown_means_no_ram_tier():
    pl = P.make_plan(host(ram_available_bytes=None), model(), "/data/m.gguf")
    assert pl.tiers.ram_bytes == 0 and any("RAM size unknown" in n for n in pl.notes)


# ---- GGUF facts ------------------------------------------------------------------------------------------------
def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, arch, layers, experts, hidden, tensors):
    """tensors: [(name, nbytes)] laid out back to back at 32-byte alignment."""
    kv = [(f"general.architecture", 8, _s(arch)), (f"{arch}.block_count", 4, struct.pack("<I", layers)),
          (f"{arch}.expert_count", 4, struct.pack("<I", experts)), (f"{arch}.embedding_length", 4, struct.pack("<I", hidden)),
          ("tokenizer.tokens", 9, struct.pack("<IQ", 8, 2) + _s("a") + _s("bb")), ("x.f", 9, struct.pack("<IQ", 6, 3) + b"\0" * 12)]
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv:
        out += _s(k) + struct.pack("<I", t) + v
    off = 0
    for name, n in tensors:
        out += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", n) + struct.pack("<I", 0) + struct.pack("<Q", off)
        off += -(-n // 32) * 32
    out += b"\0" * (-len(out) % 32)
    for name, n in tensors:
        out += b"\0" * (-(-n // 32) * 32)
    path.write_bytes(out)


def test_read_model_facts_from_gguf_headers(tmp_path):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    write_gguf(a, "toy", 5, 9, 64, [("blk.0.ffn_gate_exps.weight", 3200), ("blk.0.attn_q.weight", 640)])
    write_gguf(b, "toy", 5, 9, 64, [("blk.1.ffn_up_exps.weight", 1600)])
    f = P.read_model_facts([str(a), str(b)])
    assert (f.layers, f.experts, f.hidden) == (5, 9, 64)
    assert f.expert_bytes == 3200 + 1600 and f.dense_bytes == 640
    # layer 0 holds 3200 B over 9 experts, layer 1 holds 1600: the largest blob is ceil(3200 / 9)
    assert f.blob_max_bytes == -(-3200 // 9)


def test_non_moe_or_garbage_gguf_is_refused(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"nope")
    with pytest.raises(P.ModelError):
        P.read_model_facts([str(bad)])
    dense = tmp_path / "d.gguf"
    write_gguf(dense, "toy", 5, 9, 64, [("blk.0.attn_q.weight", 640)])
    with pytest.raises(P.ModelError):
        P.read_model_facts([str(dense)])


# ---- the test system, IQ2_XS: what the main card really holds, and that a helper holds only experts ----------------
MIB = 1 << 20
# The REAL tensor groups of the test system's IQ2_XS GGUF (MiB over 48 layers; the dense total read 2.873 GiB).  hc_* are bf16 in the file
# (n = bytes / 2); hc_attn_inject is F32 (n = bytes / 4).  Q8_0 on the card is what the engine holds on gfx12.
_GROUPS = {"attn_qkv": 475.6, "hc_attn_down": 300.0, "hc_attn_up": 300.0, "hc_ffn_down": 300.0, "hc_ffn_up": 300.0, "attn_gate": 283.1,
           "ssm_out": 282.1, "attn_q": 188.2, "ffn_gate_inp": 120.0, "attn_output": 98.4, "ple_key": 50.0, "ffn_down_shexp": 45.5,
           "ffn_gate_shexp": 42.0, "ffn_up_shexp": 41.2, "indexer.q_proj": 30.0, "ple_value": 12.5, "attn_v": 9.8, "attn_k": 8.9,
           "ssm_alpha": 8.4, "ssm_beta": 8.4, "indexer.k_proj": 7.5, "ssm_conv1d": 5.6, "small_rest": 8.6}


def host_tensors():
    t = [(f"blk.0.{g}.weight", int(m * MIB), int(m * MIB) // 2) for g, m in _GROUPS.items()]
    t += [("blk.0.hc_attn_inject.weight", int(3.8 * MIB), int(3.8 * MIB) // 4)]
    t += [(f"output_hc_{d}.weight", int(6.2 * MIB), int(6.2 * MIB) // 2) for d in ("down", "up")]
    t += [("blk.0.ffn_gate_exps.weight", 35_454_976_000, 1), ("output.weight", 322 * MIB, 1), ("token_embd.weight", 400 * MIB, 1)]
    return [t, [("per_layer_token_embd.weight", 28_800_000_000, 1)]]


HOST_MODEL = replace(P.facts_from_tensors(48, 512, 2560, host_tensors()), blob_max_bytes=1_510_400)
R9700_BOX = Gpu(0, "amd", "R9700", "gfx1201", G(32), G(31.86), gtt_total_bytes=G(15.13))
RTX3060_BOX = Gpu(0, "nvidia", "RTX 3060", "sm_86", G(12), G(11.4), pcie_gen_max=3)


def test_box_iq2xs_is_all_vram_with_the_measured_split():
    pl = P.make_plan(host(gpus=(R9700_BOX, RTX3060_BOX)), HOST_MODEL, "/m.gguf")
    assert pl.mode == "all-vram" and pl.ready and pl.tiers.ram_bytes == 0 and pl.tiers.nvme_bytes == 0
    per_expert = HOST_MODEL.expert_bytes / (48 * 512)
    main, helper = (pl.tiers.vram_bytes[("amd", 0)] / per_expert, pl.tiers.vram_bytes[("nvidia", 0)] / per_expert)
    # Measured: Strata's auto cache held 18,733 slots on the R9700, leaving 24,576 - 18,733 = 5,843 for the 3060.  The helper gets
    # the remainder, so its error is the main's error scaled by 18,733/5,843 = 3.2: 1% on the main (187 experts, 0.26 GiB) and 3% on the helper.
    assert abs(main - 18_733) <= 0.01 * 18_733, main
    assert abs(helper - 5_843) <= 0.03 * 5_843, helper
    assert round(main + helper) == 24_576                     # every expert is in VRAM


def test_counting_the_ple_table_as_dense_would_refuse_the_main_card():
    wrong = replace(HOST_MODEL, dense_bytes=32_560_092_160)           # the old bug: every non-expert tensor, PLE included
    pl = P.make_plan(host(gpus=(R9700_BOX, RTX3060_BOX)), wrong, "/m.gguf")
    assert "does not cover" in dict(pl.excluded)[("amd", 0)]


def test_helper_reserve_has_no_dense_kv_or_drafter():
    o = P.PlanOptions()
    heavy = replace(HOST_MODEL, dense_bytes=G(10), head_bytes=G(2))
    pl = P.make_plan(host(gpus=(R9700_BOX, RTX3060_BOX)), heavy, "/m.gguf", opts=replace(o, assumed_max_context=262144))
    helper = pl.helpers()[0]
    assert pl.kv_reserve_bytes[("nvidia", 0)] == o.helper_reserve_bytes == int(0.6 * (1 << 30))
    assert helper.vram_budget_bytes == RTX3060_BOX.vram_free_bytes - o.helper_reserve_bytes
    assert pl.kv_reserve_bytes[("amd", 0)] > 18 * (1 << 30)           # only the main carries the dense/KV parts


def test_ple_table_shares_a_shard_with_dense_weights(tmp_path):
    s1, s2 = tmp_path / "a.gguf", tmp_path / "b.gguf"
    write_gguf(s1, "toy", 5, 9, 64, [("blk.0.ffn_gate_exps.weight", 3200), ("blk.0.attn_q.weight", 640), ("token_embd.weight", 320),
                                     ("output.weight", 160)])
    write_gguf(s2, "toy", 5, 9, 64, [("per_layer_token_embd.weight", 6400), ("blk.1.attn_q.weight", 96)])
    f = P.read_model_facts([str(s1), str(s2)])
    assert f.dense_bytes == 640 + 96            # dense weights on both shards, including the PLE shard
    assert f.head_bytes == 160 and f.host_bytes == 320
    assert f.ple_bytes == 6400                  # only the table streams from disk
    assert f.expert_bytes == 3200


def test_dense_on_card_from_the_real_tensor_groups():
    f = HOST_MODEL
    assert f.ple_bytes == 28_800_000_000 and f.host_bytes == 400 * MIB and f.head_bytes == 322 * MIB
    o = P.PlanOptions()
    gfx12 = P.dense_on_card(o, f.hc_stored_bytes, f.hc_q8_bytes, f.dense_bytes, "rocm", "gfx1201")
    assert abs(gfx12 / MIB - 2378) <= 0.01 * 2378, gfx12 / MIB        # 2,112 MiB native projections + 266 MiB pack weights
    assert f.hc_q8_bytes < f.hc_stored_bytes * 0.55                    # Q8_0 is 8.5 bits against bf16's 16
    cuda = P.dense_on_card(o, f.hc_stored_bytes, f.hc_q8_bytes, f.dense_bytes, "cuda", "sm_86")
    assert cuda - gfx12 == f.hc_stored_bytes - f.hc_q8_bytes > 500 * MIB     # a CUDA main keeps bf16 (HC_Q8 is HIP-only)
    assert P.dense_on_card(replace(o, hc_q8=True), f.hc_stored_bytes, f.hc_q8_bytes, f.dense_bytes, "cuda", "sm_86") == gfx12


def test_without_the_q8_conversion_the_main_card_loses_about_four_hundred_experts():
    per = HOST_MODEL.expert_bytes / 24_576
    on = P.make_plan(host(gpus=(R9700_BOX, RTX3060_BOX)), HOST_MODEL, "/m.gguf")
    off = P.make_plan(host(gpus=(R9700_BOX, RTX3060_BOX)), HOST_MODEL, "/m.gguf", opts=P.PlanOptions(hc_q8=False))
    lost = (on.tiers.vram_bytes[("amd", 0)] - off.tiers.vram_bytes[("amd", 0)]) / per
    assert 350 < lost < 450, lost                   # ~563 MiB of bf16 hc_* weights / 1.44 MB per expert
