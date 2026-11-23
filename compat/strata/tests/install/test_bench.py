# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""_bench: parse the helper's real SEABRAID-CARD-BENCH line, and void the whole map on any single failure."""
from __future__ import annotations

from dataclasses import dataclass

from install import _bench as BN
from install.build import Captured, Runner

# Shape produced by strata-ext/src/card_bench.cpp format_line (percent-escaped name, %.9g numbers, nan for unmeasured).
# Decode figure is the RTX 3060 148,383 rows/s from docs/DESIGN-LOG.md; the other numbers are plausible, not measured.
OK_LINE = ("SEABRAID-CARD-BENCH v1 name=NVIDIA%20GeForce%20RTX%203060 vendor=cuda kind=helper status=ok "
           "free_vram_bytes=12348030976 layer=24 decode_rows_per_s=148383 decode_spread=0.0412 decode_iters=200 "
           "decode_rows_per_call=40 decode_rows_per_expert=4 prefill_rows_per_s=167364 prefill_spread=0.03 "
           "prefill_iters=100 prefill_rows_per_call=80 prefill_rows_per_expert=8")
UNMEASURED = ("SEABRAID-CARD-BENCH v1 name=AMD%20Radeon vendor=hip kind=helper status=unmeasured free_vram_bytes=0 layer=24 "
              "decode_rows_per_s=nan decode_spread=nan decode_iters=0 decode_rows_per_call=40 decode_rows_per_expert=4 "
              "prefill_rows_per_s=nan prefill_spread=nan prefill_iters=0 prefill_rows_per_call=80 prefill_rows_per_expert=8 "
              "note=open:%20no%20device")


@dataclass(frozen=True)
class Cand:
    vendor: str
    index: int
    device_id: int
    uuid: object
    name: str


@dataclass(frozen=True)
class Tk:
    lib_dirs: tuple = ()


class Report:
    def toolkit_for(self, vendor):
        return Tk()


class Fake(Runner):
    """A runner that executes nothing and replays scripted captures, recording the commands."""

    def __init__(self, *caps):
        super().__init__(dry_run=False, echo=lambda s: None)
        self.caps = list(caps)

    def run_capture(self, cmd, until, timeout):
        self.commands.append(cmd)
        return self.caps.pop(0)


A, B = Cand("amd", 0, 0, None, "R9700"), Cand("nvidia", 0, 0, "GPU-abc", "RTX 3090")
EXES = {"amd": "/w/build-amd/seabraid/seabraid-helper", "nvidia": "/w/build-nvidia/seabraid/seabraid-helper"}


def run(*caps, cands=(A, B), exes=EXES, runner=None):
    r = runner or Fake(*caps)
    return BN.measure(Report(), cands, exes, "/p", "/n.gguf", 48, 512, r), r


def line(rate):
    return OK_LINE.replace("decode_rows_per_s=148383", f"decode_rows_per_s={rate}")


def test_parse_reads_decode_rate_from_the_real_line_format():
    assert BN.parse("noise\n" + OK_LINE + "\nseabraid-helper: ready") == (148383.0, "")


def test_parse_reports_an_unmeasured_line_with_its_note():
    rate, why = BN.parse(UNMEASURED)
    assert rate is None and "not measured" in why and "open: no device" in why


def test_parse_without_a_line_or_with_garbage_is_none_with_a_reason():
    assert BN.parse("seabraid-helper: cannot open")[0] is None
    assert BN.parse("SEABRAID-CARD-BENCH v1 name=x decode_rows_per_s=abc")[0] is None
    assert BN.parse(line("0"))[0] is None


def test_measure_returns_a_rate_for_every_card_keyed_by_vendor_and_index():
    res, r = run(Captured(None, line(412), True, False), Captured(None, line(388), True, False))
    assert res.rates == {("amd", 0): 412.0, ("nvidia", 0): 388.0} and res.why == ""
    amd = r.commands[0]
    assert amd.argv[0] == EXES["amd"] and "--device" in amd.argv and amd.env["SEABRAID_STRATA_CARD_BENCH"] == "1"
    assert amd.env["HIP_VISIBLE_DEVICES"] == "0" and r.commands[1].env["CUDA_VISIBLE_DEVICES"] == "GPU-abc"
    assert amd.argv[amd.argv.index("--shm") + 1] != r.commands[1].argv[r.commands[1].argv.index("--shm") + 1]


def test_one_failed_card_voids_the_whole_map():
    res, _ = run(Captured(None, line(412), True, False), Captured(2, "seabraid-helper: expert layout: bad pack\n", False, False))
    assert res.rates is None and "RTX 3090" in res.why and "exited 2" in res.why and "bad pack" in res.why


def test_timeout_is_reported_as_a_timeout():
    res, _ = run(Captured(None, "loading\n", False, True))
    assert res.rates is None and "timed out" in res.why


def test_unmeasured_line_voids_the_map():
    res, _ = run(Captured(None, line(412), True, False), Captured(None, UNMEASURED, True, False))
    assert res.rates is None and "not measured" in res.why


def test_missing_helper_binary_voids_before_anything_runs():
    res, r = run(exes={"nvidia": EXES["nvidia"]})
    assert res.rates is None and "no seabraid-helper built yet (first install); rerun with --bench after the build, or with --no-build" in res.why and r.commands == []


def test_dry_run_executes_nothing_and_says_so():
    r = Runner(dry_run=True, echo=lambda s: None)
    res = BN.measure(Report(), (A,), EXES, "/p", "/n", 1, 1, r)
    assert res.rates is None and "dry-run" in res.why and r.commands == []


def test_skipped_carries_its_reason():
    assert BN.skipped("r") == BN.BenchResult(None, "r")


# ---- lead-job figures (N-card ladder M1): lower-is-better ms per decode window, one regime per concurrency class
def lead_line(rate, ms1=None, ms4=None):
    extra = "".join(f" lead{k}_window_ms={v}" for k, v in ((1, ms1), (4, ms4)) if v is not None)
    return line(rate) + extra


def test_parse_lead_reads_each_concurrency_class_and_never_the_expert_rate():
    text = lead_line(148383, 24.0, 26.5)
    assert BN.parse_lead(text, 1) == (24.0, "") and BN.parse_lead(text, 4) == (26.5, "")
    assert BN.parse(text) == (148383.0, "")                                      # the rows/s reader is untouched by the new keys


def test_parse_lead_without_a_figure_or_with_nan_or_zero_is_none_with_a_reason():
    assert BN.parse_lead(line(1), 1)[0] is None and "no lead1_window_ms" in BN.parse_lead(line(1), 1)[1]
    for bad in ("nan", "0", "-3", "abc"):
        assert BN.parse_lead(lead_line(1, bad), 1)[0] is None
    assert BN.parse_lead("noise", 1)[0] is None


def test_measure_collects_lead_figures_per_class_only_when_every_card_has_one():
    res, _ = run(Captured(None, lead_line(412, 26.7, 24.0), True, False), Captured(None, lead_line(388, 24.0, 26.0), True, False))
    assert res.lead_ms == {1: {("amd", 0): 26.7, ("nvidia", 0): 24.0}, 4: {("amd", 0): 24.0, ("nvidia", 0): 26.0}} and res.lead_why == ""
    assert res.lead_for(1) == res.lead_ms[1] and res.lead_for(8) == res.lead_ms[4] and res.lead_for(2) is None
    part, _ = run(Captured(None, lead_line(412, 26.7), True, False), Captured(None, lead_line(388, 24.0), True, False))
    assert set(part.lead_ms) == {1} and part.lead_for(4) is None and "missing" in part.lead_why
    one_missing, _ = run(Captured(None, lead_line(412, 26.7), True, False), Captured(None, line(388), True, False))
    assert one_missing.lead_ms == {} and one_missing.rates == {("amd", 0): 412.0, ("nvidia", 0): 388.0}   # rows/s survive


def test_a_bench_line_with_no_lead_figures_still_measures_rows_and_says_the_lead_job_is_missing():
    res, _ = run(Captured(None, line(412), True, False), Captured(None, line(388), True, False))
    assert res.rates and res.lead_ms == {} and res.lead_for(1) is None and "does not time it" in res.lead_why
