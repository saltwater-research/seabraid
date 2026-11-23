# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for scripts/payoff/pay_parse.py: lines in the real engine formats come back as the numbers they carry.

Each test builds text with payoff_synth (the formats of Strata v0.2.15 generate.cpp and patch 0001) and asserts the value a
reader would see in the table; a regex that drifts from a format changes a number here.  CPU only; no engine."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import payoff_synth as S  # noqa: E402
import pay_parse as P  # noqa: E402


def test_prompt_line_gives_prefill_and_decode_rates():
    log = P.parse_request(S.prompt_line(3500, 5, 410.5, 400, 63.2))
    assert log.prompt == P.Prompt(3500, 5, 3495, int(1000 * 3495 / 410.5), 410.5, 400, int(1000 * 400 / 63.2), 63.2)


def test_cancelled_prompt_line_reads_the_partial_count():
    text = "strata serve: prompt 98179 tokens = 0 reused + 12288 of 98179 read in 17565 ms (699.6 tok/s), 0 generated in 0 ms (0.0 tok/s), drafts accepted 0 of 0, 0 checkpoints (cancelled)\n"
    assert P.parse_request(text).prompt.read_tokens == 12288


def test_decode_timing_serial_and_pipelined_both_parse():
    for pipelined in (False, True):
        t = P.parse_request(S.timing_line(windows=50, cpu=6.5, stage=2.25, cpu_experts=1.5, pcie=0.75, pipelined=pipelined)).timing
        assert (t.windows, t.cpu, t.stage, t.gpu_reach_wait, t.cpu_experts, t.pcie) == (50, 6.5, 2.25, 18.4, 1.5, 0.75)


def test_hit_line_offload_is_zero_when_the_clause_is_absent_and_counted_when_present():
    assert P.parse_request(S.hit_line()).offload == 0
    log = P.parse_request(S.hit_line(60))
    assert (log.hits, log.lookups, log.offload) == (900, 1000, 60)


def test_remote_and_helper_lines_give_staging_wait_and_service():
    log = P.parse_request(S.remote_lines(staging_ms=310, wait_ms=940, helper_requests=1200))
    assert log.remote == [P.Remote(1, 4000, 1200, 310.0, 940.0)]
    assert (log.helper_requests, log.helper_us_each) == (1200, 74.8)


def test_io_line_gives_os_read_and_decode_wait():
    log = P.parse_request(S.io_line(7.5))
    assert (log.os_read_mb, log.io_wait_ms) == (12.5, 7.5)


def test_missing_lines_are_none_not_zero():
    log = P.parse_request("nothing the engine prints for a request\n")
    assert log.prompt is None and log.timing is None and log.offload is None and log.io_wait_ms is None and log.helper_requests is None
    c = P.components(log)
    assert (c.staging_ms, c.merge_wait_ms, c.local_compute_ms, c.gpu_reach_wait_ms, c.io_wait_ms) == (None,) * 5


def test_components_multiply_per_window_terms_and_split_arm_c():
    text = S.request_text("C", "decode", 70.0, 500.0)
    c = P.components(P.parse_request(text))
    assert c.staging_ms == 100 * 2.0 + 310.0      # stage term x windows + the helper's "staging+launching"
    assert c.merge_wait_ms == 940.0               # only a helper line has a merge partner
    assert c.local_compute_ms == 100 * 6.0 and c.gpu_reach_wait_ms == 100 * 18.4 and c.io_wait_ms == 7.5


def test_arm_a_has_no_merge_wait():
    c = P.components(P.parse_request(S.request_text("A", "decode", 40.0, 400.0)))
    assert c.merge_wait_ms is None and c.staging_ms == 200.0


def test_tiers_are_cumulative_and_last_wins():
    text = S.tiers_line(10, 2, 5.0) + S.tiers_line(30, 7, 12.5)
    assert P.last_tiers(text) == P.Tiers(30, 7, 12.5)


def test_startup_facts():
    s = P.parse_startup(S.startup_text(11000, helper=True))
    assert (s.cap_slots, s.cap_gib, s.helper_up, s.resident_gib) == (11000, 21.5, True, 9.8)
    s0 = P.parse_startup(S.startup_text(4000))
    assert (s0.cap_slots, s0.helper_up) == (4000, False)
    assert P.parse_startup("").cap_slots is None
