# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for tools/route_hist_report.py: synthetic dumps whose coverage, concentration and holder shares are worked
out by hand, so a wrong formula changes a number the test knows.  CPU only, stdlib + pytest; no engine, no GPU."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
import route_hist_report as R  # noqa: E402

E = 10  # experts per layer in every synthetic dump


def dump(path, prompt, decode, holder=None, n=1, prompt_tokens=3, decode_tokens=4, version=1):
    """prompt/decode: one {expert: count} dict per layer.  holder: one string of E chars per layer (default all '0')."""
    layers = len(prompt)
    doc = {"format": R.FORMAT, "version": version, "request": n, "layers": layers, "experts": E,
           "prompt_tokens": prompt_tokens, "decode_tokens": decode_tokens,
           "prompt": [sorted([e, c] for e, c in row.items()) for row in prompt],
           "decode": [sorted([e, c] for e, c in row.items()) for row in decode],
           "holder": holder or ["0" * E] * layers}
    pathlib.Path(path).write_text(json.dumps(doc))


# request A, two layers.  prompt top-2 of layer 0 is {0,1}; of layer 1 is {5}+{any}.
A_PROMPT = [{0: 10, 1: 5, 2: 1}, {5: 4}]
A_DECODE = [{0: 6, 1: 2, 3: 2}, {5: 3, 6: 1}]


@pytest.fixture
def reqs(tmp_path):
    hold = ["1" * 5 + "2" * 5, "0" * E]
    dump(tmp_path / "req-000001.json", A_PROMPT, A_DECODE, holder=hold, n=1)
    # request B has no prompt routing (its prompt came from a cache): it must not be scored for coverage
    dump(tmp_path / "req-000002.json", [{}, {}], [{9: 4}, {9: 4}], holder=["0" * 9 + "2"] * 2, n=2)
    got, bad = R.load_dir(str(tmp_path))
    assert bad == 0 and len(got) == 2
    return got


def test_top_k_breaks_ties_by_expert_id():
    assert R.top_k({3: 1, 1: 1, 2: 1}, 2) == [1, 2]
    assert R.top_k({7: 9, 2: 1}, 5) == [7, 2]


def test_coverage_known_answer(reqs):
    rows, used = R.coverage(reqs, 2)
    assert used == 1                       # B has no prompt routing
    # layer 0: prompt top-2 {0,1} covers 6+2 of 10; layer 1: prompt top-2 {5,*} covers 3 of 4
    assert rows[0] == [8, 10, 8] and rows[1] == [3, 4, 4]


def test_coverage_with_k_beyond_the_prompt_keeps_only_experts_the_prompt_touched(reqs):
    rows, _ = R.coverage(reqs, 64)
    # layer 0: the prompt touched {0,1,2}; the decode's 2 pairs on expert 3 stay uncovered.  layer 1: {5} only.
    assert [r[0] for r in rows] == [8, 3]


def test_concentration_known_answer(reqs):
    # decode phase, pooled over both requests.  top 10% of 10 experts = 1 expert per layer.
    # layer 0 decode: A {0:6,1:2,3:2} + B {9:4} -> top-1 = 6 of 14; layer 1: A {5:3,6:1} + B {9:4} -> top-1 = 4 of 8
    pooled, per_req = R.concentration(reqs, ("decode",), 0.10)
    assert pooled == pytest.approx((6 + 4) / (14 + 8))
    # per request: A -> (6+3)/(10+4); B -> 4+4 of 4+4 = 1.0; mean of the two
    assert per_req == pytest.approx(((6 + 3) / 14 + 1.0) / 2)
    # top 50% = 5 experts: everything A and B route fits in 5 per layer
    assert R.concentration(reqs, ("decode",), 0.50)[0] == pytest.approx(1.0)


def test_concentration_of_an_empty_phase_is_none_not_zero(reqs):
    assert R.concentration([reqs[1]], ("prompt",), 0.10) == (None, None)


def test_holder_share(reqs):
    h = R.holder_share(reqs)
    # decode pairs by holder: A layer 0: e0 6, e1 2, e3 2 -> '1' (experts 0-4 are '1', 5-9 '2'); A layer 1 all '0' (4);
    # B routes only expert 9 in both layers (8 pairs), held by '2'
    assert h["1"] == pytest.approx(10 / 22) and h["2"] == pytest.approx(8 / 22) and h["0"] == pytest.approx(4 / 22)
    assert sum(h.values()) == pytest.approx(1.0)


def test_loader_skips_a_wrong_file_loudly(tmp_path, capsys):
    dump(tmp_path / "req-000001.json", A_PROMPT, A_DECODE)
    dump(tmp_path / "req-000002.json", A_PROMPT, A_DECODE, version=2)
    (tmp_path / "req-000003.json").write_text("{not json")
    got, bad = R.load_dir(str(tmp_path))
    assert len(got) == 1 and bad == 2
    assert "skipping" in capsys.readouterr().err


def test_loader_rejects_an_out_of_range_expert(tmp_path):
    dump(tmp_path / "req-000001.json", [{E + 3: 1}], [{0: 1}])
    with pytest.raises(ValueError):
        R.load_one(str(tmp_path / "req-000001.json"))


def test_main_exit_codes_and_output(tmp_path, capsys):
    assert R.main([str(tmp_path)]) == 2                       # nothing to report is an error, not an empty table
    dump(tmp_path / "req-000001.json", A_PROMPT, A_DECODE)
    assert R.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for want in ("PROMPT-CONDITIONED PROMOTION", "TIERABILITY", "WHO HELD", "ALL", "chance", "oracle", "prompt-top16"):
        assert want in out
    (tmp_path / "req-000002.json").write_text("junk")
    assert R.main([str(tmp_path)]) == 1                       # reported, but a file was skipped


def test_timing_log_is_weighted_by_requests(tmp_path):
    log = tmp_path / "engine.log"
    log.write_text(
        "strata decode timing: 5 windows ...; 0.0 swaps/window; card 2: 100 layer requests, own expert work 50.0 us + waited past it 10.0 us each\n"
        "strata decode timing: 5 windows ...; 0.0 swaps/window; card 2: 300 layer requests, own expert work 70.0 us + waited past it 30.0 us each\n"
        "strata decode timing: 5 windows ...; 0.0 swaps/window\n")
    t = R.timing_from_log(str(log))
    n, own, wait = t[2]
    assert n == 400 and own == pytest.approx(65.0) and wait == pytest.approx(25.0)
