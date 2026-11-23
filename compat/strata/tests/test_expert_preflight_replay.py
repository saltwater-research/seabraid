# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks with hand-counted transfers and held-out prompt families."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import expert_preflight_replay as replay  # noqa: E402
import capture_expert_preflight as capture  # noqa: E402


def write_request(tmp_path: Path, i: int, task: str, split: str, expert: int) -> dict:
    path = tmp_path / f"routes-{i}.json"
    path.write_text(json.dumps({
        "format": "strata-route-hist", "version": 1, "request": i,
        "layers": 2, "experts": 3, "prompt_tokens": 10, "decode_tokens": 0,
        "prompt": [[[expert, 9]], [[0, 9]]], "decode": [[], []],
        "holder": ["000", "000"],  # post-request holder differs from declared starting residency
    }), encoding="utf-8")
    return {"histogram": path.name, "family": f"{task}-{i}", "split": split,
            "task": task, "effort": "deep", "prompt_tokens": 10}


@pytest.fixture
def sample(tmp_path: Path) -> tuple[Path, dict]:
    rows = [
        *(write_request(tmp_path, i, "code", "train", 1) for i in (1, 2)),
        *(write_request(tmp_path, i, "math", "train", 2) for i in (3, 4, 5)),
        write_request(tmp_path, 6, "code", "test", 1),
        write_request(tmp_path, 7, "translate", "test", 2),
    ]
    manifest = {"version": 1, "model": "synthetic-qwen", "experts": 3,
                "picks_per_token": 1, "placement": "frozen",
                "layers": [{"expert_bytes": 8, "resident": []}, {"expert_bytes": 5, "resident": [0]}],
                "requests": rows}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_task_conditioning_stages_the_right_layer_and_backs_off(sample, capsys):
    path, _ = sample
    assert replay.main([str(path), "--budget-bytes", "8"]) == 0
    report = json.loads(capsys.readouterr().out)
    policies = report["policies"]
    assert (report["train_families"], report["test_families"]) == (5, 2)
    assert policies["no_hint"]["on_demand_bytes"] == 16
    task = policies["task"]
    fields = ("on_demand_loads", "avoided_layers_with_misses", "fallback_requests", "useful_bytes", "wasted_bytes")
    assert tuple(task[k] for k in fields) == (0, 2, 1, 16, 0)
    assert report["by_family"]["code-6"]["task"]["staged_bytes"] == 8


def test_global_arms_miss_the_minority_task_under_the_same_budget(sample):
    path, _ = sample
    policies = replay.evaluate(replay.read_dataset(path), budget=8, threshold=0.9)["policies"]
    for baseline in ("static", "length"):
        assert policies[baseline]["on_demand_bytes"] == policies[baseline]["wasted_bytes"] == 8


def test_routes_that_exist_only_in_test_cannot_influence_the_stage_plan(sample):
    path, _ = sample
    write_request(path.parent, 6, "code", "test", 0)
    report = replay.evaluate(replay.read_dataset(path), budget=8, threshold=0.9)
    heldout = report["by_family"]["code-6"]["task"]
    assert heldout["wasted_bytes"] == heldout["on_demand_bytes"] == 8


def test_heldout_routes_cannot_shift_the_global_profile(sample):
    path, _ = sample
    write_request(path.parent, 7, "translate", "test", 1)
    policies = replay.evaluate(replay.read_dataset(path), budget=8, threshold=0.9)["policies"]
    assert policies["static"]["useful_bytes"] == policies["length"]["useful_bytes"] == 0
    assert policies["static_presence"]["useful_bytes"] == 0


def test_static_presence_ranks_request_coverage_over_pick_frequency(sample):
    path, manifest = sample
    for row in manifest["requests"][:5]:
        hist_path = path.parent / row["histogram"]
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        hist["prompt"][0] = [[1, 1], [0 if row["task"] == "code" else 2, 8]]
        hist_path.write_text(json.dumps(hist), encoding="utf-8")
    report = replay.evaluate(replay.read_dataset(path), budget=8, threshold=0.9)
    policies = report["by_family"]["code-6"]
    assert policies["static"]["on_demand_bytes"] == policies["static"]["wasted_bytes"] == 8
    assert policies["static_presence"]["useful_bytes"] == 8
    assert policies["static_presence"]["on_demand_bytes"] == 0


def test_static_presence_gives_repeated_requests_one_family_vote(sample):
    path, manifest = sample
    for i in range(8, 11):
        repeat = write_request(path.parent, i, "code", "train", 1)
        repeat["family"] = manifest["requests"][0]["family"]
        manifest["requests"].append(repeat)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = replay.evaluate(replay.read_dataset(path), budget=8, threshold=0.9)
    assert report["by_family"]["translate-7"]["static_presence"]["useful_bytes"] == 8


def test_length_policy_excludes_the_last_prompt_token_from_prefill(sample):
    path, _ = sample
    # The leading training expert has share 3/5: 9 picks put it below the cutoff, 10 above.
    policies = replay.evaluate(replay.read_dataset(path), budget=8, threshold=0.9998)["policies"]
    assert policies["length"]["staged_bytes"] == 0


def test_short_prompt_avoids_low_confidence_staging(sample):
    path, _ = sample
    data = replay.read_dataset(path)
    rates = replay.fit_profile([r for r in data.requests if r.split == "train"])
    assert replay.select_for_length(rates, 1, data, 8, 0.9) == set()
    assert replay.select_for_length(rates, 10, data, 8, 0.9) == {(0, 2)}
    assert replay.select_for_length(rates, 10, data, 7, 0.9) == set()


def test_high_probabilities_keep_their_ranking_and_do_not_become_certainties(sample):
    path, _ = sample
    data = replay.read_dataset(path)
    rates = {(0, 0): 0.01, (0, 2): 0.02}
    assert replay.select_for_length(rates, 4096, data, 8, 0.9) == {(0, 2)}
    assert replay.select_for_length(rates, 4096, data, 8, 1.0) == set()


def test_effort_only_narrows_a_task_with_independent_training_families(sample):
    path, manifest = sample
    manifest["requests"].extend(write_request(path.parent, i, "code", "train", 2) for i in (8, 9, 10))
    for row in manifest["requests"][-3:]:
        row["effort"] = "quick"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    policies = replay.evaluate(replay.read_dataset(path), budget=8, threshold=0.9)["policies"]
    assert policies["task"]["on_demand_loads"] == 1
    assert policies["task_effort"]["on_demand_loads"] == 0


@pytest.fixture
def capture_case(sample):
    path, plan = sample
    histograms = path.parent / "capture"
    histograms.mkdir()
    target = path.parent / "captured.json"
    plan["served_model"] = "synthetic"
    for row in plan["requests"]:
        row["prompt"] = row["family"]
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path, plan, histograms, target


@pytest.mark.parametrize("fault", ("none", "cache", "multiple", "placement", "model"))
def test_capture_binds_response_to_new_histogram(capture_case, monkeypatch, fault):
    path, plan, directory, target = capture_case
    contents = {row["prompt"]: json.loads((path.parent / row["histogram"]).read_text())
                for row in plan["requests"]}

    def submit(body):
        doc = contents[body["messages"][0]["content"]]
        doc["holder"] = ["100" if fault == "placement" else "000", "100"]
        (directory / f"req-{doc['request']}.json").write_text(json.dumps(doc))
        if fault == "multiple":
            (directory / "req-extra.json").write_text(json.dumps(doc))
        return {"model": "wrong" if fault == "model" else "synthetic", "usage": {"prompt_tokens": 10},
                "timings": {"cache_n": int(fault == "cache"), "prompt_n": 10}}

    monkeypatch.setattr(capture, "post", lambda port, timeout: submit)
    expected = 0 if fault == "none" else 2
    assert capture.main([str(path), str(directory), str(target), "--port", "18080"]) == expected
    assert target.exists() is (expected == 0)
    if expected == 0:
        assert len(replay.read_dataset(target).requests) == len(contents)


@pytest.mark.parametrize("poison", ("overlap", "duplicate file", "partial prompt", "duplicate expert"))
def test_invalid_replay_fails_without_metrics(sample, poison, capsys):
    path, manifest = sample
    requests = manifest["requests"]
    if poison == "overlap":
        requests[-1]["family"] = requests[0]["family"]
    elif poison == "duplicate file":
        requests[-1]["histogram"] = requests[0]["histogram"]
    else:
        hist_path = path.parent / requests[-1]["histogram"]
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        hist["prompt"][0] = [[1, 8]] if poison == "partial prompt" else [[2, 3], [2, 9]]
        hist_path.write_text(json.dumps(hist), encoding="utf-8")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert replay.main([str(path), "--budget-bytes", "8"]) == 2
    result = capsys.readouterr()
    assert not result.out
    assert "invalid replay" in result.err
