# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Collect a replay manifest from a dedicated loopback Strata test server.

The caller owns GPU admission, process lifecycle and the starting placement.
This tool submits a frozen prompt plan serially and binds each successful
response to exactly one new histogram. No manifest is published on failure.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from expert_preflight_replay import label, positive_int, read_dataset
from route_hist_report import load_one


class NoRedirect(HTTPRedirectHandler):
    """A loopback capture should never follow an HTTP redirect off the test server."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if type(plan["version"]) is not int or plan["version"] != 1:
        raise ValueError("capture plan version must be 1")
    label(plan["served_model"], "served_model")
    label(plan["model"], "model")
    if plan["placement"] != "frozen":
        raise ValueError("capture requires frozen placement")
    families: dict[str, str] = {}
    for row in plan["requests"]:
        family = label(row["family"], "family")
        split = row["split"]
        if split not in ("train", "test") or families.setdefault(family, split) != split:
            raise ValueError("invalid split or family in both splits")
        label(row["task"], "task")
        label(row.get("effort", "default"), "effort")
        label(row["prompt"], "prompt")
    if set(families.values()) != {"train", "test"}:
        raise ValueError("need training and test prompt families")
    return plan


def request_body(row: dict, served: str) -> dict:
    # The experiment freezes time spent generating and records effort only as a label.
    return {"model": served, "messages": [{"role": "user", "content": row["prompt"]}],
            "max_tokens": 2, "temperature": 0, "strata_checkpoint": False,
            "chat_template_kwargs": {"enable_thinking": False}}


def post(port: int, timeout: float) -> Callable[[dict], dict]:
    opener = build_opener(ProxyHandler({}), NoRedirect())
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    def submit(body: dict) -> dict:
        req = Request(url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)

    return submit


def response_tokens(response: dict, served: str) -> int:
    if "error" in response or response.get("model") != served:
        raise ValueError("server error or unexpected served model")
    tokens = positive_int(response["usage"]["prompt_tokens"], "prompt_tokens")
    timings = response["timings"]
    if type(timings["cache_n"]) is not int or timings["cache_n"] != 0:
        raise ValueError("server reused prompt tokens")
    if type(timings["prompt_n"]) is not int or timings["prompt_n"] != tokens:
        raise ValueError("server did not read the full prompt")
    return tokens


def new_histogram(directory: Path, previous: set[Path]) -> Path:
    added = set(directory.glob("req-*.json")) - previous
    if len(added) != 1:
        raise ValueError(f"expected one new histogram, found {len(added)}")
    return added.pop()


def verify_residency(hist: dict, layers: list[dict]) -> None:
    resident = [{int(e) for e, tier in enumerate(row) if tier == "1"} for row in hist["holder"]]
    if resident != [set(row["resident"]) for row in layers]:
        raise ValueError("end-of-request lead residency differs from the frozen starting map")


def capture(plan: dict, directory: Path, output_dir: Path, submit: Callable[[dict], dict]) -> dict:
    served = label(plan["served_model"], "served_model")
    previous: set[Path] = set(directory.glob("req-*.json"))
    if previous:
        raise ValueError("histogram directory must start empty")
    rows = []
    for row in plan["requests"]:
        start = time.monotonic()
        tokens = response_tokens(submit(request_body(row, served)), served)
        path = new_histogram(directory, previous)
        hist = load_one(path)
        if hist["prompt_tokens"] != tokens:
            raise ValueError("response and histogram prompt lengths differ")
        verify_residency(hist, plan["layers"])
        previous.add(path)
        rows.append({key: row[key] for key in ("family", "split", "task")})
        rows[-1].update(effort=row.get("effort", "default"), histogram=os.path.relpath(path, output_dir),
                        prompt_tokens=tokens, capture_seconds=round(time.monotonic() - start, 3))
    manifest = {field: plan[field] for field in ("version", "model", "experts", "picks_per_token", "layers", "placement")}
    return {**manifest, "requests": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("histograms", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    try:
        if args.manifest.exists() or not args.histograms.is_dir():
            raise ValueError("use an existing empty histogram directory and a new output path")
        if not 0 < args.port <= 65535 or not 0 < args.timeout <= 3600:
            raise ValueError("invalid port or timeout")
        pending = args.manifest.with_name(args.manifest.name + ".pending")
        if pending.exists() or not args.manifest.parent.is_dir():
            raise ValueError("choose a new manifest name in an existing directory")
        manifest = capture(read_plan(args.plan), args.histograms, args.manifest.parent, post(args.port, args.timeout))
        # Validate via the replay reader before giving the result its final name.
        with pending.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        read_dataset(pending)
        pending.rename(args.manifest)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"invalid capture: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
