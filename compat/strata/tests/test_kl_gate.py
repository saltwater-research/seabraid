# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for the tools/kl.py gate (--baseline/--arm) and its tools/arms.sh hook.  Synthetic LOGPOS TSVs, real
subprocess exit codes (0 PASS, 1 FAIL, 2 INVALID).  Every verdict has a file that must produce it; sabotage a bound
check and the matching test fails.  CPU only, stdlib + pytest."""
import math
import pathlib
import random
import re
import shutil
import subprocess
import sys

import pytest

TOOLS = pathlib.Path(__file__).resolve().parent.parent / "tools"
V, N = 200, 441


def logits(seed=1, n=N):
    rng = random.Random(seed)
    return [[rng.gauss(0, 3) for _ in range(V)] for _ in range(n)]


def write(path, rows, lp_shift=0.0, positions=None, lines=None):
    """One LOGPOS line per row: pos, target, logp_target, top, 0,0,0,0, then top-32 id:logprob pairs."""
    out = []
    for n, r in enumerate(rows):
        m = max(r)
        lz = math.log(sum(math.exp(x - m) for x in r)) + m
        order = sorted(range(V), key=lambda i: (-r[i], i))[:32]
        tgt = (n * 7) % V
        pos = n if positions is None else positions[n]
        cols = [str(pos), str(tgt), f"{r[tgt] - lz + lp_shift:.6f}", str(order[0]), "0", "0", "0", "0"]
        out.append("\t".join(cols + [f"{i}:{r[i] - lz:.6f}" for i in order]))
    pathlib.Path(path).write_text("".join(x + chr(10) for x in (lines if lines is not None else out)))
    return out


def gate(base, arm, *extra):
    p = subprocess.run([sys.executable, str(TOOLS / "kl.py"), "--baseline", str(base), "--arm", str(arm), *extra],
                       capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr


@pytest.fixture
def base(tmp_path):
    rows = logits()
    write(tmp_path / "base.tsv", rows)
    return tmp_path, rows


def test_identical_passes(base):
    d, rows = base
    write(d / "arm.tsv", rows)
    rc, out, _ = gate(d / "base.tsv", d / "arm.tsv")
    assert rc == 0 and out.startswith("KL-GATE PASS") and "KL 0.00000" in out and "top-1 1.0000" in out, out


def test_shifted_distribution_fails_on_kl(base):
    d, rows = base
    rng = random.Random(9)
    write(d / "arm.tsv", [[x + rng.gauss(0, 0.6) for x in r] for r in rows])
    rc, out, _ = gate(d / "base.tsv", d / "arm.tsv")
    assert rc == 1 and out.startswith("KL-GATE FAIL") and "mean KL" in out, out


def test_top1_flips_fail_on_top1_alone(base):
    d, rows = base
    flipped = []
    for n, r in enumerate(rows):  # swap the top two logits on every 4th row: argmax changes, KL stays small
        r = list(r)
        if n % 4 == 0:
            o = sorted(range(V), key=lambda i: -r[i])
            r[o[0]], r[o[1]] = r[o[1]] + 0.001, r[o[0]] - 0.001
        flipped.append(r)
    write(d / "arm.tsv", flipped)
    rc, out, _ = gate(d / "base.tsv", d / "arm.tsv", "--max-kl", "10")
    assert rc == 1 and "top-1" in out and "mean KL" not in out.split("|")[0], out


def test_thresholds_are_arguments(base):
    d, rows = base
    rng = random.Random(3)
    write(d / "arm.tsv", [[x + rng.gauss(0, 0.05) for x in r] for r in rows])
    assert gate(d / "base.tsv", d / "arm.tsv")[0] == 0
    assert gate(d / "base.tsv", d / "arm.tsv", "--max-kl", "0")[0] == 1


def test_signed_dlogp_and_t_are_reported(base):
    d, rows = base
    write(d / "arm.tsv", rows, lp_shift=-0.02)
    rc, out, _ = gate(d / "base.tsv", d / "arm.tsv")
    assert re.search(r"dlogp -0\.0200 t -\d", out), out  # constant shift: negative t


def test_truncated_file_is_invalid(base):
    d, rows = base
    write(d / "arm.tsv", rows[:-40])
    rc, out, _ = gate(d / "base.tsv", d / "arm.tsv")
    assert rc == 2 and out.startswith("KL-GATE INVALID") and "positions differ" in out, out


def test_misaligned_positions_are_invalid(base):
    d, rows = base
    write(d / "arm.tsv", rows, positions=[p + 1 for p in range(N)])
    rc, out, _ = gate(d / "base.tsv", d / "arm.tsv")
    assert rc == 2 and "positions differ" in out, out


def test_each_unjudgeable_input_is_invalid_for_its_own_reason(base):
    """The reason text is asserted so a check cannot hide behind a different, earlier one."""
    d, rows = base
    lines = write(d / "ok.tsv", rows)
    cases = {
        "empty": ([], "empty file"),
        "nan": ([lines[0].replace(lines[0].split("	")[2], "nan", 1)] + lines[1:], "NaN/inf"),
        "short": (lines[:-1] + ["5	3"], "malformed"),
        "dup": (lines + [lines[0]], "repeats position"),
        "notopk": (["	".join(lines[0].split("	")[:8])] + lines[1:], "no top-K"),
    }
    for name, (ls, reason) in cases.items():
        write(d / f"{name}.tsv", rows, lines=ls)
        rc, out, _ = gate(d / "base.tsv", d / f"{name}.tsv")
        assert rc == 2 and out.startswith("KL-GATE INVALID") and reason in out, (name, rc, out)
    rc, out, _ = gate(d / "base.tsv", d / "nonexistent.tsv")
    assert rc == 2 and "cannot read" in out, out
    rc, out, _ = gate(d / "nonexistent.tsv", d / "ok.tsv")
    assert rc == 2 and "cannot read" in out, out


def test_legacy_invocation_unchanged(base):
    d, rows = base
    write(d / "arm.tsv", rows)
    p = subprocess.run([sys.executable, str(TOOLS / "kl.py"), str(d / "base.tsv"), str(d / "arm.tsv")], capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.startswith(f"positions {N}  |dlogp| mean 0.0000"), p.stdout


def test_gate_needs_both_files(base):
    d, _ = base
    p = subprocess.run([sys.executable, str(TOOLS / "kl.py"), "--baseline", str(d / "base.tsv")], capture_output=True, text=True)
    assert p.returncode == 2


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_arms_sh_kl_gate_dry(base):
    """Dry run of arms.sh's own kl_gate function (extracted verbatim, no GPU): unset baseline is a no-op, PASS returns 0
    and logs the line, a bad arm returns non-zero and logs the FAIL line the run's RESULT.txt would carry."""
    d, rows = base
    src = (TOOLS / "arms.sh").read_text()
    fn = re.search(r"^kl_gate\(\)\{.*?return \$rc; \}", src, re.S | re.M).group(0)
    write(d / "kl_good.tsv", rows)
    rng = random.Random(4)
    write(d / "kl_bad.tsv", [[x + rng.gauss(0, 0.6) for x in r] for r in rows])
    script = f'unset KL_BASELINE KL_MAX KL_MIN_TOP1; res(){{ echo "$*"; }}\nR="{d.as_posix()}"; TOOLS="{TOOLS.as_posix()}"; PY="{sys.executable.replace(chr(92), "/")}"\n{fn}\n'
    sh = d / "dry.sh"  # a file, not bash -c: Windows command-line quoting mangles the embedded quotes

    def run(tail):
        sh.write_bytes((script + tail).encode())
        return subprocess.run([shutil.which("bash"), sh.as_posix()], capture_output=True, text=True)
    p = run('kl_gate good; echo "rc=$?"')
    assert p.stdout.strip() == "rc=0", (p.stdout, p.stderr)
    env = f'export KL_BASELINE="{(d / "base.tsv").as_posix()}"; '
    p = run(env + 'kl_gate good; echo "rc=$?"')
    assert "good KL-GATE PASS" in p.stdout and p.stdout.strip().endswith("rc=0"), p.stdout
    p = run(env + 'kl_gate bad; echo "rc=$?"')
    assert "bad KL-GATE FAIL" in p.stdout and p.stdout.strip().endswith("rc=1"), p.stdout
    p = run(env + 'kl_gate gone; echo "rc=$?"')
    assert "KL-GATE INVALID" in p.stdout and p.stdout.strip().endswith("rc=2"), p.stdout
