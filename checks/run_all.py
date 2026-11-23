#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Run every checks/check_*.py the way CI does; one line each; exit 1 if any fails.

OWNER: this file owns the checker list (the glob) and each checker's CI arguments; ci.yml and CONTRIBUTING call it,
so neither keeps a second copy.  tests/test_ci_workflow.py checks that CI runs this file.
FAILURE: a failing checker's output follows its line; every checker still runs.  check_names.py is excluded: it needs
--forbid NAME and runs only after a rename.  Stdlib only.  Run: python checks/run_all.py
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXCLUDED = {"check_names.py"}
ARGS = {"check_error_paths.py": ["--strict-runtime"]}


def checkers() -> list[str]:
    names = (os.path.basename(p) for p in glob.glob(os.path.join(HERE, "check_*.py")))
    return sorted(n for n in names if n not in EXCLUDED)


def main() -> int:
    failed = []
    for name in checkers():
        cmd = [sys.executable, os.path.join(HERE, name)] + ARGS.get(name, [])
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(HERE))
        print("%-4s %s" % ("ok" if done.returncode == 0 else "FAIL", name), flush=True)
        if done.returncode != 0:
            failed.append(name)
            print((done.stdout + done.stderr).rstrip())
    print("%d checkers, %d failed" % (len(checkers()), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
