# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The CHECKER for "the gate is pytest".  A standard without one is a wish.

``unittest discover -s tests/install`` collects only the unittest-style classes and
silently ignores every pytest-style test function in this directory -- 136 of 203 on
the day this was written -- then prints a green ``OK``.  A whole day of "128 tests,
OK" was gated on that subset.  The docstrings now say to use pytest; this file is
what makes saying it matter, by FAILING under the wrong runner instead of letting
67 tests quietly not run.

Why ``PYTEST_CURRENT_TEST`` and not ``"pytest" in sys.modules``: the obvious check
was tried first and is DECORATIVE.  ``test_plan.py`` imports pytest at module level,
so unittest's own collection puts pytest into ``sys.modules`` and the assertion
passes under the very runner it is meant to reject -- measured, not reasoned:
``unittest discover`` reported "Ran 137 tests ... OK" with that version in place.
pytest sets ``PYTEST_CURRENT_TEST`` in the environment only while it is actually
executing a test, so it distinguishes RUNNING under pytest from merely having
imported it.

This file is deliberately collected only by broad discovery.  Driving one test by
name during a sabotage (``unittest ... -p <file>.py -k <name>``, where ``Ran 1
test`` is the whole point) does not select it and is unaffected.
"""

from __future__ import annotations

import os
import unittest


class TestTheSuiteRanUnderPytest(unittest.TestCase):
    def test_this_suite_must_be_run_with_pytest(self):
        # assertTrue over a plain bool, NEVER assertIn(..., os.environ, ...).  unittest renders
        # the CONTAINER in an assertIn failure, so that form prints the entire environment --
        # including any API keys or tokens, so a failing test would leak credentials.
        # A failing test must never become a credential leak, so the membership check is reduced
        # to a bool before it reaches the assertion.
        under_pytest = "PYTEST_CURRENT_TEST" in os.environ
        self.assertTrue(
            under_pytest,
            "This suite was run with a runner that silently collects only part of it. "
            "Use the gate:  python -m pytest tests/ -q   "
            "(unittest discover reports a green OK while ~1/3 of the tests never run; "
            "unittest -k <name> is still correct for driving ONE test in a sabotage).",
        )
