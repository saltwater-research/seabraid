# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Exercise the Python surface against a built reference runtime."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from seabraid import SeabraidError, Runtime, SourceModality


@unittest.skipUnless(os.environ.get("SEABRAID_LIBRARY"), "set SEABRAID_LIBRARY for native integration")
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library = os.environ["SEABRAID_LIBRARY"]

    def test_reference_weave_for_two_rows_and_reuse(self) -> None:
        with Runtime(self.library, reference=True) as runtime:
            prepared = runtime.prepare(hidden=2, experts=4, max_tokens=3, top_k=3)
            output = prepared.weave(
                [2.0, -4.0, 0.5, 8.0], [0, 2, 1, 1, 3, 0], [0.25, 0.5, 0.25, 0.5, 0.375, 0.125],
            )
            self.assertEqual(output, [4.5, -9.0, 1.3125, 21.0])
            self.assertEqual(prepared.weave([1.0, 2.0], [1, 0, 3], [1.0, 0.0, 0.0]), [2.0, 4.0])

    def test_mixed_encoded_sources_preserve_routed_math(self) -> None:
        with Runtime(self.library, reference=True) as runtime:
            prepared = runtime.prepare(hidden=1, experts=2, max_tokens=1, top_k=1)
            mixed = SourceModality.AUDIO | SourceModality.VIDEO
            self.assertEqual(runtime.capabilities().source_modalities & mixed, mixed)
            with self.assertRaisesRegex(SeabraidError, "seabraid_session_submit: invalid argument"):
                prepared.weave([3.0], [1], [1.0], source_modalities=SourceModality(1 << 31))
            self.assertEqual(prepared.weave([3.0], [1], [1.0], source_modalities=mixed), [6.0])
            status = runtime.status()
            self.assertEqual((status.submitted, status.collected), (1, 1))
            self.assertEqual(status.last_source_modalities, mixed)
            self.assertFalse(status.outstanding)

    def test_invalid_expert_reaches_native_and_can_be_corrected(self) -> None:
        with Runtime(self.library, reference=True) as runtime:
            prepared = runtime.prepare(hidden=1, experts=2, max_tokens=1, top_k=1)
            with self.assertRaisesRegex(SeabraidError, "seabraid_session_submit: invalid argument") as raised:
                prepared.weave([2.0], [2], [1.0])
            self.assertEqual(raised.exception.status, 1)
            self.assertEqual(prepared.weave([2.0], [1], [1.0]), [4.0])

    def test_prepare_once_and_close_idempotently(self) -> None:
        with Runtime(self.library, reference=True) as runtime:
            prepared = runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
            with self.assertRaisesRegex(RuntimeError, "already prepared"):
                runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
        runtime.close()
        with self.assertRaisesRegex(RuntimeError, "runtime is closed"):
            prepared.weave([2.0], [0], [1.0])

    def test_collect_retries_same_pending_ticket_after_timeout(self) -> None:
        with Runtime(self.library, reference=True) as runtime:
            prepared = runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
            # Force the timing outcome while keeping the actual native submission pending.
            with patch.object(runtime._session.library, "seabraid_collect", return_value=3):
                with self.assertRaisesRegex(SeabraidError, "seabraid_collect: timeout") as raised:
                    prepared.weave([7.0], [0], [1.0])
                self.assertEqual(raised.exception.status, 3)
            with self.assertRaisesRegex(RuntimeError, "batch is pending"):
                prepared.weave([5.0], [0], [1.0])
            self.assertEqual(prepared.collect(), [7.0])
            self.assertEqual(prepared.weave([5.0], [0], [1.0]), [5.0])
