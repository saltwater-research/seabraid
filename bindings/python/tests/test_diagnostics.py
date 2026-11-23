# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Version, native log forwarding and native last-error text, all through mocked libraries.

Checker for seabraid.diagnostics, seabraid._version and the last-error path of seabraid._abi / seabraid._native.
"""

from __future__ import annotations

import ctypes as ct
import logging
import os
import tempfile
import unittest
from importlib import metadata
from pathlib import Path
from unittest.mock import Mock, patch

from seabraid import _abi, _version, diagnostics
from seabraid import SeabraidError, Runtime, enable_logging, disable_logging, version
from seabraid.constants import LOG_DEBUG, LOG_WARN

COMMON = ["seabraid_open", "seabraid_open_reference", "seabraid_abi_epoch", "seabraid_prepare", "seabraid_submit", "seabraid_collect", "seabraid_close"]


def library_with(*extra: str) -> Mock:
    library = Mock(spec=COMMON + list(extra))
    library.seabraid_open_reference.return_value = 0
    library.seabraid_abi_epoch.return_value = 1
    library.seabraid_close.return_value = 0
    return library


class VersionTests(unittest.TestCase):
    def test_native_version_wins_when_a_library_is_active(self) -> None:
        library = library_with("seabraid_version")
        library.seabraid_version.return_value = b"9.9.9-native"
        with patch.object(_abi, "_active", library):
            self.assertEqual(version(), "9.9.9-native")

    def test_the_version_has_one_form_before_and_after_a_native_library_loads(self) -> None:
        """R7 LOW 11: 0.5.0a2 became 0.5.0-alpha.2 once the native library was loaded."""
        library = library_with("seabraid_version")
        library.seabraid_version.return_value = b"0.5.0-alpha.2"
        with patch.object(_abi, "_active", None):
            before = version()
        with patch.object(_abi, "_active", library):
            after = version()
        self.assertEqual((before, after), (_version.package_version(), "0.5.0a2"))
        for text, expected in (("1.0.0-beta.3", "1.0.0b3"), ("1.0.0-rc.1", "1.0.0rc1"), ("1.0.0", "1.0.0"),
                               ("0.5.0a2", "0.5.0a2"), ("9.9.9-native", "9.9.9-native"),
                               ("1.2.3-alpha.4+x", "1.2.3-alpha.4+x")):
            self.assertEqual(_version.pep440(text), expected)

    def test_package_version_without_native_or_without_the_symbol(self) -> None:
        import seabraid
        with patch.object(_abi, "_active", None):
            self.assertEqual(version(), seabraid.__version__)
        with patch.object(_abi, "_active", library_with()):
            self.assertEqual(version(), seabraid.__version__)

    def test_version_file_fallback_and_unknown(self) -> None:
        missing = metadata.PackageNotFoundError("seabraid")
        with tempfile.TemporaryDirectory() as folder, patch.object(metadata, "version", side_effect=missing):
            file = Path(folder) / "VERSION"
            file.write_text("1.2.3-alpha.4\n", encoding="utf-8")
            with patch.object(_version, "_VERSION_FILE", file):
                self.assertEqual(_version.package_version(), "1.2.3a4")  # the VERSION file is semver, reported as PEP 440
            with patch.object(_version, "_VERSION_FILE", Path(folder) / "absent"):
                with self.assertLogs("seabraid", "WARNING"):
                    self.assertEqual(_version.package_version(), "0+unknown")


class LoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.library = library_with("seabraid_set_log_sink")
        self.library.seabraid_set_log_sink.return_value = 0
        active = patch.object(_abi, "_active", self.library)
        active.start()
        self.addCleanup(active.stop)
        # SEABRAID_LIBRARY (set by the linked CTest run) would make enable_logging load the real library, not the mock.
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("SEABRAID_LIBRARY", None)
        self.addCleanup(disable_logging)  # runs first (LIFO), while the mocked library is still active

    def test_records_reach_the_seabraid_logger_with_status_and_survive_gc(self) -> None:
        self.assertTrue(enable_logging("INFO"))
        callback, context, level = self.library.seabraid_set_log_sink.call_args.args
        self.assertEqual((context, level), (None, 1))
        self.assertIs(diagnostics._installed[0][1], callback)  # the module holds the callback alive
        with self.assertLogs("seabraid", "INFO") as logs:
            callback(None, 2, 4, b"provider hiccup")
            callback(None, 1, 0, None)
        first, second = logs.records
        self.assertEqual((first.levelno, first.getMessage(), first.seabraid_status), (logging.WARNING, "provider hiccup", 4))
        self.assertEqual((second.levelno, second.getMessage()), (logging.INFO, ""))

    def test_the_exit_hook_is_registered_once_across_enable_disable_cycles(self) -> None:
        """R7 NIT: disable_logging then enable_logging registered _disable_at_exit a second time."""
        with patch.object(diagnostics, "_exit_hook_registered", False), patch.object(diagnostics.atexit, "register") as register:
            for _ in range(3):
                enable_logging("INFO")
                disable_logging()
            enable_logging("INFO")
        register.assert_called_once_with(diagnostics._disable_at_exit)

    def test_disable_restores_the_default_sink(self) -> None:
        enable_logging(logging.DEBUG)
        self.assertEqual(self.library.seabraid_set_log_sink.call_args.args[2], LOG_DEBUG)
        disable_logging()
        restore, context, level = self.library.seabraid_set_log_sink.call_args.args
        self.assertFalse(restore)  # a NULL function pointer (falsy ctypes object), never None: ctypes refuses None natively
        self.assertIsInstance(restore, _abi.LogCallback)
        self.assertEqual((context, level), (None, LOG_WARN))
        self.assertEqual(diagnostics._installed, [])
        disable_logging()  # idempotent
        self.assertEqual(self.library.seabraid_set_log_sink.call_count, 2)

    def test_bad_level_status_and_missing_symbol_are_loud(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_level"):
            enable_logging("LOUD")
        self.library.seabraid_set_log_sink.return_value = 1
        with self.assertRaisesRegex(SeabraidError, "seabraid_set_log_sink: invalid argument"):
            enable_logging()
        self.assertEqual(diagnostics._installed, [])
        del self.library.seabraid_set_log_sink
        with self.assertRaisesRegex(RuntimeError, "does not expose logging"):
            enable_logging()

    def two_libraries(self):
        other = library_with("seabraid_set_log_sink")
        other.seabraid_set_log_sink.return_value = 0
        return self.library, other

    @staticmethod
    def forwarding(library) -> bool:
        """True when the library's most recent sink call left a live (non-NULL) callback installed."""
        return bool(library.seabraid_set_log_sink.call_args.args[0])

    def test_a_sink_on_a_second_library_does_not_orphan_the_first(self) -> None:
        """F12: enabling on B left A forwarding after disable_logging."""
        first, second = self.two_libraries()
        with patch.object(_abi, "resolve_library", side_effect=[first, second]):
            enable_logging("INFO")
            enable_logging("INFO")
        self.assertTrue(self.forwarding(first) and self.forwarding(second))
        disable_logging()
        self.assertFalse(self.forwarding(first))
        self.assertFalse(self.forwarding(second))
        self.assertEqual(diagnostics._installed, [])

    def test_enabling_twice_on_one_library_keeps_one_entry(self) -> None:
        with patch.object(_abi, "resolve_library", return_value=self.library):
            enable_logging("INFO")
            enable_logging("DEBUG")
        self.assertEqual(len(diagnostics._installed), 1)
        disable_logging()
        self.assertFalse(self.forwarding(self.library))

    def test_the_exit_hook_detaches_every_library(self) -> None:
        first, second = self.two_libraries()
        with patch.object(_abi, "resolve_library", side_effect=[first, second]):
            enable_logging("INFO")
            enable_logging("INFO")
        diagnostics._disable_at_exit()
        self.assertFalse(self.forwarding(first))
        self.assertFalse(self.forwarding(second))

    def test_one_library_failing_to_detach_does_not_strand_the_others(self) -> None:
        first, second = self.two_libraries()
        with patch.object(_abi, "resolve_library", side_effect=[first, second]):
            enable_logging("INFO")
            enable_logging("INFO")
        first.seabraid_set_log_sink.return_value = 1
        with self.assertRaises(SeabraidError):
            disable_logging()
        self.assertFalse(self.forwarding(second))
        self.assertEqual([entry[0] for entry in diagnostics._installed], [first])  # kept, so a later disable retries it
        first.seabraid_set_log_sink.return_value = 0
        disable_logging()
        self.assertFalse(self.forwarding(first))

    def test_no_native_library_is_logged_and_reported(self) -> None:
        with patch.object(_abi, "_active", None), patch.dict(os.environ):
            os.environ.pop("SEABRAID_LIBRARY", None)
            with self.assertLogs("seabraid", "WARNING"):
                self.assertFalse(enable_logging())


class LastErrorTests(unittest.TestCase):
    def runtime(self, library: Mock) -> Runtime:
        def open_session(out: object) -> int:
            ct.cast(out, ct.POINTER(ct.c_void_p))[0] = 5
            return 0

        library.seabraid_open_reference.side_effect = open_session
        with patch("seabraid._abi.ct.CDLL", return_value=library):
            return Runtime(__file__, reference=True)

    def test_native_text_is_appended_to_the_message(self) -> None:
        library = library_with("seabraid_last_error")
        library.seabraid_prepare.return_value = 1
        long_text = b"experts must be at least one " + b"x" * 400

        def last_error(buffer: ct.Array, capacity: int, needed: object) -> int:
            ct.cast(needed, ct.POINTER(ct.c_size_t))[0] = len(long_text)
            ct.memmove(buffer, long_text[: capacity - 1], min(len(long_text), capacity - 1))
            return 0

        library.seabraid_last_error.side_effect = last_error
        with self.runtime(library) as runtime:
            with self.assertRaises(SeabraidError) as raised:
                runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
        self.assertEqual(raised.exception.detail, long_text.decode())  # longer than the first buffer: retried
        self.assertTrue(str(raised.exception).startswith("seabraid_prepare: invalid argument (status 1): experts must"))

    def test_older_library_without_last_error_still_raises_cleanly(self) -> None:
        library = library_with()
        library.seabraid_prepare.return_value = 4
        with self.runtime(library) as runtime:
            with self.assertRaisesRegex(SeabraidError, r"^seabraid_prepare: provider failure \(status 4\)$"):
                runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)

    def test_failing_last_error_is_logged_not_swallowed(self) -> None:
        library = library_with("seabraid_last_error")
        library.seabraid_prepare.return_value = 5
        library.seabraid_last_error.return_value = 2
        with self.runtime(library) as runtime, self.assertLogs("seabraid", "WARNING") as logs:
            with self.assertRaisesRegex(SeabraidError, r"internal failure \(status 5\)$"):
                runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
        self.assertIn("seabraid_last_error failed", logs.output[0])
