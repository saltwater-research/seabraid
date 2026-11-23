# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Seabraid: an engine-neutral runtime for serving one mixture-of-experts layer from several vendors' GPUs.

OWNER   : the public surface of the Python package; every name below lives in exactly one submodule.
CONTRACT: Runtime/PreparedMoE (runtime), SeabraidError (errors), Capabilities/SessionStatus, Shape/BatchInfo (the provider
          callable's config and trailing argument), SourceModality, version(), bundled_library(),
          enable_logging()/disable_logging() (diagnostics), the placement and ranking submodules, __version__.
          Importing never loads a native library and never touches the network.
SIZE    : under 60 lines.
FAILURE : see each submodule; importing cannot fail on a missing library or missing metadata.
CHECKER : tests/test_public_api.py.
FINISH  : that test passes and examples/quickstart.py runs from a clean wheel install.
"""

from . import placement, ranking
from ._bundled import bundled_library
from ._model import BatchInfo, Capabilities, SessionStatus, Shape
from ._version import package_version
from .constants import SourceModality
from .diagnostics import disable_logging, enable_logging, version
from .errors import SeabraidError
from .runtime import PreparedMoE, Runtime

__version__ = package_version()

__all__ = [
    "BatchInfo", "Capabilities", "SeabraidError", "PreparedMoE", "Runtime", "SessionStatus", "Shape", "SourceModality",
    "__version__", "bundled_library", "disable_logging", "enable_logging", "placement", "ranking", "version",
]
