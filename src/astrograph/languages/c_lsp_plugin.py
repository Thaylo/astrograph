"""C language adapter using the LSP plugin base."""

from __future__ import annotations

from ._configured_lsp_plugin import ConfiguredLSPLanguagePluginBase


class CLSPPlugin(ConfiguredLSPLanguagePluginBase):
    """C support via an attached or spawned LSP backend."""

    LANGUAGE_ID = "c_lsp"
    LSP_LANGUAGE_ID = "c"
    FILE_EXTENSIONS = frozenset({".c", ".h"})
    SKIP_DIRS = frozenset({"build", "cmake-build-debug", "cmake-build-release"})
    DEFAULT_COMMAND = ("tcp://127.0.0.1:2087",)
    COMMAND_ENV_VAR = "ASTROGRAPH_C_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_C_LSP_TIMEOUT"
