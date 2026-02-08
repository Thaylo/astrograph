"""C++ language adapter using the LSP plugin base."""

from __future__ import annotations

from ._configured_lsp_plugin import ConfiguredLSPLanguagePluginBase


class CppLSPPlugin(ConfiguredLSPLanguagePluginBase):
    """C++ support via an attached or spawned LSP backend."""

    LANGUAGE_ID = "cpp_lsp"
    LSP_LANGUAGE_ID = "cpp"
    FILE_EXTENSIONS = frozenset({".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".ipp"})
    SKIP_DIRS = frozenset({"build", "cmake-build-debug", "cmake-build-release"})
    DEFAULT_COMMAND = ("tcp://127.0.0.1:2088",)
    COMMAND_ENV_VAR = "ASTROGRAPH_CPP_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_CPP_LSP_TIMEOUT"
