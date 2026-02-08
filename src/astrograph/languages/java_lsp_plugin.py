"""Java language adapter using the LSP plugin base."""

from __future__ import annotations

from ._configured_lsp_plugin import ConfiguredLSPLanguagePluginBase


class JavaLSPPlugin(ConfiguredLSPLanguagePluginBase):
    """Java support via an attached or spawned LSP backend."""

    LANGUAGE_ID = "java_lsp"
    LSP_LANGUAGE_ID = "java"
    FILE_EXTENSIONS = frozenset({".java"})
    SKIP_DIRS = frozenset({"build", "out", "target", ".gradle"})
    DEFAULT_COMMAND = ("tcp://127.0.0.1:2089",)
    COMMAND_ENV_VAR = "ASTROGRAPH_JAVA_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_JAVA_LSP_TIMEOUT"
