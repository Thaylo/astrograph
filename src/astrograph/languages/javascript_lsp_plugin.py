"""JavaScript language adapter using the LSP plugin base."""

from __future__ import annotations

from ._configured_lsp_plugin import ConfiguredLSPLanguagePluginBase


class JavaScriptLSPPlugin(ConfiguredLSPLanguagePluginBase):
    """JavaScript support via LSP symbols + structural graphing."""

    LANGUAGE_ID = "javascript_lsp"
    LSP_LANGUAGE_ID = "javascript"
    FILE_EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".cjs"})
    SKIP_DIRS = frozenset({"node_modules", ".next", ".nuxt", "coverage"})
    DEFAULT_COMMAND = ("typescript-language-server", "--stdio")
    COMMAND_ENV_VAR = "ASTROGRAPH_JS_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_JS_LSP_TIMEOUT"
