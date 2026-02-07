"""JavaScript language adapter using the LSP plugin base."""

from __future__ import annotations

from ._lsp_base import LSPClient, LSPLanguagePluginBase
from .lsp_client import create_subprocess_client_from_env

_DEFAULT_JS_LSP_COMMAND = ("typescript-language-server", "--stdio")
_JS_LSP_COMMAND_ENV = "ASTROGRAPH_JS_LSP_COMMAND"
_JS_LSP_TIMEOUT_ENV = "ASTROGRAPH_JS_LSP_TIMEOUT"


class JavaScriptLSPPlugin(LSPLanguagePluginBase):
    """JavaScript support via LSP symbols + structural graphing."""

    LANGUAGE_ID = "javascript_lsp"
    LSP_LANGUAGE_ID = "javascript"
    FILE_EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".cjs"})
    SKIP_DIRS = frozenset({"node_modules", ".next", ".nuxt", "coverage"})

    def __init__(self, lsp_client: LSPClient | None = None) -> None:
        super().__init__(
            lsp_client=lsp_client
            or create_subprocess_client_from_env(
                default_command=_DEFAULT_JS_LSP_COMMAND,
                command_env_var=_JS_LSP_COMMAND_ENV,
                timeout_env_var=_JS_LSP_TIMEOUT_ENV,
            )
        )
