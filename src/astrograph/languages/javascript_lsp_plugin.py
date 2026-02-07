"""JavaScript language adapter using the LSP plugin base."""

from __future__ import annotations

import os
import shlex

from ._lsp_base import LSPClient, LSPLanguagePluginBase
from .lsp_client import SubprocessLSPClient

_DEFAULT_JS_LSP_COMMAND = ("typescript-language-server", "--stdio")
_JS_LSP_COMMAND_ENV = "ASTROGRAPH_JS_LSP_COMMAND"
_JS_LSP_TIMEOUT_ENV = "ASTROGRAPH_JS_LSP_TIMEOUT"


def _default_js_lsp_client() -> LSPClient:
    """Create a default LSP client for JavaScript/TypeScript servers."""
    command_text = os.getenv(_JS_LSP_COMMAND_ENV, "")
    command = shlex.split(command_text) if command_text.strip() else list(_DEFAULT_JS_LSP_COMMAND)

    timeout_text = os.getenv(_JS_LSP_TIMEOUT_ENV, "5")
    try:
        timeout = float(timeout_text)
    except ValueError:
        timeout = 5.0

    return SubprocessLSPClient(command, request_timeout=max(timeout, 0.1))


class JavaScriptLSPPlugin(LSPLanguagePluginBase):
    """JavaScript support via LSP symbols + structural graphing."""

    def __init__(self, lsp_client: LSPClient | None = None) -> None:
        super().__init__(lsp_client=lsp_client or _default_js_lsp_client())

    @property
    def language_id(self) -> str:
        return "javascript_lsp"

    @property
    def lsp_language_id(self) -> str:
        return "javascript"

    @property
    def file_extensions(self) -> frozenset[str]:
        return frozenset({".js", ".jsx", ".mjs", ".cjs"})

    @property
    def skip_dirs(self) -> frozenset[str]:
        return frozenset({"node_modules", ".next", ".nuxt", "coverage"})
