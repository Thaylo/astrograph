"""Language plugin system for multi-language code analysis."""

from ._lsp_base import LSPClient, LSPLanguagePluginBase, LSPPosition, LSPRange, LSPSymbol
from .base import (
    ASTGraph,
    BaseLanguagePlugin,
    CodeUnit,
    LanguagePlugin,
    SemanticProfile,
    SemanticSignal,
    compute_label_histogram,
    node_match,
)
from .lsp_client import SocketLSPClient, SubprocessLSPClient
from .plugin_loader import discover_language_plugins
from .registry import LanguageRegistry

__all__ = [
    "ASTGraph",
    "BaseLanguagePlugin",
    "CodeUnit",
    "LSPClient",
    "LSPLanguagePluginBase",
    "LSPPosition",
    "LSPRange",
    "LSPSymbol",
    "LanguagePlugin",
    "SemanticProfile",
    "SemanticSignal",
    "SocketLSPClient",
    "SubprocessLSPClient",
    "LanguageRegistry",
    "compute_label_histogram",
    "discover_language_plugins",
    "node_match",
]
