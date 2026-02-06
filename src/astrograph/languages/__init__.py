"""Language plugin system for multi-language code analysis."""

from .base import (
    ASTGraph,
    BaseLanguagePlugin,
    CodeUnit,
    LanguagePlugin,
    compute_label_histogram,
    node_match,
)
from .registry import LanguageRegistry

__all__ = [
    "ASTGraph",
    "BaseLanguagePlugin",
    "CodeUnit",
    "LanguagePlugin",
    "LanguageRegistry",
    "compute_label_histogram",
    "node_match",
]
